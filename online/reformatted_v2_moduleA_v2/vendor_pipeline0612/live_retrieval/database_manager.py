import os
import pickle
import logging
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
from typing import Any, Dict, List, Callable

from .schema import DatabaseSchema
from .schema_generator import DatabaseSchemaGenerator
from .execution import execute_sql, compare_sqls, validate_sql_query, aggregate_sqls, get_execution_status
from .db_info import get_db_all_tables, get_table_all_columns, get_db_schema
from .db_values.config import LSH_N_GRAM, LSH_SIGNATURE_SIZE
from .sql_parser import get_sql_tables, get_sql_columns_dict, get_sql_condition_literals

# Keep user-provided API/DB environment variables authoritative. The repo .env is
# useful for defaults, but must not silently redirect live Module A retrieval.
load_dotenv(override=False)
DB_ROOT_PATH = Path(os.getenv("DB_ROOT_PATH", ""))


class DatabaseManager:
    """Singleton per (db_mode, db_id) for database operations.

    Adapted from DeltaRefinement runner/database_manager.py.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, db_mode=None, db_id=None):
        if db_mode is not None and db_id is not None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._init(db_mode, db_id)
                elif cls._instance.db_id != db_id:
                    cls._instance._init(db_mode, db_id)
                return cls._instance
        else:
            if cls._instance is None:
                raise ValueError("DatabaseManager instance has not been initialized.")
            return cls._instance

    def _init(self, db_mode: str, db_id: str):
        self.db_mode = db_mode
        self.db_id = db_id
        self._set_paths()
        self.lsh = None
        self.minhashes = None
        self.vector_db = None
        self.descriptive_names_vector_db = None
        self.column_groups_cache = None
        self.column_groups_cache_by_version = {}

    def _set_paths(self):
        db_root_raw = os.getenv("DB_ROOT_PATH", "")
        db_root_path = Path(db_root_raw) if db_root_raw else DB_ROOT_PATH
        self.db_path = db_root_path / f"{self.db_mode}_databases" / self.db_id / f"{self.db_id}.sqlite"
        self.db_directory_path = db_root_path / f"{self.db_mode}_databases" / self.db_id

    def get_db_path(self) -> Path:
        return self.db_path

    def set_lsh(self) -> str:
        with self._lock:
            if self.lsh is None:
                try:
                    with (self.db_directory_path / "preprocessed" / f"{self.db_id}_lsh.pkl").open("rb") as f:
                        self.lsh = pickle.load(f)
                    with (self.db_directory_path / "preprocessed" / f"{self.db_id}_minhashes.pkl").open("rb") as f:
                        self.minhashes = pickle.load(f)
                    return "success"
                except Exception as e:
                    self.lsh = "error"
                    self.minhashes = "error"
                    logging.warning(f"Error loading LSH for {self.db_id}: {e}")
                    return "error"
            elif self.lsh == "error":
                return "error"
            return "success"

    def query_lsh(
        self,
        keyword: str,
        signature_size: int = LSH_SIGNATURE_SIZE,
        n_gram: int = LSH_N_GRAM,
        top_n: int = 10,
    ) -> Dict[str, List[str]]:
        from .db_values.search import query_lsh
        status = self.set_lsh()
        if status == "success":
            return query_lsh(self.lsh, self.minhashes, keyword, signature_size, n_gram, top_n)
        raise Exception(f"LSH not available for {self.db_id}")

    def load_column_groups(
        self,
        version: str = "manual",
        artifact_root: str | Path | None = None,
    ) -> Dict[str, Any]:
        if version not in {None, "manual"}:
            raise ValueError("The public runtime supports only manual column-group archives.")
        root_key = str(Path(artifact_root).expanduser()) if artifact_root else ""
        cache_key = ("manual", root_key)
        if cache_key in self.column_groups_cache_by_version:
            return self.column_groups_cache_by_version[cache_key]
        try:
            suffix = "_column_groups_manual.pkl"
            if artifact_root:
                root = Path(artifact_root).expanduser()
                candidate_paths = [
                    root / self.db_id / "preprocessed" / f"{self.db_id}{suffix}",
                    root / self.db_id / f"{self.db_id}{suffix}",
                    root / "preprocessed" / f"{self.db_id}{suffix}",
                    root / f"{self.db_id}{suffix}",
                ]
            else:
                candidate_paths = [self.db_directory_path / "preprocessed" / f"{self.db_id}{suffix}"]
            groups_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
            with groups_path.open("rb") as f:
                groups = pickle.load(f)
            self.column_groups_cache_by_version[cache_key] = groups
            return groups
        except Exception as e:
            logging.warning(
                f"Failed to load manual column groups for {self.db_id} "
                f"from {artifact_root or self.db_directory_path}: {e}"
            )
            return {}

    def get_database_schema_string(
        self, tentative_schema: Dict[str, List[str]],
        schema_with_examples: Dict,
        schema_with_descriptions: Dict,
        include_value_description: bool = True,
    ) -> str:
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
            schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples) if schema_with_examples else None,
            schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions) if schema_with_descriptions else None,
            db_id=self.db_id,
            db_path=self.db_path,
        )
        schema_ddl_str = schema_generator.generate_schema_string(include_value_description=include_value_description)

        # Prepend FK and PK annotations before the DDL
        if tentative_schema:
            try:
                from .schema_constraints import format_foreign_keys, format_primary_keys
                table_names = list(tentative_schema.keys())
                fk_str = format_foreign_keys(table_names, self.db_id)
                pk_str = format_primary_keys(table_names, self.db_id)
                extra_parts = []
                if fk_str:
                    extra_parts.append(fk_str)
                if pk_str:
                    extra_parts.append(pk_str)
                if extra_parts:
                    schema_ddl_str = "\n\n".join(extra_parts) + "\n\n" + schema_ddl_str
            except Exception as e:
                logging.warning(f"Failed to append FK/PK info to schema: {e}")

        return schema_ddl_str

    def add_connections_to_tentative_schema(self, tentative_schema: Dict[str, List[str]]) -> Dict[str, List[str]]:
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
            db_id=self.db_id,
            db_path=self.db_path,
        )
        return schema_generator.get_schema_with_connections()

    @staticmethod
    def with_db_path(func: Callable):
        def wrapper(self, *args, **kwargs):
            return func(self.db_path, *args, **kwargs)
        return wrapper

    @classmethod
    def add_methods_to_class(cls, funcs: List[Callable]):
        for func in funcs:
            method = cls.with_db_path(func)
            setattr(cls, func.__name__, method)


# Bind utility functions as methods
functions_to_add = [
    execute_sql,
    compare_sqls,
    validate_sql_query,
    aggregate_sqls,
    get_db_all_tables,
    get_table_all_columns,
    get_db_schema,
    get_sql_tables,
    get_sql_columns_dict,
    get_sql_condition_literals,
    get_execution_status,
]
DatabaseManager.add_methods_to_class(functions_to_add)
