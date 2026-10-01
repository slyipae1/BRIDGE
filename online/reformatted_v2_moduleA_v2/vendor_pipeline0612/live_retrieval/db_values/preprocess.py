from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from datasketch import MinHashLSH
except ImportError:  # pragma: no cover - depends on local experiment env
    MinHashLSH = None

from ..execution import execute_sql
from .minhash import _create_minhash


SKIPPED_COLUMN_KEYWORDS = (
    "_id",
    " id",
    "url",
    "email",
    "web",
    "time",
    "phone",
    "date",
    "address",
)

SHORT_STRING_BEHAVIOR = "v2_whole_string_update_when_len_lt_n_gram"


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _datasketch_version() -> str | None:
    try:
        return metadata.version("datasketch")
    except metadata.PackageNotFoundError:
        return None


def _artifact_paths(preprocessed_path: Path, db_id: str) -> Dict[str, Path]:
    return {
        "unique_values": preprocessed_path / f"{db_id}_unique_values.pkl",
        "lsh": preprocessed_path / f"{db_id}_lsh.pkl",
        "minhashes": preprocessed_path / f"{db_id}_minhashes.pkl",
        "manifest": preprocessed_path / f"{db_id}_lsh_manifest.json",
    }


def get_unique_text_values(
    db_path: str | Path,
    *,
    timeout: int = 480,
) -> Dict[str, Dict[str, List[str]]]:
    """Collect distinct text values using the DeltaRefinement LSH filters."""
    db_path = Path(db_path)
    table_names = [
        table[0]
        for table in execute_sql(
            str(db_path),
            "SELECT name FROM sqlite_master WHERE type='table';",
            fetch="all",
        )
    ]

    primary_keys: List[str] = []
    for table_name in table_names:
        columns = execute_sql(
            str(db_path),
            f"PRAGMA table_info('{table_name}')",
            fetch="all",
        )
        for column in columns:
            if column[5] > 0:
                column_name = column[1]
                if column_name.lower() not in [c.lower() for c in primary_keys]:
                    primary_keys.append(column_name)

    primary_key_lower = {column.lower() for column in primary_keys}
    unique_values: Dict[str, Dict[str, List[str]]] = {}

    for table_name in table_names:
        if table_name == "sqlite_sequence":
            continue
        logging.info("Processing %s", table_name)
        columns = execute_sql(
            str(db_path),
            f"PRAGMA table_info('{table_name}')",
            fetch="all",
        )
        text_columns = [
            col[1]
            for col in columns
            if "TEXT" in str(col[2]).upper()
            and col[1].lower() not in primary_key_lower
        ]

        table_values: Dict[str, List[str]] = {}
        for column_name in text_columns:
            column_lower = column_name.lower()
            if (
                any(keyword in column_lower for keyword in SKIPPED_COLUMN_KEYWORDS)
                or column_name.endswith("Id")
            ):
                continue

            table_sql = _quote_identifier(table_name)
            column_sql = _quote_identifier(column_name)
            try:
                result = execute_sql(
                    str(db_path),
                    f"""
                    SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                    FROM (
                        SELECT DISTINCT {column_sql} AS unique_values
                        FROM {table_sql}
                        WHERE {column_sql} IS NOT NULL
                    ) AS subquery
                    """,
                    fetch="one",
                    timeout=timeout,
                )
            except Exception:
                result = (0, 0)

            sum_of_lengths, count_distinct = result
            if sum_of_lengths is None or count_distinct == 0:
                continue

            average_length = sum_of_lengths / count_distinct
            logging.info(
                "Column: %s, sum_of_lengths: %s, count_distinct: %s, average_length: %s",
                column_name,
                sum_of_lengths,
                count_distinct,
                average_length,
            )

            if (
                ("name" in column_lower and sum_of_lengths < 5_000_000)
                or (sum_of_lengths < 2_000_000 and average_length < 25)
                or count_distinct < 100
            ):
                logging.info("Fetching distinct values for %s", column_name)
                try:
                    rows = execute_sql(
                        str(db_path),
                        f"SELECT DISTINCT {column_sql} FROM {table_sql} WHERE {column_sql} IS NOT NULL",
                        fetch="all",
                        timeout=timeout,
                    )
                    table_values[column_name] = [str(value[0]) for value in rows]
                except Exception:
                    table_values[column_name] = []
                logging.info(
                    "Number of different values for %s.%s: %d",
                    table_name,
                    column_name,
                    len(table_values[column_name]),
                )

        unique_values[table_name] = table_values

    return unique_values


def make_lsh(
    unique_values: Dict[str, Dict[str, List[str]]],
    *,
    signature_size: int,
    n_gram: int,
    threshold: float,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Tuple[Any, str, str, str]]]:
    """Create a MinHash LSH from extracted unique DB values."""
    if MinHashLSH is None:
        raise ImportError("datasketch is not installed; cannot build LSH value index")

    lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    minhashes: Dict[str, Tuple[Any, str, str, str]] = {}
    total_unique_values = sum(
        len(column_values)
        for table_values in unique_values.values()
        for column_values in table_values.values()
    )
    if verbose:
        logging.info("Total unique values: %d", total_unique_values)

    for table_name, table_values in unique_values.items():
        for column_name, column_values in table_values.items():
            if verbose:
                logging.info(
                    "Processing %s - %s - %d",
                    table_name,
                    column_name,
                    len(column_values),
                )
            for value_id, value in enumerate(column_values):
                minhash = _create_minhash(signature_size, value, n_gram)
                minhash_key = f"{table_name}_{column_name}_{value_id}"
                minhashes[minhash_key] = (minhash, table_name, column_name, value)
                lsh.insert(minhash_key, minhash)

    return lsh, minhashes


def build_db_lsh(
    db_directory_path: str | Path,
    *,
    signature_size: int,
    n_gram: int,
    threshold: float,
    verbose: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build and persist v2-compatible LSH artifacts for one DB directory."""
    db_directory_path = Path(db_directory_path)
    db_id = db_directory_path.name
    db_path = db_directory_path / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    preprocessed_path = db_directory_path / "preprocessed"
    preprocessed_path.mkdir(exist_ok=True)
    artifact_paths = _artifact_paths(preprocessed_path, db_id)
    existing = [path for path in artifact_paths.values() if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"LSH artifacts already exist for {db_id}; pass overwrite=True to replace: {existing_text}"
        )

    unique_values = get_unique_text_values(db_path)
    lsh, minhashes = make_lsh(
        unique_values,
        signature_size=signature_size,
        n_gram=n_gram,
        threshold=threshold,
        verbose=verbose,
    )

    with artifact_paths["unique_values"].open("wb") as file:
        pickle.dump(unique_values, file)
    with artifact_paths["lsh"].open("wb") as file:
        pickle.dump(lsh, file)
    with artifact_paths["minhashes"].open("wb") as file:
        pickle.dump(minhashes, file)

    table_count = len(unique_values)
    column_count = sum(len(table_values) for table_values in unique_values.values())
    unique_value_count = sum(
        len(column_values)
        for table_values in unique_values.values()
        for column_values in table_values.values()
    )
    manifest: Dict[str, Any] = {
        "db_id": db_id,
        "db_path": str(db_path),
        "signature_size": signature_size,
        "n_gram": n_gram,
        "threshold": threshold,
        "unique_value_count": unique_value_count,
        "table_count": table_count,
        "column_count": column_count,
        "artifact_names": {
            key: path.name for key, path in artifact_paths.items() if key != "manifest"
        },
        "manifest_name": artifact_paths["manifest"].name,
        "short_string_behavior": SHORT_STRING_BEHAVIOR,
        "datasketch_version": _datasketch_version(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest
