from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import database_manager as database_manager_module
from .database_manager import DatabaseManager
from .dbelement_options import get_DBeleOptions
from ...module_a_base.retrieval_ablation import (
    DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    validate_retrieval_ablation_disable_channel,
)
from ...module_a_base.config import DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE
from ...telemetry import Timer, stage_timing


def _configure_db_env(db_root_path: str | Path, db_mode: str) -> Path:
    root = Path(db_root_path).expanduser().resolve()
    os.environ["DB_ROOT_PATH"] = str(root)
    os.environ["DB_ROOT_DIRECTORY"] = str(root / f"{db_mode}_databases")
    database_manager_module.DB_ROOT_PATH = root
    return root


def _schema_string(
    *,
    tentative_schema: dict[str, list[str]],
    schema_with_examples: dict[str, Any],
    schema_with_descriptions: dict[str, Any],
) -> tuple[str, str | None]:
    try:
        schema_string = DatabaseManager().get_database_schema_string(
            tentative_schema=tentative_schema,
            schema_with_examples=schema_with_examples,
            schema_with_descriptions=schema_with_descriptions,
            include_value_description=True,
        )
        return schema_string, None
    except Exception as exc:
        return "", f"schema_string_generation_failed: {exc}"


def retrieve_sql_full(
    *,
    question_id: int,
    db_id: str,
    question: str,
    evidence: str,
    current_sql: str,
    db_root_path: str | Path,
    db_mode: str = "dev",
    lsh_top_n: int = 20,
    enable_llm_fallback: bool = True,
    column_group_version: str = "manual",
    column_group_artifact_root: str | Path | None = None,
    retrieval_ablation_disable_channel: str = DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    column_retrieval_source: str = DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
) -> dict[str, Any]:
    """Return one cache-shaped question record with db_retrieval_sql_full.

    Version 0 uses current_sql as the only retrieval anchor. question/evidence
    are carried for audit and future versions, not used for recall.
    """
    retrieval_timer = Timer("module_a_retrieval_question")
    _configure_db_env(db_root_path, db_mode)
    DatabaseManager(db_mode=db_mode, db_id=db_id)
    retrieval_ablation_disable_channel = validate_retrieval_ablation_disable_channel(
        retrieval_ablation_disable_channel
    )

    old_flag = os.environ.get("MODULE_A_ENABLE_LLM_FALLBACK")
    os.environ["MODULE_A_ENABLE_LLM_FALLBACK"] = "1" if enable_llm_fallback else "0"
    try:
        (
            dbelement_options,
            tentative_schema,
            schema_with_examples,
            schema_with_descriptions,
            sql_parse_meta,
            retrieval_ablation,
        ) = get_DBeleOptions(
            sql=current_sql,
            db_id=db_id,
            LSH_top_n=lsh_top_n,
            column_group_version=column_group_version,
            column_group_artifact_root=str(column_group_artifact_root) if column_group_artifact_root else None,
            question_id=question_id,
            return_sql_parse_meta=True,
            retrieval_ablation_disable_channel=retrieval_ablation_disable_channel,
            return_retrieval_ablation=True,
            column_retrieval_source=column_retrieval_source,
        )
    finally:
        if old_flag is None:
            os.environ.pop("MODULE_A_ENABLE_LLM_FALLBACK", None)
        else:
            os.environ["MODULE_A_ENABLE_LLM_FALLBACK"] = old_flag

    schema_string, warning = _schema_string(
        tentative_schema=tentative_schema,
        schema_with_examples=schema_with_examples,
        schema_with_descriptions=schema_with_descriptions,
    )

    record: dict[str, Any] = {
        "question_id": int(question_id),
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "base_sql": current_sql,
        "db_retrieval_sql_full": {
            "source_scope": "question_sql",
            "source_group_id": None,
            "source_period_id": None,
            "dbelement_options": dbelement_options,
            "schema_string": schema_string,
            "sql_parse_meta": sql_parse_meta,
            "retrieval_ablation": retrieval_ablation,
            "retrieval_timing": stage_timing(
                retrieval_timer,
                {
                    "question_id": int(question_id),
                    "db_id": db_id,
                    "column_group_version": column_group_version,
                    "column_retrieval_source": column_retrieval_source,
                    "retrieval_ablation_disable_channel": retrieval_ablation_disable_channel,
                },
            ),
        },
    }
    if warning:
        record["warnings"] = [warning]
    return record
