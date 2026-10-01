from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION = "realtime_column_retrieval_v1"


def build_realtime_column_retrieval_prompt(
    *,
    question: str,
    evidence: str,
    target_column: str,
    schema_ddl: str,
) -> str:
    """Build the schema-level, per-COLUMN realtime retrieval prompt."""
    evidence = str(evidence or "").strip()
    evidence_section = f"## Evidence\n{evidence}\n\n" if evidence else ""
    return """You are identifying ambiguous columns for a target column in a text-to-SQL task.

Given one target column extracted from the current SQL query, the user question,
and the complete database schema, identify every other column that could plausibly
be confused with the target under the user request.

For every candidate, write an ambiguity_reason in this form:
\"when requested for <ambiguity situation> without specifying <distinguishing dimension>\"

Use only the supplied schema. Do not invent tables or columns.
Do not return the target column itself.
Return an empty options list if you cannot find any candidate.

Return exactly one JSON object and no markdown:

{{
  \"target\": \"exact target column from the input\",
  \"options\": [
    {{
      \"column\": \"table.column\",
      \"ambiguity_reason\": \"when requested for ... without specifying ...\"
    }}
  ]
}}

## User Question
{question}

{evidence_section}## Target Column
{target_column}

## Database Schema
{schema_ddl}
""".format(
        question=str(question or "").strip(),
        evidence_section=evidence_section,
        target_column=str(target_column or "").strip(),
        schema_ddl=str(schema_ddl or "").strip(),
    )


def load_schema_column_map(db_path: str | Path) -> dict[str, str]:
    """Return case-insensitive ``table.column`` -> DB-spelled references."""
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {path}")

    column_map: dict[str, str] = {}
    connection = sqlite3.connect(str(path))
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchall()
        for (table_name,) in table_rows:
            table_name = str(table_name)
            escaped_table = table_name.replace('"', '""')
            for row in connection.execute(f'PRAGMA table_info("{escaped_table}")').fetchall():
                column_name = str(row[1])
                reference = f"{table_name}.{column_name}"
                column_map[reference.casefold()] = reference
    finally:
        connection.close()
    return column_map


def parse_realtime_column_retrieval_response(
    content: str,
    *,
    target_column: str,
    schema_column_map: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one LLM response and convert it to a canonical COLUMN entry."""
    payload = _parse_json_object(content)
    returned_target = str(payload.get("target") or "").strip()
    target_column = str(target_column or "").strip()
    if returned_target != target_column:
        raise ValueError(
            "realtime retrieval target mismatch: "
            f"expected {target_column!r}, received {returned_target!r}"
        )
    raw_options = payload.get("options")
    if not isinstance(raw_options, list):
        raise ValueError("realtime retrieval response field 'options' must be a JSON list")

    target_key = target_column.casefold()
    candidates: list[dict[str, Any]] = []
    seen = {target_key}
    audit = {
        "raw_option_count": len(raw_options),
        "accepted_option_count": 0,
        "dropped_self_count": 0,
        "dropped_duplicate_count": 0,
        "dropped_unknown_column_count": 0,
        "dropped_missing_reason_count": 0,
        "dropped_malformed_option_count": 0,
    }
    target_table = target_column.split(".", 1)[0].casefold() if "." in target_column else ""

    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            audit["dropped_malformed_option_count"] += 1
            continue
        raw_column = str(raw_option.get("column") or "").strip()
        reason = str(raw_option.get("ambiguity_reason") or "").strip()
        if not raw_column:
            audit["dropped_malformed_option_count"] += 1
            continue
        canonical_column = schema_column_map.get(raw_column.casefold())
        if not canonical_column:
            audit["dropped_unknown_column_count"] += 1
            continue
        candidate_key = canonical_column.casefold()
        if candidate_key == target_key:
            audit["dropped_self_count"] += 1
            continue
        if candidate_key in seen:
            audit["dropped_duplicate_count"] += 1
            continue
        if not reason:
            audit["dropped_missing_reason_count"] += 1
            continue
        seen.add(candidate_key)
        candidate_table = canonical_column.split(".", 1)[0].casefold() if "." in canonical_column else ""
        candidates.append(
            {
                "element": canonical_column,
                "eleType": "COLUMN",
                "anchor_column": target_column,
                "ambiguity_reason": reason,
                "reasons": "realtime schema-reasoning candidate",
                "different_table": bool(target_table and candidate_table and target_table != candidate_table),
                "retrieval_source": "realtime_reasoning",
            }
        )

    audit["accepted_option_count"] = len(candidates)
    entry: dict[str, Any] = {
        "entity": "realtime_reasoning",
        "type": "COLUMN",
        "anchor_column": target_column,
        "ambiguity_reason_concise_label": "",
        "retrieval_source": "realtime_reasoning",
        "runtime_policy": {"requires_downstream_llm_reasoning": True},
        "options": [
            {
                "element": target_column,
                "eleType": "COLUMN",
                "anchor_column": target_column,
                "ambiguity_reason": "",
                "reasons": "current SQL anchor",
                "different_table": False,
                "retrieval_source": "realtime_reasoning_self",
            },
            *candidates,
        ],
    }
    return entry, audit


def _parse_json_object(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"realtime retrieval response is not a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("realtime retrieval response must be a JSON object")
    return payload
