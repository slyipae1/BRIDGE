"""Generate resumable per-table descriptive-name artifacts for a SQLite DB.

The artifact contract intentionally matches the cache read by both the public
offline schema-profile loader and the online schema-description fallback:
``preprocessed/ColGrp_artifacts/descriptive_names/{table}.json``.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request
from urllib.parse import urlparse


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "cp1252", "latin1", "iso-8859-1")
DEFAULT_SAMPLE_VALUE_LIMIT = 10
DEFAULT_SAMPLE_VALUE_MAX_CHARS = 100
CACHE_RELATIVE_DIR = Path("preprocessed") / "ColGrp_artifacts" / "descriptive_names"


def quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("commonsense evidence:", "").split())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"could not decode {path}; last error: {last_error}")


def load_schema(sqlite_path: Path) -> dict[str, list[str]]:
    with sqlite3.connect(sqlite_path) as conn:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return {
            str(table_name): [
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({quote_identifier(str(table_name))})").fetchall()
            ]
            for (table_name,) in table_rows
        }


def load_database_descriptions(
    db_dir: Path,
    schema: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, str]]]:
    description_dir = db_dir / "database_description"
    if not description_dir.is_dir():
        return {}
    table_lookup = {table.casefold(): table for table in schema}
    column_lookup = {
        (table.casefold(), column.casefold()): column
        for table, columns in schema.items()
        for column in columns
    }
    result: dict[str, dict[str, dict[str, str]]] = {}
    for csv_path in sorted(description_dir.glob("*.csv")):
        table_name = table_lookup.get(csv_path.stem.casefold(), csv_path.stem)
        result.setdefault(table_name, {})
        for row in _read_csv_rows(csv_path):
            raw_column = _clean_text(row.get("original_column_name"))
            if not raw_column:
                continue
            column_name = column_lookup.get((table_name.casefold(), raw_column.casefold()), raw_column)
            value_description = _clean_text(row.get("value_description"))
            if value_description.casefold().startswith("not useful"):
                value_description = value_description[len("not useful") :].strip()
            result[table_name][column_name] = {
                "column_name": _clean_text(row.get("column_name")),
                "column_description": _clean_text(row.get("column_description")),
                "data_format": _clean_text(row.get("data_format")),
                "value_description": value_description,
            }
    return result


def sample_column_values(
    sqlite_path: Path,
    table_name: str,
    column_name: str,
    *,
    limit: int,
    max_chars: int,
) -> list[str]:
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {quote_identifier(column_name)} FROM {quote_identifier(table_name)} "
            "WHERE " + quote_identifier(column_name) + " IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
    return [str(row[0]) for row in rows if len(str(row[0])) <= max_chars]


def build_table_info(
    *,
    sqlite_path: Path,
    table_name: str,
    columns: list[str],
    descriptions: dict[str, dict[str, dict[str, str]]],
    sample_value_limit: int,
    sample_value_max_chars: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for column_name in columns:
        item: dict[str, Any] = {"column_name": column_name}
        description = descriptions.get(table_name, {}).get(column_name, {})
        expanded_name = _clean_text(description.get("column_name"))
        if expanded_name and expanded_name != column_name:
            item["full_column_name"] = expanded_name
        for key in ("column_description", "value_description", "data_format"):
            value = _clean_text(description.get(key))
            if value:
                item[key] = value
        if item.get("data_format"):
            item["data_type"] = item["data_format"]
        samples = sample_column_values(
            sqlite_path,
            table_name,
            column_name,
            limit=sample_value_limit,
            max_chars=sample_value_max_chars,
        )
        if samples:
            item["sample_values"] = samples
        items.append(item)
    return {"table_name": table_name, "columns": items}


def render_prompt(template: str, *, table_name: str, table_info: dict[str, Any]) -> str:
    return (
        template.replace("{table_name}", table_name)
        .replace("{table_info}", json.dumps(table_info, ensure_ascii=False, indent=2))
    )


def _is_private_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").casefold()
    if host in {"localhost", "127.0.0.1"} or host.endswith(".local"):
        return True
    parts = host.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return False
    first, second = int(parts[0]), int(parts[1])
    return first == 10 or first == 192 and second == 168 or first == 172 and 16 <= second <= 31


def call_chat_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    thinking_control: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking_control in {"local_qwen", "both"}:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if thinking_control in {"dashscope", "both"}:
        payload["enable_thinking"] = False
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    opener = request.build_opener(request.ProxyHandler({})) if _is_private_base_url(base_url) else request.build_opener()
    req = request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with opener.open(req, timeout=180) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
    choices = raw_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("chat response does not contain a choice")
    content = str(((choices[0].get("message") or {}).get("content")) or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    start = content.find("{")
    if start < 0:
        raise ValueError("chat response contains no JSON object")
    parsed, _ = decoder.raw_decode(content[start:])
    if not isinstance(parsed, dict):
        raise ValueError("chat response JSON must be an object")
    return parsed, raw_response


def validate_response(response: dict[str, Any], *, table_name: str, columns: list[str]) -> dict[str, Any]:
    table_descriptive_name = _clean_text(response.get("table_descriptive_name"))
    if not table_descriptive_name:
        raise ValueError(f"{table_name}: missing table_descriptive_name")
    expected = {column.casefold(): column for column in columns}
    returned: dict[str, str] = {}
    for item in response.get("columns") or []:
        if not isinstance(item, dict):
            raise ValueError(f"{table_name}: column response entry is not an object")
        raw_column = _clean_text(item.get("column_name"))
        actual_column = expected.get(raw_column.casefold())
        descriptive_name = _clean_text(item.get("descriptive_name"))
        if not actual_column:
            raise ValueError(f"{table_name}: unexpected column {raw_column!r}")
        if not descriptive_name:
            raise ValueError(f"{table_name}.{actual_column}: empty descriptive_name")
        if actual_column in returned:
            raise ValueError(f"{table_name}.{actual_column}: duplicate response entry")
        returned[actual_column] = descriptive_name
    missing = [column for column in columns if column not in returned]
    if missing:
        raise ValueError(f"{table_name}: missing descriptive names for {missing[:5]}")
    return {
        "table_descriptive_name": table_descriptive_name,
        "columns": [
            {"column_name": column, "descriptive_name": returned[column]}
            for column in columns
        ],
    }


def artifact_path(db_dir: Path, table_name: str, *, output_dir: Path | None = None) -> Path:
    return (output_dir or db_dir / CACHE_RELATIVE_DIR) / f"{table_name}.json"


def load_valid_artifact(path: Path, *, table_name: str, columns: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok" or not isinstance(payload.get("response"), dict):
            return None
        validate_response(payload["response"], table_name=table_name, columns=columns)
        return payload
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        return None


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_chat_config(
    *,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str, str]:
    resolved_model = model or os.getenv("MODEL_NAME") or os.getenv("CHAT_MODEL_NAME")
    resolved_base_url = base_url or os.getenv("BASE_URL") or os.getenv("CHAT_BASE_URL")
    resolved_api_key = api_key or os.getenv("API_KEY") or os.getenv("CHAT_API_KEY") or "local-token"
    if not resolved_model or not resolved_base_url:
        raise ValueError("model and base URL are required through CLI or MODEL_NAME/BASE_URL environment variables")
    return resolved_model, resolved_base_url, resolved_api_key


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
