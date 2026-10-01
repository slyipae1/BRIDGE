from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "big5",
    "cp1252",
    "latin1",
    "iso-8859-1",
)

DEFAULT_SAMPLE_VALUE_LIMIT = 5
DEFAULT_SAMPLE_VALUE_MAX_CHARS = 120


def quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _clean_description_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("commonsense evidence:", "").strip()


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with Path(csv_path).open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"could not decode {csv_path} with encodings {CSV_ENCODINGS}; last error={last_error}",
    )


def load_database_descriptions(db_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    description_dir = Path(db_dir) / "database_description"
    descriptions: dict[str, dict[str, dict[str, str]]] = {}
    if not description_dir.exists():
        return descriptions
    for csv_path in sorted(description_dir.glob("*.csv")):
        table_name = csv_path.stem
        descriptions.setdefault(table_name, {})
        for row in _read_csv_rows(csv_path):
            original = _clean_description_cell(row.get("original_column_name"))
            if not original:
                continue
            descriptions[table_name][original] = {
                "column_name": _clean_description_cell(row.get("column_name")),
                "column_description": _clean_description_cell(row.get("column_description")),
                "data_format": _clean_description_cell(row.get("data_format")),
                "value_description": _clean_description_cell(row.get("value_description")),
            }
    return descriptions


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def infer_value_mode(
    column_name: str,
    data_type: str,
    distinct_count: int,
    distinct_ratio: float,
    normalized_small_values: list[str],
) -> str:
    name = str(column_name).casefold()
    dtype = str(data_type).casefold()
    small_value_set = {str(value).strip().casefold() for value in normalized_small_values if str(value).strip()}
    boolean_sets = [
        {"0", "1"},
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
    ]
    if small_value_set and any(small_value_set <= allowed for allowed in boolean_sets):
        return "BOOLEAN_LIKE"
    if any(token in name for token in ("uuid", " id", "_id", "code", "key")) or name.endswith("id"):
        return "CODE_OR_IDENTIFIER"
    if "date" in name or "date" in dtype or "time" in dtype:
        return "DATE_LIKE"
    if any(token in dtype for token in ("int", "real", "numeric", "double", "float", "decimal")):
        return "NUMERIC_MEASURE"
    if distinct_count <= 10 and distinct_count >= 2:
        return "LOW_CARD_DOMAIN"
    if distinct_ratio >= 0.95 and distinct_count >= 20:
        return "CODE_OR_IDENTIFIER"
    if "text" in dtype or "char" in dtype or not dtype:
        return "LABEL_TEXT"
    return "OTHER"


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _casefold_lookup(values: dict[str, Any]) -> dict[str, tuple[str, Any]]:
    lookup: dict[str, tuple[str, Any]] = {}
    for key, value in values.items():
        normalized = str(key).strip().casefold()
        if normalized:
            lookup[normalized] = (str(key), value)
    return lookup


def _non_empty_name(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _validate_descriptive_name_coverage(
    *,
    table_descriptive_names: dict[str, str],
    column_descriptive_names: dict[str, str],
    schema_columns: dict[str, list[str]],
    source_kind: str,
    source_path: Path,
    strict: bool,
) -> dict[str, Any]:
    expected_tables = sorted(schema_columns)
    expected_columns = sorted(
        f"{table_name}.{column_name}"
        for table_name, columns in schema_columns.items()
        for column_name in columns
    )
    missing_tables = [
        table_name
        for table_name in expected_tables
        if not _non_empty_name(table_descriptive_names.get(table_name))
    ]
    missing_columns = [
        column_ref
        for column_ref in expected_columns
        if not _non_empty_name(column_descriptive_names.get(column_ref))
    ]
    unexpected_tables = sorted(set(table_descriptive_names) - set(expected_tables))
    unexpected_columns = sorted(set(column_descriptive_names) - set(expected_columns))
    coverage = {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "strict": bool(strict),
        "expected_table_count": len(expected_tables),
        "loaded_table_count": len(table_descriptive_names),
        "expected_column_count": len(expected_columns),
        "loaded_column_count": len(column_descriptive_names),
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "unexpected_tables": unexpected_tables,
        "unexpected_columns": unexpected_columns,
    }
    if strict and (missing_tables or missing_columns or unexpected_tables or unexpected_columns):
        raise ValueError(
            "descriptive-name coverage mismatch "
            f"for {source_path}: missing_tables={missing_tables[:5]} "
            f"missing_columns={missing_columns[:5]} "
            f"unexpected_tables={unexpected_tables[:5]} "
            f"unexpected_columns={unexpected_columns[:5]}"
        )
    return coverage


def load_descriptive_name_artifacts(
    artifact_dir: Path,
    schema_columns: dict[str, list[str]],
    *,
    strict: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Load BIRD's per-table descriptive-name artifacts against a SQLite schema."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"descriptive-name artifact directory does not exist: {artifact_dir}")

    table_descriptive_names: dict[str, str] = {}
    column_descriptive_names: dict[str, str] = {}
    for table_name, columns in sorted(schema_columns.items()):
        artifact_path = artifact_dir / f"{table_name}.json"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"missing descriptive-name artifact for table {table_name}: {artifact_path}"
            )
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        response = payload.get("response")
        if payload.get("status") != "ok" or not isinstance(response, dict):
            raise ValueError(f"invalid descriptive-name artifact for table {table_name}: {artifact_path}")

        table_descriptive_name = _non_empty_name(response.get("table_descriptive_name"))
        if not table_descriptive_name:
            raise ValueError(f"empty table descriptive name for {table_name}: {artifact_path}")
        table_descriptive_names[table_name] = table_descriptive_name

        actual_by_key = {str(column_name).casefold(): str(column_name) for column_name in columns}
        seen_columns: set[str] = set()
        for item in response.get("columns") or []:
            if not isinstance(item, dict):
                raise ValueError(f"invalid column entry for {table_name}: {artifact_path}")
            artifact_column = _non_empty_name(item.get("column_name"))
            actual_column = actual_by_key.get(artifact_column.casefold())
            if not actual_column:
                raise ValueError(
                    f"unexpected descriptive-name column {artifact_column!r} for {table_name}: {artifact_path}"
                )
            descriptive_name = _non_empty_name(item.get("descriptive_name"))
            if not descriptive_name:
                raise ValueError(
                    f"empty column descriptive name for {table_name}.{actual_column}: {artifact_path}"
                )
            if actual_column in seen_columns:
                raise ValueError(
                    f"duplicate descriptive-name column {table_name}.{actual_column}: {artifact_path}"
                )
            seen_columns.add(actual_column)
            column_descriptive_names[f"{table_name}.{actual_column}"] = descriptive_name

    coverage = _validate_descriptive_name_coverage(
        table_descriptive_names=table_descriptive_names,
        column_descriptive_names=column_descriptive_names,
        schema_columns=schema_columns,
        source_kind="bird_per_table_artifacts",
        source_path=artifact_dir,
        strict=strict,
    )
    return table_descriptive_names, column_descriptive_names, coverage


def load_legacy_descriptive_name_map(
    path: Path,
    schema_columns: dict[str, list[str]],
    *,
    strict: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Load an explicitly requested legacy aggregate mapping with the same contract."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tables = payload.get("table_descriptive_names") or {}
    raw_columns = payload.get("column_descriptive_names") or {}
    if not isinstance(raw_tables, dict) or not isinstance(raw_columns, dict):
        raise ValueError(f"legacy descriptive-name map is malformed: {path}")

    table_lookup = _casefold_lookup(raw_tables)
    column_lookup = _casefold_lookup(raw_columns)
    table_descriptive_names: dict[str, str] = {}
    column_descriptive_names: dict[str, str] = {}
    for table_name, columns in sorted(schema_columns.items()):
        table_match = table_lookup.get(table_name.casefold())
        if table_match:
            table_descriptive_names[table_name] = _non_empty_name(table_match[1])
        for column_name in columns:
            column_ref = f"{table_name}.{column_name}"
            column_match = column_lookup.get(column_ref.casefold())
            if column_match:
                column_descriptive_names[column_ref] = _non_empty_name(column_match[1])

    coverage = _validate_descriptive_name_coverage(
        table_descriptive_names=table_descriptive_names,
        column_descriptive_names=column_descriptive_names,
        schema_columns=schema_columns,
        source_kind="legacy_aggregate_map",
        source_path=path,
        strict=strict,
    )
    return table_descriptive_names, column_descriptive_names, coverage


def get_unique_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    table_sql = quote_identifier(table_name)
    unique_columns: set[str] = set()
    for index_row in conn.execute(f"PRAGMA index_list({table_sql})").fetchall():
        if not bool(index_row[2]):
            continue
        index_name = str(index_row[1])
        index_columns = conn.execute(f"PRAGMA index_info({quote_identifier(index_name)})").fetchall()
        if len(index_columns) == 1:
            unique_columns.add(str(index_columns[0][2]))
    return unique_columns


def get_single_primary_key_column(conn: sqlite3.Connection, table_name: str) -> str | None:
    table_sql = quote_identifier(table_name)
    primary_key_columns = [
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_sql})").fetchall()
        if int(row[5] or 0) > 0
    ]
    if len(primary_key_columns) == 1:
        return primary_key_columns[0]
    return None


def resolve_fk_target_column(conn: sqlite3.Connection, target_table: str, raw_target_column: Any) -> str | None:
    if raw_target_column is not None:
        target_column = str(raw_target_column).strip()
        if target_column and target_column.casefold() != "none":
            return target_column
    return get_single_primary_key_column(conn, target_table)


def get_foreign_keys_by_column(conn: sqlite3.Connection, table_name: str) -> dict[str, list[dict[str, str]]]:
    table_sql = quote_identifier(table_name)
    foreign_keys: dict[str, list[dict[str, str]]] = {}
    for row in conn.execute(f"PRAGMA foreign_key_list({table_sql})").fetchall():
        target_table = str(row[2])
        target_column = resolve_fk_target_column(conn, target_table, row[4])
        foreign_keys.setdefault(str(row[3]), []).append(
            {
                "table": target_table,
                "from": str(row[3]),
                "to": target_column,
                "on_update": str(row[5]),
                "on_delete": str(row[6]),
                "match": str(row[7]),
            }
        )
    return foreign_keys


def compute_column_sample_values(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    *,
    limit: int = DEFAULT_SAMPLE_VALUE_LIMIT,
    max_chars: int = DEFAULT_SAMPLE_VALUE_MAX_CHARS,
) -> list[str]:
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column_name)
    rows = conn.execute(
        f"SELECT DISTINCT CAST({column_sql} AS TEXT) AS sample_value "
        f"FROM {table_sql} WHERE {column_sql} IS NOT NULL "
        "ORDER BY sample_value LIMIT ?",
        (max(int(limit) * 8, int(limit)),),
    ).fetchall()
    samples: list[str] = []
    for row in rows:
        value = _clean_description_cell(row[0])
        if not value or len(value) > int(max_chars) or value in samples:
            continue
        samples.append(value)
        if len(samples) >= int(limit):
            break
    return samples


def compute_column_stats(conn: sqlite3.Connection, table_name: str, column_name: str, data_type: str) -> dict[str, Any]:
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column_name)
    row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table_sql}").fetchone()[0])
    non_null_count = int(
        conn.execute(f"SELECT COUNT(*) FROM {table_sql} WHERE {column_sql} IS NOT NULL").fetchone()[0]
    )
    distinct_count = int(
        conn.execute(
            f"SELECT COUNT(DISTINCT {column_sql}) FROM {table_sql} WHERE {column_sql} IS NOT NULL"
        ).fetchone()[0]
    )
    null_count = row_count - non_null_count
    distinct_ratio = float(distinct_count) / float(non_null_count) if non_null_count else 0.0
    small_values = [
        str(row[0])
        for row in conn.execute(
            f"SELECT DISTINCT {column_sql} FROM {table_sql} WHERE {column_sql} IS NOT NULL LIMIT 20"
        ).fetchall()
    ]
    lengths = [
        float(row[0])
        for row in conn.execute(
            f"SELECT LENGTH(CAST({column_sql} AS TEXT)) FROM {table_sql} WHERE {column_sql} IS NOT NULL"
        ).fetchall()
        if row[0] is not None
    ]
    return {
        "row_count": row_count,
        "non_null_count": non_null_count,
        "null_count": null_count,
        "null_fraction": float(null_count) / float(row_count) if row_count else 0.0,
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "text_length": {
            "p10": percentile(lengths, 0.10),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
        },
        "value_mode": infer_value_mode(column_name, data_type, distinct_count, distinct_ratio, small_values),
    }


def build_schema_profile(
    db_dir: Path,
    sqlite_path: Path,
    manual_column_info_path: Path | None = None,
    *,
    strict_descriptive_names: bool = True,
    sample_value_limit: int = DEFAULT_SAMPLE_VALUE_LIMIT,
    sample_value_max_chars: int = DEFAULT_SAMPLE_VALUE_MAX_CHARS,
) -> dict[str, Any]:
    """Build a profile with complete BIRD descriptive names by default.

    ``manual_column_info_path`` is retained as a legacy argument name for
    callers that explicitly pass an aggregate map. When omitted, the canonical
    per-table BIRD descriptive-name directory is used.
    """
    db_dir = Path(db_dir)
    sqlite_path = Path(sqlite_path)
    descriptive_name_path = (
        Path(manual_column_info_path)
        if manual_column_info_path is not None
        else db_dir / "preprocessed" / "ColGrp_artifacts" / "descriptive_names"
    )
    descriptions = load_database_descriptions(db_dir)
    with sqlite3.connect(sqlite_path) as conn:
        table_names = get_table_names(conn)
        schema_columns = {
            table_name: [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()]
            for table_name in table_names
        }
        if descriptive_name_path.is_dir():
            table_descriptive_names, column_descriptive_names, descriptive_name_provenance = (
                load_descriptive_name_artifacts(
                    descriptive_name_path,
                    schema_columns,
                    strict=strict_descriptive_names,
                )
            )
        else:
            table_descriptive_names, column_descriptive_names, descriptive_name_provenance = (
                load_legacy_descriptive_name_map(
                    descriptive_name_path,
                    schema_columns,
                    strict=strict_descriptive_names,
                )
            )

        profile: dict[str, Any] = {
            "db_id": db_dir.name,
            "source_sqlite": str(sqlite_path),
            "source_descriptive_names": str(descriptive_name_path),
            "source_manual_column_info": str(descriptive_name_path)
            if manual_column_info_path is not None
            else None,
            "descriptive_name_provenance": descriptive_name_provenance,
            "sample_value_policy": {
                "limit": int(sample_value_limit),
                "max_chars": int(sample_value_max_chars),
                "selection": "distinct_non_null_text_ordered",
            },
            "tables": {},
            "columns": {},
        }
        for table_name in table_names:
            table_sql = quote_identifier(table_name)
            unique_columns = get_unique_columns(conn, table_name)
            foreign_keys_by_column = get_foreign_keys_by_column(conn, table_name)
            columns = conn.execute(f"PRAGMA table_info({table_sql})").fetchall()
            profile["tables"][table_name] = {
                "table_name": table_name,
                "table_descriptive_name": table_descriptive_names[table_name],
                "row_count": int(conn.execute(f"SELECT COUNT(*) FROM {table_sql}").fetchone()[0]),
                "columns": [str(row[1]) for row in columns],
            }
            for row in columns:
                column_name = str(row[1])
                csv_info = descriptions.get(table_name, {}).get(column_name, {})
                data_type = csv_info.get("data_format") or str(row[2] or "")
                full_name = f"{table_name}.{column_name}"
                profile["columns"][full_name] = {
                    "table": table_name,
                    "column": column_name,
                    "full_name": full_name,
                    "data_type": data_type,
                    "column_full_name": csv_info.get("column_name", column_name),
                    "descriptive_name": column_descriptive_names[full_name],
                    "column_description": csv_info.get("column_description", ""),
                    "value_description": csv_info.get("value_description", ""),
                    "constraints": {
                        "not_null": bool(row[3]),
                        "primary_key": bool(row[5]),
                        "unique": column_name in unique_columns,
                        "default_value": row[4],
                        "foreign_keys": foreign_keys_by_column.get(column_name, []),
                    },
                    "stats": compute_column_stats(conn, table_name, column_name, data_type),
                    "sample_values": compute_column_sample_values(
                        conn,
                        table_name,
                        column_name,
                        limit=sample_value_limit,
                        max_chars=sample_value_max_chars,
                    ),
                }
    profile["schema_table_count"] = len(profile["tables"])
    profile["schema_column_count"] = len(profile["columns"])
    return profile
