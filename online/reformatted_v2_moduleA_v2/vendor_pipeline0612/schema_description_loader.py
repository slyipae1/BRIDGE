from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Dict


CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "big5",
    "cp1252",
    "latin1",
    "iso-8859-1",
)


def _read_csv_rows(csv_file: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with csv_file.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"could not decode {csv_file} with encodings {CSV_ENCODINGS}; last error={last_error}",
    )


def load_tables_description(db_dir: str | Path, use_value_description: bool = True) -> Dict[str, Dict[str, Dict[str, str]]]:
    description_dir = Path(db_dir) / "database_description"
    if not description_dir.exists():
        return {}

    table_description: Dict[str, Dict[str, Dict[str, str]]] = {}
    for csv_file in sorted(description_dir.glob("*.csv")):
        table_name = csv_file.stem.strip()
        table_description.setdefault(table_name, {})
        for row in _read_csv_rows(csv_file):
            original_column_name = (row.get("original_column_name") or "").strip()
            if not original_column_name:
                continue
            value_description = (row.get("value_description") or "").replace("\n", " ").strip()
            if not use_value_description:
                value_description = ""
            table_description[table_name][original_column_name] = {
                "original_column_name": original_column_name,
                "column_name": (row.get("column_name") or "").strip(),
                "column_description": (row.get("column_description") or "").replace("\n", " ").strip(),
                "data_format": (row.get("data_format") or "").strip(),
                "value_description": value_description,
            }
    return table_description


def load_column_descriptive_names(db_dir: str | Path, db_id: str) -> Dict[str, str]:
    groups_path = Path(db_dir) / "preprocessed" / f"{db_id}_column_groups.pkl"
    if not groups_path.exists():
        return {}
    with groups_path.open("rb") as handle:
        payload = pickle.load(handle)
    return dict((payload or {}).get("column_descriptive_names", {}) or {})


def load_colgrp_artifact_descriptive_names(db_dir: str | Path) -> Dict[str, str]:
    """Load DB-local descriptive names emitted by the ColGrp preprocessing line.

    These artifacts are intentionally a fallback for graph formats, such as
    auto-constructed manual archives, that do not carry their own name map.
    The filename is the schema table name and each successful response provides
    its column-level display names.
    """
    artifact_dir = Path(db_dir) / "preprocessed" / "ColGrp_artifacts" / "descriptive_names"
    if not artifact_dir.exists():
        return {}

    descriptive_names: Dict[str, str] = {}
    for artifact_path in sorted(artifact_dir.glob("*.json")):
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") not in (None, "ok"):
            continue
        response = payload.get("response")
        if not isinstance(response, dict):
            continue

        table_name = artifact_path.stem.strip()
        if not table_name:
            continue
        for column in response.get("columns", []) or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("column_name") or "").strip()
            descriptive_name = str(column.get("descriptive_name") or "").strip()
            if column_name and descriptive_name:
                descriptive_names.setdefault(f"{table_name}.{column_name}", descriptive_name)

    return descriptive_names


def build_additional_info_by_column(db_dir: str | Path, db_id: str) -> Dict[str, Dict[str, str]]:
    db_dir = Path(db_dir)
    descriptions = load_tables_description(db_dir, use_value_description=True)
    descriptive_names = load_column_descriptive_names(db_dir, db_id)

    combined: Dict[str, Dict[str, str]] = {}
    for table_name, columns in descriptions.items():
        for column_name, info in columns.items():
            key = f"{table_name}.{column_name}"
            combined[key] = {
                "descriptive_name": descriptive_names.get(key, info.get("column_name", "") or ""),
                "column_description": info.get("column_description", "") or "",
                "value_description": info.get("value_description", "") or "",
            }

    for key, descriptive_name in descriptive_names.items():
        combined.setdefault(
            key,
            {
                "descriptive_name": descriptive_name,
                "column_description": "",
                "value_description": "",
            },
        )

    return combined
