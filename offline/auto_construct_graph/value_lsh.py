"""Build value-LSH artifacts compatible with BRIDGE online literal retrieval."""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    MinHash = None
    MinHashLSH = None


SKIPPED_COLUMN_KEYWORDS = ("_id", " id", "url", "email", "web", "time", "phone", "date", "address")
SHORT_STRING_BEHAVIOR = "v2_whole_string_update_when_len_lt_n_gram"


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _create_minhash(signature_size: int, value: str, n_gram: int) -> Any:
    if MinHash is None:
        raise ImportError("datasketch is required to build the BRIDGE value LSH index")
    minhash = MinHash(num_perm=signature_size)
    text = str(value)
    if len(text) < n_gram:
        minhash.update(text.encode("utf-8"))
        return minhash
    for index in range(len(text) - n_gram + 1):
        minhash.update(text[index:index + n_gram].encode("utf-8"))
    return minhash


def _artifact_paths(preprocessed_dir: Path, db_id: str) -> dict[str, Path]:
    return {
        "unique_values": preprocessed_dir / f"{db_id}_unique_values.pkl",
        "lsh": preprocessed_dir / f"{db_id}_lsh.pkl",
        "minhashes": preprocessed_dir / f"{db_id}_minhashes.pkl",
        "manifest": preprocessed_dir / f"{db_id}_lsh_manifest.json",
    }


def _datasketch_version() -> str | None:
    try:
        return metadata.version("datasketch")
    except metadata.PackageNotFoundError:
        return None


def _text_values(sqlite_path: Path) -> dict[str, dict[str, list[str]]]:
    unique_values: dict[str, dict[str, list[str]]] = {}
    with sqlite3.connect(sqlite_path) as connection:
        tables = [str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()]
        primary_keys = {
            str(column[1]).casefold()
            for table in tables
            for column in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            if int(column[5]) > 0
        }
        for table in tables:
            values_by_column: dict[str, list[str]] = {}
            for column in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall():
                column_name, column_type = str(column[1]), str(column[2]).upper()
                lower_name = column_name.casefold()
                if "TEXT" not in column_type or lower_name in primary_keys:
                    continue
                if any(keyword in lower_name for keyword in SKIPPED_COLUMN_KEYWORDS) or column_name.endswith("Id"):
                    continue
                table_sql, column_sql = _quote_identifier(table), _quote_identifier(column_name)
                length_sum, distinct_count = connection.execute(
                    f"SELECT SUM(LENGTH(value)), COUNT(value) FROM "
                    f"(SELECT DISTINCT {column_sql} AS value FROM {table_sql} WHERE {column_sql} IS NOT NULL)"
                ).fetchone()
                if not length_sum or not distinct_count:
                    continue
                average_length = float(length_sum) / int(distinct_count)
                if not (("name" in lower_name and length_sum < 5_000_000) or
                        (length_sum < 2_000_000 and average_length < 25) or distinct_count < 100):
                    continue
                rows = connection.execute(
                    f"SELECT DISTINCT {column_sql} FROM {table_sql} WHERE {column_sql} IS NOT NULL"
                ).fetchall()
                values_by_column[column_name] = [str(row[0]) for row in rows]
            unique_values[table] = values_by_column
    return unique_values


def build_db_lsh(
    db_directory_path: str | Path,
    *,
    signature_size: int,
    n_gram: int,
    threshold: float,
    verbose: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build v2-compatible `{db_id}_{lsh,minhashes,unique_values}` artifacts."""
    if MinHashLSH is None:
        raise ImportError("datasketch is required to build the BRIDGE value LSH index")
    db_dir = Path(db_directory_path)
    db_id = db_dir.name
    sqlite_path = db_dir / f"{db_id}.sqlite"
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")
    preprocessed_dir = db_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)
    paths = _artifact_paths(preprocessed_dir, db_id)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("LSH artifacts already exist; pass --overwrite to replace them")

    values = _text_values(sqlite_path)
    lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    minhashes: dict[str, tuple[Any, str, str, str]] = {}
    for table, values_by_column in values.items():
        for column, column_values in values_by_column.items():
            if verbose:
                logging.info("Indexing %s.%s (%d values)", table, column, len(column_values))
            for index, value in enumerate(column_values):
                minhash = _create_minhash(signature_size, value, n_gram)
                key = f"{table}_{column}_{index}"
                minhashes[key] = (minhash, table, column, value)
                lsh.insert(key, minhash)

    with paths["unique_values"].open("wb") as handle:
        pickle.dump(values, handle)
    with paths["lsh"].open("wb") as handle:
        pickle.dump(lsh, handle)
    with paths["minhashes"].open("wb") as handle:
        pickle.dump(minhashes, handle)
    column_count = sum(len(columns) for columns in values.values())
    value_count = sum(len(column_values) for columns in values.values() for column_values in columns.values())
    manifest = {
        "db_id": db_id,
        "db_path": str(sqlite_path),
        "signature_size": signature_size,
        "n_gram": n_gram,
        "threshold": threshold,
        "unique_value_count": value_count,
        "table_count": len(values),
        "column_count": column_count,
        "artifact_names": {key: path.name for key, path in paths.items() if key != "manifest"},
        "manifest_name": paths["manifest"].name,
        "short_string_behavior": SHORT_STRING_BEHAVIOR,
        "datasketch_version": _datasketch_version(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
