from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SOURCE_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_ROOT))

from auto_construct_graph.schema_profile import build_schema_profile


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_db_paths(*, db_id: str, db_dir: Path, sqlite_path: Path | None) -> tuple[Path, Path]:
    resolved_db_dir = Path(db_dir)
    resolved_sqlite = Path(sqlite_path) if sqlite_path is not None else resolved_db_dir / f"{db_id}.sqlite"
    if not resolved_db_dir.exists():
        raise FileNotFoundError(f"DB directory does not exist: {resolved_db_dir}")
    if not resolved_sqlite.exists():
        raise FileNotFoundError(f"SQLite file does not exist: {resolved_sqlite}")
    return resolved_db_dir, resolved_sqlite


def build_manifest(
    *,
    db_id: str,
    db_dir: Path,
    sqlite_path: Path,
    descriptive_names: Path | None,
    output_dir: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "db_id": db_id,
        "db_dir": str(db_dir),
        "sqlite": str(sqlite_path),
        "descriptive_names": str(descriptive_names) if descriptive_names is not None else profile.get("source_descriptive_names"),
        "descriptive_name_provenance": profile.get("descriptive_name_provenance") or {},
        "output_dir": str(output_dir),
        "schema_table_count": profile["schema_table_count"],
        "schema_column_count": profile["schema_column_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a generic BIRD DB schema profile for auto construction.")
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument(
        "--descriptive-names",
        "--manual-column-info",
        dest="descriptive_names",
        type=Path,
        help="Optional explicit per-table descriptive-name directory or legacy aggregate mapping. Defaults to the BIRD artifact directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_dir, sqlite_path = resolve_db_paths(
        db_id=str(args.db_id),
        db_dir=args.db_dir,
        sqlite_path=args.sqlite,
    )
    output_dir = args.output_dir
    profile = build_schema_profile(db_dir, sqlite_path, args.descriptive_names)
    manifest = build_manifest(
        db_id=str(args.db_id),
        db_dir=db_dir,
        sqlite_path=sqlite_path,
        descriptive_names=args.descriptive_names,
        output_dir=output_dir,
        profile=profile,
    )
    print(
        "[db-schema-profile] "
        f"db_id={args.db_id} "
        f"tables={profile['schema_table_count']} "
        f"columns={profile['schema_column_count']} "
        f"output_dir={output_dir}"
    )
    if args.check:
        return 0
    write_json(output_dir / "schema_profile.json", profile)
    write_json(output_dir / "profile_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
