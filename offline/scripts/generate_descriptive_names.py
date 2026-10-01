#!/usr/bin/env python3
"""Generate loader-compatible per-table descriptive-name caches for one SQLite DB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
if str(OFFLINE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_ROOT))

from descriptive_names import (  # noqa: E402
    DEFAULT_SAMPLE_VALUE_LIMIT,
    DEFAULT_SAMPLE_VALUE_MAX_CHARS,
    artifact_path,
    build_table_info,
    call_chat_json,
    load_database_descriptions,
    load_schema,
    load_valid_artifact,
    render_prompt,
    resolve_chat_config,
    utc_now,
    validate_response,
    write_artifact,
)


DEFAULT_TEMPLATE = OFFLINE_ROOT / "prompts" / "descriptive_name_generation.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, required=True, help="Directory containing {db_id}.sqlite.")
    parser.add_argument("--sqlite", type=Path, help="Defaults to {db_dir}/{db_id}.sqlite.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the loader-compatible DB-local cache directory.")
    parser.add_argument("--table", action="append", help="Generate only this table; repeatable.")
    parser.add_argument("--resume", action="store_true", help="Skip existing valid table artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing valid table artifacts.")
    parser.add_argument("--check", action="store_true", help="Validate cache coverage only; never call a model.")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--sample-value-limit", type=int, default=DEFAULT_SAMPLE_VALUE_LIMIT)
    parser.add_argument("--sample-value-max-chars", type=int, default=DEFAULT_SAMPLE_VALUE_MAX_CHARS)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking-control", choices=("local_qwen", "dashscope", "both", "none"), default="local_qwen")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    db_dir = args.db_dir.resolve()
    db_id = db_dir.name
    sqlite_path = (args.sqlite or db_dir / f"{db_id}.sqlite").resolve()
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")
    if args.sample_value_limit < 1 or args.sample_value_max_chars < 1:
        raise ValueError("sample limits must be positive")
    schema = load_schema(sqlite_path)
    selected_tables = list(args.table or sorted(schema))
    unknown = [table for table in selected_tables if table not in schema]
    if unknown:
        raise ValueError(f"unknown table(s): {unknown}")
    output_dir = (args.output_dir or db_dir / "preprocessed" / "ColGrp_artifacts" / "descriptive_names").resolve()

    if args.check:
        invalid = [
            table for table in selected_tables
            if load_valid_artifact(artifact_path(db_dir, table, output_dir=output_dir), table_name=table, columns=schema[table]) is None
        ]
        print(json.dumps({"db_id": db_id, "checked_tables": len(selected_tables), "invalid_or_missing": invalid}, ensure_ascii=False))
        return 1 if invalid else 0

    model, base_url, api_key = resolve_chat_config(model=args.model, base_url=args.base_url, api_key=args.api_key)
    template = args.template.read_text(encoding="utf-8")
    descriptions = load_database_descriptions(db_dir, schema)
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    for table_name in selected_tables:
        columns = schema[table_name]
        path = artifact_path(db_dir, table_name, output_dir=output_dir)
        if args.resume and load_valid_artifact(path, table_name=table_name, columns=columns) is not None:
            skipped.append(table_name)
            continue
        table_info = build_table_info(
            sqlite_path=sqlite_path,
            table_name=table_name,
            columns=columns,
            descriptions=descriptions,
            sample_value_limit=args.sample_value_limit,
            sample_value_max_chars=args.sample_value_max_chars,
        )
        request_kwargs: dict[str, Any] = {"table_name": table_name, "table_info": table_info}
        try:
            response, raw_response = call_chat_json(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=render_prompt(template, table_name=table_name, table_info=table_info),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking_control=args.thinking_control,
            )
            validated = validate_response(response, table_name=table_name, columns=columns)
            write_artifact(path, {
                "stage": "descriptive_names",
                "call_id": table_name,
                "template_name": args.template.name,
                "request_kwargs": request_kwargs,
                "status": "ok",
                "response": validated,
                "usage": raw_response.get("usage"),
                "created_at_utc": utc_now(),
            })
            completed.append(table_name)
        except Exception as exc:
            write_artifact(path, {
                "stage": "descriptive_names",
                "call_id": table_name,
                "template_name": args.template.name,
                "request_kwargs": request_kwargs,
                "status": "failed",
                "error": str(exc),
                "created_at_utc": utc_now(),
            })
            failed.append({"table": table_name, "error": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "descriptive_names_manifest.json").write_text(json.dumps({
        "db_id": db_id,
        "model": model,
        "base_url": base_url,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "updated_at_utc": utc_now(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"db_id": db_id, "completed": len(completed), "skipped": len(skipped), "failed": len(failed), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
