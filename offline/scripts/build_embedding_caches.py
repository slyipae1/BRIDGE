#!/usr/bin/env python3
"""Build resumable query/document embedding caches for one BRIDGE proposal view."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
if str(OFFLINE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_ROOT))

from auto_construct_graph.channels.embedding_channel import (  # noqa: E402
    DOCUMENT_EMBEDDING_VIEW,
    QWEN_QUERY_DOC_MODE,
    QUERY_EMBEDDING_VIEW,
    build_descriptive_name_only_embedding_records,
    build_query_embedding_text,
    embedding_text_sha256,
    query_instruction_sha256,
    write_jsonl_records,
)
from auto_construct_graph.channels.surface_name_embedding_channel import (  # noqa: E402
    SURFACE_NAME_EMBEDDING_RECORD_VERSION,
    SURFACE_NAME_QUERY_INSTRUCTION,
    build_surface_name_embedding_records,
)
from auto_construct_graph.embedding_client import (  # noqa: E402
    OpenAICompatibleEmbeddingClient,
    build_embeddings_payload,
    extract_embeddings_from_response,
    resolve_embedding_config,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_row(record: dict[str, Any], vector: list[float], *, view: str, model: str, base_url: str, instruction: str) -> dict[str, Any]:
    return {
        "column_id": record["column_id"],
        "embedding": [float(value) for value in vector],
        "embedding_model": model,
        "embedding_base_url": base_url,
        "record_version": record["record_version"],
        "embedding_text_sha256": record["embedding_text_sha256"],
        "embedding_view": view,
        "embedding_mode": QWEN_QUERY_DOC_MODE,
        "query_instruction_sha256": query_instruction_sha256(instruction),
    }


def _valid_cache(rows: list[dict[str, Any]], records: list[dict[str, Any]], *, view: str, model: str, base_url: str, instruction: str) -> dict[str, dict[str, Any]]:
    by_id = {str(record["column_id"]): record for record in records}
    expected_instruction = query_instruction_sha256(instruction)
    valid: dict[str, dict[str, Any]] = {}
    for row in rows:
        column_id = str(row.get("column_id") or "")
        record = by_id.get(column_id)
        if record and str(row.get("record_version")) == str(record["record_version"]) and str(row.get("embedding_text_sha256")) == str(record["embedding_text_sha256"]) and str(row.get("embedding_view")) == view and str(row.get("embedding_model")) == model and str(row.get("embedding_base_url")).rstrip("/") == base_url.rstrip("/") and str(row.get("query_instruction_sha256")) == expected_instruction and isinstance(row.get("embedding"), list) and row["embedding"]:
            valid[column_id] = row
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--proposal-view", choices=("schema_semantic", "surface_name"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if args.proposal_view == "schema_semantic":
        records = build_descriptive_name_only_embedding_records(profile)
        instruction = "Retrieve schema columns that could be ambiguous alternatives for an underspecified natural-language database request."
    else:
        records = build_surface_name_embedding_records(profile)
        instruction = SURFACE_NAME_QUERY_INSTRUCTION
    cfg = resolve_embedding_config()
    model = args.embedding_model or cfg.model
    base_url = (args.embedding_base_url or cfg.base_url).rstrip("/")
    api_key = args.embedding_api_key or cfg.api_key
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "records.json", {"proposal_view": args.proposal_view, "records": records})
    query_cache = _valid_cache(_load_jsonl(output / "query_embeddings.jsonl"), records, view=QUERY_EMBEDDING_VIEW, model=model, base_url=base_url, instruction=instruction)
    document_cache = _valid_cache(_load_jsonl(output / "document_embeddings.jsonl"), records, view=DOCUMENT_EMBEDDING_VIEW, model=model, base_url=base_url, instruction=instruction)
    client = OpenAICompatibleEmbeddingClient(base_url=base_url, api_key=api_key, timeout=args.timeout)
    for view, cache, path in ((DOCUMENT_EMBEDDING_VIEW, document_cache, output / "document_embeddings.jsonl"), (QUERY_EMBEDDING_VIEW, query_cache, output / "query_embeddings.jsonl")):
        missing = [record for record in records if record["column_id"] not in cache]
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start:start + args.batch_size]
            texts = [record["embedding_text"] if view == DOCUMENT_EMBEDDING_VIEW else build_query_embedding_text(record["embedding_text"], instruction) for record in batch]
            vectors = extract_embeddings_from_response(client.embeddings(build_embeddings_payload(model=model, inputs=texts)))
            if len(vectors) != len(batch):
                raise ValueError(f"embedding response length mismatch: {len(vectors)} != {len(batch)}")
            for record, vector in zip(batch, vectors):
                cache[record["column_id"]] = _cache_row(record, vector, view=view, model=model, base_url=base_url, instruction=instruction)
            write_jsonl_records(path, [cache[record["column_id"]] for record in records if record["column_id"] in cache])
    _write_json(output / "cache_manifest.json", {
        "db_id": profile.get("db_id"), "proposal_view": args.proposal_view, "record_count": len(records),
        "embedding_model": model, "embedding_base_url": base_url, "query_instruction": instruction,
        "record_version": records[0]["record_version"] if records else None,
    })
    print(json.dumps({"db_id": profile.get("db_id"), "proposal_view": args.proposal_view, "record_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
