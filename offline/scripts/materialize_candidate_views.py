#!/usr/bin/env python3
"""Materialize the three public proposal views for one database.

Surface-Name Similarity is materialized through its two internal submethods:
generic lexical matching and raw surface-name embedding similarity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
if str(OFFLINE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_ROOT))

from auto_construct_graph.candidate_channel_config import load_candidate_channel_config  # noqa: E402
from auto_construct_graph.channels.embedding_channel import extract_embedding_candidates_query_doc, load_embedding_vectors, score_query_document_pairs  # noqa: E402
from auto_construct_graph.channels.lexical_channel import extract_lexical_candidates  # noqa: E402
from auto_construct_graph.channels.surface_name_embedding_channel import extract_surface_name_embedding_candidates  # noqa: E402
from auto_construct_graph.channels.value_collision_channel import build_value_profiles, extract_value_collision_candidates  # noqa: E402
from auto_construct_graph.proposal_aggregation import write_jsonl_records  # noqa: E402


def _records(cache_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((cache_dir / "records.json").read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"invalid records cache: {cache_dir}")
    return records


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--schema-profile", type=Path, required=True)
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--schema-cache-dir", type=Path, required=True)
    parser.add_argument("--surface-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = json.loads(args.schema_profile.read_text(encoding="utf-8"))
    config = load_candidate_channel_config(args.channel_config)
    schema_records = _records(args.schema_cache_dir)
    schema_query = _jsonl(args.schema_cache_dir / "query_embeddings.jsonl")
    schema_document = _jsonl(args.schema_cache_dir / "document_embeddings.jsonl")
    surface_records = _records(args.surface_cache_dir)
    surface_query = _jsonl(args.surface_cache_dir / "query_embeddings.jsonl")
    surface_document = _jsonl(args.surface_cache_dir / "document_embeddings.jsonl")
    schema_vectors_q, schema_vectors_d = load_embedding_vectors(schema_query), load_embedding_vectors(schema_document)
    surface_vectors_q, surface_vectors_d = load_embedding_vectors(surface_query), load_embedding_vectors(surface_document)
    schema_cfg = config["channels"]["schema_embedding"]
    schema = extract_embedding_candidates_query_doc(records=schema_records, query_vectors=schema_vectors_q, document_vectors=schema_vectors_d, selection_mode=schema_cfg["selection_mode"], top_k=schema_cfg["top_k"], similarity_threshold=schema_cfg["similarity_threshold"], minimum_similarity_threshold=schema_cfg["minimum_similarity_threshold"], embedding_model=str(schema_query[0].get("embedding_model") or ""), embedding_base_url=str(schema_query[0].get("embedding_base_url") or ""))
    lexical = extract_lexical_candidates(profile, threshold_config=config)
    value_profiles = build_value_profiles(args.sqlite_path, profile)
    value, _ = extract_value_collision_candidates(value_profiles, include_literal_evidence=False, threshold_config=config)
    surface = extract_surface_name_embedding_candidates(records=surface_records, query_vectors=surface_vectors_q, document_vectors=surface_vectors_d, similarity_threshold=float(config["channels"]["surface_name_embedding"]["similarity_threshold"]), embedding_model=str(surface_query[0].get("embedding_model") or ""), embedding_base_url=str(surface_query[0].get("embedding_base_url") or ""))
    outputs = {"schema_embedding": schema, "lexical": lexical, "value_collision": value, "surface_name_embedding": surface}
    for channel, rows in outputs.items():
        write_jsonl_records(args.output_dir / channel / "candidate_pairs.jsonl", rows)
    write_jsonl_records(args.output_dir / "schema_embedding" / "all_pair_scores.jsonl", score_query_document_pairs(records=schema_records, query_vectors=schema_vectors_q, document_vectors=schema_vectors_d))
    _write_manifest(args.output_dir / "run_manifest.json", {"db_id": args.db_id, "candidate_counts": {key: len(value) for key, value in outputs.items()}, "channel_config": config})
    print(json.dumps({"db_id": args.db_id, **{key: len(value) for key, value in outputs.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
