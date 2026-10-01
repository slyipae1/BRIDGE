#!/usr/bin/env python3
"""Build a proposal-first verifier queue, or the explicit all-pairs ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
if str(OFFLINE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_ROOT))

from auto_construct_graph.all_pairs import build_all_pairs_bundles  # noqa: E402
from auto_construct_graph.proposal_aggregation import (  # noqa: E402
    build_pair_bundles,
    load_jsonl_records,
    normalize_channel_candidate,
    proposal_pair_key,
    write_jsonl_records,
)
from auto_construct_graph.types import column_ref_from_object, make_pair_key  # noqa: E402
from auto_construct_graph.verifier_value_evidence import (  # noqa: E402
    build_pair_verifier_value_evidence,
    build_verifier_value_domains,
)


CHANNELS = ("schema_embedding", "lexical", "surface_name_embedding", "value_collision")
VIEW_TO_CHANNELS = {
    "schema_semantic_similarity": {"schema_embedding"},
    "surface_name_similarity": {"lexical", "surface_name_embedding"},
    "value_domain_collision": {"value_collision"},
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    raw = row.get("pair_key")
    if isinstance(raw, list) and len(raw) == 2:
        return make_pair_key(str(raw[0]), str(raw[1]))
    return make_pair_key(column_ref_from_object(row["col1"]), column_ref_from_object(row["col2"]))


def _parse_artifact(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--candidate-artifact must be CHANNEL=PATH")
    channel, value = (part.strip() for part in raw.split("=", 1))
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}")
    path = Path(value)
    return channel, path / "candidate_pairs.jsonl" if path.is_dir() else path


def _schema_selected(record: dict[str, Any], *, max_rank: int, min_score: float) -> bool:
    scores = record.get("source_scores") or {}
    rank = scores.get("embedding_rank_min")
    score = scores.get("embedding_query_doc_similarity_max")
    try:
        return int(rank) <= max_rank and float(score) >= min_score
    except (TypeError, ValueError):
        raise ValueError(f"schema embedding candidate lacks rank/score: {record.get('candidate_id')}")


def _augment_value_evidence(
    bundles: list[dict[str, Any]], *, profile: dict[str, Any], sqlite_path: Path,
    value_pairs: set[tuple[str, str]],
) -> None:
    if not value_pairs:
        return
    domains = build_verifier_value_domains(sqlite_path, profile)
    for bundle in bundles:
        pair = proposal_pair_key(bundle)
        if pair not in value_pairs:
            continue
        evidence = build_pair_verifier_value_evidence(
            db_id=str(profile.get("db_id") or ""), left_ref=pair[0], right_ref=pair[1], value_domains=domains,
        )
        bundle["value_overlapping_stat"] = evidence["value_overlapping_stat"]
        for key, context_key in ((pair[0], "col1"), (pair[1], "col2")):
            bundle["column_context"][context_key]["sample_values"] = evidence["column_sample_values"].get(key, [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--schema-profile", type=Path, required=True)
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--candidate-artifact", action="append", default=[], help="Repeat CHANNEL=PATH.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verification-universe", choices=("proposal_union", "all_pairs"), default="proposal_union")
    parser.add_argument("--disable-proposal-view", choices=tuple(VIEW_TO_CHANNELS), default=None)
    parser.add_argument("--schema-max-rank", type=int, default=10)
    parser.add_argument("--schema-min-score", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.schema_max_rank < 1 or not -1.0 <= args.schema_min_score <= 1.0:
        raise ValueError("invalid schema selector")
    if not args.schema_profile.is_file() or not args.sqlite_path.is_file():
        raise FileNotFoundError("--schema-profile and --sqlite-path must exist")
    artifacts = dict(_parse_artifact(value) for value in args.candidate_artifact)
    if set(artifacts) != set(CHANNELS):
        raise ValueError(f"provide all candidate artifacts; missing={sorted(set(CHANNELS) - set(artifacts))}")
    for path in artifacts.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = _load_json(args.schema_profile)
    disabled = VIEW_TO_CHANNELS.get(args.disable_proposal_view, set())
    records_by_channel = {channel: load_jsonl_records(path) for channel, path in artifacts.items()}
    selected_records: dict[str, list[dict[str, Any]]] = {}
    for channel, records in records_by_channel.items():
        if channel in disabled:
            selected_records[channel] = []
        elif channel == "schema_embedding":
            selected_records[channel] = [
                row for row in records
                if _schema_selected(row, max_rank=args.schema_max_rank, min_score=args.schema_min_score)
            ]
        else:
            selected_records[channel] = records

    if args.verification_universe == "all_pairs":
        bundles = build_all_pairs_bundles(profile)
        value_pairs: set[tuple[str, str]] = set()
        supporting: dict[tuple[str, str], list[str]] = {}
    else:
        proposals = [
            normalize_channel_candidate(row, str(artifacts[channel]))
            for channel in CHANNELS
            for row in selected_records[channel]
        ]
        bundles = build_pair_bundles(proposals, profile)
        value_pairs = {_pair_key(row) for row in selected_records["value_collision"]}
        supporting = {
            proposal_pair_key(bundle): sorted({str(item["channel"]) for item in bundle.get("proposals_by_channel") or []})
            for bundle in bundles
        }
        _augment_value_evidence(bundles, profile=profile, sqlite_path=args.sqlite_path, value_pairs=value_pairs)

    queue = []
    for index, bundle in enumerate(bundles, start=1):
        pair = proposal_pair_key(bundle)
        queue.append({
            "candidate_id": bundle["candidate_id"],
            "pair_key": list(pair),
            "queue_position": index,
            "supporting_candidate_channels": supporting.get(pair, []),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(args.output_dir / "candidate_bundles.jsonl", bundles)
    write_jsonl_records(args.output_dir / "queue.jsonl", queue)
    manifest = {
        "artifact_type": "bridge_candidate_verifier_queue",
        "db_id": args.db_id,
        "verification_universe": args.verification_universe,
        "disable_proposal_view": args.disable_proposal_view,
        "schema_selector": {"rank_min_lte": args.schema_max_rank, "cosine_gte": args.schema_min_score},
        "queue_count": len(queue),
        "selected_candidate_counts": {channel: len(rows) for channel, rows in selected_records.items()},
        "schema_profile_path": str(args.schema_profile),
        "schema_profile_sha256": _sha256(args.schema_profile),
        "sqlite_path": str(args.sqlite_path),
        "candidate_artifacts": {channel: str(path) for channel, path in artifacts.items()},
        "candidate_artifact_sha256": {channel: _sha256(path) for channel, path in artifacts.items()},
        "model_visible_policy": "schema_only_no_channel_evidence",
    }
    (args.output_dir / "queue_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"db_id": args.db_id, "queue_count": len(queue), "verification_universe": args.verification_universe}, ensure_ascii=False))


if __name__ == "__main__":
    main()
