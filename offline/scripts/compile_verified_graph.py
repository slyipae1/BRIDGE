#!/usr/bin/env python3
"""Compile a proposal-first verifier ledger into a pipeline-compatible graph archive."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
if str(OFFLINE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_ROOT))

from auto_construct_graph.final_pair_verifier_v02 import decision_to_lean_edge  # noqa: E402
from auto_construct_graph.proposal_aggregation import load_jsonl_records, write_jsonl_records  # noqa: E402
from auto_construct_graph.schema_profile import get_table_names, load_descriptive_name_artifacts, quote_identifier  # noqa: E402
from auto_construct_graph.types import make_pair_key  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _schema_columns(sqlite_path: Path) -> dict[str, list[str]]:
    with sqlite3.connect(sqlite_path) as conn:
        return {
            table: [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()]
            for table in get_table_names(conn)
        }


def _latest_ok_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id and row.get("status") == "OK":
            latest[candidate_id] = row
    return latest


def _ref(column: dict[str, Any]) -> str:
    return f"{column['tab']}.{column['col']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--descriptive-name-dir", type=Path, required=True)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--verifier-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sqlite_path.is_file() or not args.queue_path.is_file() or not args.verifier_ledger.is_file():
        raise FileNotFoundError("sqlite path, queue path, and verifier ledger must exist")
    schema_columns = _schema_columns(args.sqlite_path)
    table_names, column_names, name_provenance = load_descriptive_name_artifacts(
        args.descriptive_name_dir, schema_columns, strict=True,
    )
    queue = load_jsonl_records(args.queue_path)
    queue_by_id = {str(row["candidate_id"]): row for row in queue}
    if len(queue_by_id) != len(queue):
        raise ValueError("queue has duplicate candidate IDs")
    terminal = _latest_ok_by_id(load_jsonl_records(args.verifier_ledger))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for candidate_id, queue_row in queue_by_id.items():
        decision = terminal.get(candidate_id)
        if decision is None:
            unresolved.append({"candidate_id": candidate_id, "pair_key": queue_row["pair_key"], "reason": "no_terminal_decision"})
            continue
        payload = decision.get("parsed_payload") or {}
        if payload.get("decision") == "ADD_EDGE":
            edge = decision_to_lean_edge(payload)
            edge["candidate_id"] = candidate_id
            edge["source_channels"] = list(queue_row.get("supporting_candidate_channels") or [])
            edge["verification_policy"] = "FINAL_VERIFIER"
            accepted.append(edge)
        elif payload.get("decision") == "REJECT":
            rejected.append({"candidate_id": candidate_id, "pair_key": queue_row["pair_key"], "confidence": payload.get("confidence"), "reason_summary": payload.get("reason_summary")})
        else:
            unresolved.append({"candidate_id": candidate_id, "pair_key": queue_row["pair_key"], "reason": "invalid_terminal_payload"})
    accepted.sort(key=lambda edge: make_pair_key(_ref(edge["col1"]), _ref(edge["col2"])))
    output_root = args.output_root
    middle = output_root / "intermediate" / args.db_id
    write_jsonl_records(middle / "accepted_edges.jsonl", accepted)
    write_jsonl_records(middle / "rejected_pairs.jsonl", rejected)
    write_jsonl_records(middle / "unresolved_pairs.jsonl", unresolved)
    groups = [
        {
            "group_id": str(edge["candidate_id"]),
            "latent_concept": str(edge["ambiguity_trigger_context"]),
            "ambiguity_reason_concise_label": str(edge["ambiguity_trigger_context"]),
            "columns": [_ref(edge["col1"]), _ref(edge["col2"])],
            "source_edge_count": 1,
            "source_edges": [{**edge, "edge_index": index}],
            "source_format": "bridge_llm_verified_pair",
        }
        for index, edge in enumerate(accepted)
    ]
    archive_dir = output_root / args.db_id / "preprocessed"
    archive_dir.mkdir(parents=True, exist_ok=True)
    source_name = args.source_name or output_root.name
    payload = {
        "version": "bridge_auto_construct_v1",
        "source_format": "bridge_llm_verified_pair_archive",
        "source_name": source_name,
        "db_id": args.db_id,
        "source_llm_accepted_edges_jsonl": str(middle / "accepted_edges.jsonl"),
        "source_fk_edges_jsonl": None,
        "source_er_groups_jsonl": None,
        "er_relation_groups_included": False,
        "er_deterministic_pair_edges_included": False,
        "duplicate_pair_signatures_allowed": True,
        "table_descriptive_names": table_names,
        "column_descriptive_names": column_names,
        "descriptive_name_provenance": name_provenance,
        "manual_relation_groups": groups,
    }
    json_path = archive_dir / f"{args.db_id}_column_groups_manual.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with json_path.with_suffix(".pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    manifest = {
        "artifact_type": "bridge_verified_graph_archive",
        "db_id": args.db_id,
        "source_name": source_name,
        "queue_path": str(args.queue_path),
        "verifier_ledger": str(args.verifier_ledger),
        "accepted_edge_count": len(accepted),
        "rejected_pair_count": len(rejected),
        "unresolved_pair_count": len(unresolved),
        "deterministic_fk_groups_included": False,
        "deterministic_er_groups_included": False,
        "json_output": str(json_path),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
