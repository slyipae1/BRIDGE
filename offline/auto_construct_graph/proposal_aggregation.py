from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .types import column_ref_from_object, make_pair_key, split_column_ref


CHANNEL_MAP = {
    "EMBEDDING": "SCHEMA_EMBEDDING",
    "SURFACE_NAME_EMBEDDING": "SURFACE_NAME_EMBEDDING",
    "LEXICAL": "LEXICAL_SIMILARITY",
    "VALUE_COLLISION": "VALUE_COLLISION",
}

CHANNEL_ORDER = {
    "SCHEMA_EMBEDDING": 0,
    "SURFACE_NAME_EMBEDDING": 1,
    "LEXICAL_SIMILARITY": 2,
    "VALUE_COLLISION": 3,
}


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def write_jsonl_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _column_object(column_ref: str) -> dict[str, str]:
    table, column = split_column_ref(column_ref)
    return {"tab": table, "col": column}


def _source_channel(record: dict[str, Any]) -> str:
    sources = record.get("candidate_sources") or []
    raw = str(sources[0] if sources else record.get("channel") or "").strip()
    if raw not in CHANNEL_MAP:
        raise ValueError(f"unsupported candidate source: {raw!r}")
    return CHANNEL_MAP[raw]


def _proposal_id(prefix: str, source_candidate_id: str) -> str:
    safe_id = str(source_candidate_id or "unknown").replace(":", "_").replace("/", "_")
    return f"prop_{prefix}_{safe_id}"


def _candidate_pair(record: dict[str, Any]) -> tuple[str, str]:
    return make_pair_key(
        column_ref_from_object(record.get("col1")),
        column_ref_from_object(record.get("col2")),
    )


def proposal_pair_key(proposal: dict[str, Any]) -> tuple[str, str]:
    return make_pair_key(
        column_ref_from_object(proposal.get("col1")),
        column_ref_from_object(proposal.get("col2")),
    )


def _lexical_hypothesis(evidence: dict[str, Any]) -> str:
    common_tokens = evidence.get("common_tokens") or []
    rules = evidence.get("rules") or []
    if common_tokens:
        return "The column names share lexical token(s): " + ", ".join(str(token) for token in common_tokens)
    if rules:
        return "The column names satisfy lexical rule(s): " + ", ".join(str(rule) for rule in rules)
    return "The column names have surface-form lexical similarity."


def _channel_hypothesis(channel: str, record: dict[str, Any]) -> str:
    if channel == "SCHEMA_EMBEDDING":
        return "The column descriptions are close in schema-embedding space."
    if channel == "SURFACE_NAME_EMBEDDING":
        return "The raw column names are close in instruction-aware surface-name embedding space."
    if channel == "LEXICAL_SIMILARITY":
        return _lexical_hypothesis(record.get("lexical_evidence") or {})
    if channel == "VALUE_COLLISION":
        return "The columns share an eligible normalized value domain under safe_value_collision_v1."
    raise ValueError(f"unsupported proposal channel: {channel!r}")


def _channel_evidence_payload(channel: str, record: dict[str, Any]) -> dict[str, Any]:
    if channel == "SCHEMA_EMBEDDING":
        payload = dict(record.get("embedding_evidence") or {})
        metadata = record.get("embedding_metadata") or {}
        if metadata:
            payload["embedding_metadata"] = metadata
        return payload
    if channel == "SURFACE_NAME_EMBEDDING":
        payload = dict(record.get("surface_name_embedding_evidence") or {})
        metadata = record.get("surface_name_embedding_metadata") or {}
        if metadata:
            payload["embedding_metadata"] = metadata
        return payload
    if channel == "LEXICAL_SIMILARITY":
        return dict(record.get("lexical_evidence") or {})
    if channel == "VALUE_COLLISION":
        return dict(record.get("value_collision_evidence") or {})
    raise ValueError(f"unsupported proposal channel: {channel!r}")


def _literal_evidence_for_value_collision(
    source_candidate_id: str,
    literal_evidence_by_candidate_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not literal_evidence_by_candidate_id:
        return None
    evidence = literal_evidence_by_candidate_id.get(source_candidate_id)
    if not evidence:
        return None
    matched_examples = list(evidence.get("matched_value_examples") or [])[:5]
    return {
        "literal_evidence_included": bool(evidence.get("literal_evidence_included", True)),
        "matched_value_examples": matched_examples,
        "matched_value_example_count": int(evidence.get("matched_value_example_count") or len(matched_examples)),
    }


def normalize_channel_candidate(
    record: dict[str, Any],
    source_artifact: str,
    literal_evidence_by_candidate_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    channel = _source_channel(record)
    source_candidate_id = str(record.get("candidate_id") or "")
    pair_key = _candidate_pair(record)
    evidence_refs = list(record.get("evidence_refs") or [])
    proposal_ref = f"proposal:{source_candidate_id}"
    if source_candidate_id and proposal_ref not in evidence_refs:
        evidence_refs.insert(0, proposal_ref)
    literal_evidence = None
    if channel == "VALUE_COLLISION":
        literal_evidence = _literal_evidence_for_value_collision(
            source_candidate_id,
            literal_evidence_by_candidate_id,
        )
    return {
        "proposal_id": _proposal_id(channel.lower(), source_candidate_id),
        "channel": channel,
        "source_artifact": str(source_artifact),
        "source_candidate_id": source_candidate_id,
        "pair_key": list(pair_key),
        "col1": dict(record["col1"]),
        "col2": dict(record["col2"]),
        "hypothesis": _channel_hypothesis(channel, record),
        "scores": dict(record.get("source_scores") or {}),
        "evidence_refs": evidence_refs,
        "evidence_payload": _channel_evidence_payload(channel, record),
        "literal_evidence": literal_evidence,
    }


def _compact_column_context(column_ref: str, schema_profile: dict[str, Any]) -> dict[str, Any]:
    table, column = split_column_ref(column_ref)
    profile = (schema_profile.get("columns") or {}).get(column_ref) or {}
    constraints = dict(profile.get("constraints") or {})
    stats = dict(profile.get("stats") or {})
    return {
        "ref": column_ref,
        "table": table,
        "column": column,
        "data_type": profile.get("data_type") or "",
        "descriptive_name": profile.get("descriptive_name") or profile.get("column_full_name") or column,
        "column_description": profile.get("column_description") or "",
        "sample_values": [str(value) for value in (profile.get("sample_values") or [])[:5]],
        "constraints": {
            "primary_key": bool(constraints.get("primary_key")),
            "unique": bool(constraints.get("unique")),
            "foreign_keys": constraints.get("foreign_keys") or [],
        },
        "stats": {
            "row_count": stats.get("row_count"),
            "non_null_count": stats.get("non_null_count"),
            "null_fraction": stats.get("null_fraction"),
            "distinct_count": stats.get("distinct_count"),
            "distinct_ratio": stats.get("distinct_ratio"),
            "value_mode": stats.get("value_mode"),
            "text_length": stats.get("text_length") or {},
        },
    }


def _compact_table_context(table: str, schema_profile: dict[str, Any]) -> dict[str, Any]:
    table_profile = (schema_profile.get("tables") or {}).get(table) or {}
    columns = table_profile.get("columns") or []
    return {
        "table_descriptive_name": table_profile.get("table_descriptive_name") or table_profile.get("table_name") or table,
        "row_count": table_profile.get("row_count"),
        "column_count": len(columns),
    }


def _direct_fk_between(left_ref: str, right_ref: str, schema_profile: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for source_ref, target_ref in ((left_ref, right_ref), (right_ref, left_ref)):
        source_table, source_column = split_column_ref(source_ref)
        source_col = (schema_profile.get("columns") or {}).get(source_ref) or {}
        for fk in (source_col.get("constraints") or {}).get("foreign_keys") or []:
            to_table = fk.get("to_table") or fk.get("target_table") or fk.get("table")
            to_column = fk.get("to_column") or fk.get("target_column") or fk.get("column") or fk.get("to")
            from_column = fk.get("from_column") or fk.get("source_column") or fk.get("from") or source_column
            if (
                to_table
                and to_column
                and f"{to_table}.{to_column}" == target_ref
                and str(from_column) == source_column
            ):
                facts.append({"from_column": source_ref, "to_column": target_ref, "source": "schema_profile"})
    return facts


def _declared_constraint_context(left_ref: str, right_ref: str, schema_profile: dict[str, Any]) -> dict[str, Any]:
    left_constraints = _compact_column_context(left_ref, schema_profile)["constraints"]
    right_constraints = _compact_column_context(right_ref, schema_profile)["constraints"]
    return {
        "direct_fk_between_endpoints": _direct_fk_between(left_ref, right_ref, schema_profile),
        "endpoint_column_constraints": {
            left_ref: left_constraints,
            right_ref: right_constraints,
        },
        "table_pair_fk_facts": [],
    }


def _bundle_proposal_view(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": proposal["proposal_id"],
        "channel": proposal["channel"],
        "hypothesis": proposal.get("hypothesis") or "",
        "scores": proposal.get("scores") or {},
        "evidence_refs": proposal.get("evidence_refs") or [],
        "evidence_payload": proposal.get("evidence_payload") or {},
        "literal_evidence": proposal.get("literal_evidence"),
    }


def build_pair_bundles(proposals: list[dict[str, Any]], schema_profile: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal_pair_key(proposal)].append(proposal)

    bundles: list[dict[str, Any]] = []
    for index, pair_key in enumerate(sorted(grouped), start=1):
        left_ref, right_ref = pair_key
        ordered_proposals = sorted(
            grouped[pair_key],
            key=lambda proposal: (CHANNEL_ORDER.get(str(proposal.get("channel")), 99), str(proposal.get("proposal_id"))),
        )
        left_table, _ = split_column_ref(left_ref)
        right_table, _ = split_column_ref(right_ref)
        table_names = sorted({left_table, right_table})
        bundles.append(
            {
                "candidate_id": f"cand_{index:06d}",
                "pair_key": [left_ref, right_ref],
                "col1": _column_object(left_ref),
                "col2": _column_object(right_ref),
                "verification_policy": "FINAL_VERIFIER",
                "proposals_by_channel": [_bundle_proposal_view(proposal) for proposal in ordered_proposals],
                "column_context": {
                    "col1": _compact_column_context(left_ref, schema_profile),
                    "col2": _compact_column_context(right_ref, schema_profile),
                },
                "table_context": {
                    table_name: _compact_table_context(table_name, schema_profile)
                    for table_name in table_names
                },
                "declared_constraint_context": _declared_constraint_context(left_ref, right_ref, schema_profile),
                "input_policy": {
                    "scores_are_proposal_signals_not_truth": True,
                    "channel_count_is_not_acceptance_rule": True,
                    "raw_rows_are_not_supplied": True,
                    "gold_labels_are_not_supplied": True,
                },
            }
        )
    return bundles
