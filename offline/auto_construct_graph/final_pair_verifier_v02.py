from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import (
    column_ref_from_object,
    make_pair_key,
    recover_verifier_column_object,
)


VALID_FINAL_DECISIONS = {"ADD_EDGE", "REJECT"}

REQUIRED_KEYS = {
    "candidate_id",
    "col1",
    "col2",
    "reason_summary",
    "decision",
    "ambiguity_trigger_context",
    "confidence",
}


def load_final_pair_verifier_prompt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


MODEL_SAMPLE_VALUE_LIMIT = 5
MODEL_SAMPLE_VALUE_MAX_CHARS = 25


def _deduplicate_values(values: list[Any], *, limit: int = MODEL_SAMPLE_VALUE_LIMIT) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw_text = str(value or "").strip()
        text = (
            raw_text
            if len(raw_text) <= MODEL_SAMPLE_VALUE_MAX_CHARS
            else f"{raw_text[:MODEL_SAMPLE_VALUE_MAX_CHARS]}...({len(raw_text) - MODEL_SAMPLE_VALUE_MAX_CHARS} more chars)"
        )
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def project_column_context_for_verifier(
    column_context: dict[str, Any],
    *,
    prioritized_overlap_examples: list[str],
) -> dict[str, Any]:
    context = column_context or {}
    constraints = context.get("constraints") or {}
    stats = context.get("stats") or {}
    return {
        "column": context.get("column") or "",
        "table": context.get("table") or "",
        "data_type": context.get("data_type") or "",
        "descriptive_name": context.get("descriptive_name") or "",
        "column_description": context.get("column_description") or "",
        "constraints": {
            "foreign_keys": constraints.get("foreign_keys") or [],
            "primary_key": bool(constraints.get("primary_key")),
            "unique": bool(constraints.get("unique")),
        },
        "sample_values": _deduplicate_values(
            list(prioritized_overlap_examples) + list(context.get("sample_values") or [])
        ),
        "stats": {
            "distinct_count": stats.get("distinct_count"),
            "distinct_ratio": stats.get("distinct_ratio"),
            "non_null_count": stats.get("non_null_count"),
            "null_fraction": stats.get("null_fraction"),
            "row_count": stats.get("row_count"),
            "text_length": stats.get("text_length") or {},
            "value_mode": stats.get("value_mode"),
        },
    }


def project_table_context_for_verifier(table_context: dict[str, Any]) -> dict[str, Any]:
    return {
        str(table_name): {
            "column_count": (payload or {}).get("column_count"),
            "row_count": (payload or {}).get("row_count"),
            "table_descriptive_name": (payload or {}).get("table_descriptive_name") or "",
        }
        for table_name, payload in sorted((table_context or {}).items())
    }


def project_declared_constraint_context_for_verifier(context: dict[str, Any]) -> dict[str, Any]:
    context = context or {}
    endpoint_constraints = context.get("endpoint_column_constraints") or {}
    return {
        "direct_fk_between_endpoints": context.get("direct_fk_between_endpoints") or [],
        "endpoint_column_constraints": {
            str(column_ref): {
                "foreign_keys": (constraints or {}).get("foreign_keys") or [],
                "primary_key": bool((constraints or {}).get("primary_key")),
                "unique": bool((constraints or {}).get("unique")),
            }
            for column_ref, constraints in sorted(endpoint_constraints.items())
        },
        "table_pair_fk_facts": context.get("table_pair_fk_facts") or [],
    }


def project_bundle_for_verifier_context(bundle: dict[str, Any], mode: str = "full") -> dict[str, Any]:
    if mode not in {"full", "schema_only_no_channel_evidence"}:
        raise ValueError(f"unsupported verifier bundle context mode: {mode}")

    column_context = bundle.get("column_context") or {}
    raw_value_stat = bundle.get("value_overlapping_stat")
    projected_value_stat: dict[str, Any] | None = None
    if isinstance(raw_value_stat, dict):
        projected_value_stat = {
            "unique_overlap_count": int(raw_value_stat.get("unique_overlap_count") or 0),
            "samples": _deduplicate_values(list(raw_value_stat.get("samples") or [])),
        }
    projected: dict[str, Any] = {
        "candidate_id": bundle.get("candidate_id"),
        "column_context": {
        "col1": project_column_context_for_verifier(
            column_context.get("col1") or {},
            prioritized_overlap_examples=[],
        ),
        "col2": project_column_context_for_verifier(
            column_context.get("col2") or {},
            prioritized_overlap_examples=[],
        ),
        },
        "declared_constraint_context": project_declared_constraint_context_for_verifier(
            bundle.get("declared_constraint_context") or {}
        ),
        "table_context": project_table_context_for_verifier(bundle.get("table_context") or {}),
    }
    if projected_value_stat is not None:
        projected = {
            "candidate_id": projected["candidate_id"],
            "column_context": projected["column_context"],
            "value_overlapping_stat": projected_value_stat,
            "declared_constraint_context": projected["declared_constraint_context"],
            "table_context": projected["table_context"],
        }
    if mode == "full" and bundle.get("proposals_by_channel"):
        projected["proposals_by_channel"] = bundle["proposals_by_channel"]
    return projected


def build_final_pair_verifier_messages(
    prompt_text: str,
    bundle: dict[str, Any],
    bundle_context_mode: str = "full",
) -> list[dict[str, str]]:
    verifier_bundle = project_bundle_for_verifier_context(bundle, bundle_context_mode)
    bundle_json = json.dumps(verifier_bundle, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": prompt_text.strip()},
        {
            "role": "user",
            "content": (
                "Aggregated pair proposal bundle:\n"
                f"{bundle_json}\n\n"
                "Return strict JSON only. Do not include markdown fences."
            ),
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= start:
            raw = raw[start:end + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("verifier response JSON must be an object")
    return parsed


def allowed_evidence_ids(bundle: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for proposal in bundle.get("proposals_by_channel") or []:
        proposal_id = proposal.get("proposal_id")
        if proposal_id:
            allowed.add(str(proposal_id))
        for evidence_ref in proposal.get("evidence_refs") or []:
            allowed.add(str(evidence_ref))
    if bundle.get("table_context"):
        allowed.add("table_context")
    for table_name in (bundle.get("table_context") or {}):
        allowed.add(f"table_context:{table_name}")
    if bundle.get("column_context"):
        allowed.add("column_context")
    for context_key, context_payload in (bundle.get("column_context") or {}).items():
        if isinstance(context_payload, dict):
            column_ref = context_payload.get("ref")
            if column_ref:
                allowed.add(f"column_context:{column_ref}")
        if "." in str(context_key):
            allowed.add(f"column_context:{context_key}")
    return allowed


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def supporting_evidence_ids(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    supporting_evidence = payload.get("supporting_evidence")
    if not isinstance(supporting_evidence, list):
        return ids
    for item in supporting_evidence:
        if isinstance(item, dict) and item.get("evidence_id"):
            ids.append(str(item.get("evidence_id")))
    return ids


def _payload_pair_key(payload: dict[str, Any]) -> tuple[str, str]:
    return make_pair_key(
        column_ref_from_object(payload.get("col1")),
        column_ref_from_object(payload.get("col2")),
    )


def _bundle_pair_key(bundle: dict[str, Any]) -> tuple[str, str]:
    pair_key = bundle.get("pair_key")
    if isinstance(pair_key, list) and len(pair_key) == 2:
        return make_pair_key(str(pair_key[0]), str(pair_key[1]))
    return make_pair_key(
        column_ref_from_object(bundle.get("col1")),
        column_ref_from_object(bundle.get("col2")),
    )


def _ordered_payload_pair(payload: dict[str, Any]) -> tuple[str, str]:
    return (
        column_ref_from_object(payload.get("col1")),
        column_ref_from_object(payload.get("col2")),
    )


def _ordered_bundle_pair(bundle: dict[str, Any]) -> tuple[str, str]:
    return (
        column_ref_from_object(bundle.get("col1")),
        column_ref_from_object(bundle.get("col2")),
    )


def normalize_final_verifier_payload(payload: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["col1"] = recover_verifier_column_object(payload.get("col1"), bundle.get("col1"))
    normalized["col2"] = recover_verifier_column_object(payload.get("col2"), bundle.get("col2"))
    return normalized


def validate_final_verifier_payload(payload: dict[str, Any], bundle: dict[str, Any]) -> list[str]:
    payload = normalize_final_verifier_payload(payload, bundle)
    errors: list[str] = []
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")

    extra_keys = sorted(set(payload) - REQUIRED_KEYS)
    for key in extra_keys:
        errors.append(f"unexpected key not allowed in v2 final verifier output: {key}")

    if payload.get("candidate_id") != bundle.get("candidate_id"):
        errors.append(
            f"candidate_id mismatch: expected {bundle.get('candidate_id')} got {payload.get('candidate_id')}"
        )

    decision = payload.get("decision")
    if decision not in VALID_FINAL_DECISIONS:
        errors.append(f"invalid decision: {decision}")

    try:
        if not isinstance(payload.get("col1"), dict):
            errors.append("col1 must be an object")
        if not isinstance(payload.get("col2"), dict):
            errors.append("col2 must be an object")
        if _ordered_payload_pair(payload) != _ordered_bundle_pair(bundle):
            errors.append("ordered column mismatch")
        if _payload_pair_key(payload) != _bundle_pair_key(bundle):
            errors.append("unordered pair mismatch")
    except Exception as exc:
        errors.append(f"invalid column payload: {exc}")

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"confidence must be a number in [0, 1], got {confidence!r}")

    if not _non_empty_string(payload.get("reason_summary")):
        errors.append("reason_summary must be a non-empty string")

    if decision == "ADD_EDGE":
        if not _non_empty_string(payload.get("ambiguity_trigger_context")):
            errors.append("ADD_EDGE requires non-empty ambiguity_trigger_context")
    elif decision == "REJECT":
        if payload.get("ambiguity_trigger_context") is not None:
            errors.append("REJECT requires ambiguity_trigger_context = null")

    return errors


def decision_to_lean_edge(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "col1": payload["col1"],
        "col2": payload["col2"],
        "reason_summary": payload["reason_summary"],
        "ambiguity_trigger_context": payload["ambiguity_trigger_context"],
        "confidence": float(payload.get("confidence") or 0.0),
    }
