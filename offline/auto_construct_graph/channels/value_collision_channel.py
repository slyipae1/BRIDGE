from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import re
import sqlite3
import unicodedata

from auto_construct_graph.candidate_channel_config import load_candidate_channel_config
from auto_construct_graph.schema_profile import percentile, quote_identifier
from auto_construct_graph.types import make_pair_key, split_column_ref


VALUE_COLLISION_SOURCE = "VALUE_COLLISION"
VALUE_COLLISION_POLICY_VERSION = "safe_value_collision_v1"
MISSING_NORMALIZED_VALUES = {"", "null", "none", "n/a", "na", "nan", "unknown", "not available"}
MODE_ORDER = ["CODE_OR_IDENTIFIER", "LOW_CARD_DOMAIN", "LONG_TEXT_HASH", "LABEL_TEXT"]


def normalize_value_for_collision(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.strip().casefold().split())


def is_missing_normalized_value(value: str) -> bool:
    return str(value or "").strip().casefold() in MISSING_NORMALIZED_VALUES


def classify_value_pattern(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "empty"
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text):
        return "uuid_like"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ t].*)?", text):
        return "date_like"
    if re.fullmatch(r"[+-]?\d+", text):
        return "integer_like"
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+\.\d*|\.\d+)", text):
        return "decimal_like"
    if re.fullmatch(r"[a-z]+", text):
        return "short_text" if len(text) <= 32 else "long_text"
    if re.fullmatch(r"[a-z0-9][a-z0-9_.:/+-]*", text) and any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
        return "alnum_code_like"
    if len(text) >= 80:
        return "long_text"
    return "other_text"


def _tokenize_identifier_surface(text: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return {part.casefold() for part in re.findall(r"[A-Za-z]+|\d+", spaced)}


def _has_identifier_signal(column_profile: dict[str, Any]) -> bool:
    column = str(column_profile.get("column") or "")
    description = str(column_profile.get("column_description") or "")
    name_tokens = _tokenize_identifier_surface(column)
    description_tokens = _tokenize_identifier_surface(description)
    name_folded = column.casefold()
    if name_folded == "id" or name_folded.endswith("id") or name_folded.endswith("_id"):
        return True
    if {"id", "uuid", "key", "code"} & name_tokens:
        return True
    if {"id", "identifier", "uuid", "key", "code"} & description_tokens:
        return True
    return False


def _boolean_like_domain(values: set[str]) -> bool:
    if not values:
        return False
    boolean_domains = [
        {"0", "1"},
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
    ]
    return any(values <= domain for domain in boolean_domains)


def _dominant_pattern(values: set[str]) -> str:
    if not values:
        return "empty"
    counts = Counter(classify_value_pattern(value) for value in values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _median_length(values: set[str]) -> float | None:
    if not values:
        return None
    return percentile([float(len(value)) for value in values], 0.50)


def _is_numeric_mode(column_profile: dict[str, Any]) -> bool:
    stats = column_profile.get("stats") or {}
    return str(stats.get("value_mode") or "") == "NUMERIC_MEASURE"


def _is_date_mode(column_profile: dict[str, Any], dominant_pattern: str) -> bool:
    stats = column_profile.get("stats") or {}
    return str(stats.get("value_mode") or "") == "DATE_LIKE" or dominant_pattern == "date_like"


def infer_column_eligibility(column_profile: dict[str, Any], normalized_values: set[str]) -> dict[str, Any]:
    stats = column_profile.get("stats") or {}
    constraints = column_profile.get("constraints") or {}
    value_mode = str(stats.get("value_mode") or "")
    distinct_count = len(normalized_values)
    dominant_pattern = _dominant_pattern(normalized_values)
    length_p50 = _median_length(normalized_values)
    boolean_like = _boolean_like_domain(normalized_values)
    identifier_gate = (
        value_mode == "CODE_OR_IDENTIFIER"
        or bool(constraints.get("primary_key"))
        or bool(constraints.get("unique"))
        or bool(constraints.get("foreign_keys"))
        or _has_identifier_signal(column_profile)
    )
    date_like = _is_date_mode(column_profile, dominant_pattern)
    numeric_measure_without_identifier = _is_numeric_mode(column_profile) and not identifier_gate

    modes: list[str] = []
    excluded_reasons: list[str] = []
    if distinct_count == 0:
        excluded_reasons.append("EMPTY_DOMAIN")
    if distinct_count == 1:
        excluded_reasons.append("SINGLE_VALUE_DOMAIN")
    if boolean_like:
        excluded_reasons.append("BOOLEAN_LIKE_DOMAIN")
    if date_like:
        excluded_reasons.append("DATE_LIKE_DOMAIN")
    if numeric_measure_without_identifier:
        excluded_reasons.append("NUMERIC_MEASURE")

    if identifier_gate and distinct_count >= 2 and not boolean_like and not date_like:
        modes.append("CODE_OR_IDENTIFIER")
    if (
        2 <= distinct_count <= 10
        and not boolean_like
        and not date_like
        and not numeric_measure_without_identifier
        and (length_p50 is None or length_p50 <= 32)
    ):
        modes.append("LOW_CARD_DOMAIN")
    if (
        value_mode == "LABEL_TEXT"
        and distinct_count >= 3
        and not boolean_like
        and not date_like
        and not numeric_measure_without_identifier
        and length_p50 is not None
        and length_p50 >= 80
    ):
        modes.append("LONG_TEXT_HASH")
    if (
        value_mode in {"LABEL_TEXT", "LOW_CARD_DOMAIN"}
        and distinct_count >= 3
        and not boolean_like
        and not date_like
        and not numeric_measure_without_identifier
        and (length_p50 is None or length_p50 < 80)
    ):
        modes.append("LABEL_TEXT")

    deduped_modes: list[str] = []
    for mode in MODE_ORDER:
        if mode in modes and mode not in deduped_modes:
            deduped_modes.append(mode)
    return {
        "eligibility_modes": deduped_modes,
        "excluded_reasons": sorted(set(excluded_reasons)),
        "dominant_pattern_class": dominant_pattern,
        "length_p50": length_p50,
        "boolean_like_domain": boolean_like,
    }


def build_column_value_profile(
    conn: sqlite3.Connection,
    schema_profile: dict[str, Any],
    column_ref: str,
) -> dict[str, Any]:
    column_profile = (schema_profile.get("columns") or {}).get(column_ref)
    if not isinstance(column_profile, dict):
        raise KeyError(f"column not found in schema profile: {column_ref}")
    table_name, column_name = split_column_ref(column_ref)
    normalized_values: set[str] = set()
    normalized_non_empty_count = 0
    non_null_count = 0
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column_name)
    for row in conn.execute(f"SELECT {column_sql} FROM {table_sql} WHERE {column_sql} IS NOT NULL"):
        non_null_count += 1
        normalized = normalize_value_for_collision(row[0])
        if is_missing_normalized_value(normalized):
            continue
        normalized_non_empty_count += 1
        normalized_values.add(normalized)

    eligibility = infer_column_eligibility(column_profile, normalized_values)
    return {
        "table": table_name,
        "column": column_name,
        "full_name": column_ref,
        "non_null_count": non_null_count,
        "normalized_non_empty_count": normalized_non_empty_count,
        "normalized_distinct_count": len(normalized_values),
        "normalized_distinct_values": normalized_values,
        **eligibility,
    }


def build_value_profiles(sqlite_path: Path, schema_profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    with sqlite3.connect(sqlite_path) as conn:
        return {
            column_ref: build_column_value_profile(conn, schema_profile, column_ref)
            for column_ref in sorted(schema_profile.get("columns") or {})
        }


def compute_overlap_metrics(left_values: set[str], right_values: set[str]) -> dict[str, float | int]:
    intersection = left_values & right_values
    union = left_values | right_values
    left_count = len(left_values)
    right_count = len(right_values)
    intersection_count = len(intersection)
    min_count = min(left_count, right_count)
    return {
        "left_distinct_count": left_count,
        "right_distinct_count": right_count,
        "intersection_count": intersection_count,
        "jaccard": float(intersection_count) / float(len(union)) if union else 0.0,
        "overlap_coefficient": float(intersection_count) / float(min_count) if min_count else 0.0,
        "containment_left": float(intersection_count) / float(left_count) if left_count else 0.0,
        "containment_right": float(intersection_count) / float(right_count) if right_count else 0.0,
    }


def _dynamic_min_intersection(
    left_count: int,
    right_count: int,
    thresholds: dict[str, Any],
) -> int:
    max_count = max(int(left_count), int(right_count))
    if max_count >= int(thresholds["large_domain_breakpoint"]):
        return int(thresholds["large_domain_min_intersection"])
    if max_count >= int(thresholds["medium_domain_breakpoint"]):
        return int(thresholds["medium_domain_min_intersection"])
    return int(thresholds["small_domain_min_intersection"])


def _pattern_group(pattern: str) -> str:
    if pattern in {"integer_like", "decimal_like"}:
        return "numeric"
    if pattern in {"uuid_like", "alnum_code_like"}:
        return "code"
    if pattern == "date_like":
        return "date"
    if pattern in {"short_text", "long_text", "other_text"}:
        return "text"
    return "other"


def _patterns_compatible(mode: str, left_pattern: str, right_pattern: str) -> bool:
    left_group = _pattern_group(left_pattern)
    right_group = _pattern_group(right_pattern)
    if "date" in {left_group, right_group}:
        return False
    if mode == "CODE_OR_IDENTIFIER":
        return left_group == right_group or "code" in {left_group, right_group}
    if mode in {"LABEL_TEXT", "LONG_TEXT_HASH"}:
        return left_group == "text" and right_group == "text"
    if mode == "LOW_CARD_DOMAIN":
        return left_group == right_group or "code" in {left_group, right_group}
    return False


def _passes_mode_threshold(
    mode: str,
    metrics: dict[str, float | int],
    left_pattern: str,
    right_pattern: str,
    thresholds: dict[str, Any],
) -> bool:
    intersection_count = int(metrics["intersection_count"])
    left_count = int(metrics["left_distinct_count"])
    right_count = int(metrics["right_distinct_count"])
    overlap = float(metrics["overlap_coefficient"])
    containment = max(float(metrics["containment_left"]), float(metrics["containment_right"]))

    if not _patterns_compatible(mode, left_pattern, right_pattern):
        return False
    if mode == "LOW_CARD_DOMAIN":
        return (
            intersection_count >= int(thresholds["low_card_min_intersection"])
            and overlap >= float(thresholds["low_card_overlap_threshold"])
        )
    min_intersection = _dynamic_min_intersection(left_count, right_count, thresholds)
    if mode == "LONG_TEXT_HASH":
        return (
            intersection_count >= int(thresholds["long_text_min_intersection"])
            and overlap >= float(thresholds["long_text_overlap_threshold"])
        )
    return intersection_count >= min_intersection and (
        overlap >= float(thresholds["general_overlap_threshold"])
        or containment >= float(thresholds["general_containment_threshold"])
    )


def _candidate_priority(mode: str) -> str:
    if mode == "CODE_OR_IDENTIFIER":
        return "high"
    if mode == "LOW_CARD_DOMAIN":
        return "low"
    return "medium"


def _column_obj(ref: str) -> dict[str, str]:
    table, column = split_column_ref(ref)
    return {"tab": table, "col": column}


def _rounded_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    rounded: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _make_evidence_payload(
    *,
    mode: str,
    priority: str,
    metrics: dict[str, float | int],
    left_profile: dict[str, Any],
    right_profile: dict[str, Any],
    include_literal_evidence: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "eligibility_mode": mode,
        "candidate_priority": priority,
        **_rounded_metrics(metrics),
        "left_pattern_class": left_profile.get("dominant_pattern_class"),
        "right_pattern_class": right_profile.get("dominant_pattern_class"),
        "threshold_policy": VALUE_COLLISION_POLICY_VERSION,
        "literal_evidence_included": bool(include_literal_evidence),
    }
    if include_literal_evidence:
        intersection = sorted(
            set(left_profile.get("normalized_distinct_values") or set())
            & set(right_profile.get("normalized_distinct_values") or set())
        )
        payload["matched_value_examples"] = intersection[:5]
        payload["matched_value_example_count"] = len(intersection[:5])
    return payload


def extract_value_collision_candidates(
    value_profiles: dict[str, dict[str, Any]],
    include_literal_evidence: bool = True,
    *,
    threshold_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = dict((threshold_config or load_candidate_channel_config())["channels"]["value_collision"])
    candidates: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    column_refs = sorted(value_profiles)
    for left_index, left_ref in enumerate(column_refs):
        left_profile = value_profiles[left_ref]
        left_values = set(left_profile.get("normalized_distinct_values") or set())
        left_modes = set(left_profile.get("eligibility_modes") or [])
        if not left_values or not left_modes:
            continue
        for right_ref in column_refs[left_index + 1:]:
            right_profile = value_profiles[right_ref]
            right_values = set(right_profile.get("normalized_distinct_values") or set())
            right_modes = set(right_profile.get("eligibility_modes") or [])
            shared_modes = [mode for mode in MODE_ORDER if mode in left_modes and mode in right_modes]
            if not right_values or not shared_modes:
                continue
            metrics = compute_overlap_metrics(left_values, right_values)
            if int(metrics["intersection_count"]) == 0:
                continue
            chosen_mode = None
            for mode in shared_modes:
                if _passes_mode_threshold(
                    mode,
                    metrics,
                    str(left_profile.get("dominant_pattern_class") or ""),
                    str(right_profile.get("dominant_pattern_class") or ""),
                    thresholds,
                ):
                    chosen_mode = mode
                    break
            if chosen_mode is None:
                continue
            priority = _candidate_priority(chosen_mode)
            evidence_payload = _make_evidence_payload(
                mode=chosen_mode,
                priority=priority,
                metrics=metrics,
                left_profile=left_profile,
                right_profile=right_profile,
                include_literal_evidence=include_literal_evidence,
            )
            candidate_id = f"value_collision_{len(candidates) + 1:06d}"
            candidate = {
                "candidate_id": candidate_id,
                "col1": _column_obj(left_ref),
                "col2": _column_obj(right_ref),
                "candidate_sources": [VALUE_COLLISION_SOURCE],
                "source_scores": {
                    key: value
                    for key, value in evidence_payload.items()
                    if key in {
                        "intersection_count",
                        "jaccard",
                        "overlap_coefficient",
                        "containment_left",
                        "containment_right",
                    }
                },
                "relation_hints": [
                    {
                        "family": "VALUE_DOMAIN_OVERLAP",
                        "higher_level_concept": "shared identifier or label value domain",
                    }
                ],
                "evidence_refs": [
                    f"profile:{left_ref}",
                    f"profile:{right_ref}",
                    f"value_profile:{left_ref}",
                    f"value_profile:{right_ref}",
                    f"value_collision_evidence:{candidate_id}",
                ],
                "value_collision_evidence": {
                    **{
                        key: value
                        for key, value in evidence_payload.items()
                        if key not in {"matched_value_examples", "matched_value_example_count"}
                    },
                    "resolved_thresholds": thresholds,
                },
            }
            evidence_record = {
                "candidate_id": candidate_id,
                "col1": _column_obj(left_ref),
                "col2": _column_obj(right_ref),
                **evidence_payload,
            }
            candidates.append(candidate)
            evidence_records.append(evidence_record)
    return candidates, evidence_records
