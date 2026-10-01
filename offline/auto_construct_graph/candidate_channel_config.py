"""Validated, reproducible configuration for non-deterministic candidate channels."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


CANONICAL_CANDIDATE_CHANNELS = (
    "lexical",
    "schema_embedding",
    "value_collision",
    "surface_name_embedding",
)

CHANNEL_SOURCE_BY_KEY = {
    "lexical": "LEXICAL",
    "schema_embedding": "EMBEDDING",
    "value_collision": "VALUE_COLLISION",
    "surface_name_embedding": "SURFACE_NAME_EMBEDDING",
}

CHANNEL_REPORTING_FAMILY = {
    "lexical": "surface_name_similarity",
    "schema_embedding": "schema_similarity",
    "value_collision": "value_domain_similarity",
    "surface_name_embedding": "surface_name_similarity",
}

# Public defaults are the frozen BRIDGE mainline proposal policy. The queue
# applies the additional schema rank <= 10 selector after materialization.
DEFAULT_CANDIDATE_CHANNEL_CONFIG: dict[str, Any] = {
    "config_version": "candidate_channel_config_v1",
    "channels": {
        "lexical": {
            "common_token_min_length": 3,
            "token_overlap_threshold": 0.50,
            "token_jaccard_threshold": 0.30,
            "normalized_name_min_length": 5,
            "char_ngram_dice_threshold": 0.60,
            "edit_similarity_threshold": 0.70,
        },
        "schema_embedding": {
            "document_mode": "column_descriptive_name_only",
            "selection_mode": "score_threshold",
            "top_k": None,
            "similarity_threshold": 0.55,
            "minimum_similarity_threshold": None,
        },
        "value_collision": {
            "large_domain_breakpoint": 10000,
            "large_domain_min_intersection": 30,
            "medium_domain_breakpoint": 1000,
            "medium_domain_min_intersection": 10,
            "small_domain_min_intersection": 3,
            "low_card_min_intersection": 2,
            "low_card_overlap_threshold": 1.0,
            "long_text_min_intersection": 3,
            "long_text_overlap_threshold": 0.5,
            "general_overlap_threshold": 0.5,
            "general_containment_threshold": 0.8,
        },
        "surface_name_embedding": {
            "selection_mode": "score_threshold",
            "similarity_threshold": 0.60,
            "threshold_status": "frozen_mainline",
        },
    },
}


def _finite_number(value: Any, *, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {parsed}")
    return parsed


def _positive_integer(value: Any, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {parsed}")
    return parsed


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_candidate_channel_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("candidate channel config must be a JSON object")
    channels = config.get("channels")
    if not isinstance(channels, Mapping):
        raise ValueError("candidate channel config requires an object field 'channels'")
    unknown = sorted(set(channels) - set(CANONICAL_CANDIDATE_CHANNELS))
    if unknown:
        raise ValueError(f"unknown candidate channel config sections: {unknown}")
    missing = sorted(set(CANONICAL_CANDIDATE_CHANNELS) - set(channels))
    if missing:
        raise ValueError(f"candidate channel config misses sections: {missing}")

    normalized = copy.deepcopy(dict(config))
    normalized["config_version"] = str(normalized.get("config_version") or "candidate_channel_config_v1")
    normalized_channels: dict[str, dict[str, Any]] = {}
    for key in CANONICAL_CANDIDATE_CHANNELS:
        raw = channels.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"channels.{key} must be an object")
        normalized_channels[key] = dict(raw)

    lexical = normalized_channels["lexical"]
    for field in (
        "token_overlap_threshold",
        "token_jaccard_threshold",
        "char_ngram_dice_threshold",
        "edit_similarity_threshold",
    ):
        lexical[field] = _finite_number(lexical.get(field), field=f"channels.lexical.{field}", minimum=0.0, maximum=1.0)
    for field in ("common_token_min_length", "normalized_name_min_length"):
        lexical[field] = _positive_integer(lexical.get(field), field=f"channels.lexical.{field}")

    value = normalized_channels["value_collision"]
    for field in (
        "large_domain_breakpoint",
        "large_domain_min_intersection",
        "medium_domain_breakpoint",
        "medium_domain_min_intersection",
        "small_domain_min_intersection",
        "low_card_min_intersection",
        "long_text_min_intersection",
    ):
        value[field] = _positive_integer(value.get(field), field=f"channels.value_collision.{field}")
    if value["large_domain_breakpoint"] <= value["medium_domain_breakpoint"]:
        raise ValueError("channels.value_collision.large_domain_breakpoint must exceed medium_domain_breakpoint")
    for field in (
        "low_card_overlap_threshold",
        "long_text_overlap_threshold",
        "general_overlap_threshold",
        "general_containment_threshold",
    ):
        value[field] = _finite_number(value.get(field), field=f"channels.value_collision.{field}", minimum=0.0, maximum=1.0)

    schema = normalized_channels["schema_embedding"]
    selection_mode = str(schema.get("selection_mode") or "").strip()
    if selection_mode != "score_threshold":
        raise ValueError("channels.schema_embedding.selection_mode must be 'score_threshold'")
    schema["selection_mode"] = selection_mode
    schema["document_mode"] = str(schema.get("document_mode") or "").strip()
    schema["top_k"] = None
    schema["similarity_threshold"] = _finite_number(
        schema.get("similarity_threshold"),
        field="channels.schema_embedding.similarity_threshold",
        minimum=-1.0,
        maximum=1.0,
    )
    if schema.get("minimum_similarity_threshold") is not None:
        raise ValueError("channels.schema_embedding.minimum_similarity_threshold must be null for score_threshold")
    schema["minimum_similarity_threshold"] = None

    surface = normalized_channels["surface_name_embedding"]
    if str(surface.get("selection_mode") or "").strip() != "score_threshold":
        raise ValueError("channels.surface_name_embedding.selection_mode must be 'score_threshold'")
    surface["selection_mode"] = "score_threshold"
    surface["similarity_threshold"] = _finite_number(
        surface.get("similarity_threshold"),
        field="channels.surface_name_embedding.similarity_threshold",
        minimum=-1.0,
        maximum=1.0,
    )
    surface["threshold_status"] = str(surface.get("threshold_status") or "")

    normalized["channels"] = normalized_channels
    return normalized


def load_candidate_channel_config(path: Path | None = None) -> dict[str, Any]:
    """Load an optional JSON override over the stable default candidate policy."""
    overrides: dict[str, Any] = {}
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        overrides = payload
    return validate_candidate_channel_config(_deep_merge(DEFAULT_CANDIDATE_CHANNEL_CONFIG, overrides))


def candidate_channel_config_digest(config: Mapping[str, Any]) -> str:
    validated = validate_candidate_channel_config(config)
    encoded = json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_enabled_candidate_channels(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    selected = tuple(str(item).strip() for item in raw_items if str(item).strip())
    if not selected:
        raise ValueError("enabled candidate channels must be non-empty")
    unknown = sorted(set(selected) - set(CANONICAL_CANDIDATE_CHANNELS))
    if unknown:
        raise ValueError(f"unknown enabled candidate channels: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("enabled candidate channels must not contain duplicates")
    return selected
