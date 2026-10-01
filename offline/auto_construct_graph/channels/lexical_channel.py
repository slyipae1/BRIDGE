from __future__ import annotations

from typing import Any
import difflib
import re

from auto_construct_graph.candidate_channel_config import load_candidate_channel_config
from auto_construct_graph.types import make_pair_key, split_column_ref


LEXICAL_SOURCE = "LEXICAL"


def tokenize_surface(text: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    spaced = re.sub(r"([A-Za-z])(\d)", r"\1 \2", spaced)
    spaced = re.sub(r"(\d)([A-Za-z])", r"\1 \2", spaced)
    return [part.casefold() for part in re.findall(r"[A-Za-z]+|\d+", spaced)]


def normalize_token(token: str) -> str:
    return str(token or "").casefold().strip()


def _clean_tokens(tokens: list[str]) -> list[str]:
    return [normalized for token in tokens if (normalized := normalize_token(token))]


def _char_trigrams(text: str) -> set[str]:
    if not text:
        return set()
    if len(text) < 3:
        return {text}
    return {text[index:index + 3] for index in range(len(text) - 2)}


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _rounded(value: float) -> float:
    return round(float(value), 6)


def build_column_lexical_profile(column_profile: dict[str, Any]) -> dict[str, Any]:
    tokens = _clean_tokens(tokenize_surface(str(column_profile.get("column") or "")))
    normalized_surface = " ".join(tokens)
    slot_base = None
    slot_suffix = None
    if tokens and tokens[-1].isdigit() and len(tokens) > 1:
        slot_base = " ".join(tokens[:-1])
        slot_suffix = tokens[-1]
    return {
        "ref": str(column_profile["full_name"]),
        "table": str(column_profile["table"]),
        "column": str(column_profile["column"]),
        "tokens": tokens,
        "token_set": sorted(set(tokens)),
        "normalized_surface": normalized_surface,
        "char_trigrams": sorted(_char_trigrams(normalized_surface)),
        "slot_base": slot_base,
        "slot_suffix": slot_suffix,
    }


def score_lexical_profiles(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_tokens = set(left["token_set"])
    right_tokens = set(right["token_set"])
    left_ngrams = set(left["char_trigrams"])
    right_ngrams = set(right["char_trigrams"])
    token_jaccard = _safe_divide(len(left_tokens & right_tokens), len(left_tokens | right_tokens))
    token_overlap = _safe_divide(len(left_tokens & right_tokens), min(len(left_tokens), len(right_tokens)))
    char_ngram_dice = _safe_divide(2 * len(left_ngrams & right_ngrams), len(left_ngrams) + len(right_ngrams))
    edit_similarity = difflib.SequenceMatcher(
        None,
        str(left["normalized_surface"]),
        str(right["normalized_surface"]),
    ).ratio()
    lexical_similarity = max(token_jaccard, 0.7 * token_overlap, 0.8 * char_ngram_dice, edit_similarity)
    return {
        "token_jaccard": _rounded(token_jaccard),
        "token_overlap": _rounded(token_overlap),
        "char_ngram_dice": _rounded(char_ngram_dice),
        "edit_similarity": _rounded(edit_similarity),
        "lexical_similarity": _rounded(lexical_similarity),
    }


def _column_obj(ref: str) -> dict[str, str]:
    table, column = split_column_ref(ref)
    return {"tab": table, "col": column}


def _false_positive_pattern_hint(
    *,
    rules: list[str],
    common_tokens: set[str],
    same_table: bool,
) -> str:
    if "NUMERIC_SLOT_PATTERN" in rules:
        return "NUMERIC_SLOT_PATTERN"
    if "CHAR_SIMILARITY" in rules and not common_tokens:
        return "CHAR_ONLY_MATCH"
    if len(common_tokens) == 1:
        return "SINGLE_SHARED_TOKEN"
    if not same_table:
        return "CROSS_TABLE_SURFACE_MATCH"
    if len(common_tokens) >= 2:
        return "MULTI_TOKEN_OVERLAP"
    return "OTHER_LEXICAL_SURFACE"


def _score_candidate_rules(
    left: dict[str, Any],
    right: dict[str, Any],
    scores: dict[str, float],
    thresholds: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    thresholds = (
        dict(load_candidate_channel_config()["channels"]["lexical"])
        if thresholds is None
        else dict(thresholds)
    )
    left_tokens = set(left["token_set"])
    right_tokens = set(right["token_set"])
    common_tokens = left_tokens & right_tokens
    common_long_tokens = {
        token for token in common_tokens if len(token) >= int(thresholds["common_token_min_length"])
    }
    same_table = left["table"] == right["table"]

    rules: list[str] = []
    if (
        left.get("slot_base")
        and left.get("slot_base") == right.get("slot_base")
        and left.get("slot_suffix") != right.get("slot_suffix")
    ):
        rules.append("NUMERIC_SLOT_PATTERN")
    if common_long_tokens and (
        scores["token_overlap"] >= float(thresholds["token_overlap_threshold"])
        or scores["token_jaccard"] >= float(thresholds["token_jaccard_threshold"])
    ):
        rules.append("EXACT_TOKEN_OVERLAP")
    if (
        max(len(str(left["normalized_surface"])), len(str(right["normalized_surface"])))
        >= int(thresholds["normalized_name_min_length"])
        and (
            scores["char_ngram_dice"] >= float(thresholds["char_ngram_dice_threshold"])
            or scores["edit_similarity"] >= float(thresholds["edit_similarity_threshold"])
        )
    ):
        rules.append("CHAR_SIMILARITY")

    evidence = {
        "rules": rules,
        "common_tokens": sorted(common_tokens),
        "left_tokens": left["tokens"],
        "right_tokens": right["tokens"],
        "normalized_left": left["normalized_surface"],
        "normalized_right": right["normalized_surface"],
        "false_positive_pattern_hint": _false_positive_pattern_hint(
            rules=rules,
            common_tokens=common_tokens,
            same_table=same_table,
        ),
    }
    return rules, evidence


def _relation_hints(evidence: dict[str, Any]) -> list[dict[str, str]]:
    common_tokens = evidence.get("common_tokens") or []
    if common_tokens:
        concept = "shared surface token: " + ", ".join(common_tokens[:5])
    else:
        concept = "high character-level surface similarity"
    return [{"family": "LEXICAL_SURFACE_OVERLAP", "higher_level_concept": concept}]


def extract_lexical_candidates(
    profile: dict[str, Any],
    *,
    threshold_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    thresholds = dict((threshold_config or load_candidate_channel_config())["channels"]["lexical"])
    columns = profile.get("columns") or {}
    lexical_profiles = [
        build_column_lexical_profile(columns[ref])
        for ref in sorted(columns)
        if isinstance(columns.get(ref), dict)
    ]
    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left_index, left in enumerate(lexical_profiles):
        for right in lexical_profiles[left_index + 1:]:
            pair_key = make_pair_key(left["ref"], right["ref"])
            if pair_key in seen_pairs:
                continue
            scores = score_lexical_profiles(left, right)
            rules, evidence = _score_candidate_rules(left, right, scores, thresholds)
            if not rules:
                continue
            seen_pairs.add(pair_key)
            candidates.append(
                {
                    "candidate_id": f"lexical_{len(candidates) + 1:06d}",
                    "col1": _column_obj(left["ref"]),
                    "col2": _column_obj(right["ref"]),
                    "candidate_sources": [LEXICAL_SOURCE],
                    "source_scores": scores,
                    "relation_hints": _relation_hints(evidence),
                    "evidence_refs": [f"profile:{left['ref']}", f"profile:{right['ref']}"],
                    "lexical_evidence": evidence,
                    "lexical_thresholds": thresholds,
                }
            )
    return candidates
