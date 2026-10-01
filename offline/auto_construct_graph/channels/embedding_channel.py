from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from auto_construct_graph.types import make_pair_key, split_column_ref


EMBEDDING_SOURCE = "EMBEDDING"
EMBEDDING_RECORD_VERSION = "schema_text_v0"
DESCRIPTIVE_NAME_ONLY_RECORD_VERSION = "column_descriptive_name_only_v1"
EMBEDDING_DOCUMENT_MODE_RICH_SCHEMA_TEXT = "rich_schema_text"
EMBEDDING_DOCUMENT_MODE_COLUMN_DESCRIPTIVE_NAME_ONLY = "column_descriptive_name_only"
EMBEDDING_SELECTION_TOP_K = "top_k"
EMBEDDING_SELECTION_SCORE_THRESHOLD = "score_threshold"
EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR = "top_k_or_score_threshold_with_floor"
SYMMETRIC_DOC_DOC_MODE = "symmetric_doc_doc"
QWEN_QUERY_DOC_MODE = "qwen_query_doc"
DOCUMENT_EMBEDDING_VIEW = "document"
QUERY_EMBEDDING_VIEW = "query"
QWEN_QUERY_INSTRUCTION = (
    "Retrieve schema columns that could be ambiguous alternatives for an underspecified natural-language database request."
)
FORBIDDEN_PROFILE_FIELDS = [
    "sample_values",
    "top_values",
    "distinct_values",
    "value_description",
]


def _column_obj(ref: str) -> dict[str, str]:
    table, column = split_column_ref(ref)
    return {"tab": table, "col": column}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_embedding_document_mode(document_mode: str | None = None) -> str:
    """Return one supported embedding document contract."""
    mode = str(document_mode or EMBEDDING_DOCUMENT_MODE_RICH_SCHEMA_TEXT).strip()
    if mode in {
        EMBEDDING_DOCUMENT_MODE_RICH_SCHEMA_TEXT,
        EMBEDDING_DOCUMENT_MODE_COLUMN_DESCRIPTIVE_NAME_ONLY,
    }:
        return mode
    raise ValueError(
        "document_mode must be one of "
        f"{EMBEDDING_DOCUMENT_MODE_RICH_SCHEMA_TEXT!r}, "
        f"{EMBEDDING_DOCUMENT_MODE_COLUMN_DESCRIPTIVE_NAME_ONLY!r}"
    )


def embedding_text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def build_query_embedding_text(document_text: str, instruction: str = QWEN_QUERY_INSTRUCTION) -> str:
    return f"Instruct: {instruction}\nQuery: {document_text}"


def query_instruction_sha256(instruction: str = QWEN_QUERY_INSTRUCTION) -> str:
    return embedding_text_sha256(instruction)


def abstract_value_semantics(column_profile: dict[str, Any]) -> str:
    stats = column_profile.get("stats") or {}
    constraints = column_profile.get("constraints") or {}
    value_mode = str(stats.get("value_mode") or "OTHER")
    null_fraction = _safe_float(stats.get("null_fraction"))
    distinct_ratio = _safe_float(stats.get("distinct_ratio"))
    text_length = stats.get("text_length") or {}
    p50_length = _safe_float(text_length.get("p50"), default=-1.0)

    labels: list[str] = []
    mode_labels = {
        "BOOLEAN_LIKE": "boolean-like categorical value",
        "CODE_OR_IDENTIFIER": "code or identifier",
        "DATE_LIKE": "date or time value",
        "NUMERIC_MEASURE": "numeric measure",
        "LOW_CARD_DOMAIN": "low-cardinality categorical label",
        "LABEL_TEXT": "text label",
        "OTHER": "other schema value",
    }
    labels.append(mode_labels.get(value_mode, "other schema value"))

    if bool(constraints.get("primary_key")):
        labels.append("primary key")
    if bool(constraints.get("unique")):
        labels.append("unique column")
    if constraints.get("foreign_keys"):
        labels.append("foreign key column")
    if distinct_ratio >= 0.95:
        labels.append("mostly unique values")
    elif 0.0 < distinct_ratio <= 0.05:
        labels.append("many repeated values")
    if null_fraction >= 0.90:
        labels.append("high-null column")
    elif null_fraction <= 0.01:
        labels.append("mostly populated column")
    if value_mode == "LABEL_TEXT" and p50_length >= 80:
        labels.append("long free-form text")
    elif value_mode == "LABEL_TEXT" and 0 <= p50_length <= 32:
        labels.append("short text label")

    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            deduped.append(label)
            seen.add(label)
    return "; ".join(deduped)


def build_embedding_record(profile: dict[str, Any], column_ref: str, column_profile: dict[str, Any]) -> dict[str, Any]:
    table_name = str(column_profile.get("table") or "")
    table_profile = (profile.get("tables") or {}).get(table_name) or {}
    field_values = [
        ("table name", table_name),
        ("table descriptive name", table_profile.get("table_descriptive_name")),
        ("column name", column_profile.get("column")),
        ("column descriptive name", column_profile.get("descriptive_name")),
        ("column description", column_profile.get("column_description")),
        ("abstract value semantics", abstract_value_semantics(column_profile)),
    ]
    lines = [f"{label}: {cleaned}" for label, value in field_values if (cleaned := _clean_text(value))]
    embedding_text = "\n".join(lines)
    return {
        "column_id": str(column_ref),
        "table_name": table_name,
        "column_name": str(column_profile.get("column") or ""),
        "embedding_text": embedding_text,
        "embedding_text_sha256": embedding_text_sha256(embedding_text),
        "record_version": EMBEDDING_RECORD_VERSION,
        "document_mode": EMBEDDING_DOCUMENT_MODE_RICH_SCHEMA_TEXT,
        "included_fields": [
            "table_name",
            "table_descriptive_name",
            "column_name",
            "column_descriptive_name",
            "column_description",
            "abstract_value_semantics",
        ],
        "excluded_fields": [
            "sample_values",
            "top_values",
            "distinct_values",
            "value_description",
            "gold_labels",
        ],
    }


def build_embedding_records(
    profile: dict[str, Any],
    *,
    document_mode: str = EMBEDDING_DOCUMENT_MODE_RICH_SCHEMA_TEXT,
) -> list[dict[str, Any]]:
    mode = resolve_embedding_document_mode(document_mode)
    columns = profile.get("columns") or {}
    if mode == EMBEDDING_DOCUMENT_MODE_COLUMN_DESCRIPTIVE_NAME_ONLY:
        return build_descriptive_name_only_embedding_records(profile)
    return [
        build_embedding_record(profile, ref, columns[ref])
        for ref in sorted(columns)
        if isinstance(columns.get(ref), dict)
    ]


def build_descriptive_name_only_embedding_record(
    column_ref: str,
    column_profile: dict[str, Any],
) -> dict[str, Any]:
    """Create the minimal POC record whose document is only the descriptive name."""
    descriptive_name = _clean_text(column_profile.get("descriptive_name"))
    if not descriptive_name:
        raise ValueError(f"column has no descriptive name for embedding POC: {column_ref}")
    return {
        "column_id": str(column_ref),
        "table_name": str(column_profile.get("table") or ""),
        "column_name": str(column_profile.get("column") or ""),
        "column_descriptive_name": descriptive_name,
        "embedding_text": descriptive_name,
        "embedding_text_sha256": embedding_text_sha256(descriptive_name),
        "record_version": DESCRIPTIVE_NAME_ONLY_RECORD_VERSION,
        "document_mode": EMBEDDING_DOCUMENT_MODE_COLUMN_DESCRIPTIVE_NAME_ONLY,
        "included_fields": ["column_descriptive_name_only"],
        "excluded_fields": [
            "table_name",
            "table_descriptive_name",
            "column_name",
            "column_description",
            "abstract_value_semantics",
            "sample_values",
            "top_values",
            "distinct_values",
            "value_description",
            "gold_labels",
        ],
    }


def build_descriptive_name_only_embedding_records(profile: dict[str, Any]) -> list[dict[str, Any]]:
    columns = profile.get("columns") or {}
    return [
        build_descriptive_name_only_embedding_record(ref, columns[ref])
        for ref in sorted(columns)
        if isinstance(columns.get(ref), dict)
    ]


def audit_descriptive_name_only_embedding_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_column_ids: list[str] = []
    seen_column_ids: set[str] = set()
    invalid_records: list[dict[str, str]] = []
    for record in records:
        column_id = str(record.get("column_id") or "")
        if column_id in seen_column_ids:
            duplicate_column_ids.append(column_id)
        seen_column_ids.add(column_id)
        descriptive_name = _clean_text(record.get("column_descriptive_name"))
        embedding_text = str(record.get("embedding_text") or "")
        if (
            str(record.get("record_version") or "") != DESCRIPTIVE_NAME_ONLY_RECORD_VERSION
            or str(record.get("document_mode") or "")
            not in {"", EMBEDDING_DOCUMENT_MODE_COLUMN_DESCRIPTIVE_NAME_ONLY}
            or not descriptive_name
            or embedding_text != descriptive_name
        ):
            invalid_records.append(
                {
                    "column_id": column_id,
                    "record_version": str(record.get("record_version") or ""),
                    "embedding_text": embedding_text,
                    "column_descriptive_name": descriptive_name,
                }
            )
    return {
        "record_count": len(records),
        "record_version": DESCRIPTIVE_NAME_ONLY_RECORD_VERSION,
        "document_text_policy": "embedding_text equals column_descriptive_name exactly",
        "duplicate_column_id_count": len(duplicate_column_ids),
        "duplicate_column_ids": sorted(set(duplicate_column_ids)),
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records,
    }


def ensure_descriptive_name_only_record_audit_passes(audit: dict[str, Any]) -> None:
    if int(audit.get("duplicate_column_id_count") or 0):
        raise ValueError(f"embedding POC records contain duplicate column ids: {audit['duplicate_column_ids'][:5]}")
    if int(audit.get("invalid_record_count") or 0):
        raise ValueError(f"embedding POC records violate descriptive-name-only policy: {audit['invalid_records'][:3]}")


def audit_embedding_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_hits: list[dict[str, Any]] = []
    empty_count = 0
    max_text_length = 0
    seen_column_ids: set[str] = set()
    duplicate_column_ids: set[str] = set()

    forbidden_text_labels = {
        "sample value",
        "sample values",
        "top value",
        "top values",
        "distinct value",
        "distinct values",
        "value description",
        "gold label",
        "gold labels",
    }

    for record in records:
        column_id = str(record.get("column_id") or "")
        if column_id in seen_column_ids:
            duplicate_column_ids.add(column_id)
        seen_column_ids.add(column_id)

        embedding_text = str(record.get("embedding_text") or "")
        max_text_length = max(max_text_length, len(embedding_text))
        if not embedding_text.strip():
            empty_count += 1

        included_fields = {str(field) for field in record.get("included_fields") or []}
        for field in FORBIDDEN_PROFILE_FIELDS:
            if field in included_fields:
                forbidden_hits.append({"column_id": column_id, "kind": "included_field", "field": field})

        lowered = embedding_text.casefold()
        for label in sorted(forbidden_text_labels):
            if label in lowered:
                forbidden_hits.append({"column_id": column_id, "kind": "embedding_text_label", "field": label})

    return {
        "record_count": len(records),
        "record_version": EMBEDDING_RECORD_VERSION,
        "forbidden_profile_fields": list(FORBIDDEN_PROFILE_FIELDS),
        "forbidden_field_hits": forbidden_hits,
        "max_text_length": max_text_length,
        "empty_embedding_text_count": empty_count,
        "duplicate_column_id_count": len(duplicate_column_ids),
        "duplicate_column_ids": sorted(duplicate_column_ids),
    }


def ensure_record_audit_passes(audit: dict[str, Any]) -> None:
    if audit.get("forbidden_field_hits"):
        raise ValueError(f"embedding record audit found forbidden field hits: {audit['forbidden_field_hits'][:5]}")
    if int(audit.get("empty_embedding_text_count") or 0):
        raise ValueError("embedding record audit found empty embedding text")
    if int(audit.get("duplicate_column_id_count") or 0):
        raise ValueError("embedding record audit found duplicate column ids")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_embedding_vectors(rows_or_path: Iterable[dict[str, Any]] | Path) -> dict[str, list[float]]:
    if isinstance(rows_or_path, (str, Path)):
        rows = load_jsonl_records(Path(rows_or_path))
    else:
        rows = list(rows_or_path)
    vectors: dict[str, list[float]] = {}
    for row in rows:
        column_id = str(row.get("column_id") or "")
        raw_vector = row.get("embedding")
        if raw_vector is None:
            raw_vector = row.get("vector")
        if not column_id:
            raise ValueError(f"embedding row has no column_id: {row!r}")
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError(f"embedding row has no vector: {column_id}")
        vectors[column_id] = [float(value) for value in raw_vector]
    return vectors


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("embedding vector norm is zero")
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding vector dimensions differ: {len(left)} != {len(right)}")
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def resolve_embedding_candidate_selection(
    *,
    selection_mode: str = EMBEDDING_SELECTION_TOP_K,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    minimum_similarity_threshold: float | None = None,
) -> dict[str, int | float | str | None]:
    mode = str(selection_mode or "").strip()
    if mode == EMBEDDING_SELECTION_TOP_K:
        if similarity_threshold is not None:
            raise ValueError("similarity_threshold is only valid when selection_mode='score_threshold'")
        if minimum_similarity_threshold is not None:
            raise ValueError("minimum_similarity_threshold is only valid for the hybrid selection mode")
        resolved_top_k = 20 if top_k is None else int(top_k)
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive when selection_mode='top_k'")
        return {
            "selection_mode": mode,
            "top_k": resolved_top_k,
            "similarity_threshold": None,
            "minimum_similarity_threshold": None,
        }
    if mode == EMBEDDING_SELECTION_SCORE_THRESHOLD:
        if top_k is not None:
            raise ValueError("top_k must be omitted when selection_mode='score_threshold'")
        if minimum_similarity_threshold is not None:
            raise ValueError("minimum_similarity_threshold is only valid for the hybrid selection mode")
        if similarity_threshold is None:
            raise ValueError("similarity_threshold is required when selection_mode='score_threshold'")
        threshold = float(similarity_threshold)
        if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
            raise ValueError("similarity_threshold must be a finite cosine value in [-1, 1]")
        return {
            "selection_mode": mode,
            "top_k": None,
            "similarity_threshold": threshold,
            "minimum_similarity_threshold": None,
        }
    if mode == EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR:
        if top_k is None or int(top_k) <= 0:
            raise ValueError("top_k must be positive when selection_mode='top_k_or_score_threshold_with_floor'")
        if similarity_threshold is None:
            raise ValueError(
                "similarity_threshold is required when selection_mode='top_k_or_score_threshold_with_floor'"
            )
        if minimum_similarity_threshold is None:
            raise ValueError(
                "minimum_similarity_threshold is required when selection_mode='top_k_or_score_threshold_with_floor'"
            )
        high_threshold = float(similarity_threshold)
        floor_threshold = float(minimum_similarity_threshold)
        if not math.isfinite(high_threshold) or high_threshold < -1.0 or high_threshold > 1.0:
            raise ValueError("similarity_threshold must be a finite cosine value in [-1, 1]")
        if not math.isfinite(floor_threshold) or floor_threshold < -1.0 or floor_threshold > 1.0:
            raise ValueError("minimum_similarity_threshold must be a finite cosine value in [-1, 1]")
        if floor_threshold > high_threshold:
            raise ValueError("minimum_similarity_threshold must not exceed similarity_threshold")
        return {
            "selection_mode": mode,
            "top_k": int(top_k),
            "similarity_threshold": high_threshold,
            "minimum_similarity_threshold": floor_threshold,
        }
    raise ValueError(
        "selection_mode must be one of "
        f"{EMBEDDING_SELECTION_TOP_K!r}, {EMBEDDING_SELECTION_SCORE_THRESHOLD!r}, "
        f"{EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR!r}"
    )


def _rank_maps(column_ids: list[str], normalized_vectors: dict[str, list[float]]) -> dict[str, dict[str, dict[str, float | int]]]:
    ranks: dict[str, dict[str, dict[str, float | int]]] = {}
    for left in column_ids:
        scored: list[tuple[str, float]] = []
        for right in column_ids:
            if left == right:
                continue
            scored.append((right, _cosine(normalized_vectors[left], normalized_vectors[right])))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranks[left] = {
            right: {"rank": index, "cosine": cosine}
            for index, (right, cosine) in enumerate(scored, start=1)
        }
    return ranks


def _query_doc_rank_maps(
    column_ids: list[str],
    normalized_query_vectors: dict[str, list[float]],
    normalized_document_vectors: dict[str, list[float]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    ranks: dict[str, dict[str, dict[str, float | int]]] = {}
    for query_id in column_ids:
        scored: list[tuple[str, float]] = []
        for document_id in column_ids:
            if query_id == document_id:
                continue
            scored.append((document_id, _cosine(normalized_query_vectors[query_id], normalized_document_vectors[document_id])))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranks[query_id] = {
            document_id: {"rank": index, "cosine": cosine}
            for index, (document_id, cosine) in enumerate(scored, start=1)
        }
    return ranks


def score_query_document_pairs(
    *,
    records: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    document_vectors: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Score every unordered pair with the Qwen query/document contract.

    The returned rows are threshold-independent so the same cache can support
    candidate selection, all-pairs verifier priority, and threshold studies.
    """
    column_ids = sorted(str(record["column_id"]) for record in records)
    if len(set(column_ids)) != len(column_ids):
        raise ValueError("embedding records contain duplicate column_id values")
    missing_query = [column_id for column_id in column_ids if column_id not in query_vectors]
    missing_document = [column_id for column_id in column_ids if column_id not in document_vectors]
    if missing_query:
        raise ValueError(f"missing query embedding vectors for {len(missing_query)} columns: {missing_query[:5]}")
    if missing_document:
        raise ValueError(f"missing document embedding vectors for {len(missing_document)} columns: {missing_document[:5]}")
    normalized_query_vectors = {column_id: _l2_normalize(query_vectors[column_id]) for column_id in column_ids}
    normalized_document_vectors = {column_id: _l2_normalize(document_vectors[column_id]) for column_id in column_ids}
    ranks = _query_doc_rank_maps(column_ids, normalized_query_vectors, normalized_document_vectors)

    rows: list[dict[str, Any]] = []
    for left_index, left_ref in enumerate(column_ids):
        for right_ref in column_ids[left_index + 1:]:
            left_to_right = ranks[left_ref][right_ref]
            right_to_left = ranks[right_ref][left_ref]
            left_cosine = float(left_to_right["cosine"])
            right_cosine = float(right_to_left["cosine"])
            rows.append(
                {
                    "pair_key": [left_ref, right_ref],
                    "col1": _column_obj(left_ref),
                    "col2": _column_obj(right_ref),
                    "query_to_doc_similarity_left_to_right": _rounded(left_cosine),
                    "query_to_doc_similarity_right_to_left": _rounded(right_cosine),
                    "query_doc_similarity_max": _rounded(max(left_cosine, right_cosine)),
                    "rank_query_left_to_doc_right": int(left_to_right["rank"]),
                    "rank_query_right_to_doc_left": int(right_to_left["rank"]),
                    "rank_min": min(int(left_to_right["rank"]), int(right_to_left["rank"])),
                }
            )
    return rows


def extract_embedding_candidates(
    *,
    records: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    selection_mode: str = EMBEDDING_SELECTION_TOP_K,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    minimum_similarity_threshold: float | None = None,
    embedding_model: str = "",
    embedding_base_url: str = "",
) -> list[dict[str, Any]]:
    selection = resolve_embedding_candidate_selection(
        selection_mode=selection_mode,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        minimum_similarity_threshold=minimum_similarity_threshold,
    )
    column_ids = [str(record["column_id"]) for record in records]
    missing = [column_id for column_id in column_ids if column_id not in vectors]
    if missing:
        raise ValueError(f"missing embedding vectors for {len(missing)} columns: {missing[:5]}")
    normalized_vectors = {column_id: _l2_normalize(vectors[column_id]) for column_id in column_ids}
    ranks = _rank_maps(column_ids, normalized_vectors)
    record_by_column_id = {str(record["column_id"]): record for record in records}

    pair_records: list[dict[str, Any]] = []
    for left_index, left in enumerate(column_ids):
        for right in column_ids[left_index + 1:]:
            left_rank = int(ranks[left][right]["rank"])
            right_rank = int(ranks[right][left]["rank"])
            cosine = float(ranks[left][right]["cosine"])
            selected_by_top_k = (
                selection["selection_mode"]
                in {EMBEDDING_SELECTION_TOP_K, EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR}
                and min(left_rank, right_rank) <= int(selection["top_k"])
            )
            selected_by_threshold = (
                selection["selection_mode"]
                in {EMBEDDING_SELECTION_SCORE_THRESHOLD, EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR}
                and cosine >= float(selection["similarity_threshold"])
            )
            passes_minimum_similarity_threshold = (
                selection["selection_mode"] != EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR
                or cosine >= float(selection["minimum_similarity_threshold"])
            )
            if not passes_minimum_similarity_threshold or (not selected_by_top_k and not selected_by_threshold):
                continue
            pair_key = make_pair_key(left, right)
            left_ref, right_ref = pair_key
            left_record = record_by_column_id[left_ref]
            right_record = record_by_column_id[right_ref]
            pair_records.append(
                {
                    "candidate_id": f"embedding_{len(pair_records) + 1:06d}",
                    "col1": _column_obj(left_ref),
                    "col2": _column_obj(right_ref),
                    "candidate_sources": [EMBEDDING_SOURCE],
                    "source_scores": {
                        "embedding_cosine": _rounded(cosine),
                        "embedding_rank_min": min(left_rank, right_rank),
                        "embedding_rank_left_to_right": left_rank if left_ref == left else right_rank,
                        "embedding_rank_right_to_left": right_rank if left_ref == left else left_rank,
                    },
                    "relation_hints": [
                        {
                            "family": "SCHEMA_SEMANTIC_NEIGHBOR",
                            "higher_level_concept": "nearby schema-intensional column descriptions",
                        }
                    ],
                    "evidence_refs": [f"profile:{left_ref}", f"profile:{right_ref}"],
                    "embedding_evidence": {
                        "kind": "embedding_similarity",
                        "cosine": _rounded(cosine),
                        "rank_left_to_right": left_rank if left_ref == left else right_rank,
                        "rank_right_to_left": right_rank if left_ref == left else left_rank,
                        "top_k": selection["top_k"],
                        "similarity_threshold": selection["similarity_threshold"],
                        "minimum_similarity_threshold": selection["minimum_similarity_threshold"],
                        "embedding_record_version": left_record.get("record_version"),
                        "left_embedding_text_sha256": left_record.get("embedding_text_sha256"),
                        "right_embedding_text_sha256": right_record.get("embedding_text_sha256"),
                    },
                    "embedding_metadata": {
                        "embedding_model": embedding_model,
                        "embedding_base_url": embedding_base_url,
                        "selection_mode": selection["selection_mode"],
                        "selected_by_top_k": selected_by_top_k,
                        "selected_by_threshold": selected_by_threshold,
                        "passes_minimum_similarity_threshold": passes_minimum_similarity_threshold,
                    },
                }
            )
    return pair_records


def extract_embedding_candidates_query_doc(
    *,
    records: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    document_vectors: dict[str, list[float]],
    selection_mode: str = EMBEDDING_SELECTION_TOP_K,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    minimum_similarity_threshold: float | None = None,
    embedding_model: str = "",
    embedding_base_url: str = "",
    query_instruction: str = QWEN_QUERY_INSTRUCTION,
) -> list[dict[str, Any]]:
    selection = resolve_embedding_candidate_selection(
        selection_mode=selection_mode,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        minimum_similarity_threshold=minimum_similarity_threshold,
    )
    column_ids = [str(record["column_id"]) for record in records]
    missing_query = [column_id for column_id in column_ids if column_id not in query_vectors]
    missing_document = [column_id for column_id in column_ids if column_id not in document_vectors]
    if missing_query:
        raise ValueError(f"missing query embedding vectors for {len(missing_query)} columns: {missing_query[:5]}")
    if missing_document:
        raise ValueError(f"missing document embedding vectors for {len(missing_document)} columns: {missing_document[:5]}")
    normalized_query_vectors = {column_id: _l2_normalize(query_vectors[column_id]) for column_id in column_ids}
    normalized_document_vectors = {column_id: _l2_normalize(document_vectors[column_id]) for column_id in column_ids}
    ranks = _query_doc_rank_maps(column_ids, normalized_query_vectors, normalized_document_vectors)
    record_by_column_id = {str(record["column_id"]): record for record in records}

    pair_records: list[dict[str, Any]] = []
    for left_index, left in enumerate(column_ids):
        for right in column_ids[left_index + 1:]:
            left_to_right = ranks[left][right]
            right_to_left = ranks[right][left]
            left_rank = int(left_to_right["rank"])
            right_rank = int(right_to_left["rank"])
            left_cosine = float(left_to_right["cosine"])
            right_cosine = float(right_to_left["cosine"])
            selected_by_top_k = (
                selection["selection_mode"]
                in {EMBEDDING_SELECTION_TOP_K, EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR}
                and min(left_rank, right_rank) <= int(selection["top_k"])
            )
            selected_by_threshold = (
                selection["selection_mode"]
                in {EMBEDDING_SELECTION_SCORE_THRESHOLD, EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR}
                and max(left_cosine, right_cosine) >= float(selection["similarity_threshold"])
            )
            passes_minimum_similarity_threshold = (
                selection["selection_mode"] != EMBEDDING_SELECTION_TOP_K_OR_SCORE_THRESHOLD_WITH_FLOOR
                or max(left_cosine, right_cosine) >= float(selection["minimum_similarity_threshold"])
            )
            if not passes_minimum_similarity_threshold or (not selected_by_top_k and not selected_by_threshold):
                continue
            pair_key = make_pair_key(left, right)
            left_ref, right_ref = pair_key
            original_left_is_pair_left = left_ref == left
            pair_left_record = record_by_column_id[left_ref]
            pair_right_record = record_by_column_id[right_ref]
            pair_left_to_right_rank = left_rank if original_left_is_pair_left else right_rank
            pair_right_to_left_rank = right_rank if original_left_is_pair_left else left_rank
            pair_left_to_right_cosine = left_cosine if original_left_is_pair_left else right_cosine
            pair_right_to_left_cosine = right_cosine if original_left_is_pair_left else left_cosine
            pair_records.append(
                {
                    "candidate_id": f"embedding_{len(pair_records) + 1:06d}",
                    "col1": _column_obj(left_ref),
                    "col2": _column_obj(right_ref),
                    "candidate_sources": [EMBEDDING_SOURCE],
                    "source_scores": {
                        "embedding_query_doc_similarity_max": _rounded(max(left_cosine, right_cosine)),
                        "embedding_rank_min": min(left_rank, right_rank),
                        "embedding_rank_query_left_to_doc_right": pair_left_to_right_rank,
                        "embedding_rank_query_right_to_doc_left": pair_right_to_left_rank,
                    },
                    "relation_hints": [
                        {
                            "family": "SCHEMA_SEMANTIC_NEIGHBOR",
                            "higher_level_concept": "instruction-aware query-to-schema-document column retrieval",
                        }
                    ],
                    "evidence_refs": [f"profile:{left_ref}", f"profile:{right_ref}"],
                    "embedding_evidence": {
                        "kind": "embedding_similarity",
                        "embedding_mode": QWEN_QUERY_DOC_MODE,
                        "query_to_doc_similarity_left_to_right": _rounded(pair_left_to_right_cosine),
                        "query_to_doc_similarity_right_to_left": _rounded(pair_right_to_left_cosine),
                        "rank_query_left_to_doc_right": pair_left_to_right_rank,
                        "rank_query_right_to_doc_left": pair_right_to_left_rank,
                        "top_k": selection["top_k"],
                        "similarity_threshold": selection["similarity_threshold"],
                        "minimum_similarity_threshold": selection["minimum_similarity_threshold"],
                        "embedding_record_version": pair_left_record.get("record_version"),
                        "left_embedding_text_sha256": pair_left_record.get("embedding_text_sha256"),
                        "right_embedding_text_sha256": pair_right_record.get("embedding_text_sha256"),
                        "query_instruction_sha256": query_instruction_sha256(query_instruction),
                    },
                    "embedding_metadata": {
                        "embedding_model": embedding_model,
                        "embedding_base_url": embedding_base_url,
                        "embedding_mode": QWEN_QUERY_DOC_MODE,
                        "selection_mode": selection["selection_mode"],
                        "selected_by_top_k": selected_by_top_k,
                        "selected_by_threshold": selected_by_threshold,
                        "passes_minimum_similarity_threshold": passes_minimum_similarity_threshold,
                    },
                }
            )
    return pair_records
