"""Raw column-name embedding candidate channel.

This channel intentionally has a narrower input contract than schema
embedding. Documents are raw identifiers only; the Qwen query side carries
the ambiguity-oriented retrieval instruction.
"""

from __future__ import annotations

from typing import Any

from auto_construct_graph.channels.embedding_channel import (
    DOCUMENT_EMBEDDING_VIEW,
    QWEN_QUERY_DOC_MODE,
    QUERY_EMBEDDING_VIEW,
    _column_obj,
    embedding_text_sha256,
    query_instruction_sha256,
    score_query_document_pairs,
)
from auto_construct_graph.types import make_pair_key


SURFACE_NAME_EMBEDDING_SOURCE = "SURFACE_NAME_EMBEDDING"
SURFACE_NAME_EMBEDDING_RECORD_VERSION = "surface_name_raw_column_name_v1"
SURFACE_NAME_QUERY_INSTRUCTION = (
    "Retrieve likely ambiguous database column names, especially when an underspecified "
    "user-mentioned field could plausibly refer to either column name."
)
SURFACE_NAME_DOCUMENT_POLICY = "embedding_text equals the raw column name exactly"


def build_surface_name_embedding_record(column_ref: str, column_profile: dict[str, Any]) -> dict[str, Any]:
    raw_column_name = str(column_profile.get("column") or "").strip()
    if not raw_column_name:
        raise ValueError(f"column has no raw column name: {column_ref}")
    return {
        "column_id": str(column_ref),
        "table_name": str(column_profile.get("table") or ""),
        "column_name": raw_column_name,
        "embedding_text": raw_column_name,
        "embedding_text_sha256": embedding_text_sha256(raw_column_name),
        "record_version": SURFACE_NAME_EMBEDDING_RECORD_VERSION,
        "document_policy": SURFACE_NAME_DOCUMENT_POLICY,
        "included_fields": ["raw_column_name_only"],
        "excluded_fields": [
            "table_name",
            "table_descriptive_name",
            "column_descriptive_name",
            "column_description",
            "statistics",
            "sample_values",
            "value_description",
            "gold_labels",
        ],
    }


def build_surface_name_embedding_records(profile: dict[str, Any]) -> list[dict[str, Any]]:
    columns = profile.get("columns") or {}
    return [
        build_surface_name_embedding_record(column_ref, columns[column_ref])
        for column_ref in sorted(columns)
        if isinstance(columns.get(column_ref), dict)
    ]


def audit_surface_name_embedding_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    duplicate_column_ids: list[str] = []
    invalid_records: list[dict[str, Any]] = []
    for record in records:
        column_id = str(record.get("column_id") or "")
        if column_id in seen:
            duplicate_column_ids.append(column_id)
        seen.add(column_id)
        raw_column_name = str(record.get("column_name") or "").strip()
        if (
            str(record.get("record_version") or "") != SURFACE_NAME_EMBEDDING_RECORD_VERSION
            or not raw_column_name
            or str(record.get("embedding_text") or "") != raw_column_name
        ):
            invalid_records.append(
                {
                    "column_id": column_id,
                    "column_name": raw_column_name,
                    "embedding_text": record.get("embedding_text"),
                    "record_version": record.get("record_version"),
                }
            )
    return {
        "record_count": len(records),
        "record_version": SURFACE_NAME_EMBEDDING_RECORD_VERSION,
        "document_text_policy": SURFACE_NAME_DOCUMENT_POLICY,
        "duplicate_column_id_count": len(duplicate_column_ids),
        "duplicate_column_ids": sorted(set(duplicate_column_ids)),
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records,
    }


def ensure_surface_name_embedding_record_audit_passes(audit: dict[str, Any]) -> None:
    if int(audit.get("duplicate_column_id_count") or 0):
        raise ValueError(f"surface-name records contain duplicate ids: {audit['duplicate_column_ids'][:5]}")
    if int(audit.get("invalid_record_count") or 0):
        raise ValueError(f"surface-name records violate raw-name policy: {audit['invalid_records'][:3]}")


def extract_surface_name_embedding_candidates(
    *,
    records: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    document_vectors: dict[str, list[float]],
    similarity_threshold: float,
    embedding_model: str = "",
    embedding_base_url: str = "",
) -> list[dict[str, Any]]:
    threshold = float(similarity_threshold)
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("surface-name similarity_threshold must be in [-1, 1]")
    records_by_ref = {str(record["column_id"]): record for record in records}
    candidates: list[dict[str, Any]] = []
    for score in score_query_document_pairs(
        records=records,
        query_vectors=query_vectors,
        document_vectors=document_vectors,
    ):
        if float(score["query_doc_similarity_max"]) < threshold:
            continue
        left_ref, right_ref = make_pair_key(*score["pair_key"])
        left_record = records_by_ref[left_ref]
        right_record = records_by_ref[right_ref]
        candidates.append(
            {
                "candidate_id": f"surface_name_embedding_{len(candidates) + 1:06d}",
                "col1": _column_obj(left_ref),
                "col2": _column_obj(right_ref),
                "candidate_sources": [SURFACE_NAME_EMBEDDING_SOURCE],
                "source_scores": {
                    "surface_name_embedding_similarity_max": score["query_doc_similarity_max"],
                    "surface_name_embedding_rank_min": score["rank_min"],
                    "surface_name_embedding_rank_query_left_to_doc_right": score[
                        "rank_query_left_to_doc_right"
                    ],
                    "surface_name_embedding_rank_query_right_to_doc_left": score[
                        "rank_query_right_to_doc_left"
                    ],
                },
                "relation_hints": [
                    {
                        "family": "SURFACE_NAME_SEMANTIC_NEIGHBOR",
                        "higher_level_concept": "instruction-aware raw column-name similarity",
                    }
                ],
                "evidence_refs": [f"profile:{left_ref}", f"profile:{right_ref}"],
                "surface_name_embedding_evidence": {
                    "kind": "surface_name_embedding_similarity",
                    "embedding_mode": QWEN_QUERY_DOC_MODE,
                    "query_to_doc_similarity_left_to_right": score["query_to_doc_similarity_left_to_right"],
                    "query_to_doc_similarity_right_to_left": score["query_to_doc_similarity_right_to_left"],
                    "similarity_max": score["query_doc_similarity_max"],
                    "rank_query_left_to_doc_right": score["rank_query_left_to_doc_right"],
                    "rank_query_right_to_doc_left": score["rank_query_right_to_doc_left"],
                    "similarity_threshold": threshold,
                    "embedding_record_version": SURFACE_NAME_EMBEDDING_RECORD_VERSION,
                    "left_embedding_text_sha256": left_record["embedding_text_sha256"],
                    "right_embedding_text_sha256": right_record["embedding_text_sha256"],
                    "query_instruction_sha256": query_instruction_sha256(SURFACE_NAME_QUERY_INSTRUCTION),
                },
                "surface_name_embedding_metadata": {
                    "embedding_model": embedding_model,
                    "embedding_base_url": embedding_base_url,
                    "embedding_mode": QWEN_QUERY_DOC_MODE,
                    "selection_mode": "score_threshold",
                    "embedding_views": [DOCUMENT_EMBEDDING_VIEW, QUERY_EMBEDDING_VIEW],
                },
            }
        )
    return candidates
