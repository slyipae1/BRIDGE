"""Explicit exhaustive-pair verifier queue used only by the retained ablation."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .proposal_aggregation import _compact_column_context, _compact_table_context, _declared_constraint_context
from .types import make_pair_key, split_column_ref


def _column_object(column_ref: str) -> dict[str, str]:
    table, column = split_column_ref(column_ref)
    return {"tab": table, "col": column}


def _column_refs(schema_profile: dict[str, Any]) -> list[str]:
    return sorted(str(ref) for ref in (schema_profile.get("columns") or {}) if str(ref).strip())


def build_all_pairs_bundles(schema_profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return schema-only contexts for every unordered column pair."""
    bundles: list[dict[str, Any]] = []
    for index, pair in enumerate(combinations(_column_refs(schema_profile), 2), start=1):
        left_ref, right_ref = make_pair_key(*pair)
        table_names = sorted({split_column_ref(left_ref)[0], split_column_ref(right_ref)[0]})
        bundles.append(
            {
                "candidate_id": f"all_pairs_{index:06d}",
                "pair_key": [left_ref, right_ref],
                "col1": _column_object(left_ref),
                "col2": _column_object(right_ref),
                "verification_policy": "FINAL_VERIFIER",
                "proposals_by_channel": [],
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
                    "candidate_generation_signals_are_not_supplied": True,
                    "raw_rows_are_not_supplied": True,
                    "gold_labels_are_not_supplied": True,
                },
            }
        )
    return bundles
