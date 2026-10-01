from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple


# `by_anchor` is retained only for internal COLUMN-literal component assembly.
# The public runner fixes actual detection requests to `by_all`.
MODULE_A_DETECTION_MODES = ("by_all", "by_anchor")


@dataclass(frozen=True)
class DetectionSlice:
    anchor_index: int
    anchor_type: str
    anchor_key: str
    anchor_label: str
    dbelement_options: List[Dict[str, Any]]
    source_entry_indices: List[int]
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        metadata = {
            "anchor_index": self.anchor_index,
            "anchor_type": self.anchor_type,
            "anchor_key": self.anchor_key,
            "anchor_label": self.anchor_label,
            "source_entry_indices": list(self.source_entry_indices),
            "slice_dbelement_option_count": len(self.dbelement_options),
        }
        # ADD: Optional metadata is omitted for ordinary slices so the default
        # request-artifact shape remains unchanged.
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return metadata


def build_detection_slices(
    dbelement_options: Iterable[Dict[str, Any]],
    *,
    detection_mode: str,
) -> List[DetectionSlice]:
    if detection_mode not in MODULE_A_DETECTION_MODES:
        raise ValueError(f"Unsupported Module A detection mode: {detection_mode}")

    entries = list(dbelement_options or [])
    if detection_mode == "by_all":
        return [
            DetectionSlice(
                anchor_index=0,
                anchor_type="ALL",
                anchor_key="__all__",
                anchor_label="all retrieved DB elements",
                dbelement_options=entries,
                source_entry_indices=list(range(len(entries))),
            )
        ]

    grouped: Dict[str, Tuple[str, str, List[Dict[str, Any]], List[int]]] = {}
    ordered_keys: List[str] = []

    for entry_index, entry in enumerate(entries):
        entry_type = str(entry.get("type", "") or "").upper()
        if entry_type == "COLUMN":
            anchor = _column_anchor_for_entry(entry)
            if not anchor:
                continue
            key = f"COLUMN::{anchor}"
            label = anchor
            anchor_type = "COLUMN"
        elif entry_type == "VALUE":
            literal = _value_literal_for_entry(entry)
            original_cols = _value_original_columns(entry)
            original_col_label = "|".join(original_cols) if original_cols else "unknown_col"
            key = f"VALUE::{literal}::{original_col_label}::entry_{entry_index}"
            label = f"{literal} @ {original_col_label}"
            anchor_type = "VALUE"
        else:
            continue

        if key not in grouped:
            ordered_keys.append(key)
            grouped[key] = (anchor_type, label, [], [])
        grouped[key][2].append(entry)
        grouped[key][3].append(entry_index)

    slices: List[DetectionSlice] = []
    for anchor_index, key in enumerate(ordered_keys):
        anchor_type, label, grouped_entries, source_indices = grouped[key]
        slices.append(
            DetectionSlice(
                anchor_index=anchor_index,
                anchor_type=anchor_type,
                anchor_key=key,
                anchor_label=label,
                dbelement_options=list(grouped_entries),
                source_entry_indices=list(source_indices),
            )
        )
    return slices


def dedupe_raw_detection_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        key = _raw_detection_key(entry)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _column_anchor_for_entry(entry: Dict[str, Any]) -> str:
    anchor = str(entry.get("anchor_column", "") or "").strip()
    if anchor:
        return anchor
    for option in entry.get("options", []) or []:
        option_anchor = str(option.get("anchor_column", "") or "").strip()
        if option_anchor:
            return option_anchor
    return ""


def _value_literal_for_entry(entry: Dict[str, Any]) -> str:
    literal = str(entry.get("entity", "") or "").strip()
    if literal:
        return literal
    for option in entry.get("options", []) or []:
        for key in ("value", "element"):
            value = str(option.get(key, "") or "").strip()
            if value:
                return value
    return "unknown_value"


def _value_original_columns(entry: Dict[str, Any]) -> List[str]:
    columns: List[str] = []
    seen = set()
    for option in entry.get("options", []) or []:
        if not option.get("is_original_anchor"):
            continue
        col = str(option.get("col", "") or "").strip()
        if not col or col in seen:
            continue
        seen.add(col)
        columns.append(col)
    return columns


def _raw_detection_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    ambiguity_type = _normalize_text(entry.get("ambiguity_type", ""))
    ambiguity_context = _normalize_text(entry.get("ambiguity_context", ""))
    choices = []
    for choice in entry.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        choices.append(
            (
                _normalize_text(choice.get("nl_description", "")),
                _normalize_text(choice.get("sql_snippet", "")),
            )
        )
    return ambiguity_type, ambiguity_context, tuple(choices)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
