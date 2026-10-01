"""Detection-only COLUMN/VALUE aggregation for Module A.

This module is intentionally invoked only when ``--apply-col-lit-aggregation``
is set. It derives prompt slices from a copy of live retrieval output and never
mutates the retrieval payload used for resume, recall analysis, or provenance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sqlglot import exp

from ..vendor_pipeline0612.live_retrieval.db_info import (
    get_db_all_tables,
    get_table_all_columns,
)
from ..vendor_pipeline0612.live_retrieval.dbelement_options import _clean_literal
from ..vendor_pipeline0612.live_retrieval.sql_parser import (
    _CONDITION_EXP_TYPES,
    _condition_parent_for_literal,
    _get_main_parent,
    _is_date_format_literal,
    _iter_condition_literal_nodes,
    _literal_text,
    _literal_validation_value,
    fixed_parse_one,
    get_sql_columns_dict,
)
from .config import COL_LIT_AGGREGATION_MODES
from .detection_slices import DetectionSlice, build_detection_slices


def validate_col_lit_aggregation_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    if mode not in COL_LIT_AGGREGATION_MODES:
        raise ValueError(
            "Unsupported column-literal aggregation mode: "
            f"{mode!r}; expected one of {COL_LIT_AGGREGATION_MODES}"
        )
    return mode


def _uses_pruning(mode: str) -> bool:
    return mode in {"prune", "prune_combine"}


def _uses_combination(mode: str) -> bool:
    return mode in {"combine", "prune_combine"}


def _canonical(value: Any) -> str:
    return str(value or "").strip().casefold()


def _display(value: Any) -> str:
    return str(value or "").strip()


def _column_parts(column_ref: str) -> tuple[str, str] | None:
    if "." not in str(column_ref or ""):
        return None
    table, column = str(column_ref).split(".", 1)
    table, column = table.strip(), column.strip()
    return (table, column) if table and column else None


def _literal_kind(literal: exp.Expression) -> str:
    if isinstance(literal, exp.Literal):
        return "textual" if literal.is_string else "numeric"
    if isinstance(literal, exp.Boolean):
        return "boolean"
    return "other"


def _column_anchor(entry: dict[str, Any]) -> str:
    anchor = _display(entry.get("anchor_column"))
    if anchor:
        return anchor
    for option in entry.get("options") or []:
        if isinstance(option, dict) and _display(option.get("anchor_column")):
            return _display(option.get("anchor_column"))
    return ""


@dataclass(frozen=True)
class LiteralTrace:
    source_column: str
    literal_value: str
    literal_kind: str
    column_sql_occurrences: int
    unresolved_same_name_occurrences: int
    source_resolution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "literal_value": self.literal_value,
            "literal_kind": self.literal_kind,
            "column_sql_occurrences": self.column_sql_occurrences,
            "unresolved_same_name_occurrences": self.unresolved_same_name_occurrences,
            "source_resolution": self.source_resolution,
        }


@dataclass
class ValueAnchor:
    entry_index: int
    original_column: str
    original_value: str
    existence: bool | None
    option_count: int
    original_anchor_count: int
    option_columns: set[str]
    trace_matches: list[LiteralTrace] = field(default_factory=list)

    @property
    def is_textual(self) -> bool:
        return bool(self.trace_matches) and all(
            trace.literal_kind == "textual" for trace in self.trace_matches
        )

    @property
    def source_occurrence_count(self) -> int | None:
        if not self.trace_matches:
            return None
        return max(trace.column_sql_occurrences for trace in self.trace_matches)

    @property
    def unresolved_source_occurrences(self) -> int:
        return max(
            (trace.unresolved_same_name_occurrences for trace in self.trace_matches),
            default=0,
        )

    @property
    def no_action_reason(self) -> str | None:
        if self.original_anchor_count != 1:
            return "original_anchor_not_unique"
        if self.existence is not True:
            return "existence_false_or_unknown"
        if not self.trace_matches:
            return "literal_trace_unmatched"
        if not self.is_textual:
            return "literal_not_textual"
        if self.unresolved_source_occurrences:
            return "source_occurrence_unresolved"
        if self.source_occurrence_count != 1:
            return "column_occurs_multiple_times_or_unresolved"
        return None

    @property
    def prune_eligible(self) -> bool:
        return self.no_action_reason is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_index": self.entry_index,
            "original_column": self.original_column,
            "original_value": self.original_value,
            "existence": self.existence,
            "option_count": self.option_count,
            "original_anchor_count": self.original_anchor_count,
            "option_columns": sorted(self.option_columns),
            "literal_trace": [trace.as_dict() for trace in self.trace_matches],
            "prune_eligible": self.prune_eligible,
            "no_action_reason": self.no_action_reason,
        }


@dataclass
class ColumnLiteralComponent:
    anchor_column: str
    column_entry_indices: list[int]
    values: list[ValueAnchor]
    column_candidates: dict[str, str]
    missing_value_trace_keys: set[tuple[str, str]] = field(default_factory=set)

    @property
    def component_id(self) -> str:
        return f"COLUMN_VALUE_COMPONENT::{self.anchor_column}"

    @property
    def source_key(self) -> str:
        return _canonical(self.anchor_column)

    @property
    def source_entry_indices(self) -> list[int]:
        return [*self.column_entry_indices, *(value.entry_index for value in self.values)]

    @property
    def no_action_reasons(self) -> list[str]:
        reasons = {
            value.no_action_reason for value in self.values if value.no_action_reason
        }
        if self.missing_value_trace_keys:
            reasons.add("literal_without_rendered_value_anchor")
        if self.source_key not in self.column_candidates:
            reasons.add("source_column_missing_from_candidates")
        return sorted(reasons)

    @property
    def pruning_applies(self) -> bool:
        return (
            bool(self.values)
            and not self.missing_value_trace_keys
            and self.source_key in self.column_candidates
            and all(value.prune_eligible for value in self.values)
        )

    @property
    def permitted_candidate_columns(self) -> set[str]:
        permitted = {self.source_key}
        for value in self.values:
            permitted.update(value.option_columns)
        return permitted

    @property
    def pruned_candidates(self) -> dict[str, str]:
        if not self.pruning_applies:
            return dict(self.column_candidates)
        permitted = self.permitted_candidate_columns
        return {
            key: display
            for key, display in self.column_candidates.items()
            if key in permitted
        }

    @property
    def singleton_skip(self) -> bool:
        return (
            self.pruning_applies
            and len(self.pruned_candidates) == 1
            and all(
                value.option_count == 1 and value.original_anchor_count == 1
                for value in self.values
            )
        )

    def as_dict(self, *, apply_pruning: bool) -> dict[str, Any]:
        effective_candidates = (
            self.pruned_candidates if apply_pruning else self.column_candidates
        )
        removed = [
            display
            for key, display in self.column_candidates.items()
            if key not in effective_candidates
        ]
        return {
            "component_id": self.component_id,
            "anchor_column": self.anchor_column,
            "column_entry_indices": list(self.column_entry_indices),
            "value_entry_indices": [value.entry_index for value in self.values],
            "values": [value.as_dict() for value in self.values],
            "missing_value_trace_keys": [
                {"source_column": source, "literal": literal}
                for source, literal in sorted(self.missing_value_trace_keys)
            ],
            "pruning_applies": self.pruning_applies,
            "no_action_reasons": self.no_action_reasons,
            "column_candidates_before": list(self.column_candidates.values()),
            "column_candidates_after": list(effective_candidates.values()),
            "removed_column_candidates": removed,
            "singleton_skip": self.singleton_skip,
        }


@dataclass(frozen=True)
class ColLitAggregationPlan:
    detection_slices: list[DetectionSlice]
    audit: dict[str, Any]


def _resolve_column_occurrence(
    column: exp.Column,
    *,
    parsed_sql: exp.Expression,
    db_path: Path,
    actual_tables: set[str],
) -> tuple[str | None, str]:
    """Resolve physical SQL references conservatively for occurrence counting."""
    column_name = _display(column.name)
    table_token = _display(column.table)
    actual_by_key = {_canonical(table): str(table) for table in actual_tables}

    if table_token:
        token_key = _canonical(table_token)
        for table in parsed_sql.find_all(exp.Table):
            table_name = _display(table.name)
            alias = _display(table.alias or table_name)
            if _canonical(alias) != token_key and _canonical(table_name) != token_key:
                continue
            actual_name = actual_by_key.get(_canonical(table_name))
            if actual_name:
                return f"{actual_name}.{column_name}", "qualified_or_alias"
            return None, "cte_or_nonphysical_alias"
        actual_name = actual_by_key.get(token_key)
        if actual_name:
            return f"{actual_name}.{column_name}", "direct_table"
        return None, "unknown_qualified_table"

    main_parent = _get_main_parent(column)
    matches: list[str] = []
    for table in parsed_sql.find_all(exp.Table):
        if _get_main_parent(table) != main_parent:
            continue
        table_name = _display(table.name)
        actual_name = actual_by_key.get(_canonical(table_name))
        if not actual_name:
            continue
        try:
            candidates = get_table_all_columns(str(db_path), actual_name)
        except Exception:
            continue
        if _canonical(column_name) in {_canonical(item) for item in candidates}:
            matches.append(actual_name)
    if len(matches) == 1:
        return f"{matches[0]}.{column_name}", "unqualified_unique"
    if len(matches) > 1:
        return None, "unqualified_ambiguous"
    return None, "unqualified_unresolved"


def _trace_sql_literals(
    *,
    current_sql: str,
    db_path: str | Path | None,
) -> tuple[list[LiteralTrace], dict[str, Any]]:
    """Build source-column/literal traces only for an enabled aggregation run."""
    diagnostics: dict[str, Any] = {"status": "ok", "error": None}
    path = Path(str(db_path or "")).expanduser()
    if not str(path) or not path.is_file():
        return [], {"status": "no_db_file", "error": str(db_path or "")}
    try:
        actual_tables = {str(table) for table in get_db_all_tables(str(path))}
        columns_dict = get_sql_columns_dict(str(path), current_sql)
        parsed_sql = fixed_parse_one(current_sql, read="sqlite")
    except Exception as exc:
        return [], {"status": "parse_failed", "error": f"{type(exc).__name__}: {exc}"}

    source_occurrences: Counter[str] = Counter()
    unresolved_by_column_name: Counter[str] = Counter()
    source_resolution: dict[str, str] = {}
    for column in parsed_sql.find_all(exp.Column):
        source_ref, resolution = _resolve_column_occurrence(
            column,
            parsed_sql=parsed_sql,
            db_path=path,
            actual_tables=actual_tables,
        )
        if source_ref:
            source_key = _canonical(source_ref)
            source_occurrences[source_key] += 1
            source_resolution.setdefault(source_key, resolution)
        else:
            unresolved_by_column_name[_canonical(column.name)] += 1

    traces: list[LiteralTrace] = []
    for literal in _iter_condition_literal_nodes(parsed_sql):
        if _is_date_format_literal(literal):
            continue
        parent_context = _condition_parent_for_literal(literal)
        if parent_context is None:
            continue
        raw_literal = _literal_text(literal)
        retrieval_literal = _literal_validation_value(raw_literal, parent_context)
        cleaned_literal = _clean_literal(str(retrieval_literal))
        if not cleaned_literal:
            continue
        for condition_column in parent_context.find_all(exp.Column):
            column_name = _display(condition_column.name)
            for table_name, column_names in columns_dict.items():
                if _canonical(column_name) not in {_canonical(item) for item in column_names}:
                    continue
                source_ref = f"{_display(table_name)}.{column_name}"
                source_key = _canonical(source_ref)
                traces.append(
                    LiteralTrace(
                        source_column=source_ref,
                        literal_value=cleaned_literal,
                        literal_kind=_literal_kind(literal),
                        column_sql_occurrences=source_occurrences.get(source_key, 0),
                        unresolved_same_name_occurrences=unresolved_by_column_name[
                            _canonical(column_name)
                        ],
                        source_resolution=source_resolution.get(source_key, "unresolved"),
                    )
                )
    diagnostics.update(
        trace_count=len(traces),
        unresolved_column_occurrence_count=sum(unresolved_by_column_name.values()),
    )
    return traces, diagnostics


def _value_anchor(
    *,
    entry_index: int,
    entry: dict[str, Any],
    traces_by_source_literal: dict[tuple[str, str], list[LiteralTrace]],
) -> ValueAnchor | None:
    options = [option for option in (entry.get("options") or []) if isinstance(option, dict)]
    originals = [option for option in options if option.get("is_original_anchor")]
    if len(originals) != 1:
        return None
    original = originals[0]
    original_column = _display(original.get("col"))
    original_value = _clean_literal(_display(original.get("value")))
    if not original_column or not original_value:
        return None
    option_columns = {
        _canonical(option.get("col"))
        for option in options
        if _display(option.get("col"))
    }
    return ValueAnchor(
        entry_index=entry_index,
        original_column=original_column,
        original_value=original_value,
        existence=original.get("existence"),
        option_count=len(options),
        original_anchor_count=len(originals),
        option_columns=option_columns,
        trace_matches=list(
            traces_by_source_literal.get((_canonical(original_column), original_value), [])
        ),
    )


def _build_components(
    *,
    entries: list[dict[str, Any]],
    traces: list[LiteralTrace],
) -> tuple[list[ColumnLiteralComponent], dict[str, Any]]:
    trace_by_source_literal: dict[tuple[str, str], list[LiteralTrace]] = defaultdict(list)
    for trace in traces:
        trace_by_source_literal[(_canonical(trace.source_column), trace.literal_value)].append(trace)

    column_entries: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    value_anchors: list[ValueAnchor] = []
    invalid_value_entry_indices: list[int] = []
    for entry_index, entry in enumerate(entries):
        entry_type = _canonical(entry.get("type")).upper()
        if entry_type == "COLUMN":
            anchor = _column_anchor(entry)
            if anchor:
                column_entries[_canonical(anchor)].append((entry_index, entry))
        elif entry_type == "VALUE":
            anchor = _value_anchor(
                entry_index=entry_index,
                entry=entry,
                traces_by_source_literal=trace_by_source_literal,
            )
            if anchor is None:
                invalid_value_entry_indices.append(entry_index)
            else:
                value_anchors.append(anchor)

    values_by_column: dict[str, list[ValueAnchor]] = defaultdict(list)
    unlinked_value_entries: list[int] = []
    for value in value_anchors:
        source_key = _canonical(value.original_column)
        if source_key in column_entries:
            values_by_column[source_key].append(value)
        else:
            unlinked_value_entries.append(value.entry_index)

    components: list[ColumnLiteralComponent] = []
    for source_key, values in values_by_column.items():
        grouped_entries = column_entries[source_key]
        anchor_column = _column_anchor(grouped_entries[0][1])
        candidates: dict[str, str] = {}
        for _, entry in grouped_entries:
            for option in entry.get("options") or []:
                if not isinstance(option, dict):
                    continue
                candidate = _display(option.get("element"))
                if candidate:
                    candidates.setdefault(_canonical(candidate), candidate)
        rendered_literals = {
            (source_key, value.original_value) for value in values
        }
        missing_value_trace_keys = {
            key for key in trace_by_source_literal if key[0] == source_key and key not in rendered_literals
        }
        components.append(
            ColumnLiteralComponent(
                anchor_column=anchor_column,
                column_entry_indices=[index for index, _ in grouped_entries],
                values=values,
                column_candidates=candidates,
                missing_value_trace_keys=missing_value_trace_keys,
            )
        )
    return components, {
        "column_anchor_count": len(column_entries),
        "value_entry_count": len(value_anchors),
        "invalid_value_entry_indices": invalid_value_entry_indices,
        "unlinked_value_entry_indices": unlinked_value_entries,
    }


def _apply_pruning(
    entries: list[dict[str, Any]],
    components: Iterable[ColumnLiteralComponent],
) -> list[dict[str, Any]]:
    derived_entries = deepcopy(entries)
    for component in components:
        if not component.pruning_applies:
            continue
        permitted = set(component.pruned_candidates)
        for entry_index in component.column_entry_indices:
            options = derived_entries[entry_index].get("options") or []
            derived_entries[entry_index]["options"] = [
                option
                for option in options
                if not isinstance(option, dict)
                or _canonical(option.get("eleType") or option.get("type")) != "column"
                or _canonical(option.get("element")) in permitted
            ]
    return derived_entries


def _slice_metadata(mode: str, components: Iterable[ColumnLiteralComponent]) -> dict[str, Any]:
    records = list(components)
    if not records:
        return {}
    return {
        "col_lit_aggregation": {
            "mode": mode,
            "component_ids": [component.component_id for component in records],
            "component_actions": [
                "prune" if component.pruning_applies else "no_action"
                for component in records
            ],
        }
    }


def _remap_standard_slices(
    *,
    entries: list[dict[str, Any]],
    source_indices: list[int],
    mode: str,
    components_by_source_index: dict[int, ColumnLiteralComponent],
) -> list[DetectionSlice]:
    base_slices = build_detection_slices(entries, detection_mode="by_anchor")
    remapped: list[DetectionSlice] = []
    for slice_info in base_slices:
        original_indices = [source_indices[index] for index in slice_info.source_entry_indices]
        linked_components = {
            component.component_id: component
            for index, component in components_by_source_index.items()
            if index in original_indices
        }
        remapped.append(
            DetectionSlice(
                anchor_index=slice_info.anchor_index,
                anchor_type=slice_info.anchor_type,
                anchor_key=slice_info.anchor_key,
                anchor_label=slice_info.anchor_label,
                dbelement_options=slice_info.dbelement_options,
                source_entry_indices=original_indices,
                extra_metadata=_slice_metadata(mode, linked_components.values()),
            )
        )
    return remapped


def _ordered_component_entry_indices(component: ColumnLiteralComponent) -> list[int]:
    return [*component.column_entry_indices, *(value.entry_index for value in component.values)]


def _build_by_anchor_slices(
    *,
    entries: list[dict[str, Any]],
    active_indices: list[int],
    components: list[ColumnLiteralComponent],
    mode: str,
) -> list[DetectionSlice]:
    active_set = set(active_indices)
    components_by_source_index = {
        index: component
        for component in components
        for index in component.source_entry_indices
        if index in active_set
    }
    if not _uses_combination(mode):
        return _remap_standard_slices(
            entries=[entries[index] for index in active_indices],
            source_indices=active_indices,
            mode=mode,
            components_by_source_index=components_by_source_index,
        )

    combined_components = [
        component
        for component in components
        if any(index in active_set for index in component.source_entry_indices)
    ]
    combined_source_indices = {
        index for component in combined_components for index in component.source_entry_indices
    }
    normal_indices = [index for index in active_indices if index not in combined_source_indices]
    standard_slices = _remap_standard_slices(
        entries=[entries[index] for index in normal_indices],
        source_indices=normal_indices,
        mode=mode,
        components_by_source_index={},
    )
    positioned: list[tuple[int, int, DetectionSlice]] = []
    for slice_info in standard_slices:
        positioned.append((min(slice_info.source_entry_indices), 1, slice_info))
    for component in combined_components:
        source_indices = [
            index for index in _ordered_component_entry_indices(component) if index in active_set
        ]
        if not source_indices:
            continue
        positioned.append(
            (
                min(source_indices),
                0,
                DetectionSlice(
                    anchor_index=0,
                    anchor_type="COLUMN_VALUE_COMPONENT",
                    anchor_key=component.component_id,
                    anchor_label=component.anchor_column,
                    dbelement_options=[entries[index] for index in source_indices],
                    source_entry_indices=source_indices,
                    extra_metadata=_slice_metadata(mode, [component]),
                ),
            )
        )
    positioned.sort(key=lambda item: (item[0], item[1], item[2].anchor_key))
    return [
        DetectionSlice(
            anchor_index=index,
            anchor_type=slice_info.anchor_type,
            anchor_key=slice_info.anchor_key,
            anchor_label=slice_info.anchor_label,
            dbelement_options=slice_info.dbelement_options,
            source_entry_indices=slice_info.source_entry_indices,
            extra_metadata=slice_info.extra_metadata,
        )
        for index, (_, _, slice_info) in enumerate(positioned)
    ]


def _build_by_all_slice(
    *,
    entries: list[dict[str, Any]],
    active_indices: list[int],
    components: list[ColumnLiteralComponent],
    mode: str,
) -> list[DetectionSlice]:
    active_set = set(active_indices)
    ordered_indices = list(active_indices)
    if _uses_combination(mode):
        component_by_index = {
            index: component
            for component in components
            for index in component.source_entry_indices
            if index in active_set
        }
        emitted_components: set[str] = set()
        ordered_indices = []
        for index in active_indices:
            component = component_by_index.get(index)
            if component is None:
                ordered_indices.append(index)
                continue
            if component.component_id in emitted_components:
                continue
            emitted_components.add(component.component_id)
            ordered_indices.extend(
                item for item in _ordered_component_entry_indices(component) if item in active_set
            )
    return [
        DetectionSlice(
            anchor_index=0,
            anchor_type="ALL",
            anchor_key="__all__",
            anchor_label="all retrieved DB elements",
            dbelement_options=[entries[index] for index in ordered_indices],
            source_entry_indices=ordered_indices,
            extra_metadata={
                "col_lit_aggregation": {
                    "mode": mode,
                    "component_count": len(components),
                    "combined": _uses_combination(mode),
                }
            },
        )
    ]


def build_col_lit_aggregation_plan(
    *,
    dbelement_options: Iterable[dict[str, Any]],
    current_sql: str,
    db_path: str | Path | None,
    detection_mode: str,
    mode: str,
) -> ColLitAggregationPlan:
    """Build derived detection slices for an active aggregation setting."""
    validate_col_lit_aggregation_mode(mode)
    if detection_mode not in {"by_all", "by_anchor"}:
        raise ValueError(f"Unsupported Module A detection mode: {detection_mode}")
    entries = deepcopy(list(dbelement_options or []))
    traces, trace_diagnostics = _trace_sql_literals(
        current_sql=current_sql,
        db_path=db_path,
    )
    components, linkage = _build_components(entries=entries, traces=traces)
    derived_entries = _apply_pruning(entries, components) if _uses_pruning(mode) else entries
    skipped = [
        component for component in components if _uses_pruning(mode) and component.singleton_skip
    ]
    skipped_indices = {index for component in skipped for index in component.source_entry_indices}
    active_indices = [index for index in range(len(derived_entries)) if index not in skipped_indices]
    if detection_mode == "by_all":
        slices = _build_by_all_slice(
            entries=derived_entries,
            active_indices=active_indices,
            components=components,
            mode=mode,
        ) if active_indices else []
    else:
        slices = _build_by_anchor_slices(
            entries=derived_entries,
            active_indices=active_indices,
            components=components,
            mode=mode,
        )
    prune_eligible_component_count = sum(
        component.pruning_applies for component in components
    )
    audit = {
        "mode": mode,
        "trace": trace_diagnostics,
        "linkage": linkage,
        "component_count": len(components),
        "prune_eligible_component_count": prune_eligible_component_count,
        "pruning_component_count": (
            prune_eligible_component_count if _uses_pruning(mode) else 0
        ),
        "no_action_component_count": sum(not component.pruning_applies for component in components),
        "singleton_skip_component_ids": [component.component_id for component in skipped],
        "detection_slice_count": len(slices),
        "components": [
            component.as_dict(apply_pruning=_uses_pruning(mode))
            for component in components
        ],
    }
    return ColLitAggregationPlan(detection_slices=slices, audit=audit)
