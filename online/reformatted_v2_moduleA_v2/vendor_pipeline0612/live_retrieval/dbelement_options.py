from __future__ import annotations

import difflib
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from .database_manager import DatabaseManager
from .db_values.config import LSH_N_GRAM, LSH_SIGNATURE_SIZE

from .sql_parser import (
    get_db_aware_tables_and_columns_from_sql,
    get_sql_condition_literals,
    validate_literal_existence_records,
)
from ..schema_description_loader import (
    load_colgrp_artifact_descriptive_names,
    load_tables_description,
)
from .database_profiler import get_db_profiler
from .llm_fallback import extract_elements_with_llm
from ...telemetry import elapsed_s, monotonic_s
from ...module_a_base.retrieval_ablation import (
    DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    retrieval_channel_enabled,
    validate_retrieval_ablation_disable_channel,
)
from ...module_a_base.config import (
    DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP,
)
from sqlglot import exp



def _normalize_sql(sql: str) -> str:
    """Remove comments and trailing semicolon from SQL."""
    if not isinstance(sql, str):
        return ""
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    sql = sql.strip().rstrip(";")
    # Many generated SQLs incorrectly use double quotes for string literals.
    # Normalize only obvious condition-side literals to avoid mis-parsing them as identifiers.
    sql = re.sub(r'(?i)(=\s*)"([^"]*)"', lambda m: f"{m.group(1)}'{m.group(2)}'", sql)
    sql = re.sub(r'(?i)(LIKE\s*)"([^"]*)"', lambda m: f"{m.group(1)}'{m.group(2)}'", sql)
    return sql


def _sql_needs_literal_extraction(sql: str) -> bool:
    """Return True when SQL contains condition-side literal anchors."""
    try:
        from .fixed_parse_one import fixed_parse_one

        parsed = fixed_parse_one(sql, read="sqlite")
        literal_nodes = list(parsed.find_all(exp.Literal))
        literal_nodes.extend(parsed.find_all(exp.Boolean))
        for literal in literal_nodes:
            if isinstance(literal, exp.Literal) and literal.is_string and str(literal.this or "").startswith("%"):
                parent = literal.parent
                is_date_format = False
                while parent is not None:
                    if parent.__class__.__name__.lower() in {"timetostr", "strtotime"}:
                        is_date_format = True
                        break
                    if isinstance(parent, (exp.Select, exp.Subquery)):
                        break
                    parent = parent.parent
                if is_date_format:
                    continue
            parent = literal.parent
            if parent is None:
                continue
            parent_sql = str(parent).upper()
            if any(token in parent_sql for token in ("=", "IN", "LIKE", "BETWEEN", ">", "<")):
                return True
        return False
    except Exception:
        # Conservative regex fallback — only match quoted (string) values.
        return bool(
            re.search(r"""(?ix)
            (?:=|<>|!=|>=|<=|>|<)\s*(?:'[^']*'|"[^"]*")
            | \bLIKE\b\s*(?:'[^']*'|"[^"]*")
            | \bBETWEEN\b\s*(?:'[^']*'|"[^"]*")\s+\bAND\b
            """, sql)
        )


def _is_unknown_llm_literal_target(table_name: Any, column_name: Any) -> bool:
    table_text = str(table_name or "").strip()
    column_text = str(column_name or "").strip()
    unknown_values = {"", "unknown", "none", "null"}
    return table_text.casefold() in unknown_values or column_text.casefold() in unknown_values


def _split_llm_literals_for_validation(
    llm_literals: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, Dict[str, List[str]]], List[Dict[str, Any]]]:
    mapped_literals: Dict[str, Dict[str, List[str]]] = {}
    unmatched_literals: List[Dict[str, Any]] = []
    for table_name, col_literals in (llm_literals or {}).items():
        for column_name, values in (col_literals or {}).items():
            for value in values or []:
                if _is_unknown_llm_literal_target(table_name, column_name):
                    unmatched_literals.append(
                        {
                            "literal": value,
                            "table": str(table_name or ""),
                            "column": str(column_name or ""),
                            "source": "llm_fallback",
                            "reason": "llm_unmatched_unknown_column",
                        }
                    )
                    continue
                mapped_literals.setdefault(str(table_name), {}).setdefault(str(column_name), [])
                if value not in mapped_literals[str(table_name)][str(column_name)]:
                    mapped_literals[str(table_name)][str(column_name)].append(value)
    return mapped_literals, unmatched_literals


def _column_ref_key(column_ref: str) -> str:
    return str(column_ref or "").strip().casefold()


def _split_column_ref(column_ref: str) -> Tuple[str, str] | None:
    if not isinstance(column_ref, str) or "." not in column_ref:
        return None
    table_name, column_name = column_ref.split(".", 1)
    table_name = table_name.strip()
    column_name = column_name.strip()
    if not table_name or not column_name:
        return None
    return table_name, column_name


def _load_schema_case_map() -> Dict[Tuple[str, str], Tuple[str, str]]:
    try:
        schema = DatabaseManager().get_db_schema()
    except Exception:
        return {}

    case_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for table_name, columns in (schema or {}).items():
        for column_name in columns or []:
            case_map[(str(table_name).casefold(), str(column_name).casefold())] = (
                str(table_name),
                str(column_name),
            )
    return case_map


def _canonicalize_column_ref(
    column_ref: str,
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> str:
    parts = _split_column_ref(column_ref)
    if not parts:
        return column_ref
    table_name, column_name = parts
    actual = (schema_case_map or {}).get((table_name.casefold(), column_name.casefold()))
    if not actual:
        return f"{table_name}.{column_name}"
    actual_table, actual_column = actual
    return f"{actual_table}.{actual_column}"


def _lookup_column_descriptive_name(
    column_descriptive_names: Dict[str, str],
    column_ref: str,
    fallback: str = "",
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> str:
    if not isinstance(column_descriptive_names, dict):
        return fallback

    raw_ref = str(column_ref or "").strip()
    candidate_refs: List[str] = []
    if raw_ref:
        candidate_refs.append(raw_ref)
        canonical_ref = _canonicalize_column_ref(raw_ref, schema_case_map)
        if canonical_ref and canonical_ref not in candidate_refs:
            candidate_refs.append(canonical_ref)

    for candidate_ref in candidate_refs:
        descriptive_name = column_descriptive_names.get(candidate_ref)
        if descriptive_name:
            return str(descriptive_name)

    names_by_key = {
        str(key).strip().casefold(): str(value)
        for key, value in column_descriptive_names.items()
        if str(key).strip() and value
    }
    for candidate_ref in candidate_refs:
        descriptive_name = names_by_key.get(candidate_ref.casefold())
        if descriptive_name:
            return descriptive_name

    return fallback


def _dedupe_column_refs(
    column_refs: List[str],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for column_ref in column_refs:
        canonical = _canonicalize_column_ref(column_ref, schema_case_map)
        key = _column_ref_key(canonical)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(canonical)
    return deduped


def _qualify_within_table_column_ref(table_name: str, column_ref: str) -> str:
    column_ref = str(column_ref or "").strip()
    if not column_ref:
        return ""
    if "." in column_ref:
        return column_ref
    return f"{table_name}.{column_ref}"


def _canonicalize_schema_dict(
    schema_dict: Dict[str, List[str]],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[str, List[str]]:
    canonical_schema: Dict[str, List[str]] = {}
    seen_by_table: Dict[Tuple[str, str], Set[str]] = {}
    for table_name, columns in (schema_dict or {}).items():
        for column_name in columns or []:
            actual = (schema_case_map or {}).get((str(table_name).casefold(), str(column_name).casefold()))
            if actual:
                actual_table, actual_column = actual
                table_key = ("schema", actual_table.casefold())
            else:
                actual_table, actual_column = str(table_name), str(column_name)
                table_key = ("raw", actual_table)
            if table_key not in seen_by_table:
                canonical_schema[actual_table] = []
                seen_by_table[table_key] = set()
            if actual_column.casefold() not in seen_by_table[table_key]:
                canonical_schema[actual_table].append(actual_column)
                seen_by_table[table_key].add(actual_column.casefold())
    return canonical_schema


def _canonicalize_column_option(
    option: Dict[str, Any],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    normalized = dict(option)
    element = normalized.get("element", "")
    if isinstance(element, str):
        normalized["element"] = _canonicalize_column_ref(element, schema_case_map)
    return normalized


def _column_option_dedup_key(option: Dict[str, Any]) -> Tuple[str, str]:
    return ("column", str(option.get("element", "")).casefold())


def _dedupe_dbelement_entry_options(
    option_group: Dict[str, Any],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    if str(option_group.get("type", "")).upper() != "COLUMN":
        return option_group

    deduped_group = dict(option_group)
    seen: Set[Tuple[str, str]] = set()
    deduped_options: List[Dict[str, Any]] = []
    for option in option_group.get("options", []) or []:
        if str(option.get("eleType", option_group.get("type", ""))).upper() != "COLUMN":
            deduped_options.append(option)
            continue
        canonical_option = _canonicalize_column_option(option, schema_case_map)
        key = _column_option_dedup_key(canonical_option)
        if key in seen:
            continue
        seen.add(key)
        deduped_options.append(canonical_option)
    deduped_group["options"] = deduped_options
    return deduped_group


def _merge_unique_dbelement_options(
    part_options: List[Dict[str, Any]],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    dbelement_options: List[Dict[str, Any]] = []
    for option in part_options:
        deduped_option = _dedupe_dbelement_entry_options(option, schema_case_map)
        # MOD: do not merge entries here; the new behavior is strictly scoped
        # to case-insensitive dedup inside each COLUMN entry's options list.
        dbelement_options.append(deduped_option)
    return dbelement_options


########################### Column Group ##########################################
COLUMN_GROUP_VERSIONS = ("manual",)


def _manual_relation_groups(column_groups: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups = column_groups.get("manual_relation_groups")
    if groups is None:
        groups = column_groups.get("relation_hyperedges")
    return [group for group in (groups or []) if isinstance(group, dict)]


def _manual_group_label(group: Dict[str, Any]) -> str:
    return str(
        group.get("ambiguity_reason_concise_label")
        or group.get("latent_concept")
        or group.get("edge_id")
        or group.get("group_id")
        or "manual ambiguity group"
    )


def _manual_group_id(group: Dict[str, Any], group_index: int) -> str:
    return str(group.get("group_id") or group.get("edge_id") or f"manual_group_{group_index:04d}")


def _manual_column_description(
    column_descriptive_names: Dict[str, str],
    column_ref: str,
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> str:
    fallback = column_ref.split(".", 1)[1] if "." in column_ref else column_ref
    return _lookup_column_descriptive_name(
        column_descriptive_names,
        column_ref,
        fallback=fallback,
        schema_case_map=schema_case_map,
    )


def _manual_group_to_dbelement_entry(
    *,
    group: Dict[str, Any],
    group_index: int,
    anchor_ref: str,
    option_columns: List[str],
    column_descriptive_names: Dict[str, str],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    label = _manual_group_label(group)
    group_id = _manual_group_id(group, group_index)
    anchor_parts = _split_column_ref(anchor_ref)
    anchor_table = anchor_parts[0] if anchor_parts else ""
    relation_group = dict(group)
    relation_group.setdefault("edge_id", group_id)
    relation_group.setdefault("source", "manual_ambiguity_graph")
    relation_group.setdefault("ambiguity_reason_concise_label", label)
    relation_group.setdefault("columns", list(option_columns))

    return {
        "entity": label,
        "type": "COLUMN",
        "anchor_column": anchor_ref,
        "ambiguity_reason_concise_label": label,
        "relation_edge_id": group_id,
        "relation_subtype": "manual_untyped",
        "options": [
            {
                "element": opt_column,
                "eleType": "COLUMN",
                "anchor_column": anchor_ref,
                "descriptive_name": _manual_column_description(
                    column_descriptive_names,
                    opt_column,
                    schema_case_map,
                ),
                "ambiguity_reason": label,
                "reasons": f"member of manual ambiguity group: {label}",
                "different_table": (
                    opt_column.split(".", 1)[0].casefold() != anchor_table.casefold()
                    if "." in opt_column and anchor_table
                    else False
                ),
            }
            for opt_column in option_columns
        ],
        "relation_group": relation_group,
        "runtime_policy": group.get("runtime_policy", {"requires_downstream_llm_reasoning": True}),
        "confidence": group.get("confidence", "manual"),
    }


def _manual_option_sort_key(anchor_ref: str):
    anchor_table, anchor_column = _split_column_ref(anchor_ref) or ("", anchor_ref)

    def sort_key(column_ref: str) -> tuple:
        table_name, column_name = _split_column_ref(column_ref) or ("", column_ref)
        if table_name.casefold() == anchor_table.casefold() and column_name.casefold() == anchor_column.casefold():
            return (0, column_ref)
        if column_name.casefold() == anchor_column.casefold():
            return (1, column_ref)
        if table_name.casefold() == anchor_table.casefold():
            return (2, column_ref)
        return (3, table_name.casefold(), column_ref)

    return sort_key


def _identify_column_groups_manual(
    tables: List[str],
    columns_dict: Dict[str, List[str]],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    *,
    column_group_artifact_root: str | None = None,
) -> List[Dict[str, Any]]:
    del tables
    database_manager = DatabaseManager()
    if column_group_artifact_root:
        column_groups = database_manager.load_column_groups(
            version="manual",
            artifact_root=column_group_artifact_root,
        )
    else:
        column_groups = database_manager.load_column_groups(version="manual")
    if not column_groups:
        logging.warning("No manual column groups available for identifying column options")
        return []

    relation_groups = _manual_relation_groups(column_groups)
    column_descriptive_names = dict(column_groups.get("column_descriptive_names", {}) or {})
    db_directory_path = getattr(database_manager, "db_directory_path", None)
    fallback_descriptive_names = (
        load_colgrp_artifact_descriptive_names(db_directory_path)
        if db_directory_path is not None
        else {}
    )
    for column_ref, descriptive_name in fallback_descriptive_names.items():
        if not str(column_descriptive_names.get(column_ref) or "").strip():
            column_descriptive_names[column_ref] = descriptive_name
    column_options: List[Dict[str, Any]] = []
    for table_name, columns in columns_dict.items():
        for column_name in columns or []:
            anchor_ref = _canonicalize_column_ref(f"{table_name}.{column_name}", schema_case_map)
            anchor_key = _column_ref_key(anchor_ref)
            for group_index, group in enumerate(relation_groups, start=1):
                edge_columns = _dedupe_column_refs(list(group.get("columns", []) or []), schema_case_map)
                if anchor_key not in {_column_ref_key(column) for column in edge_columns}:
                    continue
                if len(edge_columns) < 2:
                    continue
                option_columns = list(edge_columns)
                option_columns.sort(key=_manual_option_sort_key(anchor_ref))
                column_options.append(
                    _manual_group_to_dbelement_entry(
                        group=group,
                        group_index=group_index,
                        anchor_ref=anchor_ref,
                        option_columns=option_columns,
                        column_descriptive_names=column_descriptive_names,
                        schema_case_map=schema_case_map,
                    )
                )
    return column_options


def _identify_column_groups(
    tables: List[str],
    columns_dict: Dict[str, List[str]],
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    *,
    column_group_version: str | None = None,
    relation_types: List[str] | None = None,
    column_group_artifact_root: str | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve candidates from the supplied public manual graph archive."""
    del relation_types
    if column_group_version not in {None, "manual"}:
        raise ValueError("The public runtime supports only manual column-group archives.")
    return _identify_column_groups_manual(
        tables,
        columns_dict,
        schema_case_map=schema_case_map,
        column_group_artifact_root=column_group_artifact_root,
    )

############################ Value ###########################################
def _coerce_literal_record(raw_literal: Any) -> Dict[str, Any]:
    if isinstance(raw_literal, dict):
        existence = raw_literal.get("existence")
        if existence not in {True, False, None}:
            existence = None
        record = {
            "literal": raw_literal.get("literal", ""),
            "existence": existence,
            "matched_value": raw_literal.get("matched_value"),
            "source": raw_literal.get("source", "sql_parser"),
        }
        for key in (
            "table",
            "column",
            "reason",
            "retrieval_literal",
            "retrieval_candidates",
            "candidate_matched_value",
            "candidate_source",
        ):
            if key in raw_literal:
                record[key] = raw_literal.get(key)
        return record
    return {
        "literal": raw_literal,
        "existence": None,
        "matched_value": None,
        "source": "legacy_literal",
    }


def _format_anchor_value_option(
    *,
    literal: str,
    original_column_key: str,
    existence: Optional[bool],
) -> Dict[str, Any]:
    if existence is True:
        element = f"VALUE `{literal}` exist in COLUMN `{original_column_key}`"
        reasons = f"literal '{literal}' verified in original SQL column {original_column_key}"
    elif existence is False:
        element = f"VALUE `{literal}` not exist in COLUMN `{original_column_key}`"
        reasons = f"WARNING of current SQL: literal '{literal}' not found in original SQL column {original_column_key}; see if another column or value should be used."
    else:
        element = f"VALUE `{literal}` in COLUMN `{original_column_key}` without verify actual existence"
        reasons = f"literal '{literal}' mapped to original SQL column {original_column_key}, but actual DB existence was not verified."
    return {
        "element": element,
        "col": original_column_key,
        "value": literal,
        "eleType": "VALUE",
        "reasons": reasons,
        "different_table": False,
        "existence": existence,
        "is_original_anchor": True,
    }


def _dedupe_value_candidates_preserve_order(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        # MOD: value retrieval is case-sensitive; keep DB casing variants such
        # as `legal` vs `Legal` even when they come from the same column.
        key = (
            str(candidate.get("col", "")).casefold(),
            str(candidate.get("value", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _normalize_value_candidate_records(candidates: Any) -> List[Dict[str, Any]]:
    if isinstance(candidates, dict):
        return [
            {
                "col": col,
                "value": value,
                "retrieval_channel": "lsh",
                "score": None,
            }
            for col, value in candidates.items()
        ]
    return list(candidates or [])


def _build_value_retrieval_probes(
    cleaned_literal: str,
    literal_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = [
        {
            "value": cleaned_literal,
            "source": "anchor",
            "score": None,
            "is_anchor": True,
        }
    ]
    seen_values: Set[str] = {cleaned_literal}

    raw_candidates = literal_record.get("retrieval_candidates") or []
    if not raw_candidates and literal_record.get("candidate_matched_value") is not None:
        raw_candidates = [
            {
                "value": literal_record.get("candidate_matched_value"),
                "source": literal_record.get("candidate_source") or "substring_lookup",
                "score": None,
            }
        ]

    for raw_candidate in raw_candidates:
        if isinstance(raw_candidate, dict):
            candidate_raw_value = raw_candidate.get("value")
            candidate_source = raw_candidate.get("source") or "substring_lookup"
            candidate_score = raw_candidate.get("score")
        else:
            candidate_raw_value = raw_candidate
            candidate_source = "substring_lookup"
            candidate_score = None
        if candidate_raw_value is None:
            continue
        candidate_value = _clean_literal(str(candidate_raw_value))
        if not candidate_value or candidate_value in seen_values:
            continue
        probes.append(
            {
                "value": candidate_value,
                "source": candidate_source,
                "score": candidate_score,
                "is_anchor": False,
            }
        )
        seen_values.add(candidate_value)
        if len(probes) >= 4:
            break
    return probes


def _identify_value_options(
    literals: Dict[str, Dict[str, List[Any]]],
    LSH_top_n: int = 20,
    db_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Any]]]]:
    """
    Identify value options for literals found in a SQL part.

    Args:
        literals: Dictionary of table -> column -> list of literal values from get_sql_condition_literals

    Returns:
        List of value option dictionaries
        Schema with examples
    """
    value_options = []

    if not literals:
        return [], {}

    # Process literals organized by table and column
    schema_with_examples = {}
    for table_name, column_literals in literals.items():
        for column_name, literal_values in column_literals.items():
            for literal in literal_values:
                literal_record = _coerce_literal_record(literal)
                literal_value = literal_record.get("retrieval_literal", literal_record.get("literal", ""))
                # Clean the literal
                if type(literal_value) == str:
                    cleaned_literal = _clean_literal(literal_value)
                elif len(str(literal_value)) < 2 or len(str(literal_value)) > 100:
                    continue
                else:
                    cleaned_literal = str(literal_value).strip()

                if not cleaned_literal:
                    continue
                original_column_key = f"{table_name}.{column_name}"
                options = [
                    _format_anchor_value_option(
                        literal=cleaned_literal,
                        original_column_key=original_column_key,
                        existence=literal_record.get("existence"),
                    )
                ]
                seen_value_options: Set[Tuple[str, str]] = {
                    (original_column_key.casefold(), cleaned_literal)
                }

                retrieval_probes = _build_value_retrieval_probes(cleaned_literal, literal_record)
                for retrieval_probe in retrieval_probes:
                    probe_value = retrieval_probe["value"]
                    if not retrieval_probe.get("is_anchor"):
                        candidate_key = (original_column_key.casefold(), probe_value)
                        if candidate_key not in seen_value_options:
                            options.append({
                                "element": f"VALUE `{probe_value}` exists in COLUMN `{original_column_key}`",
                                "col": original_column_key,
                                "value": probe_value,
                                "eleType": "VALUE",
                                "reasons": (
                                    f"literal '{cleaned_literal}' has DB-backed retrieval candidate "
                                    f"'{probe_value}' from {retrieval_probe.get('source')}; "
                                    "this is not exact existence validation for the original anchor."
                                ),
                                "different_table": False,
                                "existence": True,
                                "is_original_anchor": False,
                                "source_anchor": cleaned_literal,
                                "probe_source": retrieval_probe.get("source"),
                                "score": retrieval_probe.get("score"),
                            })
                            seen_value_options.add(candidate_key)

                    # Find alternative columns that might contain this value
                    candidate_value_records, lsh_results_for_schema, lookup_succeeded = _find_columns_for_value(
                        probe_value,
                        LSH_top_n,
                    )
                    candidate_value_records = _normalize_value_candidate_records(candidate_value_records)
                    # update overall schema with examples for all candidates
                    for t, cols in lsh_results_for_schema.items():
                        for c, vals in cols.items():
                            if t not in schema_with_examples:
                                schema_with_examples[t] = {}
                            if c not in schema_with_examples[t]:
                                schema_with_examples[t][c] = []
                            for v in vals:
                                if v not in schema_with_examples[t][c]:
                                    schema_with_examples[t][c].append(v)
                    for candidate in candidate_value_records:
                        col = candidate["col"]
                        most_similar_value = candidate["value"]
                        if (
                            col.casefold() == original_column_key.casefold()
                            and str(most_similar_value) == cleaned_literal
                        ):
                            continue
                        candidate_key = (str(col).casefold(), str(most_similar_value))
                        if candidate_key in seen_value_options:
                            continue
                        options.append({
                            "element": f"VALUE `{most_similar_value}` exists in COLUMN `{col}`",
                            "col": col,
                            "value": most_similar_value,
                            "eleType": "VALUE",
                            "reasons": (
                                f"literal '{cleaned_literal}' used retrieval probe '{probe_value}' "
                                f"and found '{most_similar_value}' in column {col} via "
                                f"{candidate.get('retrieval_channel')}"
                            ),
                            "different_table": col.split(".")[0].lower() != table_name.lower() if "." in col else False,
                            "existence": True,
                            "is_original_anchor": False,
                            "retrieval_channel": candidate.get("retrieval_channel"),
                            "score": candidate.get("score"),
                            "source_anchor": cleaned_literal,
                            "retrieval_probe": probe_value,
                            "probe_source": retrieval_probe.get("source"),
                        })
                        seen_value_options.add(candidate_key)

                value_options.append({
                    "entity": cleaned_literal,
                    "type": "VALUE",
                    "options": options,
                })

    return value_options, schema_with_examples

def _clean_literal(literal: str) -> str:
    """
    Clean literal value by removing quotes and trimming.

    Args:
        literal: Raw literal from SQL

    Returns:
        Cleaned literal
    """
    if not literal:
        return ""

    # Remove surrounding quotes
    cleaned = literal.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
        (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1]

    # Skip if empty or too short/long
    if not cleaned or len(cleaned) < 2 or len(cleaned) > 100:
        return ""

    return cleaned


def _find_columns_for_value(
    value: str,
    LSH_top_n: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Any]]], bool]:
    """
    Find columns that might contain the given value and return the most similar value found.

    Args:
        state: Current system state
        value: Value to search for

    Returns:
        List[Dict[str, Any]]: Candidate value records with col/value/channel metadata
        Dict[str, Dict[str, List[Any]]]: Schema examples harvested from lookup results
        bool: Whether the lookup backend succeeded
    """
    schema_with_examples = {}  # table -> column -> list of similar values from LSH and schema examples
    raw_candidates: List[Dict[str, Any]] = []
    lookup_succeeded = False

    manager = DatabaseManager()
    try:
        lsh_results = manager.query_lsh(
            value,
            signature_size=LSH_SIGNATURE_SIZE,
            n_gram=LSH_N_GRAM,
            top_n=LSH_top_n,
        )
        lookup_succeeded = True
        for table_name, column_values in lsh_results.items():
            for column_name, values in column_values.items():
                column_key = f"{table_name}.{column_name}"
                best_similarity = 0.0
                best_value = None
                for returned_value in values:
                    if _values_are_similar(value, str(returned_value)):
                        similarity = _calculate_similarity(value, str(returned_value))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_value = str(returned_value)
                if best_value is not None:
                    raw_candidates.append(
                        {
                            "col": column_key,
                            "value": best_value,
                            "retrieval_channel": "lsh",
                            "score": best_similarity,
                        }
                    )
                schema_with_examples.setdefault(table_name, {}).setdefault(column_name, [])
                for value_example in values:
                    if value_example not in schema_with_examples[table_name][column_name]:
                        schema_with_examples[table_name][column_name].append(value_example)
    except Exception as exc:
        logging.warning(f"Failed to query LSH for value '{value}': {exc}")

    return _dedupe_value_candidates_preserve_order(raw_candidates), schema_with_examples, lookup_succeeded

def _calculate_similarity(value1: str, value2: str) -> float:
    """
    Calculate similarity score between two values.

    Args:
        value1: First value
        value2: Second value

    Returns:
        Similarity score between 0 and 1
    """
    v1 = str(value1).lower().strip()
    v2 = str(value2).lower().strip()

    return difflib.SequenceMatcher(None, v1, v2).ratio()

def _values_are_similar(value1: str, value2: str) -> bool:
    """
    Check if two values are similar enough to be considered the same.

    Args:
        value1: First value
        value2: Second value

    Returns:
        True if values are similar
    """
    if not value1 or not value2:
        return False

    # Normalize values
    v1 = str(value1).lower().strip()
    v2 = str(value2).lower().strip()

    # Exact match
    if v1 == v2:
        return True

    # Check if one contains the other AND is at least 3 characters long
    if (len(v1) >= 3 and v1 in v2) or (len(v2) >= 3 and v2 in v1):
        return True

    return False


# ------------------------------------------------------------------
# LLM fallback for column/value extraction (when SQL parsing fails)
# ------------------------------------------------------------------

def _llm_extract_elements(sql: str, db_id: str) -> Optional[Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]]:
    """Use LLM to extract columns and literal values from SQL when parsing fails.

    Returns (columns_dict, literals) in the same format expected by
    _identify_column_groups and _identify_value_options, or None on failure.

    columns_dict = {table_name: [column_name, ...]}
    literals = {table_name: {column_name: [value_str, ...]}}
    """
    if os.getenv("MODULE_A_ENABLE_LLM_FALLBACK", "1").lower() in {"0", "false", "no"}:
        return None
    return extract_elements_with_llm(sql, db_id)


######################### Main Interface #########################################
def get_DBeleOptions(
    sql: str,
    db_id: str,
    grp_context: Dict = {},
    LSH_top_n: int = 8,
    filter=False,
    column_group_version: str | None = None,
    column_group_relation_types: List[str] | None = None,
    column_group_artifact_root: Optional[str] = None,
    question_id: Optional[int] = None,
    return_sql_parse_meta: bool = False,
    retrieval_ablation_disable_channel: str = DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    return_retrieval_ablation: bool = False,
    column_retrieval_source: str = DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, Dict[str, List[Any]]], Dict[str, Dict[str, Dict[str, str]]]]:

    """
    grp_context["TARGETED"]
    grp_context["unitwise_partialsql"]

    """
    sql = _normalize_sql(sql)
    retrieval_ablation_disable_channel = validate_retrieval_ablation_disable_channel(
        retrieval_ablation_disable_channel
    )
    if column_retrieval_source not in MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES:
        raise ValueError(
            "column_retrieval_source must be one of "
            f"{MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES}, got {column_retrieval_source!r}"
        )
    column_channel_enabled = retrieval_channel_enabled(
        disable_channel=retrieval_ablation_disable_channel,
        channel="column",
    )
    value_channel_enabled = retrieval_channel_enabled(
        disable_channel=retrieval_ablation_disable_channel,
        channel="value",
    )
    sql_parse_meta: Dict[str, Any] = {
        "source": "none",
        "tables": [],
        "columns_dict": {},
    }
    parse_timing: Dict[str, Any] = {
        "column_parse_elapsed_s": 0.0,
        "literal_parse_elapsed_s": 0.0,
        "llm_fallback_used": False,
        "llm_fallback_elapsed_s": 0.0,
        "total_parse_elapsed_s": 0.0,
    }
    retrieval_ablation: Dict[str, Any] = {
        "disable_channel": retrieval_ablation_disable_channel,
        "column_retrieval_source": column_retrieval_source,
        "channels": {
            "column": {
                "enabled_for_detection": column_channel_enabled,
                "candidate_generation_executed": False,
            },
            "value": {
                "enabled_for_detection": value_channel_enabled,
                "candidate_generation_executed": False,
            },
        },
        "parsed_anchor_trace": {
            "columns": [],
            "literals": [],
            "unmatched_literals": [],
        },
        "active_dbelement_entry_counts": {"COLUMN": 0, "VALUE": 0},
    }

    def _return_result(
        result: Tuple[
            List[Dict[str, Any]],
            Dict[str, List[str]],
            Dict[str, Dict[str, List[Any]]],
            Dict[str, Dict[str, Dict[str, str]]],
        ],
    ):
        extras: List[Any] = []
        if return_sql_parse_meta:
            extras.append(sql_parse_meta)
        if return_retrieval_ablation:
            extras.append(retrieval_ablation)
        if extras:
            return (*result, *extras)
        return result

    def _record_sql_parse_meta(
        source: str,
        tables_value: List[str],
        columns_value: Dict[str, List[str]],
    ) -> None:
        nonlocal sql_parse_meta
        sql_parse_meta = {
            "source": source,
            "tables": [str(table) for table in (tables_value or [])],
            "columns_dict": {
                str(table): [str(column) for column in (columns or [])]
                for table, columns in (columns_value or {}).items()
            },
            "timing": _current_parse_timing(),
        }

    def _current_parse_timing() -> Dict[str, Any]:
        total = (
            float(parse_timing.get("column_parse_elapsed_s") or 0.0)
            + float(parse_timing.get("literal_parse_elapsed_s") or 0.0)
            + float(parse_timing.get("llm_fallback_elapsed_s") or 0.0)
        )
        timing = dict(parse_timing)
        timing["total_parse_elapsed_s"] = round(total, 6)
        return timing

    def _refresh_sql_parse_timing() -> None:
        sql_parse_meta["timing"] = _current_parse_timing()

    def _refresh_retrieval_ablation_trace(
        columns_value: Dict[str, List[str]],
        literal_existence_value: Dict[str, Dict[str, List[Any]]],
        unmatched_literals_value: List[Dict[str, Any]],
    ) -> None:
        retrieval_ablation["parsed_anchor_trace"] = {
            "columns": [
                {
                    "table": str(table_name),
                    "column": str(column_name),
                    "anchor_ref": f"{table_name}.{column_name}",
                }
                for table_name, columns in (columns_value or {}).items()
                for column_name in (columns or [])
            ],
            "literals": [
                {
                    "literal": literal_record.get("literal"),
                    "retrieval_literal": literal_record.get("retrieval_literal"),
                    "table": str(literal_record.get("table") or table_name),
                    "column": str(literal_record.get("column") or column_name),
                    "existence": literal_record.get("existence"),
                    "matched_value": literal_record.get("matched_value"),
                    "source": literal_record.get("source"),
                    "reason": literal_record.get("reason"),
                }
                for table_name, column_literals in (literal_existence_value or {}).items()
                for column_name, records in (column_literals or {}).items()
                for literal_record in (_coerce_literal_record(record) for record in (records or []))
            ],
            "unmatched_literals": [
                {
                    "literal": record.get("literal"),
                    "parent_sql": record.get("parent_sql"),
                    "reason": record.get("reason"),
                    "source": record.get("source"),
                }
                for record in (unmatched_literals_value or [])
                if isinstance(record, dict)
            ],
        }

    def _refresh_active_entry_counts(options: List[Dict[str, Any]]) -> None:
        retrieval_ablation["active_dbelement_entry_counts"] = {
            "COLUMN": sum(
                1 for option in (options or []) if str(option.get("type", "")).upper() == "COLUMN"
            ),
            "VALUE": sum(
                1 for option in (options or []) if str(option.get("type", "")).upper() == "VALUE"
            ),
        }

    if not sql:
        _refresh_sql_parse_timing()
        _refresh_active_entry_counts([])
        return _return_result(([], {}, {}, {}))

    dbelement_options = []
    column_options = []
    value_options = []
    tentative_schema: Dict[str, List[str]] = {}
    schema_with_examples: Dict[str, Dict[str, List[Any]]] = {}
    schema_with_descriptions: Dict[str, Dict[str, Dict[str, str]]] = {}

    def _identify_column_groups_with_optional_artifact_root(
        input_tables: List[str],
        input_columns_dict: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"schema_case_map": schema_case_map}
        if column_group_version is not None or column_group_relation_types is not None:
            kwargs["column_group_version"] = column_group_version
            kwargs["relation_types"] = column_group_relation_types
        if column_group_artifact_root:
            kwargs["column_group_artifact_root"] = column_group_artifact_root
        return _identify_column_groups(input_tables, input_columns_dict, **kwargs)

    try:
        schema_case_map = _load_schema_case_map()
        # Parse SQL to extract tables, columns, and literals
        tried_llm_fallback = False
        used_llm_fallback = False
        try:
            column_parse_start = monotonic_s()
            tables, columns_dict = get_db_aware_tables_and_columns_from_sql(db_id, sql)
            parse_timing["column_parse_elapsed_s"] = elapsed_s(column_parse_start)
            if columns_dict:
                tentative_schema = _canonicalize_schema_dict(columns_dict, schema_case_map)
                _record_sql_parse_meta("db_aware_parser", tables, columns_dict)
            else:
                tables, columns_dict = [], {}
        except Exception as e:
            parse_timing["column_parse_elapsed_s"] = elapsed_s(locals().get("column_parse_start", monotonic_s()))
            logging.warning(f"Failed to parse SQL for columns with DB-aware parser '{sql}': {e}")
            tables, columns_dict = [], {}

        try:
            literal_parse_start = monotonic_s()
            literal_parser_kwargs: Dict[str, Any] = {
                "filter": True,
                "return_details": True,
            }
            if not value_channel_enabled:
                literal_parser_kwargs["include_retrieval_candidates"] = False
            literals, literal_existence, unmatched_literals = get_sql_condition_literals(
                db_id,
                sql,
                **literal_parser_kwargs,
            )
            parse_timing["literal_parse_elapsed_s"] = elapsed_s(literal_parse_start)
        except Exception as e:
            parse_timing["literal_parse_elapsed_s"] = elapsed_s(locals().get("literal_parse_start", monotonic_s()))
            logging.warning(f"Failed to parse SQL for literals '{sql}': {e}")
            literals = {}
            literal_existence = {}
            unmatched_literals = []

        # NOTE: Unmatched SQL literals are collected for the future NLQ/LLM entity merge path.
        if unmatched_literals:
            print(f"[MODA-LITERAL] unmatched_literals={unmatched_literals}")

        # LLM fallback: only try to recover the missing dimension that really matters.
        columns_ok = bool(columns_dict) and bool(tables)
        needs_literals = _sql_needs_literal_extraction(sql)
        literals_ok = bool(literals) or not needs_literals
        if not columns_ok or (needs_literals and not literals_ok):
            logging.info(f"SQL parsing incomplete for '{sql[:80]}' (cols_ok={columns_ok}, lits_ok={literals_ok}), trying LLM fallback")
            tried_llm_fallback = True
            llm_fallback_start = monotonic_s()
            llm_result = _llm_extract_elements(sql, db_id)
            parse_timing["llm_fallback_elapsed_s"] = elapsed_s(llm_fallback_start)
            if llm_result:
                llm_columns, llm_literals = llm_result
                if not columns_ok and llm_columns:
                    columns_dict = llm_columns
                    tables = list(columns_dict.keys())
                    tentative_schema = _canonicalize_schema_dict(columns_dict, schema_case_map)
                    _record_sql_parse_meta("llm_fallback", tables, columns_dict)
                    used_llm_fallback = True
                    logging.info(f"LLM fallback provided {sum(len(v) for v in columns_dict.values())} columns")
                if needs_literals and not literals_ok and llm_literals:
                    mapped_llm_literals, unmatched_llm_literals = _split_llm_literals_for_validation(llm_literals)
                    if unmatched_llm_literals:
                        unmatched_literals.extend(unmatched_llm_literals)
                    literals = mapped_llm_literals
                    literal_existence = validate_literal_existence_records(
                        db_id,
                        mapped_llm_literals,
                        source="llm_fallback",
                    )
                    used_llm_fallback = True
                    logging.info(
                        "LLM fallback provided %s mapped literals and %s unmatched literals",
                        sum(len(vv) for v in mapped_llm_literals.values() for vv in v.values())
                        if mapped_llm_literals
                        else 0,
                        len(unmatched_llm_literals),
                    )
        parse_timing["llm_fallback_used"] = bool(used_llm_fallback)
        _refresh_sql_parse_timing()
        _refresh_retrieval_ablation_trace(columns_dict, literal_existence, unmatched_literals)

        # Find column groups for identified columns
        if column_channel_enabled and column_retrieval_source == MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP:
            column_options = _identify_column_groups_with_optional_artifact_root(tables, columns_dict)
            retrieval_ablation["channels"]["column"]["candidate_generation_executed"] = True
        elif column_channel_enabled:
            # ADD: Parsed anchors remain available while the stage-level
            # realtime retriever materializes COLUMN candidates in batch.
            retrieval_ablation["channels"]["column"]["candidate_generation_deferred_to"] = (
                "module_a_realtime_column_retrieval"
            )

        # Find value options for identified literals
        if value_channel_enabled:
            value_options, schema_with_examples = _identify_value_options(
                literal_existence,
                LSH_top_n,
                db_id,
            )
            retrieval_ablation["channels"]["value"]["candidate_generation_executed"] = True

        # Combine options
        part_options = column_options + value_options

        # MOD: Deduplicate SQLite identifiers case-insensitively while keeping DB-case spelling.
        dbelement_options = _merge_unique_dbelement_options(part_options, schema_case_map)

        tentative_schema = update_tentative_schema_from_parts(dbelement_options, tentative_schema, schema_case_map)
        DatabaseManager().add_connections_to_tentative_schema(tentative_schema)
        schema_with_examples, schema_with_descriptions = expand_schema_attributes(tentative_schema, schema_with_examples, {}, db_id)

        # filter by grp_context["TARGETED"] if exists, overwrite dbelement_options to only include options that related to columns in grp_context["TARGETED"]
        if filter and "TARGETED" in grp_context and "unitwise_partialsql" in grp_context:
            targeted_unitids = grp_context["TARGETED"]
            unitwise_partialsql = grp_context["unitwise_partialsql"]
            targeted_units = [unit_sql for unit_id, unit_sql in unitwise_partialsql.items() if unit_id in targeted_unitids]
            targeted_units_lower = [u.lower() for u in targeted_units]
            filtered_columns_dict = {}
            for tab, cols in columns_dict.items():
                for col in cols:
                    if any(col.lower() in u for u in targeted_units_lower):
                        if tab not in filtered_columns_dict:
                            filtered_columns_dict[tab] = []
                        if col not in filtered_columns_dict[tab]:
                            filtered_columns_dict[tab].append(col)
            filtered_literal_existence = {}
            for tab, col_literals in literal_existence.items():
                for col, records in col_literals.items():
                    for record in records:
                        literal_record = _coerce_literal_record(record)
                        lit = literal_record.get("literal", "")
                        if any(str(lit).lower() in u for u in targeted_units_lower):
                            if tab not in filtered_literal_existence:
                                filtered_literal_existence[tab] = {}
                            if col not in filtered_literal_existence[tab]:
                                filtered_literal_existence[tab][col] = []
                            if literal_record not in filtered_literal_existence[tab][col]:
                                filtered_literal_existence[tab][col].append(literal_record)
            # re-run option identification with filtered columns and literals
            if (
                column_channel_enabled
                and column_retrieval_source == MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP
            ):
                column_options = _identify_column_groups_with_optional_artifact_root(
                    tables,
                    filtered_columns_dict,
                )
            if value_channel_enabled:
                value_options, schema_with_examples = _identify_value_options(
                    filtered_literal_existence,
                    LSH_top_n,
                    db_id,
                )
            part_options = column_options + value_options
            dbelement_options = _merge_unique_dbelement_options(part_options, schema_case_map)
        else:
            pass # keep all options without filtering

    except Exception as e:
        logging.warning(f"Failed to parse SQL part '{sql}': {e}")

    _refresh_active_entry_counts(dbelement_options)
    return _return_result((dbelement_options, tentative_schema, schema_with_examples, schema_with_descriptions))

############################### Update to schema ##################################
def update_tentative_schema_from_parts(
    dbelement_options: List,
    tentative_schema: Dict,
    schema_case_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[str, List[str]]:
    """
    Update tentative schema to include any new columns found in part options.
    Also update schema_with_examples and schema_with_descriptions.

    Args:
        state: Current system state
    """

    tentative_schema = _canonicalize_schema_dict(tentative_schema, schema_case_map)

    def _add_column_ref(column_ref: str) -> None:
        canonical_ref = _canonicalize_column_ref(column_ref, schema_case_map)
        parts = _split_column_ref(canonical_ref)
        if not parts:
            return
        table_name, column_name = parts
        actual_table = next(
            (table for table in tentative_schema if table.casefold() == table_name.casefold()),
            table_name,
        )
        if actual_table not in tentative_schema:
            tentative_schema[actual_table] = []
        if column_name.casefold() not in [col.casefold() for col in tentative_schema[actual_table]]:
            tentative_schema[actual_table].append(column_name)

    for entity_dict in dbelement_options:
        for option in entity_dict.get("options", []):
            element = option.get("element", "")
            ele_type = option.get("eleType", "")

            if ele_type == "COLUMN" and "." in element:
                _add_column_ref(element)

            elif ele_type == "VALUE":
                # Use the "col" field for more reliable parsing
                col = option.get("col", "")
                if col and "." in col:
                    _add_column_ref(col)

    return tentative_schema


def expand_schema_attributes(tentative_schema: Dict, schema_with_examples: Dict, schema_with_descriptions: Dict, db_id) -> Tuple[Dict[str, Dict[str, List[Any]]], Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Expand schema_with_examples and schema_with_descriptions for new columns.

    Args:
        state: Current system state
    """
    # Load table descriptions
    db_path = DatabaseManager().db_directory_path
    table_descriptions = load_tables_description(db_path, use_value_description=True)

    # Get column descriptive names from column groups
    column_groups = DatabaseManager().load_column_groups()
    column_descriptive_names = column_groups.get("column_descriptive_names", {})

    # Get database profiler for this db_id
    db_profiler = get_db_profiler(db_id)
    db_manager = DatabaseManager()

    for table_name, columns in tentative_schema.items():
        for column_name in columns:
            column_key = f"{table_name}.{column_name}"

            # Initialize schema structures if needed
            if table_name not in schema_with_examples:
                schema_with_examples[table_name] = {}

            if table_name not in schema_with_descriptions:
                schema_with_descriptions[table_name] = {}

            # Update schema_with_examples with LSH results
            if column_name not in schema_with_examples[table_name]:
                schema_with_examples[table_name][column_name] = []

            # If no examples from LSH, get sample values from database using db_profiler
            if not schema_with_examples[table_name][column_name]:
                try:
                    sample_sql = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT 5"

                    # Use db_profiler for execution with caching
                    result = db_profiler.execute_sql(
                        db_manager.db_path,
                        sample_sql,
                        store=True,  # Enable caching
                        timeout=30
                    )

                    if result and len(result) > 0:
                        sample_values = [str(row[0]) for row in result]
                        # if totale length of combinded string of examples excced 100, try to reduce number of examples to fit in 100, if single example still excceed, do not add
                        total_length = sum(len(str(val)) for val in sample_values)
                        while total_length > 100 and len(sample_values) > 1:
                            sample_values.pop()  # Remove the last example
                            total_length = sum(len(str(val)) for val in sample_values)
                        if total_length > 100 and len(sample_values) == 1:
                            sample_values = []  # Do not add if single example still exceeds limit
                        schema_with_examples[table_name][column_name] = sample_values
                        logging.debug(f"Retrieved {len(sample_values)} sample values for {table_name}.{column_name}")
                    else:
                        logging.debug(f"No sample values found for {table_name}.{column_name}")

                except Exception as e:
                    logging.debug(f"Could not get examples for {table_name}.{column_name}: {e}")

            # Update schema_with_descriptions
            if column_name not in schema_with_descriptions[table_name]:
                # Get description info from table descriptions
                desc_info = {}
                if (table_name in table_descriptions and
                    column_name in table_descriptions[table_name]):
                    desc_info = table_descriptions[table_name][column_name]

                # Get descriptive name from column groups
                descriptive_name = _lookup_column_descriptive_name(
                    column_descriptive_names,
                    column_key,
                    fallback="",
                )

                schema_with_descriptions[table_name][column_name] = {
                    "column_name": desc_info.get("column_name", "") or descriptive_name,
                    "column_description": desc_info.get("column_description", ""),
                    "value_description": desc_info.get("value_description", ""),
                }

    return schema_with_examples, schema_with_descriptions
