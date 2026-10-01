from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set


DDL_MODE_PLAIN = "Sphinteract_plain"


def enrich_schema_ddl(
    schema_ddl: str,
    *,
    additional_info_by_column: Dict[str, str],
    ddl_mode: str,
    subset_only: bool,
    current_sql: str,
    dbelement_options: List[dict] | None,
) -> str:
    if ddl_mode == DDL_MODE_PLAIN:
        return schema_ddl

    allowed_columns = set(additional_info_by_column.keys())
    if subset_only:
        sql_columns = extract_sql_column_refs(current_sql)
        dbelement_columns = extract_dbelement_columns(dbelement_options or [])
        allowed_columns = allowed_columns & sql_columns & dbelement_columns

    return _append_inline_info(schema_ddl, additional_info_by_column, allowed_columns)


def extract_sql_column_refs(sql: str) -> Set[str]:
    refs: Set[str] = set()
    for table, column in re.findall(r"([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)", sql or ""):
        refs.add(f"{table}.{column}")
    return refs


def extract_dbelement_columns(dbelement_options: Iterable[dict]) -> Set[str]:
    refs: Set[str] = set()
    for entry in dbelement_options or []:
        for option in entry.get("options", []) or []:
            col = option.get("col")
            if isinstance(col, str) and "." in col:
                refs.add(col)
    return refs


def _append_inline_info(schema_ddl: str, info_by_column: Dict[str, str], allowed_columns: Set[str]) -> str:
    current_table = ""
    enriched_lines: List[str] = []

    for raw_line in schema_ddl.splitlines():
        line = raw_line
        create_match = re.match(r"\s*CREATE TABLE\s+[`\"]?([\w -]+)[`\"]?\s*$", line)
        if create_match:
            current_table = create_match.group(1)
            enriched_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith(("(", ")", "foreign key", "primary key")):
            enriched_lines.append(line)
            continue

        column_name = _extract_column_name(stripped)
        if not column_name or not current_table:
            enriched_lines.append(line)
            continue

        key = f"{current_table}.{column_name}"
        extra = info_by_column.get(key, "").strip()
        if key in allowed_columns and extra:
            enriched_lines.append(f"{line} -- {extra}")
        else:
            enriched_lines.append(line)

    return "\n".join(enriched_lines)


def _extract_column_name(stripped_line: str) -> str:
    token = stripped_line.rstrip(",").split(" ", 1)[0]
    return token.strip("`\"")
