from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


SCHEMA_FILTER_NONE = "none"
SCHEMA_FILTER_PROMPT_COLUMNS = "prompt_columns"
MODULE_A_SCHEMA_FILTER_MODES = (
    SCHEMA_FILTER_NONE,
    SCHEMA_FILTER_PROMPT_COLUMNS,
)


@dataclass(frozen=True)
class _CreateTableBlock:
    table_name: str
    body: str


def filter_schema_for_detection(
    schema_ddl: str,
    *,
    db_path: str | None,
    current_sql: str,
    prompt_dbelement_options: Iterable[dict] | None,
    mode: str,
    sql_columns_by_table: dict | None = None,
) -> tuple[str, dict]:
    if mode not in MODULE_A_SCHEMA_FILTER_MODES:
        raise ValueError(f"Unsupported Module A schema filter mode: {mode}")

    meta = _base_meta(schema_ddl, mode)
    if mode == SCHEMA_FILTER_NONE:
        meta.update(
            {
                "filter_action": "no_action",
                "fallback_reason": None,
                "sql_column_count": 0,
                "special_column_count": 0,
                "dbelement_column_count": 0,
                "selected_column_count": 0,
                "output_table_count": meta["input_table_count"],
                "output_column_count": meta["input_column_count"],
            }
        )
        return schema_ddl, meta

    try:
        schema_columns = _load_schema_columns(db_path, schema_ddl)
        sql_columns = _collect_sql_columns(
            db_path,
            current_sql,
            schema_columns,
            sql_columns_by_table=sql_columns_by_table,
        )
        special_columns = _collect_special_columns(db_path, schema_columns, schema_ddl)
        dbelement_columns = _collect_dbelement_columns(prompt_dbelement_options or [], schema_columns)
        selected_columns = sql_columns | special_columns | dbelement_columns

        filtered_schema = _filter_schema_ddl(schema_ddl, selected_columns, schema_columns)
        output_table_count, output_column_count = _count_schema(filtered_schema)
        meta.update(
            {
                "filter_action": "filtered",
                "fallback_reason": None,
                "sql_column_count": len(sql_columns),
                "special_column_count": len(special_columns),
                "dbelement_column_count": len(dbelement_columns),
                "selected_column_count": len(selected_columns),
                "output_table_count": output_table_count,
                "output_column_count": output_column_count,
            }
        )
        return filtered_schema, meta
    except Exception as exc:
        meta.update(
            {
                "filter_action": "fallback_full_schema",
                "fallback_reason": f"{exc.__class__.__name__}: {exc}",
                "sql_column_count": 0,
                "special_column_count": 0,
                "dbelement_column_count": 0,
                "selected_column_count": 0,
                "output_table_count": meta["input_table_count"],
                "output_column_count": meta["input_column_count"],
            }
        )
        return schema_ddl, meta


def _base_meta(schema_ddl: str, mode: str) -> dict:
    input_table_count, input_column_count = _count_schema(schema_ddl)
    return {
        "schema_filter_mode": mode,
        "input_table_count": input_table_count,
        "input_column_count": input_column_count,
    }


def _load_schema_columns(db_path: str | None, schema_ddl: str) -> Dict[str, str]:
    from_db = _load_schema_columns_from_db(db_path)
    if from_db:
        return from_db
    return _load_schema_columns_from_ddl(schema_ddl)


def _load_schema_columns_from_db(db_path: str | None) -> Dict[str, str]:
    if not db_path:
        return {}
    path = Path(str(db_path)).expanduser()
    if not path.exists():
        return {}

    conn = sqlite3.connect(str(path))
    try:
        tables = _list_user_tables(conn)
        columns: Dict[str, str] = {}
        for table in tables:
            for row in conn.execute(f'PRAGMA table_info("{_escape_identifier(table)}")'):
                column = str(row[1] or "").strip()
                if not column:
                    continue
                ref = f"{table}.{column}"
                columns[_normalize_ref(ref)] = ref
        return columns
    finally:
        conn.close()


def _load_schema_columns_from_ddl(schema_ddl: str) -> Dict[str, str]:
    columns: Dict[str, str] = {}
    for block in _iter_create_table_blocks(schema_ddl):
        for definition in _split_top_level(block.body):
            definition = definition.strip()
            if _is_table_constraint(definition):
                continue
            column_name = _extract_identifier(definition)
            if not column_name:
                continue
            ref = f"{block.table_name}.{column_name}"
            columns[_normalize_ref(ref)] = ref
    return columns


def _collect_sql_columns(
    db_path: str | None,
    current_sql: str,
    schema_columns: Dict[str, str],
    *,
    sql_columns_by_table: dict | None = None,
) -> Set[str]:
    refs: Set[str] = set()
    refs.update(_collect_supplied_sql_columns(sql_columns_by_table, schema_columns))
    if refs:
        return refs

    if db_path:
        try:
            from ..vendor_pipeline0612.live_retrieval.sql_parser import get_sql_columns_dict

            columns_by_table = get_sql_columns_dict(str(db_path), current_sql or "")
            for table, columns in (columns_by_table or {}).items():
                for column in columns or []:
                    _add_canonical_ref(refs, f"{table}.{column}", schema_columns)
        except Exception:
            pass

    if refs:
        return refs
    return _refs_from_text(current_sql, schema_columns)


def _collect_supplied_sql_columns(
    sql_columns_by_table: dict | None,
    schema_columns: Dict[str, str],
) -> Set[str]:
    refs: Set[str] = set()
    if not isinstance(sql_columns_by_table, dict):
        return refs
    for table, columns in sql_columns_by_table.items():
        if not isinstance(columns, (list, tuple, set)):
            continue
        for column in columns:
            column_text = str(column or "").strip()
            if not column_text:
                continue
            if "." in column_text:
                _add_canonical_ref(refs, column_text, schema_columns)
            else:
                _add_canonical_ref(refs, f"{table}.{column_text}", schema_columns)
    return refs


def _collect_special_columns(
    db_path: str | None,
    schema_columns: Dict[str, str],
    schema_ddl: str,
) -> Set[str]:
    if not db_path:
        return _collect_special_columns_from_ddl(schema_ddl, schema_columns)
    path = Path(str(db_path)).expanduser()
    if not path.exists():
        return _collect_special_columns_from_ddl(schema_ddl, schema_columns)

    refs: Set[str] = set()
    conn = sqlite3.connect(str(path))
    try:
        tables = _list_user_tables(conn)
        for table in tables:
            table_info = list(conn.execute(f'PRAGMA table_info("{_escape_identifier(table)}")'))
            for row in table_info:
                column = str(row[1] or "").strip()
                is_pk = bool(row[5])
                if column and is_pk:
                    _add_canonical_ref(refs, f"{table}.{column}", schema_columns)

            for row in conn.execute(f'PRAGMA foreign_key_list("{_escape_identifier(table)}")'):
                source_column = str(row[3] or "").strip()
                target_table = str(row[2] or "").strip()
                target_column = str(row[4] or "").strip()
                if source_column:
                    _add_canonical_ref(refs, f"{table}.{source_column}", schema_columns)
                if target_table and target_column:
                    _add_canonical_ref(refs, f"{target_table}.{target_column}", schema_columns)

            for row in conn.execute(f'PRAGMA index_list("{_escape_identifier(table)}")'):
                index_name = str(row[1] or "").strip()
                is_unique = bool(row[2])
                if not index_name or not is_unique:
                    continue
                for index_row in conn.execute(f'PRAGMA index_info("{_escape_identifier(index_name)}")'):
                    index_column = str(index_row[2] or "").strip()
                    if index_column:
                        _add_canonical_ref(refs, f"{table}.{index_column}", schema_columns)
    finally:
        conn.close()
    return refs


def _collect_special_columns_from_ddl(
    schema_ddl: str,
    schema_columns: Dict[str, str],
) -> Set[str]:
    refs: Set[str] = set()
    for block in _iter_create_table_blocks(schema_ddl):
        for raw_definition in _split_top_level(block.body):
            definition = raw_definition.strip()
            if not definition:
                continue
            if _is_table_constraint(definition):
                for column in _local_special_constraint_columns(definition):
                    _add_canonical_ref(refs, f"{block.table_name}.{column}", schema_columns)
                for target_ref in _referenced_columns_in_constraint(definition):
                    _add_canonical_ref(refs, target_ref, schema_columns)
                continue

            column_name = _extract_identifier(definition)
            if not column_name:
                continue
            if _definition_has_special_inline_constraint(definition):
                _add_canonical_ref(refs, f"{block.table_name}.{column_name}", schema_columns)
            for target_ref in _referenced_columns_in_constraint(definition):
                _add_canonical_ref(refs, target_ref, schema_columns)
    return refs


def _collect_dbelement_columns(
    prompt_dbelement_options: Iterable[dict],
    schema_columns: Dict[str, str],
) -> Set[str]:
    refs: Set[str] = set()
    for entry in prompt_dbelement_options or []:
        for text in _iter_string_values(entry):
            refs.update(_refs_from_text(text, schema_columns))
    return refs


def _filter_schema_ddl(
    schema_ddl: str,
    selected_columns: Set[str],
    schema_columns: Dict[str, str],
) -> str:
    blocks = list(_iter_create_table_blocks(schema_ddl))
    if not blocks:
        raise ValueError("no CREATE TABLE blocks found")

    selected_by_table: Dict[str, Set[str]] = {}
    for ref in selected_columns:
        canonical = _canonical_ref(ref, schema_columns)
        if not canonical or "." not in canonical:
            continue
        table, column = canonical.split(".", 1)
        selected_by_table.setdefault(_normalize_identifier(table), set()).add(
            _normalize_identifier(column)
        )

    rendered_blocks: List[str] = []
    for block in blocks:
        selected_columns_for_table = selected_by_table.get(_normalize_identifier(block.table_name), set())
        if not selected_columns_for_table:
            continue

        kept_definitions: List[str] = []
        for raw_definition in _split_top_level(block.body):
            definition = raw_definition.strip()
            if not definition:
                continue
            if _is_table_constraint(definition):
                constraint_columns = {
                    _normalize_identifier(column)
                    for column in _columns_in_constraint(definition)
                }
                if constraint_columns & selected_columns_for_table:
                    kept_definitions.append(_strip_trailing_comma(definition))
                continue

            column_name = _extract_identifier(definition)
            if (
                column_name
                and _normalize_identifier(column_name) in selected_columns_for_table
            ):
                kept_definitions.append(_strip_trailing_comma(definition))

        if not kept_definitions:
            continue
        rendered_blocks.append(_render_create_table(block.table_name, kept_definitions))

    return "\n".join(rendered_blocks)


def _render_create_table(table_name: str, definitions: Sequence[str]) -> str:
    lines = [f"CREATE TABLE {table_name}", "("]
    for index, definition in enumerate(definitions):
        suffix = "," if index < len(definitions) - 1 else ""
        lines.append(f"    {definition}{suffix}")
    lines.append(")")
    return "\n".join(lines)


def _count_schema(schema_ddl: str) -> tuple[int, int]:
    try:
        blocks = list(_iter_create_table_blocks(schema_ddl))
        table_count = len(blocks)
        column_count = 0
        for block in blocks:
            for definition in _split_top_level(block.body):
                definition = definition.strip()
                if definition and not _is_table_constraint(definition):
                    column_count += 1
        return table_count, column_count
    except Exception:
        return 0, 0


def _iter_create_table_blocks(schema_ddl: str) -> Iterable[_CreateTableBlock]:
    ddl = schema_ddl or ""
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w -]*))\s*\(",
        re.IGNORECASE,
    )
    position = 0
    while True:
        match = pattern.search(ddl, position)
        if not match:
            break
        open_paren_index = match.end() - 1
        close_paren_index = _find_matching_paren(ddl, open_paren_index)
        if close_paren_index < 0:
            break
        table_name = _clean_identifier(match.group("name"))
        yield _CreateTableBlock(
            table_name=table_name,
            body=ddl[open_paren_index + 1 : close_paren_index],
        )
        position = close_paren_index + 1


def _find_matching_paren(text: str, open_paren_index: int) -> int:
    depth = 0
    quote_char = ""
    bracket_quote = False
    index = open_paren_index
    while index < len(text):
        char = text[index]
        if quote_char:
            if bracket_quote and char == "]":
                quote_char = ""
                bracket_quote = False
            elif char == quote_char:
                if index + 1 < len(text) and text[index + 1] == quote_char:
                    index += 1
                else:
                    quote_char = ""
            index += 1
            continue

        if char in {"'", '"', "`"}:
            quote_char = char
        elif char == "[":
            quote_char = "]"
            bracket_quote = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _split_top_level(body: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote_char = ""
    bracket_quote = False
    index = 0
    while index < len(body):
        char = body[index]
        if quote_char:
            if bracket_quote and char == "]":
                quote_char = ""
                bracket_quote = False
            elif char == quote_char:
                if index + 1 < len(body) and body[index + 1] == quote_char:
                    index += 1
                else:
                    quote_char = ""
            index += 1
            continue

        if char in {"'", '"', "`"}:
            quote_char = char
        elif char == "[":
            quote_char = "]"
            bracket_quote = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
        index += 1
    parts.append(body[start:])
    return parts


def _is_table_constraint(definition: str) -> bool:
    first_token = _extract_identifier(definition).lower()
    return first_token in {
        "constraint",
        "primary",
        "foreign",
        "unique",
        "check",
        "exclude",
    }


def _columns_in_constraint(definition: str) -> List[str]:
    columns: List[str] = []
    for group in re.findall(r"\(([^()]+)\)", definition):
        for part in _split_top_level(group):
            candidate = _clean_identifier(part.strip().split()[0]) if part.strip() else ""
            if candidate and candidate.upper() not in {"ASC", "DESC"}:
                columns.append(candidate)
    return columns


def _definition_has_special_inline_constraint(definition: str) -> bool:
    return bool(
        re.search(
            r"\bPRIMARY\s+KEY\b|\bUNIQUE\b|\bREFERENCES\b",
            definition or "",
            re.IGNORECASE,
        )
    )


def _local_special_constraint_columns(definition: str) -> List[str]:
    if re.search(r"\bFOREIGN\s+KEY\b", definition or "", re.IGNORECASE):
        match = re.search(r"\bFOREIGN\s+KEY\s*\(([^()]+)\)", definition, re.IGNORECASE)
        return _columns_from_parenthesized_group(match.group(1) if match else "")
    if re.search(r"\bPRIMARY\s+KEY\b|\bUNIQUE\b", definition or "", re.IGNORECASE):
        return _columns_in_constraint(definition)
    return []


def _referenced_columns_in_constraint(definition: str) -> List[str]:
    refs: List[str] = []
    pattern = re.compile(
        r"\bREFERENCES\s+(?P<table>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w -]*)\s*\((?P<columns>[^()]+)\)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(definition or ""):
        table = _clean_identifier(match.group("table"))
        for column in _columns_from_parenthesized_group(match.group("columns")):
            refs.append(f"{table}.{column}")
    return refs


def _columns_from_parenthesized_group(group: str) -> List[str]:
    columns: List[str] = []
    for part in _split_top_level(group or ""):
        token = part.strip().split()[0] if part.strip() else ""
        candidate = _clean_identifier(token)
        if candidate and candidate.upper() not in {"ASC", "DESC"}:
            columns.append(candidate)
    return columns


def _extract_identifier(definition: str) -> str:
    stripped = definition.strip()
    if not stripped:
        return ""
    if stripped[0] in {'"', "`"}:
        closing = stripped.find(stripped[0], 1)
        if closing > 0:
            return stripped[1:closing]
    if stripped[0] == "[":
        closing = stripped.find("]", 1)
        if closing > 0:
            return stripped[1:closing]
    return _clean_identifier(stripped.split(None, 1)[0].rstrip(","))


def _refs_from_text(text: Any, schema_columns: Dict[str, str]) -> Set[str]:
    if not isinstance(text, str):
        return set()
    refs: Set[str] = set()
    text_lc = text.lower()
    for normalized, canonical in schema_columns.items():
        if normalized in text_lc:
            refs.add(canonical)

    generic_pattern = re.compile(
        r"([A-Za-z_][\w -]*)\.([A-Za-z_][\w ()/%+\-&]*[A-Za-z0-9_\)])"
    )
    for table, column in generic_pattern.findall(text):
        column = column.split(":", 1)[0].strip()
        _add_canonical_ref(refs, f"{table.strip()}.{column}", schema_columns)
    return refs


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_string_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_string_values(nested)


def _add_canonical_ref(refs: Set[str], ref: str, schema_columns: Dict[str, str]) -> None:
    canonical = _canonical_ref(ref, schema_columns)
    if canonical:
        refs.add(canonical)


def _canonical_ref(ref: str, schema_columns: Dict[str, str]) -> str:
    cleaned = _clean_ref(ref)
    if "." not in cleaned:
        return ""
    return schema_columns.get(_normalize_ref(cleaned), cleaned)


def _clean_ref(ref: str) -> str:
    cleaned = str(ref or "").strip().strip("`\"[]")
    cleaned = cleaned.split(":", 1)[0].strip()
    if "." not in cleaned:
        return cleaned
    table, column = cleaned.split(".", 1)
    return f"{_clean_identifier(table)}.{_clean_identifier(column)}"


def _normalize_ref(ref: str) -> str:
    return _clean_ref(ref).lower()


def _normalize_identifier(identifier: str) -> str:
    return _clean_identifier(identifier).lower()


def _clean_identifier(identifier: str) -> str:
    return str(identifier or "").strip().strip("`\"[]").rstrip(",")


def _strip_trailing_comma(definition: str) -> str:
    return definition.strip().rstrip(",")


def _escape_identifier(identifier: str) -> str:
    return str(identifier).replace('"', '""')


def _list_user_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]
