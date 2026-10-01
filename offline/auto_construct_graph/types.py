from __future__ import annotations

from typing import Any


ColumnRef = str
PairKey = tuple[str, str]


def split_column_ref(ref: str) -> tuple[str, str]:
    text = str(ref or "").strip()
    if "." not in text:
        raise ValueError(f"column ref must be table.column, got: {ref!r}")
    table, column = text.split(".", 1)
    table = table.strip()
    column = column.strip()
    if not table or not column:
        raise ValueError(f"column ref must have non-empty table and column: {ref!r}")
    return table, column


def column_ref_from_object(value: Any) -> str:
    if isinstance(value, str):
        split_column_ref(value)
        return value.strip()
    if isinstance(value, dict):
        table = str(value.get("tab") or value.get("table") or "").strip()
        column = str(value.get("col") or value.get("column") or "").strip()
        if table and column:
            return f"{table}.{column}"
    raise ValueError(f"unsupported column reference: {value!r}")


def recover_verifier_column_object(value: Any, expected: Any) -> Any:
    """Return canonical {tab, col} only for unambiguous verifier column slips."""
    if not isinstance(value, dict):
        return value
    expected_ref = column_ref_from_object(expected)
    expected_table, expected_column = split_column_ref(expected_ref)
    table = str(value.get("tab") or value.get("table") or "").strip()
    if table != expected_table:
        return value

    try:
        if column_ref_from_object(value) == expected_ref:
            return {"tab": expected_table, "col": expected_column}
    except ValueError:
        pass

    non_table_items = [(key, val) for key, val in value.items() if key not in {"tab", "table"}]
    if len(non_table_items) != 1:
        return value
    key, raw_val = non_table_items[0]
    val = str(raw_val or "").strip()
    if str(key).strip() == expected_column or val == expected_column:
        return {"tab": expected_table, "col": expected_column}
    return value


def make_pair_key(col1: str, col2: str) -> PairKey:
    left = str(col1 or "").strip()
    right = str(col2 or "").strip()
    split_column_ref(left)
    split_column_ref(right)
    if left.casefold() == right.casefold():
        raise ValueError(f"pair endpoints must be distinct: {left!r}, {right!r}")
    return tuple(sorted((left, right)))
