import logging
import sqlite3
import sqlvalidator
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from func_timeout import func_timeout, FunctionTimedOut
import os

from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify

from .db_info import get_table_all_columns, get_db_all_tables

from .fixed_parse_one import fixed_parse_one

from dotenv import load_dotenv
load_dotenv()

def format_sql_query(query, meta_time_out = 10):
    try:
        return func_timeout(meta_time_out, sqlvalidator.format_sql, args=(query))
    except FunctionTimedOut:
        print(f"Timeout in format_sql_query: {query}")
        return query
    except Exception:
        return query


def get_sql_tables(db_path: str, sql: str) -> List[str]:
    """
    Retrieves table names involved in an SQL query.

    Args:
        db_path (str): Path to the database file.
        sql (str): The SQL query string.

    Returns:
        List[str]: List of table names involved in the SQL query.
    """
    db_tables = get_db_all_tables(db_path)
    try:
        parsed_tables = list(fixed_parse_one(sql, read='sqlite').find_all(exp.Table))
        correct_tables = [
            str(table.name).strip().replace('\"', '').replace('`', '')
            for table in parsed_tables
            if str(table.name).strip().lower() in [db_table.lower() for db_table in db_tables]
        ]
        return correct_tables
    except Exception as e:
        logging.critical(f"Error in get_sql_tables: {e}\nSQL: {sql}")
        raise e

def _get_main_parent(expression: exp.Expression) -> Optional[exp.Expression]:
    """
    Retrieves the main parent expression for a given SQL expression.

    Args:
        expression (exp.Expression): The SQL expression.

    Returns:
        Optional[exp.Expression]: The main parent expression or None if not found.
    """
    parent = expression.parent
    while parent and not isinstance(parent, exp.Subquery):
        parent = parent.parent
    return parent

def _get_table_with_alias(parsed_sql: exp.Expression, alias: str) -> Optional[exp.Table]:
    """
    Retrieves the table associated with a given alias.

    Args:
        parsed_sql (exp.Expression): The parsed SQL expression.
        alias (str): The table alias.

    Returns:
        Optional[exp.Table]: The table associated with the alias or None if not found.
    """
    return next((table for table in parsed_sql.find_all(exp.Table) if table.alias == alias), None)

def get_sql_columns_dict(db_path: str, sql: str) -> Dict[str, List[str]]:
    """
    Retrieves a dictionary of tables and their respective columns involved in an SQL query.

    Args:
        db_path (str): Path to the database file.
        sql (str): The SQL query string.

    Returns:
        Dict[str, List[str]]: Dictionary of tables and their columns.
    """
    sql = qualify(fixed_parse_one(sql, read='sqlite'), qualify_columns=True, validate_qualify_columns=False) if isinstance(sql, str) else sql
    columns_dict = {}

    sub_queries = [subq for subq in sql.find_all(exp.Subquery) if subq != sql]
    for sub_query in sub_queries:
        subq_columns_dict = get_sql_columns_dict(db_path, sub_query)
        for table, columns in subq_columns_dict.items():
            if table not in columns_dict:
                columns_dict[table] = columns
            else:
                columns_dict[table].extend([col for col in columns if col.lower() not in [c.lower() for c in columns_dict[table]]])

    for column in sql.find_all(exp.Column):
        column_name = column.name
        table_alias = column.table
        table_name = None

        if table_alias:
            table = _get_table_with_alias(sql, table_alias)
            if table:
                table_name = table.name
            else:
                direct_table = next(
                    (
                        candidate_table.name
                        for candidate_table in sql.find_all(exp.Table)
                        if candidate_table.name.lower() == table_alias.lower()
                    ),
                    None,
                )
                table_name = direct_table or table_alias

        if not table_name:
            candidate_tables = [t for t in sql.find_all(exp.Table) if _get_main_parent(t) == _get_main_parent(column)]
            for candidate_table in candidate_tables:
                table_columns = get_table_all_columns(db_path, candidate_table.name)
                if column_name.lower() in [col.lower() for col in table_columns]:
                    table_name = candidate_table.name
                    break

        if table_name:
            if table_name not in columns_dict:
                columns_dict[table_name] = []
            if column_name.lower() not in [c.lower() for c in columns_dict[table_name]]:
                columns_dict[table_name].append(column_name)

    return columns_dict


def get_db_aware_tables_and_columns_from_sql(db_id: str, sql: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """Resolve SQL tables and columns using the actual SQLite schema when possible."""
    db_root = os.getenv("DB_ROOT_DIRECTORY", "")
    db_path = os.path.abspath(os.path.expanduser(os.path.join(db_root, db_id, f"{db_id}.sqlite")))

    columns_dict = get_sql_columns_dict(db_path=db_path, sql=sql)
    tables = list(columns_dict.keys())

    if not tables:
        tables = [
            str(table).strip().replace('"', '').replace("`", "").lower()
            for table in get_sql_tables(db_path, sql)
        ]

    return tables, columns_dict

def _check_value_exists(db_path: str, table_name: str, column_name: str, value: str) -> Optional[Any]:
    """
    Return the stored DB value only when the candidate exactly equals a row value.

    This intentionally does not use LIKE/substring matching, because the returned
    value drives Module A's `exist in` / `not exist in` label.
    """
    try:
        if not db_path or not os.path.exists(db_path):
            return None

        conn = sqlite3.connect(str(db_path), timeout=5)
        cursor = conn.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        rows = cursor.fetchall()
        actual_column = column_name
        col_type = ""
        for row in rows:
            if str(row[1]).lower() == str(column_name).lower():
                actual_column = row[1]
                col_type = (row[2] or "").upper()
                break
        else:
            conn.close()
            return None

        cursor.execute(
            f'SELECT "{actual_column}" FROM "{table_name}" WHERE "{actual_column}" = ? LIMIT 1',
            (value,),
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row[0]

        if any(token in col_type for token in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")):
            cursor.execute(
                f'SELECT "{actual_column}" FROM "{table_name}" WHERE CAST("{actual_column}" AS TEXT) = ? LIMIT 1',
                (str(value),),
            )
            row = cursor.fetchone()
            if row:
                conn.close()
                return row[0]

        conn.close()
        return None
    except Exception:
        return None

_CONDITION_EXP_TYPES = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.In,
    exp.Like,
    exp.ILike,
    exp.Between,
)

_CONDITION_BOUNDARY_TYPES = (
    exp.Select,
    exp.Subquery,
    exp.Union,
    exp.Except,
    exp.Intersect,
)


def _literal_text(literal: exp.Expression) -> str:
    if isinstance(literal, exp.Literal):
        return str(literal.this)
    return str(literal)


def _iter_condition_literal_nodes(parsed_sql: exp.Expression) -> List[exp.Expression]:
    literal_nodes: List[exp.Expression] = list(parsed_sql.find_all(exp.Literal))
    literal_nodes.extend(parsed_sql.find_all(exp.Boolean))
    return literal_nodes


def _is_date_format_literal(literal: exp.Expression) -> bool:
    if not isinstance(literal, exp.Literal) or not literal.is_string:
        return False
    literal_value = str(literal.this or "")
    if not literal_value.startswith("%"):
        return False

    parent = literal.parent
    while parent is not None and not isinstance(parent, _CONDITION_EXP_TYPES):
        parent_class_name = parent.__class__.__name__.lower()
        if parent_class_name in {"timetostr", "strtotime"}:
            return True
        parent = parent.parent
    return False


def _condition_parent_for_literal(literal: exp.Literal) -> Optional[exp.Expression]:
    parent = literal.parent
    while parent is not None:
        if isinstance(parent, _CONDITION_EXP_TYPES) and list(parent.find_all(exp.Column)):
            return parent
        if isinstance(parent, _CONDITION_BOUNDARY_TYPES):
            return None
        parent = parent.parent
    return None


def _literal_validation_value(literal_value: str, parent_context: exp.Expression) -> str:
    if isinstance(parent_context, (exp.Like, exp.ILike)):
        return literal_value.replace("%", "")
    return literal_value


def _append_unique_literal(
    used_entities: Dict[str, Dict[str, List[str]]],
    table_name: str,
    column_name: str,
    literal_value: str,
) -> None:
    used_entities.setdefault(table_name, {}).setdefault(column_name, [])
    if literal_value not in used_entities[table_name][column_name]:
        used_entities[table_name][column_name].append(literal_value)


def _append_unique_literal_existence(
    literal_existence: Dict[str, Dict[str, List[Dict[str, Any]]]],
    table_name: str,
    column_name: str,
    record: Dict[str, Any],
) -> None:
    literal_existence.setdefault(table_name, {}).setdefault(column_name, [])
    record_key = (
        str(record.get("table")),
        str(record.get("column")),
        str(record.get("literal")),
        record.get("existence"),
        str(record.get("matched_value")),
        str(record.get("source")),
        str(record.get("reason")),
    )
    existing_keys = {
        (
            str(existing.get("table")),
            str(existing.get("column")),
            str(existing.get("literal")),
            existing.get("existence"),
            str(existing.get("matched_value")),
            str(existing.get("source")),
            str(existing.get("reason")),
        )
        for existing in literal_existence[table_name][column_name]
    }
    if record_key not in existing_keys:
        literal_existence[table_name][column_name].append(record)


def validate_literal_existence_records(
    db_id: str,
    literals: Dict[str, Dict[str, List[Any]]],
    *,
    source: str = "sql_parser",
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    db_root = os.getenv("DB_ROOT_DIRECTORY", "")
    db_path = os.path.abspath(os.path.expanduser(os.path.join(db_root, db_id, f"{db_id}.sqlite")))
    literal_existence: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    try:
        actual_db_tables = set(table.lower() for table in get_db_all_tables(db_path))
    except Exception:
        actual_db_tables = set()

    for table_name, column_literals in (literals or {}).items():
        for column_name, literal_values in (column_literals or {}).items():
            for literal_value in literal_values or []:
                existence: Optional[bool] = None
                matched_value: Optional[Any] = None
                retrieval_candidates: List[Dict[str, Any]] = []
                if str(table_name).lower() in actual_db_tables:
                    validation_value = str(literal_value)
                    value_check = _check_value_exists(db_path, table_name, column_name, validation_value)
                    existence = value_check is not None
                    matched_value = value_check if value_check is not None else None
                    retrieval_candidates = _find_substring_value_candidates(
                        db_path,
                        table_name,
                        column_name,
                        validation_value,
                    )
                _append_unique_literal_existence(
                    literal_existence,
                    table_name,
                    column_name,
                    {
                        "literal": literal_value,
                        "retrieval_literal": validation_value,
                        "table": table_name,
                        "column": column_name,
                        "existence": existence,
                        "matched_value": matched_value,
                        "source": source,
                        "reason": "mapped_condition_literal" if existence is not None else "mapped_condition_literal_unverified",
                        "retrieval_candidates": retrieval_candidates,
                        "candidate_matched_value": (
                            retrieval_candidates[0]["value"] if retrieval_candidates else None
                        ),
                        "candidate_source": (
                            "substring_lookup" if retrieval_candidates else None
                        ),
                    },
                )
    return literal_existence


def _normalized_edit_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, str(left).casefold(), str(right).casefold()).ratio()


def _find_substring_value_candidates(
    db_path: str,
    table_name: str,
    column_name: str,
    value: str,
    *,
    max_candidates: int = 3,
) -> List[Dict[str, Any]]:
    try:
        if not db_path or not os.path.exists(db_path) or not str(value):
            return []

        conn = sqlite3.connect(str(db_path), timeout=5)
        cursor = conn.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        rows = cursor.fetchall()
        actual_column = None
        for row in rows:
            if str(row[1]).lower() == str(column_name).lower():
                actual_column = row[1]
                break
        if actual_column is None:
            conn.close()
            return []

        cursor.execute(
            f'SELECT DISTINCT "{actual_column}" FROM "{table_name}" '
            f'WHERE CAST("{actual_column}" AS TEXT) LIKE ?',
            (f"%{value}%",),
        )
        rows = cursor.fetchall()
        conn.close()

        scored_candidates: List[Dict[str, Any]] = []
        seen_values = set()
        anchor_value = str(value)
        for row in rows:
            candidate_value = row[0]
            if candidate_value is None:
                continue
            candidate_text = str(candidate_value)
            if not candidate_text or candidate_text == anchor_value:
                continue
            if candidate_text in seen_values:
                continue
            seen_values.add(candidate_text)
            score = _normalized_edit_similarity(anchor_value, candidate_text)
            scored_candidates.append(
                {
                    "value": candidate_value,
                    "source": "substring_lookup",
                    "score": score,
                    "table": table_name,
                    "column": actual_column,
                }
            )

        scored_candidates.sort(
            key=lambda candidate: (
                -float(candidate.get("score") or 0.0),
                abs(len(str(candidate.get("value", ""))) - len(anchor_value)),
                str(candidate.get("value", "")),
            )
        )
        return scored_candidates[:max_candidates]
    except Exception:
        return []


def _append_unique_unmatched_literal(
    unmatched_literals: List[Dict[str, Any]],
    literal_value: str,
    parent_sql: str,
    reason: str,
    *,
    source: str = "sql_parser",
) -> None:
    record = {
        "literal": literal_value,
        "parent_sql": parent_sql,
        "reason": reason,
        "source": source,
    }
    if record not in unmatched_literals:
        unmatched_literals.append(record)


def get_sql_condition_literals(
    db_id: str,
    sql: str,
    filter=False,
    return_details: bool = False,
    include_retrieval_candidates: bool = True,
) -> Any:
    """
    Retrieves literals used in SQL query conditions and checks their existence in the database.

    Args:
        db_id (str): Database ID.
        sql (str): The SQL query string.
        filter (bool): If True, only find textual literals with exact matching or LIKE operations.
        return_details (bool): If True, return mapped literal-existence records and unmatched literals.
        include_retrieval_candidates (bool): If False, retain mapped-literal parsing and exact
            existence checks but skip substring candidate collection used by VALUE retrieval.

    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary of tables and their columns with condition literals.
    """
    # get path from .env
    db_root = os.getenv("DB_ROOT_DIRECTORY", "")
    db_path = os.path.abspath(os.path.expanduser(os.path.join(db_root, db_id, f"{db_id}.sqlite")))

    try:
        # Get actual database tables to distinguish from CTE names
        actual_db_tables = set(table.lower() for table in get_db_all_tables(db_path))

        columns_dict = get_sql_columns_dict(db_path=db_path, sql=sql)
        used_entities: Dict[str, Dict[str, List[str]]] = {}
        literal_existence: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        unmatched_literals: List[Dict[str, Any]] = []
        parsed_sql = fixed_parse_one(sql, read="sqlite")

        for literal in _iter_condition_literal_nodes(parsed_sql):
            if _is_date_format_literal(literal):
                continue

            literal_value = _literal_text(literal)
            parent_context = _condition_parent_for_literal(literal)
            parent_sql = str(literal.parent) if literal.parent is not None else str(literal)
            if parent_context is None:
                _append_unique_unmatched_literal(
                    unmatched_literals,
                    literal_value,
                    parent_sql,
                    "no_column_context",
                )
                continue

            mapped_any = False
            for column_exp in parent_context.find_all(exp.Column):
                column_name = column_exp.name
                for table_name, column_names in columns_dict.items():
                    if column_name.lower() not in [col.lower() for col in column_names]:
                        continue

                    mapped_any = True
                    existence: Optional[bool] = None
                    matched_value: Optional[Any] = None
                    retrieval_candidates: List[Dict[str, Any]] = []
                    if table_name.lower() in actual_db_tables:
                        validation_value = _literal_validation_value(literal_value, parent_context)
                        try:
                            value_check = _check_value_exists(db_path, table_name, column_name, validation_value)
                            if include_retrieval_candidates:
                                retrieval_candidates = _find_substring_value_candidates(
                                    db_path,
                                    table_name,
                                    column_name,
                                    validation_value,
                                )
                        except Exception as exc:
                            logging.debug(
                                "Failed to verify literal value %r in %s.%s: %s",
                                validation_value,
                                table_name,
                                column_name,
                                exc,
                            )
                            value_check = None
                            retrieval_candidates = []
                            existence = None
                        else:
                            existence = value_check is not None
                            matched_value = value_check if value_check is not None else None

                    _append_unique_literal(used_entities, table_name, column_name, literal_value)
                    _append_unique_literal_existence(
                        literal_existence,
                        table_name,
                        column_name,
                        {
                        "literal": literal_value,
                        "retrieval_literal": validation_value,
                        "table": table_name,
                        "column": column_name,
                        "existence": existence,
                            "matched_value": matched_value,
                            "source": "sql_parser",
                            "reason": (
                                "mapped_condition_literal"
                                if existence is not None
                                else "mapped_condition_literal_unverified"
                            ),
                            "retrieval_candidates": retrieval_candidates,
                            "candidate_matched_value": (
                                retrieval_candidates[0]["value"] if retrieval_candidates else None
                            ),
                            "candidate_source": (
                                "substring_lookup" if retrieval_candidates else None
                            ),
                        },
                    )

            if not mapped_any:
                _append_unique_unmatched_literal(
                    unmatched_literals,
                    literal_value,
                    str(parent_context),
                    "no_matching_table_column",
                )

        logging.debug(f"Extracted SQL condition literals: {used_entities}")
        if return_details:
            return used_entities, literal_existence, unmatched_literals
        return used_entities

    except Exception as e:
        logging.critical(f"Error in get_sql_condition_literals: {e}\nSQL {sql}\n")
        raise e

def validate_sql_syntax(sql: str) -> bool:
    """
    Validate if SQL has correct syntax and can be parsed.

    Args:
        sql (str): SQL query to validate

    Returns:
        bool: True if SQL is syntactically valid
    """
    try:
        import sqlglot

        # Try to parse the SQL
        parsed = fixed_parse_one(sql, dialect="sqlite")
        return parsed is not None

    except Exception as e:
        logging.debug(f"SQL validation failed: {e}")
        return False
