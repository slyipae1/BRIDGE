"""Format foreign key and primary key information from cached database schema.

Uses DatabaseSchemaGenerator.CACHED_DB_SCHEMA which is populated
during schema generation (via PRAGMA table_info and PRAGMA foreign_key_list).
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _get_cached_schema(db_id: str):
    """Get the cached database schema for a given db_id, if loaded."""
    from .schema_generator import DatabaseSchemaGenerator
    return DatabaseSchemaGenerator.CACHED_DB_SCHEMA.get(db_id)


def format_foreign_keys(table_names: List[str], db_id: str) -> str:
    """Format a markdown section listing foreign key relationships.

    For each table in table_names, finds columns with foreign key references
    and formats them as readable lines.

    Returns empty string if no FK info is available.
    """
    cached = _get_cached_schema(db_id)
    if not cached:
        logger.debug(f"No cached schema for {db_id}, cannot extract FK info")
        return ""

    lines = ["## Foreign Key Relationships"]
    found_any = False

    # Track shown relationships to avoid duplicates
    shown = set()

    for table_name in table_names:
        table_schema = cached.tables.get(table_name)
        if not table_schema:
            continue
        for col_name, col_info in table_schema.columns.items():
            # Foreign keys FROM this column (the FK constraint is defined on this table)
            for dest_table, dest_col in col_info.foreign_keys:
                rel = (table_name, col_name, dest_table, dest_col)
                if rel not in shown:
                    lines.append(f"- **{table_name}.{col_name}** → {dest_table}.{dest_col}")
                    shown.add(rel)
                    found_any = True
            # Referenced BY other tables (FK defined elsewhere, pointing to this column)
            for src_table, src_col in col_info.referenced_by:
                rel = (src_table, src_col, table_name, col_name)
                if rel not in shown:
                    lines.append(f"- **{src_table}.{src_col}** → {table_name}.{col_name}")
                    shown.add(rel)
                    found_any = True

    if not found_any:
        lines.append("(no foreign key relationships found in the selected tables)")

    return "\n".join(lines)


def format_primary_keys(table_names: List[str], db_id: str) -> str:
    """Format a markdown section listing primary keys for each table.

    Returns empty string if no PK info is available.
    """
    cached = _get_cached_schema(db_id)
    if not cached:
        logger.debug(f"No cached schema for {db_id}, cannot extract PK info")
        return ""

    lines = ["## Primary Keys"]
    found_any = False

    for table_name in table_names:
        table_schema = cached.tables.get(table_name)
        if not table_schema:
            continue
        pk_cols = [name for name, info in table_schema.columns.items() if info.primary_key]
        if pk_cols:
            lines.append(f"- **{table_name}**: {', '.join(pk_cols)}")
            found_any = True

    if not found_any:
        lines.append("(no primary key information available)")

    return "\n".join(lines)
