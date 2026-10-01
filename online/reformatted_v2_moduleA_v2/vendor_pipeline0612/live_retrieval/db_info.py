import logging
import os
from typing import List, Dict

from .execution import execute_sql

def get_db_all_tables(db_path: str) -> List[str]:
    """
    Retrieves all table names from the database.

    Args:
        db_path (str): The path to the database file.

    Returns:
        List[str]: A list of table names.
    """
    try:
        raw_table_names = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0].replace('\"', '').replace('`', '') for table in raw_table_names if table[0] != "sqlite_sequence"]
    except Exception as e:
        logging.error(f"Error in get_db_all_tables: {e}")
        raise e

def get_table_all_columns(db_path: str, table_name: str) -> List[str]:
    """
    Retrieves all column names for a given table.

    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table.

    Returns:
        List[str]: A list of column names.
    """
    try:
        table_info_rows = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`);")
        return [row[1].replace('\"', '').replace('`', '') for row in table_info_rows]
    except Exception as e:
        logging.error(f"Error in get_table_all_columns: {e}\nTable: {table_name}")
        raise e

def get_db_schema(db_path: str) -> Dict[str, List[str]]:
    """
    Retrieves the schema of the database.

    Args:
        db_path (str): The path to the database file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of column names.
    """
    try:
        table_names = get_db_all_tables(db_path)
        return {table_name: get_table_all_columns(db_path, table_name) for table_name in table_names}
    except Exception as e:
        logging.error(f"Error in get_db_schema: {e}")
        raise e


def get_db_columnGroups (db_path: str):
    import pickle
    db_id = db_path.split("/")[-1].split(".")[0]
    db_directory_path = "/".join(db_path.split("/")[:-1])
    try:
        # groups_path = db_directory_path / "preprocessed" / f"{db_id}_column_groups.pkl"
        groups_path = os.path.join(db_directory_path, "preprocessed", f"{db_id}_column_groups.pkl")

        with open(groups_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"Failed to load column groups for {db_id}: {e}")
        return {}

    """
    # Get within-table and cross-table groups
    within_table_groups = column_groups.get("within_table_groups", {})
    cross_table_groups = column_groups.get("cross_table_groups", [])
    similar_name_groups = column_groups.get("similar_name_groups", [])
    column_descriptive_names = column_groups.get("column_descriptive_names", {})
    column_key = f"{table_name}.{column_name}"

    for group in any_of_above:
            group_columns_list = group.get("columns", [])
            if column_key in group_columns_list:
            # exist in group
    """
