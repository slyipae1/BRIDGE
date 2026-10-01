import logging
import random
from typing import Dict, Any, Optional
from .execution import execute_sql
import threading


class DatabaseProfiler:
    """
    Database profiler that maintains a cache of SQL execution results for performance optimization.
    Thread-safe implementation for concurrent access.
    """

    def __init__(self, db_id: str):
        self.db_id = db_id
        self.cache: Dict[str, Any] = {}  # SQL query -> execution result
        self.cache_limit: int = 1000  # Maximum number of cached queries
        self._cache_lock = threading.Lock()
        logging.info(f"DatabaseProfiler initialized for db_id: {db_id}")

    def execute_sql(self, db_path: str, sql: str, store: bool = False, timeout: int = 30) -> Any:
        """
        Execute SQL query with optional caching.

        Args:
            db_path: Path to database
            sql: SQL query to execute
            store: Whether to store result in cache
            timeout: Query timeout in seconds

        Returns:
            Query execution result (original format)
        """
        cache_key = f"{db_path}||{sql.strip()}"

        # Check cache first if storing is enabled
        if store:
            with self._cache_lock:
                if cache_key in self.cache:
                    logging.debug(f"Cache hit for SQL: {sql[:50]}...")
                    # Return the original result format
                    return self._denormalize_result(self.cache[cache_key])

        # Execute query
        try:
            result = execute_sql(db_path, sql, timeout=timeout)

            # Store in cache if requested (store normalized version)
            if store:
                with self._cache_lock:
                    self._manage_cache_size()
                    # Store normalized result for comparison purposes
                    normalized_result = self._normalize_result_for_comparison(result)
                    self.cache[cache_key] = {
                        'normalized': normalized_result,
                        'original': result
                    }
                    logging.debug(f"Cached result for SQL: {sql[:50]}...")

            return result

        except Exception as e:
            # print(f"ERROR: db_profiler SQL execution failed for db_id {self.db_id}: {e} - SQL: \n {sql[:30]}")
            raise e

    def _normalize_result_for_comparison(self, result):
        """
        Normalize result for comparison - convert to set of tuples to ignore order and duplicates.

        Args:
            result: Raw execution result

        Returns:
            Normalized result for comparison
        """
        try:
            if isinstance(result, list):
                if result and isinstance(result[0], (list, tuple)):
                    # Convert each row to tuple and create set to ignore order and duplicates
                    return frozenset(tuple(row) for row in result)
                else:
                    # Handle simple list
                    return frozenset(result)
            else:
                # Handle single values
                return result
        except Exception as e:
            logging.warning(f"Could not normalize result for comparison: {e}")
            return result

    def _denormalize_result(self, cached_data):
        """
        Convert cached data back to original format.

        Args:
            cached_data: Data from cache

        Returns:
            Original result format
        """
        if isinstance(cached_data, dict) and 'original' in cached_data:
            return cached_data['original']
        return cached_data

    def get_normalized_result(self, db_path: str, sql: str, timeout: int = 30):
        """
        Get normalized result for comparison purposes.

        Args:
            db_path: Path to database
            sql: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Normalized result for comparison
        """
        cache_key = f"{db_path}||{sql.strip()}"

        # Check cache first
        with self._cache_lock:
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if isinstance(cached_data, dict) and 'normalized' in cached_data:
                    return cached_data['normalized']

        # Execute and normalize
        result = self.execute_sql(db_path, sql, store=True, timeout=timeout)
        return self._normalize_result_for_comparison(result)

    def _manage_cache_size(self):
        """Remove 1/3 of cache entries randomly when limit is reached."""
        if len(self.cache) >= self.cache_limit:
            keys_to_remove = random.sample(list(self.cache.keys()), len(self.cache) // 3)
            for key in keys_to_remove:
                del self.cache[key]
            logging.info(f"Cache cleanup for db_id {self.db_id}: removed {len(keys_to_remove)} entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "db_id": self.db_id,
                "cache_size": len(self.cache),
                "cache_limit": self.cache_limit,
                "cache_utilization": len(self.cache) / self.cache_limit * 100
            }

    def clear_cache(self):
        """Clear all cached results."""
        with self._cache_lock:
            self.cache.clear()
            logging.info(f"DatabaseProfiler cache cleared for db_id: {self.db_id}")


######################################################################################################################

### Singleton management for each db_id
_DB_PROFILERS: Dict[str, DatabaseProfiler] = {}
_DB_PROFILERS_LOCK = threading.Lock()

def get_db_profiler(db_id: str) -> DatabaseProfiler:
    """
    Get the DatabaseProfiler singleton instance for a specific database ID.
    Creates a new instance if one doesn't exist for the given db_id.

    Args:
        db_id: The database identifier

    Returns:
        DatabaseProfiler: Singleton profiler instance for the db_id
    """
    with _DB_PROFILERS_LOCK:
        if db_id not in _DB_PROFILERS:
            _DB_PROFILERS[db_id] = DatabaseProfiler(db_id)
            logging.info(f"Created new DatabaseProfiler singleton for db_id: {db_id}")
        return _DB_PROFILERS[db_id]

def clear_all_profilers():
    """Clear all profiler instances (useful for testing or cleanup)."""
    with _DB_PROFILERS_LOCK:
        for profiler in _DB_PROFILERS.values():
            profiler.clear_cache()
        _DB_PROFILERS.clear()
        logging.info("Cleared all DatabaseProfiler instances")

def get_all_profiler_stats() -> Dict[str, Dict[str, Any]]:
    """Get cache statistics for all profilers."""
    with _DB_PROFILERS_LOCK:
        return {db_id: profiler.get_cache_stats() for db_id, profiler in _DB_PROFILERS.items()}
