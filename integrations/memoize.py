import sqlite3
import hashlib
import json
import functools
from typing import Callable, Any, Optional

## Originally from https://www.kevinkatz.io/posts/memoize-to-sqlite

def memoize_to_sqlite(func_name: str, filename: str = "cache.db"):
    """
    Memoization decorator that caches the output of a method in a SQLite
    database.
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any):
            with SQLiteMemoization(filename) as memoizer:
                return memoizer.fetch_or_compute(func, func_name, *args, **kwargs)
        return wrapped
    return decorator

class SQLiteMemoization:
    def __init__(self, filename: str):
        self.filename = filename
        self.connection: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "SQLiteMemoization":
        self.connection = sqlite3.connect(self.filename)
        self._initialize_database()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        if self.connection:
            self.connection.close()
        self.connection = None

    def _initialize_database(self):
        if self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, result TEXT)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS cache_ndx ON cache(hash)"
            )

    def fetch_or_compute(self, func: Callable[..., Any], func_name: str, *args: Any, **kwargs: Any) -> Any:
        arg_hash = self._compute_hash(func_name, *args, **kwargs)

        result = self._fetch_from_cache(arg_hash)
        if result is not None:
            return result

        return self._compute_and_cache_result(func, arg_hash, *args, **kwargs)

    def _compute_hash(self, func_name: str, *args: Any, **kwargs: Any) -> str:
        data = f"{func_name}:{repr(args)}:{repr(kwargs)}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _fetch_from_cache(self, arg_hash: str) -> Optional[Any]:
        if not self.connection:
            return None
        cursor = self.connection.cursor()
        cursor.execute("SELECT result FROM cache WHERE hash = ?", (arg_hash,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def _compute_and_cache_result(self, func: Callable[..., Any], arg_hash: str, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self._cache_result(arg_hash, result)
        return result

    def _cache_result(self, arg_hash: str, result: Any):
        if self.connection:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO cache (hash, result) VALUES (?, ?)",
                (arg_hash, json.dumps(result))
            )
            self.connection.commit()