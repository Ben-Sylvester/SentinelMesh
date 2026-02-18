import sqlite3
import threading
from pathlib import Path
from core.persistence.schema import init_schema

DB_PATH = Path("data/learning_state.db")
DB_PATH.parent.mkdir(exist_ok=True)

# B-17: Thread-safe shared connection.
# - check_same_thread=False: allow the connection from multiple async tasks.
# - WAL journal mode: allows concurrent readers + one writer without blocking.
# - A threading.Lock serialises all writes explicitly.
_DB: sqlite3.Connection | None = None
_DB_LOCK = threading.Lock()


def get_db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        with _DB_LOCK:
            if _DB is None:   # double-checked locking
                conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-8000")   # 8 MB page cache
                init_schema(conn)
                _DB = conn
    return _DB


def execute(query: str, params: tuple = (), commit: bool = False):
    """Thread-safe execute — acquires write lock for mutations."""
    conn = get_db()
    if commit:
        with _DB_LOCK:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur.fetchall()
    else:
        # Reads don't need the write lock (WAL allows concurrent reads)
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
