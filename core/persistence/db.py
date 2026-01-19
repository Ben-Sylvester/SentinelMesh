import sqlite3
from pathlib import Path

DB_PATH = Path("data/learning_state.db")
DB_PATH.parent.mkdir(exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
