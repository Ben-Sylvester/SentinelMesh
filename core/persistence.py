# This will:
# Open SQLite DB
# Auto-create tables
# Save / load state

# Be dependency-free
# Each learning component will:
# Load state on startup
# Persist on update


import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/learning.db")
DB_PATH.parent.mkdir(exist_ok=True)

class Persistence:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS kv_store (
            namespace TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (namespace, key)
        )
        """)

        self.conn.commit()

    # -------------------------
    # Generic KV Interface
    # -------------------------

    def save(self, namespace: str, key: str, value: dict):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO kv_store(namespace, key, value)
        VALUES (?, ?, ?)
        """, (namespace, key, json.dumps(value)))
        self.conn.commit()

    def load(self, namespace: str, key: str):
        cur = self.conn.cursor()
        cur.execute("""
        SELECT value FROM kv_store
        WHERE namespace=? AND key=?
        """, (namespace, key))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def load_namespace(self, namespace: str):
        cur = self.conn.cursor()
        cur.execute("""
        SELECT key, value FROM kv_store WHERE namespace=?
        """, (namespace,))
        rows = cur.fetchall()
        return {k: json.loads(v) for k, v in rows}


# Singleton
persistence = Persistence()
