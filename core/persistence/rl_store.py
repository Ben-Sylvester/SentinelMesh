from typing import Dict
from core.persistence.db import execute


class RLStore:
    def __init__(self):
        execute("""
        CREATE TABLE IF NOT EXISTS rl_qvalues (
            state TEXT,
            action TEXT,
            value REAL,
            count INTEGER,
            PRIMARY KEY (state, action)
        )
        """, commit=True)

    def load(self) -> Dict:
        rows = execute("SELECT state, action, value, count FROM rl_qvalues")
        q = {}
        for s, a, v, c in rows:
            q.setdefault(s, {})[a] = {"value": v, "count": c}
        return q

    def save(self, state: str, action: str, value: float, count: int):
        execute("""
        INSERT INTO rl_qvalues(state, action, value, count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(state, action)
        DO UPDATE SET value=?, count=?
        """, (state, action, value, count, value, count), commit=True)
