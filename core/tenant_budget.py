import sqlite3
from datetime import date
from pathlib import Path

DB_PATH = Path("tenant_budget.db")


class TenantBudget:
    def __init__(self):
        self._init_db()

    def _connect(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spend (
                    tenant_id TEXT,
                    day TEXT,
                    total REAL,
                    PRIMARY KEY (tenant_id, day)
                )
            """)
            conn.commit()

    def _today(self):
        return date.today().isoformat()

    def get_spend(self, tenant_id: str) -> float:
        today = self._today()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT total FROM spend
                WHERE tenant_id = ? AND day = ?
            """, (tenant_id, today)).fetchone()

        return row[0] if row else 0.0

    def can_spend(self, tenant_id: str, limit: float, cost: float) -> bool:
        return (self.get_spend(tenant_id) + cost) <= limit

    def record(self, tenant_id: str, cost: float):
        today = self._today()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO spend(tenant_id, day, total)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id, day)
                DO UPDATE SET total = total + excluded.total
            """, (tenant_id, today, cost))
            conn.commit()
