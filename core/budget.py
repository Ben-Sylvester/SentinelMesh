import sqlite3
from datetime import date
from pathlib import Path

DB_PATH = Path("budget.db")


class BudgetManager:
    def __init__(
        self,
        daily_limit_usd: float = 5.00,
        per_request_limit_usd: float = 0.20,
    ):
        self.daily_limit = daily_limit_usd
        self.per_request_limit = per_request_limit_usd
        self._init_db()

    def _connect(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_spend (
                    day TEXT PRIMARY KEY,
                    total REAL NOT NULL
                )
            """)
            conn.commit()

    def _today(self) -> str:
        return date.today().isoformat()

    def get_today_spend(self) -> float:
        today = self._today()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT total FROM daily_spend WHERE day = ?",
                (today,)
            ).fetchone()

        return row[0] if row else 0.0

    def can_spend(self, estimated_cost: float) -> bool:
        if estimated_cost > self.per_request_limit:
            return False

        today_total = self.get_today_spend()
        return (today_total + estimated_cost) <= self.daily_limit

    def record_spend(self, amount: float):
        today = self._today()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO daily_spend(day, total)
                VALUES (?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    total = total + excluded.total
            """, (today, amount))
            conn.commit()
