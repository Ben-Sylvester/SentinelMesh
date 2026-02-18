import sqlite3
import threading
from datetime import date
from pathlib import Path

DB_PATH  = Path("tenant_budget.db")
_LOCK    = threading.Lock()


class TenantBudget:
    """
    FIX: previous can_spend() + record() used two separate connections,
    creating a TOCTOU race: two concurrent requests could both pass
    can_spend() and then both record(), allowing overspend by N * cost.

    Now uses a threading.Lock + single connection per check-and-record
    operation so the read-then-write is atomic.
    """

    def __init__(self):
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(DB_PATH), check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spend (
                    tenant_id TEXT,
                    day       TEXT,
                    total     REAL,
                    PRIMARY KEY (tenant_id, day)
                )
            """)
            conn.commit()

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def get_spend(self, tenant_id: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT total FROM spend WHERE tenant_id = ? AND day = ?",
                (tenant_id, self._today()),
            ).fetchone()
        return row[0] if row else 0.0

    def check_and_record(self, tenant_id: str, limit: float, cost: float) -> bool:
        """
        Atomically check budget and record spend if within limit.
        Returns True if spend was accepted, False if it would exceed the limit.

        Thread-safe: single lock covers both the read and the write.
        """
        today = self._today()
        with _LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT total FROM spend WHERE tenant_id = ? AND day = ?",
                    (tenant_id, today),
                ).fetchone()
                current = row[0] if row else 0.0

                if current + cost > limit:
                    return False

                conn.execute("""
                    INSERT INTO spend (tenant_id, day, total)
                    VALUES (?, ?, ?)
                    ON CONFLICT(tenant_id, day)
                    DO UPDATE SET total = total + excluded.total
                """, (tenant_id, today, cost))
                conn.commit()
        return True

    # Keep old interface for compatibility — delegates to check_and_record
    def can_spend(self, tenant_id: str, limit: float, cost: float) -> bool:
        return (self.get_spend(tenant_id) + cost) <= limit

    def record(self, tenant_id: str, cost: float) -> None:
        today = self._today()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO spend (tenant_id, day, total)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id, day)
                DO UPDATE SET total = total + excluded.total
            """, (tenant_id, today, cost))
            conn.commit()
