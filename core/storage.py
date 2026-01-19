import sqlite3
from pathlib import Path

DB_PATH = Path("orchestrator.db")


class BanditStorage:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    strategy_name TEXT PRIMARY KEY,
                    count INTEGER NOT NULL,
                    total_reward REAL NOT NULL
                )
            """)
            conn.commit()

    def load(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT strategy_name, count, total_reward FROM bandit_arms"
            ).fetchall()

        return {
            name: {"count": count, "total_reward": total_reward}
            for name, count, total_reward in rows
        }

    def save_arm(self, strategy_name: str, count: int, total_reward: float):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO bandit_arms(strategy_name, count, total_reward)
                VALUES (?, ?, ?)
                ON CONFLICT(strategy_name) DO UPDATE SET
                    count = excluded.count,
                    total_reward = excluded.total_reward
            """, (strategy_name, count, total_reward))
            conn.commit()
