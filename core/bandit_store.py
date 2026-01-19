import sqlite3
from pathlib import Path

DB_PATH = Path("bandit.db")


class BanditStore:
    def __init__(self):
        self._init_db()

    def _connect(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS arms (
                    name TEXT PRIMARY KEY,
                    pulls INTEGER NOT NULL,
                    reward REAL NOT NULL
                )
            """)
            conn.commit()

    def load_arms(self, strategy_names):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, pulls, reward FROM arms"
            ).fetchall()

        data = {r[0]: {"pulls": r[1], "reward": r[2]} for r in rows}

        # Initialize missing arms
        for name in strategy_names:
            if name not in data:
                self.save_arm(name, pulls=0, reward=0.0)
                data[name] = {"pulls": 0, "reward": 0.0}

        return data

    def save_arm(self, name, pulls, reward):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO arms(name, pulls, reward)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    pulls = excluded.pulls,
                    reward = excluded.reward
            """, (name, pulls, reward))
            conn.commit()
