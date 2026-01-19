import sqlite3
import secrets
from pathlib import Path

DB_PATH = Path("tenants.db")


class TenantStore:
    def __init__(self):
        self._init_db()

    def _connect(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    daily_limit REAL NOT NULL,
                    requests_per_minute INTEGER NOT NULL
                )
            """)
            conn.commit()

    # -------------------------
    # Admin
    # -------------------------

    def create_tenant(
        self,
        name: str,
        daily_limit: float = 5.0,
        rpm: int = 60
    ):
        tenant_id = secrets.token_hex(8)
        api_key = f"sk_{secrets.token_hex(16)}"

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO tenants(id, name, api_key, daily_limit, requests_per_minute)
                VALUES (?, ?, ?, ?, ?)
            """, (tenant_id, name, api_key, daily_limit, rpm))
            conn.commit()

        return {
            "tenant_id": tenant_id,
            "api_key": api_key,
            "daily_limit": daily_limit,
            "rpm": rpm
        }

    def get_by_key(self, api_key: str):
        with self._connect() as conn:
            row = conn.execute("""
                SELECT id, name, daily_limit, requests_per_minute
                FROM tenants WHERE api_key = ?
            """, (api_key,)).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "daily_limit": row[2],
            "rpm": row[3],
        }
