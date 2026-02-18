import json
from core.persistence.db import execute, _DB_LOCK, get_db


def load_world_model(world_model) -> None:
    """
    Load world model beliefs from SQLite.

    Previously called ensure_schema() which ran DROP TABLE on every invocation,
    destroying all persisted learning. Fixed to use CREATE TABLE IF NOT EXISTS
    (handled by schema.py init on DB creation).
    """
    rows = execute("SELECT signature, data FROM world_model")
    for row in rows:
        sig  = row["signature"] if hasattr(row, "__getitem__") else row[0]
        data = row["data"]      if hasattr(row, "__getitem__") else row[1]
        world_model.beliefs[sig] = json.loads(data)


def save_world_model(world_model) -> None:
    db = get_db()
    with _DB_LOCK:
        for signature, data in world_model.beliefs.items():
            db.execute(
                """
                INSERT INTO world_model (signature, data)
                VALUES (?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    data       = excluded.data,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (signature, json.dumps(data)),
            )
        db.commit()
