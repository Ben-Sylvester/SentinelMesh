import json
from .db import get_db

def save_world_model(world_model):
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS world_model (
            signature TEXT PRIMARY KEY,
            data TEXT
        )
    """)
    for sig, data in world_model.beliefs.items():
        db.execute(
            "REPLACE INTO world_model VALUES (?, ?)",
            (sig, json.dumps(data))
        )
    db.commit()


def load_world_model(world_model):
    db = get_db()
    rows = db.execute("SELECT * FROM world_model").fetchall()
    for row in rows:
        world_model.beliefs[row["signature"]] = json.loads(row["data"])
