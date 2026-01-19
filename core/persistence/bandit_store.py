import json
from .db import get_db

def save_bandit(bandit):
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS bandit_state (
            arm TEXT PRIMARY KEY,
            data TEXT
        )
    """)
    for arm, state in bandit.arms.items():
        db.execute(
            "REPLACE INTO bandit_state VALUES (?, ?)",
            (arm, json.dumps(state))
        )
    db.commit()


def load_bandit(bandit):
    db = get_db()
    rows = db.execute("SELECT * FROM bandit_state").fetchall()
    for row in rows:
        if row["arm"] in bandit.arms:
            bandit.arms[row["arm"]] = json.loads(row["data"])
