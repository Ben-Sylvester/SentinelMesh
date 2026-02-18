import json
from core.persistence.db import execute


def save_bandit(bandit) -> None:
    """
    B-09 fix: uses arm/data columns (aligned with schema.py).
    Previous version created a conflicting arm/data schema on the same
    DB that schema.py had created as id/state — now unified.
    """
    for arm, state in bandit.arms.items():
        execute(
            """
            INSERT INTO bandit_state (arm, data)
            VALUES (?, ?)
            ON CONFLICT(arm) DO UPDATE SET data = excluded.data,
                                           updated_at = CURRENT_TIMESTAMP
            """,
            (arm, json.dumps(state)),
            commit=True,
        )


def load_bandit(bandit) -> None:
    rows = execute("SELECT arm, data FROM bandit_state")
    for row in rows:
        arm_name = row["arm"] if hasattr(row, "__getitem__") else row[0]
        arm_data_raw = row["data"] if hasattr(row, "__getitem__") else row[1]
        if arm_name in bandit.arms:
            bandit.arms[arm_name] = json.loads(arm_data_raw)
