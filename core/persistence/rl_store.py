from collections import defaultdict, namedtuple
from core.persistence.db import execute
import json


# Canonical serialization helpers — used by both RLStore and Router
# so keys are ALWAYS encoded/decoded the same way.

def encode_key(key: tuple) -> str:
    """Encode a tuple key to a stable JSON string for SQLite storage."""
    return json.dumps(list(key))          # ('abc', 'balanced') → '["abc", "balanced"]'


def decode_key(s: str) -> tuple:
    """Decode a JSON string back to a tuple key."""
    return tuple(json.loads(s))


class RLStore:
    """
    Persists and loads Q-values.

    FIX: Previous version serialized state/action as str(tuple), e.g.
    "('abc123', 'balanced')". RLPolicy.q uses the actual tuple as the dict
    key. str(tuple) != tuple in Python, so every Q-value loaded from DB was
    immediately unreachable — the RL policy had complete amnesia after every
    restart.

    Now uses json.dumps(list(key)) → json.loads(s) → tuple() for a round-
    trippable encoding: ('abc123', 'balanced') ↔ '["abc123", "balanced"]'.
    """

    def load_rl(self) -> defaultdict:
        rows = execute("SELECT state, action, value FROM rl_qvalues")
        q: defaultdict = defaultdict(lambda: defaultdict(float))
        for row in rows:
            state_raw  = row["state"]  if hasattr(row, "__getitem__") else row[0]
            action_raw = row["action"] if hasattr(row, "__getitem__") else row[1]
            value      = row["value"]  if hasattr(row, "__getitem__") else row[2]
            try:
                state  = decode_key(state_raw)
                action = decode_key(action_raw)
            except Exception:
                # Gracefully skip rows written with the old str(tuple) format
                continue
            q[state][action] = float(value)
        return q

    def save_rl(self, state, action, value: float, count: int = 0) -> None:
        execute(
            """
            INSERT INTO rl_qvalues (state, action, value, count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(state, action) DO UPDATE SET
                value      = excluded.value,
                count      = excluded.count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (encode_key(state), encode_key(action), value, count),
            commit=True,
        )
