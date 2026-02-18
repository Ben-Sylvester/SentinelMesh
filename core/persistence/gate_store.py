import json
import numpy as np
from core.persistence.db import execute


class GateStore:
    """
    Persists RetrievalGate weights to SQLite.
    Replaces previous file-based (gate.json) implementation.
    """
    _KEY = "retrieval_gate"

    def save(self, gate) -> None:
        payload = json.dumps({"weights": gate.w.tolist()})
        execute(
            """
            INSERT INTO meta_policy (key, data)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET data = excluded.data,
                                           updated_at = CURRENT_TIMESTAMP
            """,
            (self._KEY, payload),
            commit=True,
        )

    def load(self, gate) -> None:
        rows = execute(
            "SELECT data FROM meta_policy WHERE key = ?",
            (self._KEY,),
        )
        if rows:
            raw  = rows[0]["data"] if hasattr(rows[0], "__getitem__") else rows[0][0]
            data = json.loads(raw)
            gate.w = np.array(data["weights"])
