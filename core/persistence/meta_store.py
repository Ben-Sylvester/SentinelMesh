import json
from core.persistence.db import execute


class MetaStore:
    """
    Persists MetaPolicy values to SQLite.
    Replaces previous file-based (meta.json) implementation which had
    no atomic write guarantees and would lose data on concurrent access.
    """

    _KEY = "meta_policy"

    def save(self, meta) -> None:
        payload = json.dumps({
            "values":  meta.values,
            "epsilon": meta.epsilon,
        })
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

    def load(self, meta) -> None:
        rows = execute(
            "SELECT data FROM meta_policy WHERE key = ?",
            (self._KEY,),
        )
        if rows:
            data = json.loads(rows[0]["data"] if hasattr(rows[0], "__getitem__") else rows[0][0])
            meta.values  = data.get("values",  meta.values)
            meta.epsilon = data.get("epsilon", meta.epsilon)
