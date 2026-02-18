import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

from core.persistence.db import execute, _DB_LOCK, get_db

logger = logging.getLogger(__name__)


class TraceStore:
    """
    B-18 fix: replaced file-based storage (traces.json) with SQLite.

    Previous implementation:
    - Read entire JSON file → append → write back on every request: O(n) disk I/O
    - No atomic write; concurrent requests caused lost writes
    - File corruption possible under load

    New implementation:
    - Append-only INSERT into traces table (O(1) per write)
    - SQLite WAL mode handles concurrent writers safely (see db.py)
    - last() query uses SQL LIMIT for O(log n) access
    """

    def __init__(self, ws_manager=None):
        self.ws_manager = ws_manager
        # Schema is initialized by db.py/schema.py on first connection

    def log(self, trace) -> None:
        """Persist a trace record. Accepts dict or Pydantic model."""
        if hasattr(trace, "dict"):
            trace = trace.dict()

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **trace,
        }

        execute(
            "INSERT INTO traces (timestamp, strategy, data) VALUES (?, ?, ?)",
            (
                record["timestamp"],
                record.get("strategy"),
                json.dumps(record),
            ),
            commit=True,
        )

        if self.ws_manager:
            asyncio.create_task(
                self.ws_manager.broadcast({"type": "trace", "payload": record})
            )

    def last(self, limit: int = 100) -> list:
        rows = execute(
            "SELECT data FROM traces ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [json.loads(r["data"] if hasattr(r, "__getitem__") else r[0]) for r in rows]

    def strategy_timeseries(self) -> dict:
        rows = execute(
            "SELECT strategy, data FROM traces ORDER BY id DESC LIMIT 1000"
        )
        data: dict = {}
        for row in rows:
            strategy = row["strategy"] if hasattr(row, "__getitem__") else row[0]
            raw      = row["data"]     if hasattr(row, "__getitem__") else row[1]
            if not strategy:
                continue
            record = json.loads(raw)
            data.setdefault(strategy, []).append({
                "reward":    record.get("reward", 0),
                "timestamp": record.get("timestamp"),
            })
        return data

    def model_roi(self) -> dict:
        rows = execute(
            "SELECT data FROM traces ORDER BY id DESC LIMIT 5000"
        )
        stats: dict = {}
        for row in rows:
            raw    = row["data"] if hasattr(row, "__getitem__") else row[0]
            record = json.loads(raw)
            models = record.get("models_used", [])
            cost   = record.get("cost_usd",   0.0001)
            reward = record.get("reward",      0)
            for model in models:
                s = stats.setdefault(model, {"reward": 0.0, "cost": 0.0, "count": 0})
                s["reward"] += reward
                s["cost"]   += cost
                s["count"]  += 1

        return {
            model: {
                "avg_reward": s["reward"] / max(s["count"], 1),
                "avg_cost":   s["cost"]   / max(s["count"], 1),
                "roi":        s["reward"] / max(s["cost"], 0.0001),
            }
            for model, s in stats.items()
        }
