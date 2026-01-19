import json
import asyncio
from pathlib import Path
from datetime import datetime

TRACE_FILE = Path("traces.json")


class TraceStore:
    def __init__(self, ws_manager=None):
        self.traces = []
        self.ws_manager = ws_manager

        # Load existing traces into memory (optional but useful)
        if TRACE_FILE.exists():
            try:
                self.traces = json.loads(TRACE_FILE.read_text())
            except Exception:
                self.traces = []

    def log(self, trace: dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **trace
        }

        # In-memory
        self.traces.append(record)

        # Persist to disk
        if TRACE_FILE.exists():
            data = json.loads(TRACE_FILE.read_text())
        else:
            data = []

        data.append(record)
        TRACE_FILE.write_text(json.dumps(data, indent=2))

        # 🔴 Realtime push
        if self.ws_manager:
            asyncio.create_task(
                self.ws_manager.broadcast({
                    "type": "trace",
                    "payload": record
                })
            )

    def load(self):
        if not TRACE_FILE.exists():
            return []
        return json.loads(TRACE_FILE.read_text())

    def strategy_timeseries(self):
        data = {}
        for trace in self.traces:
            strategy = trace.get("strategy")
            if not strategy:
                continue

            data.setdefault(strategy, []).append({
                "reward": trace.get("reward", 0),
                "timestamp": trace.get("timestamp")
            })
        return data

    def model_roi(self):
        stats = {}

        for trace in self.traces:
            models = trace.get("models_used", [])
            cost = trace.get("cost_usd", 0.0001)
            reward = trace.get("reward", 0)

            for model in models:
                s = stats.setdefault(model, {"reward": 0.0, "cost": 0.0, "count": 0})
                s["reward"] += reward
                s["cost"] += cost
                s["count"] += 1

        roi = {}
        for model, s in stats.items():
            roi[model] = {
                "avg_reward": s["reward"] / max(s["count"], 1),
                "avg_cost": s["cost"] / max(s["count"], 1),
                "roi": s["reward"] / max(s["cost"], 0.0001),
            }

        return roi
