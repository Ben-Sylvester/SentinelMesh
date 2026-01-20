# Belief normalization
# Decay over time
# Persistence
# Trend tracking

import hashlib
import numpy as np
import hashlib
import numpy as np

from core.persistence.world_model_store import (
    save_world_model,
    load_world_model,
)


class WorldModel:
    def __init__(self):
        self.beliefs = {}
        load_world_model(self)   

    def task_signature(self, features):
        rounded = np.round(features, 2)
        return hashlib.md5(rounded.tobytes()).hexdigest()

    def recommend(self, signature):
        strategies = self.beliefs.get(signature)
        if not strategies:
            return None
        return max(
            strategies.items(),
            key=lambda x: x[1]["avg_reward"]
        )[0]

    def update(self, signature, strategy, reward):
        sig = self.beliefs.setdefault(signature, {})
        entry = sig.setdefault(
            strategy,
            {"avg_reward": 0.0, "count": 0}
        )

        entry["count"] += 1
        entry["avg_reward"] += (reward - entry["avg_reward"]) / entry["count"]

        # persist immediately (safe, cheap)
        save_world_model(self)

    def stats(self):
        return {
            sig: {
                strat: dict(stats)
                for strat, stats in strategies.items()
            }
            for sig, strategies in self.beliefs.items()
        }
