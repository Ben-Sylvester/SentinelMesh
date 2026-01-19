# Belief normalization
# Decay over time
# Persistence
# Trend tracking


from core.persistence.world_model_store import WorldModelStore


class WorldModel:
    def __init__(self):
        self.store = WorldModelStore()
        self.beliefs = self.store.load()

    def task_signature(self, features):
        return str(features)

    def recommend(self, signature):
        strategies = self.beliefs.get(signature)
        if not strategies:
            return None
        return max(strategies.items(), key=lambda x: x[1]["avg_reward"])[0]

    def update(self, signature, strategy, reward):
        sig = self.beliefs.setdefault(signature, {})
        entry = sig.setdefault(strategy, {"avg_reward": 0.0, "count": 0})

        entry["count"] += 1
        entry["avg_reward"] += (reward - entry["avg_reward"]) / entry["count"]

        self.store.save(signature, strategy, entry["avg_reward"], entry["count"])

    def stats(self):
        return self.beliefs
