# Moving average reward
# Confidence weighting
# Natural decay
# Adaptive exploitation
from core.persistence.bandit_store import BanditStore


class ContextualBandit:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.arms = {}
        self.store = BanditStore()
        self._load()
        self.total_steps += 1
        exploration_weight = max(0.1, 1.0 / (1 + self.total_steps * 0.01))


    def _load(self):
        data = self.store.load()
        for arm, stats in data.items():
            self.arms[arm] = stats

    def register_arm(self, name: str):
        if name not in self.arms:
            self.arms[name] = {"pulls": 0, "reward": 0.0}
            self.store.save(name, 0, 0.0)

    def select(self, features):
        # Simple UCB selection
        import math

        total = sum(a["pulls"] for a in self.arms.values()) + 1
        best_arm = None
        best_score = -1e9

        for name, stats in self.arms.items():
            pulls = max(stats["pulls"], 1)
            avg = stats["reward"]
            bonus = math.sqrt(2 * math.log(total) / pulls)
            score = avg + bonus

            if score > best_score:
                best_score = score
                best_arm = name

        return best_arm

    def update(self, name: str, reward: float):
        arm = self.arms[name]
        arm["pulls"] += 1
        arm["reward"] += (reward - arm["reward"]) / arm["pulls"]

        self.store.save(name, arm["pulls"], arm["reward"])

    def stats(self):
        return self.arms
