# Moving average reward
# Confidence weighting
# Natural decay
# Adaptive exploitation
import math
from typing import Dict
from core.persistence.bandit_store import save_bandit, load_bandit


class ContextualBandit:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.arms: Dict[str, Dict] = {}
        self.total_steps = 0

        # Load persisted state
        load_bandit(self)

    # ----------------------------
    # Arm Management
    # ----------------------------

    def register_arm(self, name: str):
        if name not in self.arms:
            self.arms[name] = {
                "pulls": 0,
                "reward": 0.0,
            }

    # ----------------------------
    # Selection (UCB)
    # ----------------------------

    def select(self, features):
        self.total_steps += 1

        total_pulls = sum(a["pulls"] for a in self.arms.values()) + 1
        best_arm = None
        best_score = float("-inf")

        for name, stats in self.arms.items():
            pulls = max(stats["pulls"], 1)
            avg_reward = stats["reward"]

            exploration = math.sqrt(2 * math.log(total_pulls) / pulls)
            score = avg_reward + exploration

            if score > best_score:
                best_score = score
                best_arm = name

        return best_arm

    # ----------------------------
    # Learning Update
    # ----------------------------

    def update(self, name: str, features, reward: float):
        arm = self.arms[name]
        arm["pulls"] += 1
        arm["reward"] += (reward - arm["reward"]) / arm["pulls"]

        # Persist after update
        save_bandit(self)

    # ----------------------------
    # Observability
    # ----------------------------

    def stats(self):
        return self.arms

    @property
    def last_scores(self):
        return {
            name: arm["reward"]
            for name, arm in self.arms.items()
        }
