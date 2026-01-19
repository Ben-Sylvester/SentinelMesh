import random
import math
from core.bandit_store import BanditStore
from typing import Dict


class BanditArm:
    def __init__(self, count: int = 0, total_reward: float = 0.0):
        self.count = count
        self.total_reward = total_reward

    @property
    def average_reward(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_reward / self.count


class EpsilonGreedyBandit:
    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon
        self.arms: Dict[str, BanditArm] = {}

    def register_arm(self, name: str, count: int = 0, total_reward: float = 0.0):
        if name not in self.arms:
            self.arms[name] = BanditArm(count, total_reward)

    def select_arm(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(self.arms.keys()))

        return max(
            self.arms.items(),
            key=lambda x: x[1].average_reward
        )[0]

    def update(self, name: str, reward: float):
        arm = self.arms[name]
        arm.count += 1
        arm.total_reward += reward



class UCBPolicy:
    def __init__(self, strategies):
        self.strategies = strategies
        self.store = BanditStore()
        self.arms = self.store.load_arms(strategies)

    def select_arm(self):
        total_pulls = sum(a["pulls"] for a in self.arms.values()) + 1

        scores = {}

        for name, data in self.arms.items():
            pulls = data["pulls"]

            # Cold start → force exploration
            if pulls == 0:
                return name

            avg_reward = data["reward"] / pulls
            bonus = math.sqrt(2 * math.log(total_pulls) / pulls)

            scores[name] = avg_reward + bonus

        # Pick max score
        return max(scores, key=scores.get)

    def update(self, name, reward):
        arm = self.arms[name]
        arm["pulls"] += 1
        arm["reward"] += reward

        self.store.save_arm(
            name=name,
            pulls=arm["pulls"],
            reward=arm["reward"]
        )
