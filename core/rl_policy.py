import random
from collections import defaultdict, deque
import numpy as np

class RLPolicy:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.q = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.reward_history = defaultdict(lambda: deque(maxlen=100))

    def normalize_reward(self, state, reward):
        hist = self.reward_history[state]
        hist.append(reward)

        if len(hist) < 5:
            return reward

        mean = np.mean(hist)
        std = np.std(hist) + 1e-6
        return float((reward - mean) / std)

    def select(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)

        qs = np.array([self.q[state][a] for a in actions])
        temp = max(0.3, self.epsilon)
        probs = np.exp(qs / temp) / np.sum(np.exp(qs / temp))
        return np.random.choice(actions, p=probs)

    def stats(self):
        total_states = len(self.q)
        total_actions = sum(len(v) for v in self.q.values())

        all_qs = [q for state in self.q.values() for q in state.values()]
        avg_q = sum(all_qs) / len(all_qs) if all_qs else 0.0

        return {
            "states": total_states,
            "actions": total_actions,
            "epsilon": round(self.epsilon, 4),
            "avg_q": round(avg_q, 4),
        }


    def update(self, state, action, reward, next_state):
        reward = self.normalize_reward(state, reward)

        best_next = max(self.q[next_state].values(), default=0.0)
        self.q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q[state][action]
        )

        self.epsilon = max(0.05, self.epsilon * 0.995)
