# LinUCB contextual bandit
# Per-arm linear model: A (d×d), b (d×1)
# Score: θ_a^T x + α * sqrt(x^T A_a^{-1} x)
import math
import random
import numpy as np
from typing import Dict
# FIX: __init__ no longer calls load_bandit — Router is the single loader.
# This eliminates the double-load that occurred when Router also called load_bandit.


class ContextualBandit:
    """
    LinUCB contextual bandit. Learns a per-arm linear model over the feature
    space so strategy selection is informed by task characteristics.

    LinUCB formula:
        score_a = θ_a · x  +  α · sqrt(x^T A_a^{-1} x)
        θ_a     = A_a^{-1} b_a  (ridge regression estimate)
        α       controls exploration width
    """

    def __init__(self, feature_dim: int, alpha: float = 1.0):
        self.feature_dim = feature_dim
        self.alpha       = alpha
        self.arms: Dict[str, Dict] = {}
        self.total_steps = 0
        # load_bandit is NOT called here — Router.__init__ calls it once
        # after register_arm() so loaded state overwrites fresh initializations.

    # ----------------------------
    # Arm Management
    # ----------------------------

    def register_arm(self, name: str) -> None:
        """Register an arm with fresh LinUCB state. No-op if already registered."""
        if name not in self.arms:
            d = self.feature_dim
            self.arms[name] = {
                "A":      np.eye(d).tolist(),
                "b":      np.zeros(d).tolist(),
                "pulls":  0,
                "reward": 0.0,
            }

    def _ensure_linucb_keys(self, arm: dict) -> dict:
        """
        Migrate arms loaded from DB that were saved under the old UCB1 schema
        (only had 'pulls' and 'reward'). Adds A/b matrices if absent so the
        arm can participate in LinUCB scoring immediately.
        """
        d = self.feature_dim
        if "A" not in arm:
            arm["A"] = np.eye(d).tolist()
        if "b" not in arm:
            arm["b"] = np.zeros(d).tolist()
        return arm

    # ----------------------------
    # Selection (LinUCB)
    # FIX: cold start now uses random selection across all unexplored arms
    # instead of always returning the first arm, ensuring proper initial exploration.
    # ----------------------------

    def select(self, features: np.ndarray) -> str:
        self.total_steps += 1
        x = features.astype(float)

        unexplored = [name for name, s in self.arms.items() if s["pulls"] == 0]
        if unexplored:
            # Random cold-start: spread initial exploration evenly
            return random.choice(unexplored)

        best_arm   = None
        best_score = float("-inf")

        for name, stats in self.arms.items():
            stats = self._ensure_linucb_keys(stats)
            A     = np.array(stats["A"])
            b     = np.array(stats["b"])

            A_inv        = np.linalg.inv(A)
            theta        = A_inv @ b
            exploitation = theta @ x
            exploration  = self.alpha * math.sqrt(max(0.0, x @ A_inv @ x))
            score        = exploitation + exploration

            if score > best_score:
                best_score = score
                best_arm   = name

        return best_arm

    # ----------------------------
    # In-memory update only (no I/O)
    # FIX: update() previously called save_bandit() directly, causing:
    #   1. Persistence inside the asyncio lock (blocks event loop)
    #   2. Double-save (Router's _persist also saves bandit after the lock)
    # Router._persist_sync() is now the single persistence owner.
    # ----------------------------

    def update_memory(self, name: str, features: np.ndarray, reward: float) -> None:
        """Update LinUCB matrices and running stats in memory only. No I/O."""
        arm = self.arms[name]
        arm = self._ensure_linucb_keys(arm)
        x   = features.astype(float)

        A  = np.array(arm["A"])
        b  = np.array(arm["b"])
        A += np.outer(x, x)
        b += reward * x

        arm["A"]      = A.tolist()
        arm["b"]      = b.tolist()
        arm["pulls"] += 1
        arm["reward"] += (reward - arm["reward"]) / arm["pulls"]

    # Keep old name as alias for any external callers
    def update(self, name: str, features: np.ndarray, reward: float) -> None:
        """Alias for update_memory — persistence is handled by Router._persist_sync."""
        self.update_memory(name, features, reward)

    # ----------------------------
    # Observability
    # ----------------------------

    def stats(self) -> Dict:
        return {
            name: {"pulls": s["pulls"], "avg_reward": round(s["reward"], 4)}
            for name, s in self.arms.items()
        }

    @property
    def last_scores(self) -> Dict[str, float]:
        return {name: arm["reward"] for name, arm in self.arms.items()}
