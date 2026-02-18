# Belief normalization, decay, persistence, trend tracking
import hashlib
import json
import numpy as np
import time


def _action_key(strategy: str, retrieval_flag: bool) -> str:
    """
    FIX: tuple keys are not JSON-serializable. Encode as a stable string key
    so beliefs can be persisted to/from SQLite JSON without TypeError.
    Format: "strategy_name|0" or "strategy_name|1"
    """
    return f"{strategy}|{int(retrieval_flag)}"


def _decode_action_key(key: str):
    """Decode string key back to (strategy, retrieval_flag) tuple."""
    parts = key.rsplit("|", 1)
    return parts[0], bool(int(parts[1]))


class WorldModel:
    def __init__(self, decay: float = 0.995):
        self.beliefs: dict = {}
        self.decay = decay
        # load_world_model NOT called here — Router is sole loader (B-14)

    def task_signature(self, features: np.ndarray) -> str:
        rounded = np.round(features, 2)
        return hashlib.md5(rounded.tobytes()).hexdigest()

    def recommend(self, signature: str):
        strategies = self.beliefs.get(signature)
        if not strategies:
            return None, None, 0.0

        best_key   = None
        best_score = -float("inf")
        best_conf  = 0.0

        for key, data in strategies.items():
            effective_count = data["count"]
            if effective_count < 0.5:
                continue

            variance   = data["m2"] / max(effective_count - 1, 1e-6)
            confidence = effective_count / (effective_count + variance + 1e-6)
            score      = data["mean"]

            if score > best_score:
                best_score = score
                best_key   = key
                best_conf  = confidence

        if best_key is None:
            return None, None, 0.0

        strategy, retrieval_flag = _decode_action_key(best_key)
        return strategy, retrieval_flag, best_conf

    def update(
        self,
        signature:     str,
        strategy:      str,
        retrieval_flag: bool,
        reward:        float,
    ) -> None:
        """
        B-03 fix: explicit (strategy, retrieval_flag) args.
        B-28 fix: count is not decayed — only mean/m2 decay.
        FIX: uses string keys so beliefs are JSON-serializable.
        """
        key = _action_key(strategy, retrieval_flag)
        sig = self.beliefs.setdefault(signature, {})

        entry = sig.setdefault(key, {
            "mean":        0.0,
            "m2":          0.0,
            "count":       0,
            "trend":       0.0,
            "last_update": time.time(),
        })

        if retrieval_flag:
            reward -= 0.02

        # Decay running stats (not count)
        entry["mean"] *= self.decay
        entry["m2"]   *= self.decay

        # Welford online update
        entry["count"] += 1
        delta           = reward - entry["mean"]
        entry["mean"]  += delta / entry["count"]
        delta2          = reward - entry["mean"]
        entry["m2"]    += delta * delta2

        # EMA trend
        entry["trend"]       = 0.9 * entry["trend"] + 0.1 * delta
        entry["last_update"] = time.time()

    def normalized_scores(self, signature: str) -> dict:
        strategies = self.beliefs.get(signature)
        if not strategies:
            return {}
        means   = np.array([v["mean"] for v in strategies.values()])
        min_m, max_m = means.min(), means.max()
        norm = {}
        for key, data in strategies.items():
            norm[key] = 0.5 if max_m - min_m < 1e-6 else (data["mean"] - min_m) / (max_m - min_m)
        return norm

    def stats(self) -> dict:
        """
        FIX: beliefs used to have tuple keys — not JSON serializable.
        Now uses string keys throughout; stats() is safe to return from FastAPI endpoints.
        """
        return self.beliefs
