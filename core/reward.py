# Stable bounds
# Cost & latency never overpower quality
# RL stays numerically sane


from core.models import StrategyResult
# -----------------------------
# Canonical Reward Computation
# -----------------------------

def compute_reward(result: StrategyResult) -> float:
    """
    Normalized reward ∈ [-1, 1]

    Reward balances:
    - quality (confidence)
    - cost
    - latency

    Higher is better.
    """

    # Defensive defaults
    confidence = float(result.confidence or 0.0)
    cost = float(result.cost_usd or 0.0)
    latency = float(result.latency_ms or 0.0)

    # Normalization caps (tunable, conservative)
    MAX_COST = 0.20        # USD
    MAX_LATENCY = 3000.0   # ms

    cost_penalty = min(cost / MAX_COST, 1.0)
    latency_penalty = min(latency / MAX_LATENCY, 1.0)

    # Weighted reward (quality dominant)
    reward = (
        1.2 * confidence
        - 0.4 * cost_penalty
        - 0.2 * latency_penalty
    )

    return normalize_reward(reward)


# -----------------------------
# Reward Normalization
# -----------------------------

def normalize_reward(r: float) -> float:
    """
    Clamp reward to stable range.
    """
    return round(max(-1.0, min(1.0, r)), 4)
