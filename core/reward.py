# Stable bounds
# Cost & latency never overpower quality
# RL stays numerically sane

from typing import Optional, Tuple
from core.models import StrategyResult


# -----------------------------
# Canonical Reward Computation
# -----------------------------

def compute_reward(
    result: StrategyResult,
    latency: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Returns a 3-tuple: (quality_score, latency_penalty, cost_penalty)
    Each component is normalized to [0, 1] so the caller controls weights.

    B-02 fix: accepts (result, latency) and returns a 3-tuple compatible
    with RewardComponents(*raw) unpacking in Router._update().
    """
    # Defensive defaults
    confidence = float(result.confidence or 0.0)
    cost = float(result.cost_usd or 0.0)
    # Prefer explicit latency arg; fall back to result field
    latency_ms = float(latency if latency is not None else (result.latency_ms or 0.0))

    # Normalization caps (tunable, conservative)
    MAX_COST    = 0.20       # USD
    MAX_LATENCY = 3000.0     # ms

    quality_score   = max(0.0, min(1.0, confidence))
    cost_penalty    = min(cost / MAX_COST, 1.0)
    latency_penalty = min(latency_ms / MAX_LATENCY, 1.0)

    return (
        round(quality_score,   4),
        round(latency_penalty, 4),
        round(cost_penalty,    4),
    )


# -----------------------------
# Scalar Reward (convenience)
# -----------------------------

def scalar_reward(
    result: StrategyResult,
    latency: Optional[float] = None,
    w_quality: float = 0.6,
    w_latency: float = 0.2,
    w_cost: float = 0.2,
) -> float:
    """
    Weighted scalar reward ∈ [-1, 1]. Weights are explicit, not buried.
    """
    quality, lat_pen, cost_pen = compute_reward(result, latency)
    r = w_quality * quality - w_latency * lat_pen - w_cost * cost_pen
    return round(max(-1.0, min(1.0, r)), 4)
