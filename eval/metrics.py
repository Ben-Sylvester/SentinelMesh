# def score_output(output: str, expected_contains: str) -> float:
#     """
#     Simple proxy quality metric.
#     """
#     output = output.lower()
#     expected_contains = expected_contains.lower()

#     if expected_contains in output:
#         return 1.0
#     return 0.0


# def compute_efficiency(cost: float, latency: float) -> float:
#     """
#     Lower cost + lower latency = higher efficiency.
#     """
#     return 1.0 / (1.0 + cost + (latency / 1000.0))

import numpy as np
from collections import Counter

def aggregate(results):
    if not results:
        return {}

    return {
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_cost": float(np.mean([r["cost"] for r in results])),
        "avg_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
        "success_rate": sum(r["success"] for r in results) / len(results),
        "strategy_distribution": dict(Counter(r["strategy"] for r in results)),
        "total_runs": len(results),
    }
