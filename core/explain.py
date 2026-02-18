from typing import Optional, Dict
from core.models import ExplainTrace, StrategyResult


def build_trace(
    features:         dict,
    strategy_name:    str,
    reason:           str,
    result:           StrategyResult,
    selection_source: Optional[str]  = None,
    template_name:    Optional[str]  = None,
    bandit_scores:    Optional[Dict] = None,
    reward:           Optional[float] = None,
) -> ExplainTrace:
    """
    B-08 fix: ExplainTrace now has all fields this function passes.
    FIX: reward is forwarded so eval/runner.py trace["reward"] works.
    """
    disagreement = None
    if isinstance(result.raw_outputs, dict):
        disagreement = result.raw_outputs.get("disagreement")

    return ExplainTrace(
        features=features,
        strategy=strategy_name,
        models_used=result.models_used,
        reason=reason,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        confidence=result.confidence,
        disagreement=disagreement,
        selection_source=selection_source,
        template=template_name,
        bandit_scores=bandit_scores,
        reward=reward,
    )
