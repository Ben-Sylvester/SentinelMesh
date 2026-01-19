from core.models import ExplainTrace, StrategyResult


def build_trace(features: dict, strategy_name: str, reason: str, result: StrategyResult):
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
        disagreement=disagreement
    )
