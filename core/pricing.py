# 1) Pricing registry (per model)
# 2) Budget manager (daily + per-request)
# 3) Enforcement inside execution flow
# If a request exceeds limits:
# Either downgrade strategy
# Or reject safely with explanation


# USD per 1K tokens (example pricing – adjust anytime)

MODEL_PRICING = {
    "openai:gpt-4.1-mini": {
        "input": 0.00015,
        "output": 0.00060,
    },
    "openai-vision:gpt-4.1-mini": {
        "input": 0.00020,
        "output": 0.00080,
    },
    "mock-local": {
        "input": 0.0,
        "output": 0.0,
    }
}


def estimate_cost(model_name: str, tokens: int) -> float:
    """
    Simple blended estimate.
    """
    pricing = MODEL_PRICING.get(model_name)

    if not pricing:
        # Unknown model → conservative fallback
        return round(tokens * 0.000002, 6)

    avg_price = (pricing["input"] + pricing["output"]) / 2
    return round((tokens / 1000) * avg_price, 6)
