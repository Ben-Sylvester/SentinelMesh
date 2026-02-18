# B-24 fix: complete pricing table for all models registered in app.py.
# Previously only gpt-4.1-mini had real prices; all others fell back to
# the generic 0.000002/token rate, making cost tracking unreliable.

# USD per 1K tokens (input + output averaged for blended estimate)
# Sources: official API pricing pages as of Q1 2026
MODEL_PRICING = {
    # OpenAI
    "openai:gpt-4.1-mini": {
        "input":  0.00015,
        "output": 0.00060,
    },
    "openai:gpt-4.1": {
        "input":  0.00200,
        "output": 0.00800,
    },
    "openai:gpt-3.5-turbo": {
        "input":  0.00050,
        "output": 0.00150,
    },
    "openai-vision:gpt-4.1-mini": {
        "input":  0.00020,
        "output": 0.00080,
    },

    # Anthropic
    "anthropic:claude-3-haiku": {
        "input":  0.00025,
        "output": 0.00125,
    },
    "anthropic:claude-3-sonnet": {
        "input":  0.00300,
        "output": 0.01500,
    },
    "anthropic:claude-3-opus": {
        "input":  0.01500,
        "output": 0.07500,
    },

    # Mistral
    "mistral:mistral-small": {
        "input":  0.00020,
        "output": 0.00060,
    },
    "mistral:mistral-medium": {
        "input":  0.00270,
        "output": 0.00810,
    },
    "mistral:mistral-large": {
        "input":  0.00800,
        "output": 0.02400,
    },

    # Google
    "google:gemini-1.5-flash": {
        "input":  0.00007,
        "output": 0.00021,
    },
    "google:gemini-1.5-pro": {
        "input":  0.00350,
        "output": 0.01050,
    },
    "google-vision:gemini-1.5-pro": {
        "input":  0.00350,
        "output": 0.01050,
    },

    # Local / free models
    "local:llama3": {
        "input":  0.0,
        "output": 0.0,
    },
    "mock-local": {
        "input":  0.0,
        "output": 0.0,
    },
}


def estimate_cost(model_name: str, tokens: int) -> float:
    """
    Blended (input+output average) cost estimate.
    Falls back to a conservative rate for unknown models and logs a warning.
    """
    pricing = MODEL_PRICING.get(model_name)

    if not pricing:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown model '%s' — using fallback pricing. Add to MODEL_PRICING.",
            model_name,
        )
        return round(tokens * 0.000002, 6)

    avg_price = (pricing["input"] + pricing["output"]) / 2
    return round((tokens / 1000) * avg_price, 6)
