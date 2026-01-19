import os
import time
from adapters.base import ModelAdapter
from core.models import ModelOutput


class AnthropicAdapter(ModelAdapter):
    """
    Stubbed adapter (safe if no API key yet).
    Replace with real Anthropic SDK later.
    """

    def __init__(self, model: str):
        self.model = model

    async def run(self, prompt: str, context=None) -> ModelOutput:
        start = time.time()

        # ⚠️ Placeholder behavior
        text = f"[Claude:{self.model}] {prompt[:200]}"

        latency = (time.time() - start) * 1000

        return ModelOutput(
            output=text,
            cost_usd=0.002,
            latency_ms=int(latency),
            confidence=0.70,
            models_used=[self.model],
        )
