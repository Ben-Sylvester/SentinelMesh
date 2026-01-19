import time
from adapters.base import ModelAdapter
from core.models import ModelOutput


class GoogleAdapter(ModelAdapter):
    """
    Google Gemini adapter (stub version).
    Plug real SDK later.
    """

    def __init__(self, model: str):
        self.model = model

    async def run(self, prompt: str, context=None) -> ModelOutput:
        start = time.time()

        text = f"[Gemini:{self.model}] {prompt[:200]}"

        latency = (time.time() - start) * 1000

        return ModelOutput(
            output=text,
            cost_usd=0.0015,
            latency_ms=int(latency),
            confidence=0.68,
            models_used=[self.model],
        )
