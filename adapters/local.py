import time
from adapters.base import ModelAdapter
from core.models import ModelOutput


class LocalAdapter(ModelAdapter):
    """
    Local model stub (Ollama / vLLM ready later).
    """

    def __init__(self, model: str):
        self.model = model

    async def run(self, prompt: str, context=None) -> ModelOutput:
        start = time.time()

        text = f"[Local:{self.model}] {prompt[:200]}"

        latency = (time.time() - start) * 1000

        return ModelOutput(
            output=text,
            cost_usd=0.0,  # local is free
            latency_ms=int(latency),
            confidence=0.55,
            models_used=[self.model],
        )
