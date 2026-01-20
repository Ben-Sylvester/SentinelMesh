import time
from adapters.base import ModelAdapter
from core.models import StrategyResult


class MistralAdapter(ModelAdapter):
    """
    Stub adapter for Mistral.
    """

    def __init__(self, model: str):
        self.model = model

    async def run(self, prompt: str, context=None) -> StrategyResult:
        start = time.time()

        text = f"[Mistral:{self.model}] {prompt[:200]}"

        latency = (time.time() - start) * 1000

        return StrategyResult(
            output=text,
            cost_usd=0.001,
            latency_ms=int(latency),
            confidence=0.65,
            models_used=[self.model],
        )
