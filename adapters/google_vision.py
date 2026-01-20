from adapters.base import ModelAdapter
from core.models import StrategyResult

class GoogleVisionAdapter(ModelAdapter):
    def __init__(self, model_name="gemini-1.5-pro-vision"):
        self.model_name = model_name

    async def run(self, image_bytes, prompt, context=None) -> StrategyResult:
        # TODO: integrate Gemini Vision SDK
        # Mock implementation for now
        return StrategyResult(
            output={
                "objects": ["chart", "table"],
                "text": ["Revenue Q3"],
                "layout": {}
            },
            cost_usd=0.002,
            latency_ms=420,
            confidence=0.85,
            models_used=[self.model_name]
        )
