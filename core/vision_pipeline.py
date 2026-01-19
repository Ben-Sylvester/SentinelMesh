import json
import time
from core.models import StrategyResult
from core.pricing import estimate_cost
from adapters.vision_openai import OpenAIVisionAdapter
from adapters.base import ModelAdapter


class VisionReasoningPipeline:
    def __init__(
        self,
        vision_model: OpenAIVisionAdapter,
        reasoning_model: ModelAdapter
    ):
        self.vision_model = vision_model
        self.reasoning_model = reasoning_model
        self.name = "vision → reasoning"

    async def execute(self, prompt: str, image_bytes: bytes) -> StrategyResult:
        start = time.time()

        # ---- Vision step ----
        vision_result = await self.vision_model.run(image_bytes, {})

        if vision_result.error:
            raise RuntimeError(vision_result.error)

        try:
            structured = json.loads(vision_result.output)
        except Exception:
            structured = {"raw": vision_result.output}

        # ---- Reasoning step ----
        reasoning_prompt = f"""
User goal:
{prompt}

Extracted visual data:
{json.dumps(structured, indent=2)}

Reason over this data and produce a clear answer.
"""

        reasoning_result = await self.reasoning_model.run(reasoning_prompt, {})

        latency = int((time.time() - start) * 1000)

        vision_cost = estimate_cost(self.vision_model.name, vision_result.tokens)
        reasoning_cost = estimate_cost(self.reasoning_model.name, reasoning_result.tokens)
        cost = round(vision_cost + reasoning_cost, 6)

        return StrategyResult(
            output=reasoning_result.output,
            models_used=[self.vision_model.name, self.reasoning_model.name],
            cost_usd=cost,
            latency_ms=latency,
            confidence=0.7,
            raw_outputs={
                "vision": structured,
                "reasoning": reasoning_result.output
            }
        )
