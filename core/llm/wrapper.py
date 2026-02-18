import time
from core.llm.types import LLMInput, LLMOutput
from adapters.base import ModelAdapter


class LLMWrapper:
    def __init__(self, model: ModelAdapter):
        self.model = model

    async def generate(self, llm_input: LLMInput) -> LLMOutput:
        start = time.time()

        result = await self.model.run(
            llm_input.prompt,
            {
                "temperature": llm_input.temperature,
                "max_tokens":  llm_input.max_tokens,
                "images":      llm_input.images,
            },
        )

        # FIX: result is now ModelResult (output, tokens, latency_ms, error)
        # Previously used getattr(result, "tokens", 0) which masked the fact
        # that adapters were returning StrategyResult (which had no tokens field)
        # causing all cost estimates to be $0. Now that adapters correctly return
        # ModelResult, direct attribute access is safe and correct.
        latency = int((time.time() - start) * 1000)

        return LLMOutput(
            text=result.output or "",      # guard against None on error
            model_name=self.model.name,
            latency_ms=result.latency_ms or latency,
            tokens=result.tokens,          # ModelResult.tokens is always set
        )
