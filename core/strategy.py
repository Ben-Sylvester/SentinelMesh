import time
from typing import List
import asyncio
from core.pricing import estimate_cost
from core.metrics import disagreement_score
from core.models import StrategyResult
from adapters.base import ModelAdapter

# Strategy interface
class Strategy:
    name: str

    async def execute(self, prompt: str) -> StrategyResult:
        raise NotImplementedError
    
    def execute_sync(self, prompt: str):
        return asyncio.run(self.execute(prompt, {}))



class SingleModelStrategy(Strategy):
    def __init__(self, model: ModelAdapter):
        self.model = model
        self.name = f"single({model.name})"

    async def execute(self, prompt: str) -> StrategyResult:
        start = time.time()
        result = await self.model.run(prompt, {})

        latency = int((time.time() - start) * 1000)
        cost = estimate_cost(self.model.name, result.tokens)

        return StrategyResult(
            output=result.output,
            models_used=[self.model.name],
            cost_usd=cost,  # fake cost for now
            latency_ms=latency,
            confidence=0.5,
            raw_outputs={self.model.name: result.output}
        )


# Parallel Strategy
class ParallelVoteStrategy(Strategy):
    def __init__(self, models: list[ModelAdapter]):
        self.models = models
        self.name = "parallel_vote"

    async def execute(self, prompt: str) -> StrategyResult:
        start = time.time()

        tasks = [m.run(prompt, {}) for m in self.models]
        results = await asyncio.gather(*tasks)

        latency = int((time.time() - start) * 1000)

        raw_outputs = {
            model.name: result.output
            for model, result in zip(self.models, results)
        }

        disagreement = disagreement_score(raw_outputs)

        # Naive vote: pick shortest answer (placeholder logic)
        final_output = min(raw_outputs.values(), key=len)

        total_tokens = sum(r.tokens for r in results)
        total_cost = 0.0
        for model, r in zip(self.models, results):
            total_cost += estimate_cost(model.name, r.tokens)

        total_cost = round(total_cost, 6)


        confidence = round(1.0 - disagreement, 3)

        return StrategyResult(
            output=final_output,
            models_used=[m.name for m in self.models],
            cost_usd=total_cost,
            latency_ms=latency,
            confidence=confidence,
            raw_outputs={
                "responses": raw_outputs,
                "disagreement": disagreement
            }
        )
