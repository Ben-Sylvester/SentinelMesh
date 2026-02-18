import time
from typing import List
import asyncio
import numpy as np
from core.pricing import estimate_cost
from core.metrics import disagreement_score, text_similarity
from core.models import StrategyResult
from adapters.base import ModelAdapter

# LLM Wrapper
from core.llm.wrapper import LLMWrapper
from core.llm.types import LLMInput


# Strategy interface
class Strategy:
    name: str

    async def execute(self, prompt: str) -> StrategyResult:
        raise NotImplementedError

    def execute_sync(self, prompt: str) -> StrategyResult:
        # B-05/B-16 fix: removed the spurious {} second argument.
        # execute() only accepts prompt. Using asyncio.get_event_loop() + 
        # run_until_complete avoids the RuntimeError when called from within
        # an existing event loop (e.g. self-play running inside FastAPI).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside an async context — use a fresh thread with its own loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.execute(prompt))
                return future.result()
        else:
            return asyncio.run(self.execute(prompt))


# Single Model Strategy
class SingleModelStrategy(Strategy):
    def __init__(self, model: ModelAdapter):
        self.model = model
        self.llm   = LLMWrapper(model)
        self.name  = f"single({model.name})"

    async def execute(self, prompt: str) -> StrategyResult:
        # B-05 fix: execute() accepts only prompt (no {} context dict)
        llm_output = await self.llm.generate(LLMInput(prompt=prompt))
        cost = estimate_cost(self.model.name, llm_output.tokens)
        return StrategyResult(
            output=llm_output.text,
            models_used=[self.model.name],
            cost_usd=cost,
            latency_ms=llm_output.latency_ms,
            confidence=0.5,
            raw_outputs={self.model.name: llm_output.text},
        )


# Parallel Ensemble Strategy with proper consensus selection
class ParallelVoteStrategy(Strategy):
    """
    B-13 fix: was selecting min(outputs, key=len) — the shortest response.
    Now selects the response with the lowest average pairwise disagreement
    (i.e. the most "central" / consensus response among all models).
    This is a true voting mechanism: the response that agrees most with
    all other responses wins.
    """

    def __init__(self, models: List[ModelAdapter]):
        self.models   = models
        self.wrappers = [LLMWrapper(m) for m in models]
        self.name     = "parallel_vote"

    async def execute(self, prompt: str) -> StrategyResult:
        start = time.time()

        results = await asyncio.gather(*[
            w.generate(LLMInput(prompt=prompt))
            for w in self.wrappers
        ])

        latency = int((time.time() - start) * 1000)

        raw_outputs = {r.model_name: r.text for r in results}
        disagreement = disagreement_score(raw_outputs)

        # Consensus selection: pick the response most similar to all others
        final_output = self._consensus_output(raw_outputs)

        total_cost = round(
            sum(estimate_cost(r.model_name, r.tokens) for r in results), 6
        )
        confidence = round(1.0 - disagreement, 3)

        return StrategyResult(
            output=final_output,
            models_used=[m.name for m in self.models],
            cost_usd=total_cost,
            latency_ms=latency,
            confidence=confidence,
            raw_outputs={
                "responses":    raw_outputs,
                "disagreement": disagreement,
            },
        )

    @staticmethod
    def _consensus_output(outputs: dict) -> str:
        """
        Select the response with the minimum average pairwise distance
        to all other responses — the Condorcet-style consensus answer.
        Falls back to the first response if all are identical or there is
        only one model.
        """
        if len(outputs) <= 1:
            return next(iter(outputs.values()))

        values = list(outputs.values())
        best_idx   = 0
        best_score = float("inf")

        for i, candidate in enumerate(values):
            avg_distance = np.mean([
                1.0 - text_similarity(candidate, other)
                for j, other in enumerate(values)
                if i != j
            ])
            if avg_distance < best_score:
                best_score = avg_distance
                best_idx   = i

        return values[best_idx]
