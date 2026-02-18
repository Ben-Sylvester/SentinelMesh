import asyncio
import random
import time
from adapters.base import ModelAdapter
from core.models import ModelResult    # FIX: was missing import — caused NameError


class MockModel(ModelAdapter):
    """
    Mock adapter for local testing without API keys.
    FIX: returns ModelResult (not StrategyResult).
    FIX: fake_delay uses asyncio.sleep instead of time.sleep (was blocking the event loop).
    """
    def __init__(self, name: str = "mock-local"):
        self.name = name

    async def run(self, prompt: str, context: dict = None) -> ModelResult:
        start = time.time()
        await self._fake_delay()
        latency = int((time.time() - start) * 1000)
        return ModelResult(
            output=f"[{self.name}] response to: {prompt[:60]}",
            tokens=random.randint(50, 150),
            latency_ms=latency,
        )

    async def _fake_delay(self):
        # FIX: was time.sleep() — blocking in async context
        await asyncio.sleep(random.uniform(0.05, 0.2))
