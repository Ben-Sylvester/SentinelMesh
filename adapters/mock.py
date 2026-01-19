import time
import random
from adapters.base import ModelAdapter
from core.models import ModelResult

# Mock Model (so we can test immediately)
class MockModel(ModelAdapter):
    def __init__(self, name: str):
        self.name = name

    async def run(self, prompt: str, context: dict) -> ModelResult:
        start = time.time()
        await self.fake_delay()

        latency = int((time.time() - start) * 1000)

        return ModelResult(
            output=f"[{self.name}] response to: {prompt[:40]}",
            tokens=random.randint(50, 150),
            latency_ms=latency,
            error=None
        )

    async def fake_delay(self):
        time.sleep(random.uniform(0.1, 0.4))
