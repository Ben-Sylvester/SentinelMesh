import time
import os
from openai import AsyncOpenAI

from adapters.base import ModelAdapter
from core.models import ModelResult


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = f"openai:{model}"

    async def run(self, prompt: str, context: dict) -> ModelResult:
        start = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            latency = int((time.time() - start) * 1000)

            message = response.choices[0].message.content
            usage = response.usage

            tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)

            return ModelResult(
                output=message,
                tokens=tokens,
                latency_ms=latency,
                error=None
            )

        except Exception as e:
            latency = int((time.time() - start) * 1000)

            return ModelResult(
                output=None,
                tokens=0,
                latency_ms=latency,
                error=str(e)
            )
