import os
import time
from adapters.base import ModelAdapter
from core.models import ModelResult


class MistralAdapter(ModelAdapter):
    """
    Mistral adapter. Uses real SDK when MISTRAL_API_KEY is set.
    FIX: returns ModelResult, self.name already set correctly.
    """
    def __init__(self, model: str = "mistral-small"):
        self.model    = model
        self.provider = "mistral"
        self.name     = f"mistral:{model}"

    async def run(self, prompt: str, context: dict = None) -> ModelResult:
        start   = time.time()
        api_key = os.getenv("MISTRAL_API_KEY")

        if api_key:
            try:
                from mistralai.async_client import MistralAsyncClient
                from mistralai.models.chat_completion import ChatMessage
                client   = MistralAsyncClient(api_key=api_key)
                response = await client.chat(
                    model=self.model,
                    messages=[ChatMessage(role="user", content=prompt)],
                )
                latency = int((time.time() - start) * 1000)
                text    = response.choices[0].message.content
                tokens  = response.usage.total_tokens if response.usage else 0
                return ModelResult(output=text, tokens=tokens, latency_ms=latency)
            except Exception as e:
                latency = int((time.time() - start) * 1000)
                return ModelResult(output=None, tokens=0, latency_ms=latency, error=str(e))

        text    = f"[{self.name}] {prompt[:200]}"
        latency = int((time.time() - start) * 1000)
        return ModelResult(output=text, tokens=len(prompt.split()) * 2, latency_ms=latency)
