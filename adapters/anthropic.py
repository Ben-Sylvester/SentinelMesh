import os
import time
from adapters.base import ModelAdapter
from core.models import ModelResult


class AnthropicAdapter(ModelAdapter):
    """
    Anthropic Claude adapter.
    Uses the real SDK when ANTHROPIC_API_KEY is set; stubs otherwise.
    FIX: returns ModelResult (not StrategyResult), sets self.name.
    """
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self.name  = f"anthropic:{model}"    # FIX: was missing → LLMWrapper crashed

    async def run(self, prompt: str, context: dict = None) -> ModelResult:
        start   = time.time()
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if api_key:
            try:
                import anthropic
                client   = anthropic.AsyncAnthropic(api_key=api_key)
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=context.get("max_tokens", 1024) if context else 1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                latency = int((time.time() - start) * 1000)
                text    = response.content[0].text
                tokens  = response.usage.input_tokens + response.usage.output_tokens
                return ModelResult(output=text, tokens=tokens, latency_ms=latency)
            except Exception as e:
                latency = int((time.time() - start) * 1000)
                return ModelResult(output=None, tokens=0, latency_ms=latency, error=str(e))

        # Stub when no key
        text    = f"[Claude:{self.model}] {prompt[:200]}"
        latency = int((time.time() - start) * 1000)
        return ModelResult(output=text, tokens=len(prompt.split()) * 2, latency_ms=latency)
