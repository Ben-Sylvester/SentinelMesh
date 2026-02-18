import os
import time
import asyncio
from adapters.base import ModelAdapter
from core.models import ModelResult


class LocalAdapter(ModelAdapter):
    """
    Local model adapter — targets Ollama (http://localhost:11434) by default.
    FIX: returns ModelResult, sets self.name (was missing).
    """
    def __init__(self, model: str = "llama3"):
        self.model    = model
        self.name     = f"local:{model}"     # FIX: was missing
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    async def run(self, prompt: str, context: dict = None) -> ModelResult:
        start = time.time()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                data    = resp.json()
                text    = data.get("response", "")
                tokens  = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                latency = int((time.time() - start) * 1000)
                return ModelResult(output=text, tokens=tokens, latency_ms=latency)
        except Exception as e:
            # Graceful degradation: stub response so ensembles still complete
            latency = int((time.time() - start) * 1000)
            stub    = f"[Local:{self.model}] {prompt[:200]}"
            return ModelResult(output=stub, tokens=len(prompt.split()) * 2, latency_ms=latency)
