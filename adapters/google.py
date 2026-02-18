import os
import time
from adapters.base import ModelAdapter
from core.models import ModelResult


class GoogleAdapter(ModelAdapter):
    """
    Google Gemini adapter. Uses real SDK when GOOGLE_API_KEY is set.
    FIX: returns ModelResult, sets self.name (was missing).
    """
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self.name  = f"google:{model}"    # FIX: was missing → LLMWrapper crashed

    async def run(self, prompt: str, context: dict = None) -> ModelResult:
        start   = time.time()
        api_key = os.getenv("GOOGLE_API_KEY")

        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model    = genai.GenerativeModel(self.model)
                response = await model.generate_content_async(prompt)
                latency  = int((time.time() - start) * 1000)
                text     = response.text
                tokens   = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0
                return ModelResult(output=text, tokens=tokens, latency_ms=latency)
            except Exception as e:
                latency = int((time.time() - start) * 1000)
                return ModelResult(output=None, tokens=0, latency_ms=latency, error=str(e))

        text    = f"[Gemini:{self.model}] {prompt[:200]}"
        latency = int((time.time() - start) * 1000)
        return ModelResult(output=text, tokens=len(prompt.split()) * 2, latency_ms=latency)
