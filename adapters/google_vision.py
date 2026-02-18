import os
import base64
import time
from adapters.base import ModelAdapter
from core.models import ModelResult


class GoogleVisionAdapter(ModelAdapter):
    """
    Google Gemini Vision adapter.
    FIX: sets self.name (was missing — VisionReasoningPipeline calls
    estimate_cost(self.vision_model.name, ...) which would AttributeError).
    Returns ModelResult, not StrategyResult.
    """
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model = model
        self.name  = f"google-vision:{model}"   # FIX: was missing

    async def run(self, image_bytes: bytes, context: dict = None) -> ModelResult:
        start   = time.time()
        api_key = os.getenv("GOOGLE_API_KEY")

        if api_key:
            try:
                import google.generativeai as genai
                from google.generativeai.types import content_types
                genai.configure(api_key=api_key)

                model    = genai.GenerativeModel(self.model)
                img_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}
                prompt   = (
                    "Extract structured content from this image as JSON with keys: "
                    "objects (list), text (list), tables (list), layout (object). "
                    "Return ONLY valid JSON. No prose."
                )
                response = await model.generate_content_async([prompt, img_part])
                latency  = int((time.time() - start) * 1000)
                tokens   = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0
                return ModelResult(output=response.text, tokens=tokens, latency_ms=latency)
            except Exception as e:
                latency = int((time.time() - start) * 1000)
                return ModelResult(output=None, tokens=0, latency_ms=latency, error=str(e))

        # Stub
        import json
        latency = int((time.time() - start) * 1000)
        stub    = json.dumps({"objects": ["chart", "table"], "text": [], "tables": [], "layout": {}})
        return ModelResult(output=stub, tokens=50, latency_ms=latency)
