import base64
import time
import os
from openai import AsyncOpenAI
from adapters.base import ModelAdapter
from core.models import ModelResult


VISION_SCHEMA_PROMPT = """
You are a vision extraction model.
Return ONLY valid JSON with this schema:
{
  "objects": [string],
  "text": [string],
  "tables": [string],
  "layout": object
}
No prose. No markdown. JSON only.
"""


class OpenAIVisionAdapter(ModelAdapter):
    """
    FIX: message content types corrected from 'input_text'/'input_image'
    (not valid OpenAI API types) to 'text' and 'image_url' (correct types).
    Returns ModelResult instead of StrategyResult.
    """
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = model
        self.name   = f"openai-vision:{model}"

    async def run(self, image_bytes: bytes, context: dict = None) -> ModelResult:
        start   = time.time()
        encoded = base64.b64encode(image_bytes).decode()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": VISION_SCHEMA_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            # FIX: correct content types for OpenAI vision API
                            {"type": "text",      "text": "Extract structured content."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        ],
                    },
                ],
                temperature=0,
            )
            latency = int((time.time() - start) * 1000)
            content = response.choices[0].message.content
            tokens  = response.usage.total_tokens
            return ModelResult(output=content, tokens=tokens, latency_ms=latency)
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return ModelResult(output=None, tokens=0, latency_ms=latency, error=str(e))
