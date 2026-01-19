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
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = f"openai-vision:{model}"

    async def run(self, image_bytes: bytes, context: dict) -> ModelResult:
        start = time.time()
        encoded = base64.b64encode(image_bytes).decode()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": VISION_SCHEMA_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Extract structured content."},
                            {"type": "input_image", "image_base64": encoded}
                        ]
                    }
                ],
                temperature=0
            )

            latency = int((time.time() - start) * 1000)
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens

            return ModelResult(
                output=content,
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
