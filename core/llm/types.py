from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LLMInput:
    prompt: str
    images: Optional[List[str]] = None
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class LLMOutput:
    text: str
    model_name: str
    latency_ms: int
    tokens: int
    cost_usd: Optional[float] = None
    confidence: Optional[float] = None