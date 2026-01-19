from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ModelResult(BaseModel):
    output: Any
    tokens: int
    latency_ms: int
    error: Optional[str] = None


class StrategyResult(BaseModel):
    output: Any
    models_used: List[str]
    cost_usd: float
    latency_ms: int
    confidence: float
    raw_outputs: Dict[str, Any]


class ExplainTrace(BaseModel):
    features: Dict[str, Any]
    strategy: str
    models_used: List[str]
    reason: str
    cost_usd: float
    latency_ms: int
    confidence: float
    disagreement: Optional[float] = None


class OrchestratorResponse(BaseModel):
    output: Any
    trace: ExplainTrace
