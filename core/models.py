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
    tool_call: Optional[Dict[str, Any]] = None   # B-15: enables ToolAgent loop


class ExplainTrace(BaseModel):
    features: Dict[str, Any]
    strategy: str
    models_used: List[str]
    reason: str
    cost_usd: float
    latency_ms: int
    confidence: float
    disagreement: Optional[float] = None
    # B-08: previously missing — caused ValidationError on every build_trace() call
    selection_source: Optional[str] = None
    template: Optional[str] = None
    bandit_scores: Optional[Dict[str, Any]] = None
    # FIX: reward included so eval/runner.py can access it without KeyError
    reward: Optional[float] = None


class OrchestratorResponse(BaseModel):
    output: Any
    trace: ExplainTrace
