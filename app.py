from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from adapters.openai import OpenAIAdapter
from adapters.anthropic import AnthropicAdapter
from adapters.mistral import MistralAdapter
from adapters.local import LocalAdapter
from adapters.google import GoogleAdapter
from adapters.google_vision import GoogleVisionAdapter
from adapters.vision_openai import OpenAIVisionAdapter

from core.router import Router
from core.explain import build_trace
from core.models import OrchestratorResponse
from core.strategy import SingleModelStrategy, ParallelVoteStrategy
from core.vision_pipeline import VisionReasoningPipeline
from core.trace_store import TraceStore
from core.retrieval.retriever import Retriever
from core.retrieval.vector_index import VectorIndex
from core.tenants import TenantStore
from core.rate_limit import RateLimiter
from core.tenant_budget import TenantBudget
from core.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

app = FastAPI(title="SentinelMesh — Explainable Multi-Model Orchestrator")

# --------------------------------------------------
# Models
# --------------------------------------------------
openai_gpt4   = OpenAIAdapter("gpt-4.1-mini")
openai_gpt35  = OpenAIAdapter("gpt-3.5-turbo")
claude        = AnthropicAdapter("claude-3-haiku")
mistral       = MistralAdapter("mistral-small")
google_gemini = GoogleAdapter("gemini-1.5-flash")
local_llama   = LocalAdapter("llama3")
vision_openai = OpenAIVisionAdapter("gpt-4.1-mini")
google_vision = GoogleVisionAdapter()

# --------------------------------------------------
# Strategies
# --------------------------------------------------
single_openai     = SingleModelStrategy(openai_gpt4)
fast_cheap        = SingleModelStrategy(mistral)
parallel_ensemble = ParallelVoteStrategy([openai_gpt4, claude, mistral, local_llama, google_gemini])
high_accuracy     = ParallelVoteStrategy([openai_gpt4, openai_gpt35, google_gemini, claude])

# --------------------------------------------------
# B-04 fix: Retriever instance (not class) passed to Router
# --------------------------------------------------
class _NoOpEmbedder:
    """Stub — replace with a real sentence-transformer in production."""
    def embed(self, text: str):
        import numpy as np
        return np.zeros(384)

retriever = Retriever(VectorIndex(dim=384), _NoOpEmbedder())

router = Router(
    {
        "single_openai":     single_openai,
        "fast_cheap":        fast_cheap,
        "parallel_ensemble": parallel_ensemble,
        "high_accuracy":     high_accuracy,
    },
    retriever,
)

vision_pipeline = VisionReasoningPipeline(
    vision_model=google_vision,
    reasoning_model=openai_gpt4,
)

ws_manager    = WebSocketManager()
trace_store   = TraceStore(ws_manager=ws_manager)
tenants       = TenantStore()
rate_limiter  = RateLimiter()
tenant_budget = TenantBudget()

# --------------------------------------------------
# B-26 fix: Admin auth
# --------------------------------------------------
ADMIN_API_KEY     = os.getenv("ADMIN_API_KEY", "")
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

def require_admin(admin_key: str = Depends(_admin_key_header)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY not configured.")
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    return admin_key

def resolve_tenant(x_api_key: str):
    tenant = tenants.get_by_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant

async def safe_broadcast(payload: dict):
    try:
        await ws_manager.broadcast(payload)
    except Exception as exc:
        logger.warning("WebSocket broadcast failed: %s", exc)


class RunRequest(BaseModel):
    prompt: str


# --------------------------------------------------
# B-01 fix: /run uses router.route_with_metadata()
# Budget check uses atomic check_and_record (TOCTOU fix)
# FIX: removed dead imports (should_escalate, time)
# --------------------------------------------------
@app.post("/run", response_model=OrchestratorResponse)
async def run(req: RunRequest, x_api_key: str = Header(...)):
    tenant = resolve_tenant(x_api_key)

    if not await rate_limiter.allow_async(tenant["id"], tenant["rpm"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    result, ctx, reward = await router.route_with_metadata(req.prompt)
    meta = router.decision_metadata(ctx)

    # FIX: atomic budget check-and-record prevents TOCTOU overspend
    if not tenant_budget.check_and_record(
        tenant_id=tenant["id"],
        limit=tenant["daily_limit"],
        cost=result.cost_usd,
    ):
        raise HTTPException(status_code=402, detail="Tenant daily budget exceeded")

    trace = build_trace(
        features={
            "tenant_id":        tenant["id"],
            "belief_signature": meta["signature"],
            "selection_source": meta["selection_source"],
        },
        strategy_name=meta["strategy"],
        reason=f'{meta["selection_source"]} | reward={round(reward, 4)}',
        result=result,
        selection_source=meta["selection_source"],
        bandit_scores=meta.get("bandit_scores"),
        reward=round(reward, 4),
    )

    trace_store.log(trace)
    asyncio.create_task(safe_broadcast({"type": "trace", "payload": trace.dict()}))
    return OrchestratorResponse(output=result.output, trace=trace)


@app.post("/run_vision", response_model=OrchestratorResponse)
async def run_vision(
    prompt:    str        = Form(...),
    image:     UploadFile = File(...),
    x_api_key: str        = Header(...),
):
    tenant      = resolve_tenant(x_api_key)
    if not await rate_limiter.allow_async(tenant["id"], tenant["rpm"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    image_bytes = await image.read()
    result      = await vision_pipeline.execute(prompt, image_bytes)

    if not tenant_budget.check_and_record(
        tenant_id=tenant["id"],
        limit=tenant["daily_limit"],
        cost=result.cost_usd,
    ):
        raise HTTPException(status_code=402, detail="Tenant daily budget exceeded")

    trace = build_trace(
        features={"multimodal": True, "image_size": len(image_bytes), "tenant_id": tenant["id"]},
        strategy_name="vision→reasoning",
        reason="vision → reasoning pipeline",
        result=result,
    )

    trace_store.log(trace)
    asyncio.create_task(safe_broadcast({"type": "trace", "payload": trace.dict()}))
    return OrchestratorResponse(output=result.output, trace=trace)


# B-06 fix: /query — no strategy_name param
@app.get("/query")
async def query(task: str):
    result = await router.route(task=task)
    return result.dict()


@app.get("/stats")
def stats():
    return router.stats()

@app.get("/admin/world-model")
def world_model_stats():
    return router.world_model.stats()

@app.get("/metrics/beliefs")
def belief_heatmap():
    return router.world_model.stats()

@app.get("/metrics/strategies")
def strategy_stats():
    return router.stats()

@app.get("/metrics/traces")
def recent_traces(limit: int = 100):
    return trace_store.last(limit)

@app.get("/metrics/roi")
def model_roi():
    return trace_store.model_roi()

@app.get("/metrics/strategy-drift")
def strategy_drift():
    return trace_store.strategy_timeseries()

# B-10 fix: single rl-stats route
@app.get("/admin/rl-stats")
def rl_stats():
    return router.rl.stats()

# B-26 fix: admin auth required
@app.post("/admin/create-tenant")
def create_tenant(name: str, _: str = Depends(require_admin)):
    return tenants.create_tenant(name=name)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-LEARNING SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/learning/stats")
async def learning_stats():
    """Get comprehensive learning system statistics."""
    return {
        "continuous_learning": router.continuous_learner.stats(),
        "model_builder": router.model_builder.stats(),
        "domain_adaptation": router.domain_adapter.stats(),
        "independence": router.independence_scheduler.stats(),
        "evolution": router.evolution_engine.stats(),
    }


@app.post("/learning/trigger-training")
async def trigger_training(admin_key: str = Header(None, alias="x-admin-key")):
    """Manually trigger model training on collected data."""
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(401, "Unauthorized")
    
    # Cluster training data
    router.continuous_learner.compute_clusters(n_clusters=20)
    
    # Train models for each cluster
    trained = []
    for cluster_id in range(20):
        examples = router.continuous_learner.get_training_batch(cluster_id=cluster_id, limit=1000)
        if len(examples) >= 100:
            model = router.model_builder.train_cluster_model(cluster_id, examples)
            if model:
                trained.append(model.name)
    
    return {"status": "training_complete", "models_trained": trained}


@app.post("/learning/evolve-models")
async def evolve_models(admin_key: str = Header(None, alias="x-admin-key")):
    """Trigger evolutionary model improvement."""
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(401, "Unauthorized")
    
    # Get all active models
    active_models = [m for m in router.model_builder.models.values() if m.active]
    
    if not active_models:
        return {"status": "no_models_to_evolve"}
    
    # Evolve each model
    evolved = []
    for model in active_models:
        base_model = {
            "path": str(model.weights_path),
            "accuracy": model.accuracy,
            "latency_ms": model.avg_latency_ms,
            "size_mb": model.size_mb,
        }
        
        # Mock evaluation function
        def evaluate_fn(path):
            return {"accuracy": 0.90, "latency_ms": 80, "size_mb": 50}
        
        mutants = router.evolution_engine.evolve(base_model, evaluate_fn, n_mutations=3)
        best = router.evolution_engine.select_best(mutants, keep_top_n=1)
        
        if best:
            evolved.append(best[0])
    
    return {"status": "evolution_complete", "mutants_evaluated": len(evolved)}


@app.get("/learning/independence-progress")
async def independence_progress():
    """Get detailed independence progress metrics."""
    scheduler = router.independence_scheduler
    
    return {
        "current_level": scheduler.current_level.name,
        "target_pct": scheduler.get_target_percentage(),
        "actual_pct": scheduler.get_actual_percentage(),
        "cost_savings": scheduler.estimate_cost_savings(),
        "requests_breakdown": {
            "total": scheduler.total_requests,
            "self_model": scheduler.self_model_requests,
            "external": scheduler.external_requests,
        },
    }


@app.get("/learning/domain-detection")
async def domain_detection():
    """Get current industry detection and domain statistics."""
    adapter = router.domain_adapter
    detected = adapter.detect_industry()
    
    return {
        "detected_industry": detected,
        "confidence": adapter.detection_confidence,
        "vocabulary_suggestions": adapter.suggest_vocabulary_expansion(),
        "compliance_requirements": adapter.get_compliance_requirements(),
        "top_domain_terms": adapter.extract_domain_terms(top_n=50),
    }
