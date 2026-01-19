import time
import asyncio
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# ----------------------------
# Model Adapters
# ----------------------------

from adapters.openai import OpenAIAdapter
from adapters.anthropic import AnthropicAdapter
from adapters.mistral import MistralAdapter
from adapters.local import LocalAdapter
from adapters.google import GoogleAdapter
from adapters.google_vision import GoogleVisionAdapter
from adapters.vision_openai import OpenAIVisionAdapter

# ----------------------------
# Core Orchestrator
# ----------------------------

from core.router import Router
from core.explain import build_trace
from core.escalation import should_escalate
from core.models import OrchestratorResponse
from core.strategy import SingleModelStrategy, ParallelVoteStrategy
from core.vision_pipeline import VisionReasoningPipeline
from core.trace_store import TraceStore

# ----------------------------
# Multi-Tenant Infrastructure
# ----------------------------

from core.tenants import TenantStore
from core.rate_limit import RateLimiter
from core.tenant_budget import TenantBudget

# ----------------------------
# WebSocket
# ----------------------------

from core.websocket_manager import WebSocketManager


app = FastAPI(title="SentinelMesh — Explainable Multi-Model Orchestrator")

# --------------------------------------------------
# Model Setup (5 LLM + 2 VLM)
# --------------------------------------------------

openai_gpt4 = OpenAIAdapter("gpt-4.1-mini")
openai_gpt35 = OpenAIAdapter("gpt-3.5-turbo")

claude = AnthropicAdapter("claude-3-haiku")
mistral = MistralAdapter("mistral-small")
google_gemini = GoogleAdapter("gemini-1.5-flash")
local_llama = LocalAdapter("llama3")

vision_openai = OpenAIVisionAdapter("gpt-4.1-mini")
google_vision = GoogleVisionAdapter()

# --------------------------------------------------
# Strategies
# --------------------------------------------------

single_openai = SingleModelStrategy(openai_gpt4)
fast_cheap = SingleModelStrategy(mistral)

parallel_ensemble = ParallelVoteStrategy([
    openai_gpt4,
    claude,
    mistral,
    local_llama,
    google_gemini
])

high_accuracy = ParallelVoteStrategy([
    openai_gpt4,
    openai_gpt35,
    google_gemini,
    claude
])

router = Router({
    "single_openai": single_openai,
    "fast_cheap": fast_cheap,
    "parallel_ensemble": parallel_ensemble,
    "high_accuracy": high_accuracy,
})

# --------------------------------------------------
# Vision Pipeline
# --------------------------------------------------

vision_pipeline = VisionReasoningPipeline(
    vision_model=google_vision,
    reasoning_model=openai_gpt4
)

# --------------------------------------------------
# Observability
# --------------------------------------------------

ws_manager = WebSocketManager()
trace_store = TraceStore()

# --------------------------------------------------
# Multi-Tenant Systems
# --------------------------------------------------

tenants = TenantStore()
rate_limiter = RateLimiter()
tenant_budget = TenantBudget()

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def resolve_tenant(x_api_key: str):
    tenant = tenants.get_by_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


async def safe_broadcast(payload: dict):
    """
    Prevent websocket failures from blocking API execution.
    """
    try:
        await ws_manager.broadcast(payload)
    except Exception as e:
        print("⚠️ WebSocket broadcast failed:", e)


# --------------------------------------------------
# Schemas
# --------------------------------------------------

class RunRequest(BaseModel):
    prompt: str


# --------------------------------------------------
# Run Endpoint
# --------------------------------------------------

@app.post("/run", response_model=OrchestratorResponse)
async def run(req: RunRequest, x_api_key: str = Header(...)):
    tenant = resolve_tenant(x_api_key)

    if not rate_limiter.allow(tenant["id"], tenant["rpm"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # 🔍 Feature extraction
    features = router.extract_features(req.prompt)

    # ▶️ Strategy selection
    strategy = router.select_strategy(features)

    start_time = time.time()
    result = await strategy.execute(req.prompt, {})
    latency = round(time.time() - start_time, 4)

    escalated = False
    escalation_reason = None

    # 🔁 Escalation
    if should_escalate(result):
        next_strategy = router.escalate(strategy.name)
        if next_strategy:
            escalated = True
            escalation_reason = f"disagreement → {next_strategy.name}"

            start_time = time.time()
            result = await next_strategy.execute(req.prompt, {})
            latency = round(time.time() - start_time, 4)
            strategy = next_strategy

    # 🧠 Learning update
    reward = router.update(strategy.name, result)

    # 💰 Budget enforcement
    if not tenant_budget.can_spend(
        tenant_id=tenant["id"],
        limit=tenant["daily_limit"],
        cost=result.cost_usd
    ):
        raise HTTPException(status_code=402, detail="Tenant daily budget exceeded")

    tenant_budget.record(tenant["id"], result.cost_usd)

    # 🧾 Belief snapshot
    belief = router.belief_snapshot()

    # 🧾 Trace
    trace = build_trace(
        features={
            "length": features.get("length"),
            "has_code": features.get("has_code"),
            "tenant_id": tenant["id"],
            "latency_sec": latency,
            "belief_signature": belief.get("task_signature"),
            "belief_recommendation": belief.get("belief_recommendation"),
            "belief_stats": belief.get("belief_stats"),
            "node": "sentinelmesh-core"
        },
        strategy_name=strategy.name,
        reason=f"bandit | reward={round(reward, 4)}"
               + (f" | {escalation_reason}" if escalated else ""),
        result=result
    )

    trace_store.log(trace)

    # ⚡ Async broadcast
    asyncio.create_task(
        safe_broadcast({
            "type": "trace",
            "payload": trace
        })
    )

    return OrchestratorResponse(
        output=result.output,
        trace=trace
    )


# --------------------------------------------------
# Vision Endpoint
# --------------------------------------------------

@app.post("/run_vision", response_model=OrchestratorResponse)
async def run_vision(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    tenant = resolve_tenant(x_api_key)

    if not rate_limiter.allow(tenant["id"], tenant["rpm"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    image_bytes = await image.read()

    start_time = time.time()
    result = await vision_pipeline.execute(prompt, image_bytes)
    latency = round(time.time() - start_time, 4)

    # 💰 Budget enforcement
    if not tenant_budget.can_spend(
        tenant_id=tenant["id"],
        limit=tenant["daily_limit"],
        cost=result.cost_usd
    ):
        raise HTTPException(status_code=402, detail="Tenant daily budget exceeded")

    tenant_budget.record(tenant["id"], result.cost_usd)

    belief = router.belief_snapshot()

    trace = build_trace(
        features={
            "multimodal": True,
            "image_size": len(image_bytes),
            "tenant_id": tenant["id"],
            "latency_sec": latency,
            "belief_signature": belief.get("task_signature"),
            "belief_recommendation": belief.get("belief_recommendation"),
            "belief_stats": belief.get("belief_stats"),
            "node": "sentinelmesh-vision"
        },
        strategy_name=vision_pipeline.name,
        reason="vision → reasoning pipeline",
        result=result
    )

    trace_store.log(trace)

    asyncio.create_task(
        safe_broadcast({
            "type": "trace",
            "payload": trace
        })
    )

    return OrchestratorResponse(
        output=result.output,
        trace=trace
    )


# --------------------------------------------------
# Metrics / Admin
# --------------------------------------------------

@app.post("/admin/create-tenant")
def create_tenant(name: str):
    return tenants.create_tenant(name=name)


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


@app.get("/admin/rl-stats")
def rl_stats():
    return router.rl.stats()


# --------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


#...............................
# Re-enforment Learning States
@app.get("/admin/rl-stats")
def rl_stats():
    return router.rl.stats()
