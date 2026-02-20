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


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY & CONTEXT SYSTEM ENDPOINTS (NEW - V2.0)
# ══════════════════════════════════════════════════════════════════════════════

from core.memory.memory_manager import MemoryManager

# Initialize memory system
memory_manager = MemoryManager()

@app.post("/memory/store")
async def store_memory(
    user_id: str,
    session_id: str,
    prompt: str,
    response: str,
    metadata: Optional[dict] = None
):
    """Store an interaction in long-term memory."""
    await memory_manager.store_interaction(
        user_id=user_id,
        session_id=session_id,
        prompt=prompt,
        response=response,
        metadata=metadata
    )
    return {"status": "stored", "session_id": session_id}


@app.post("/memory/recall")
async def recall_memory(
    user_id: str,
    current_prompt: str,
    k: int = 5,
    session_id: Optional[str] = None
):
    """Recall relevant context from past interactions."""
    memories = await memory_manager.recall_context(
        user_id=user_id,
        current_prompt=current_prompt,
        k=k,
        session_id=session_id
    )
    return {
        "memories": memories,
        "count": len(memories),
        "formatted_context": memory_manager.format_context_for_prompt(memories)
    }


@app.get("/memory/conversation/{session_id}")
async def get_conversation(session_id: str, limit: Optional[int] = None):
    """Get full conversation history for a session."""
    history = memory_manager.get_conversation_history(session_id, limit)
    return {
        "session_id": session_id,
        "messages": [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in history
        ],
        "count": len(history)
    }


@app.get("/memory/preferences/{user_id}")
async def get_preferences(user_id: str):
    """Get learned preferences for a user."""
    prefs = memory_manager.get_user_preferences(user_id)
    return {"user_id": user_id, "preferences": prefs}


@app.delete("/memory/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    memory_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
    return memory_manager.stats()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING API ENDPOINTS (NEW - V2.0)
# ══════════════════════════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse
from core.streaming import StreamingManager

streaming_manager = StreamingManager()

@app.post("/stream")
async def stream_completion(prompt: str, model: str = "gpt-4"):
    """
    Stream response token-by-token using Server-Sent Events.
    
    Usage:
        const eventSource = new EventSource('/stream?prompt=Hello&model=gpt-4');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.content);
            if (data.done) eventSource.close();
        };
    """
    async def event_generator():
        # Select adapter (simplified - in production use router)
        from adapters.openai import OpenAIAdapter
        adapter = OpenAIAdapter(model)
        
        async for chunk in streaming_manager.stream_response(adapter, prompt):
            yield streaming_manager.format_sse(chunk)
            if chunk.done:
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED /run ENDPOINT WITH MEMORY (V2.0)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/v2/run")
async def run_with_memory(
    prompt: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_memory: bool = True,
    stream: bool = False
):
    """
    Enhanced /run endpoint with memory support.
    
    - Recalls relevant context automatically
    - Stores interaction for future recall
    - Supports streaming responses
    - Learns user preferences
    """
    # Recall relevant context if memory enabled
    context_str = ""
    if use_memory and user_id:
        memories = await memory_manager.recall_context(
            user_id=user_id,
            current_prompt=prompt,
            k=3
        )
        if memories:
            context_str = memory_manager.format_context_for_prompt(memories)
    
    # Enhance prompt with context
    enhanced_prompt = prompt
    if context_str:
        enhanced_prompt = f"{context_str}\n\n## Current Request:\n{prompt}"
    
    # Route request
    result, ctx, reward = await router.route_with_metadata(enhanced_prompt)
    
    # Store in memory
    if use_memory and user_id and session_id:
        await memory_manager.store_interaction(
            user_id=user_id,
            session_id=session_id,
            prompt=prompt,
            response=result.output or "",
            metadata={
                "strategy": ctx.strategy_name,
                "cost_usd": result.cost_usd,
                "reward": reward
            }
        )
    
    # Build trace
    trace = build_trace(ctx.features, ctx.strategy_name, ctx.reason, result, 
                       ctx.selection_source, ctx.template_name, ctx.bandit_scores, reward)
    
    return {
        "output": result.output,
        "trace": trace.dict(),
        "memory_used": bool(context_str),
        "memories_recalled": len(memories) if use_memory and user_id else 0
    }



# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 UPGRADES: MEMORY + STREAMING + VISUAL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

from core.memory import MemoryManager
from core.streaming import StreamManager
from core.multimodal import VisionManager, VisionPipelineIntegration
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File

# Initialize Phase 1 systems
memory_manager = MemoryManager()
stream_manager = StreamManager()
vision_manager = VisionManager(router=router)
vision_pipeline_integration = VisionPipelineIntegration(vision_manager, router)

logger.info("✅ Phase 1 upgrades loaded: Memory, Streaming, Visual Intelligence")


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/memory/context/{user_id}")
async def get_memory_context(
    user_id: str,
    prompt: str,
    k: int = 5
):
    """
    Retrieve relevant context from user's memory.
    
    Returns:
    - memories: List of relevant past interactions
    - preferences: User preferences
    - memory_count: Number of memories found
    """
    context = await memory_manager.recall_context(user_id, prompt, k=k)
    return context


@app.post("/memory/preference")
async def set_user_preference(
    user_id: str,
    key: str,
    value: str
):
    """Set a user preference manually."""
    memory_manager.set_user_preference(user_id, key, value)
    return {"status": "preference_set", "user_id": user_id, "key": key, "value": value}


@app.get("/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
    return memory_manager.stats()


@app.get("/memory/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    last_n: int = 10
):
    """Get conversation history for a session."""
    history = memory_manager.get_conversation_history(session_id, last_n)
    return {"session_id": session_id, "messages": history}


@app.delete("/memory/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    memory_manager.clear_session(session_id)
    return {"status": "session_cleared", "session_id": session_id}


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/stream")
async def stream_response(
    prompt: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Stream response token-by-token using Server-Sent Events.
    
    Usage:
    ```javascript
    const eventSource = new EventSource('/stream?prompt=Hello');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data);
    };
    ```
    """
    async def generate():
        async for chunk in stream_manager.stream_response(prompt, router, user_id, session_id):
            data = {
                "type": chunk.type,
                "content": chunk.content,
                "metadata": chunk.metadata
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/stream-with-memory")
async def stream_with_memory(
    prompt: str,
    user_id: str,
    session_id: str
):
    """
    Stream response with memory context.
    Combines streaming with stateful conversations.
    """
    async def generate():
        async for chunk in stream_manager.stream_with_memory(
            prompt, router, memory_manager, user_id, session_id
        ):
            data = {
                "type": chunk.type,
                "content": chunk.content,
                "metadata": chunk.metadata
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL INTELLIGENCE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/vision/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = "Describe this image in detail",
    quality: str = "balanced"
):
    """
    Analyze an image with a vision model.
    
    Parameters:
    - image: Image file (JPEG, PNG, etc.)
    - prompt: Analysis prompt
    - quality: "cheap", "balanced", or "accurate"
    
    Returns:
    - analysis: Image analysis
    - model_used: Vision model used
    - cost_usd: Cost of analysis
    """
    image_data = await image.read()
    result = await vision_manager.analyze(image_data, prompt, quality)
    
    return {
        "analysis": result.output,
        "model_used": result.model_used,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "metadata": result.metadata
    }


@app.post("/vision/generate")
async def generate_image(
    prompt: str,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard"
):
    """
    Generate an image from a text prompt.
    
    Parameters:
    - prompt: Image generation prompt
    - model: "dall-e-3" or "stable-diffusion-xl"
    - size: Image size (e.g., "1024x1024")
    - quality: "standard" or "hd"
    
    Returns:
    - image_url: URL to generated image
    - cost_usd: Cost of generation
    """
    result = await vision_manager.generate(prompt, model, size, quality)
    
    return {
        "image_url": result.output,
        "model_used": result.model_used,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "metadata": result.metadata
    }


@app.post("/vision/ocr")
async def extract_text_from_image(
    image: UploadFile = File(...),
    provider: str = "tesseract"
):
    """
    Extract text from an image using OCR.
    
    Parameters:
    - image: Image file
    - provider: "tesseract", "azure-ocr", or "google-vision"
    
    Returns:
    - text: Extracted text
    - cost_usd: Cost of OCR
    """
    image_data = await image.read()
    result = await vision_manager.extract_text(image_data, provider)
    
    return {
        "text": result.output,
        "provider": result.model_used,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms
    }


@app.post("/vision/query")
async def visual_query(
    image: UploadFile = File(...),
    query: str = "What is in this image?"
):
    """
    Process a visual query using two-stage pipeline:
    1. Vision model analyzes image
    2. Reasoning model synthesizes answer
    """
    image_data = await image.read()
    result = await vision_pipeline_integration.process_visual_query(image_data, query)
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED /run ENDPOINT WITH MEMORY
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/run-with-memory")
async def run_with_memory(
    prompt: str,
    user_id: str,
    session_id: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
):
    """
    Enhanced /run endpoint with memory context.
    
    Automatically:
    - Recalls relevant past interactions
    - Applies user preferences
    - Stores interaction in memory
    
    Returns:
    - output: Response
    - trace: Explain trace
    - context: Memory context used
    """
    # Recall context
    context = await memory_manager.recall_context(user_id, prompt, k=3)
    
    # Augment prompt with context
    augmented_prompt = prompt
    if context["memories"]:
        memory_text = "\n".join([
            f"Previous: {m['prompt']} -> {m['response'][:100]}..."
            for m in context["memories"][:2]
        ])
        augmented_prompt = f"Context:\n{memory_text}\n\nCurrent: {prompt}"
    
    # Apply preferences
    prefs = context["preferences"]
    if prefs.get("response_length") == "short":
        augmented_prompt += "\n(Keep response brief)"
    
    # Route request
    result, ctx, reward = await router.route_with_metadata(augmented_prompt)
    
    # Store in memory
    await memory_manager.store_interaction(
        user_id=user_id,
        session_id=session_id,
        prompt=prompt,
        response=result.output or "",
        metadata={"cost": result.cost_usd, "reward": reward}
    )
    
    # Build trace
    trace = build_trace(
        features=ctx.features.tolist(),
        strategy_name=ctx.strategy_name,
        reason=f"Selected by {ctx.selection_source}",
        result=result,
        selection_source=ctx.selection_source,
        template_name=ctx.template_name,
        bandit_scores=None,
        reward=reward
    )
    
    return {
        "output": result.output,
        "trace": trace,
        "context": {
            "memories_used": len(context["memories"]),
            "preferences_applied": prefs
        }
    }


logger.info("✅ Phase 1 API endpoints registered")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 UPGRADES: SEMANTIC CACHE + FUNCTION CALLING + PROMPT LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

from core.cache import SemanticCache, CacheMiddleware
from core.functions import FunctionRegistry, FunctionCallingRouter, FunctionExecutor
from core.prompts.prompt_library import PromptLibrary

# Initialize Phase 2 systems
semantic_cache = SemanticCache()
cache_middleware = CacheMiddleware(router, semantic_cache)
function_registry = FunctionRegistry()
function_executor = FunctionExecutor(function_registry)
function_router = FunctionCallingRouter(router, function_registry)
prompt_library = PromptLibrary()

logger.info("✅ Phase 2 upgrades loaded: Cache, Functions, Prompt Library")


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CACHE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/run-cached")
async def run_with_cache(
    prompt: str,
    similarity_threshold: float = 0.95,
    ttl: Optional[int] = None
):
    """
    Run request with semantic caching.
    
    Checks cache for semantically similar prompts before calling LLM.
    Can reduce costs by 30-45%.
    """
    # Try cache first
    cached = await semantic_cache.get(prompt, similarity_threshold)
    
    if cached:
        return {
            "output": cached["response"],
            "cached": True,
            "similarity": cached["similarity"],
            "hits": cached["hits"],
            "cost_usd": 0.0,
            "trace": {
                "strategy": "cache",
                "cached_at": cached["cached_at"]
            }
        }
    
    # Cache miss - call router
    result, ctx, reward = await router.route_with_metadata(prompt)
    
    # Store in cache
    if result.output:
        await semantic_cache.set(
            prompt=prompt,
            response=result.output,
            ttl=ttl,
            metadata={
                "cost": result.cost_usd,
                "models": result.models_used
            }
        )
    
    trace = build_trace(
        features=ctx.features.tolist(),
        strategy_name=ctx.strategy_name,
        reason=f"Selected by {ctx.selection_source}",
        result=result,
        selection_source=ctx.selection_source,
        template_name=ctx.template_name,
        bandit_scores=None,
        reward=reward
    )
    
    return {
        "output": result.output,
        "cached": False,
        "cost_usd": result.cost_usd,
        "trace": trace
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get semantic cache statistics."""
    stats = semantic_cache.get_stats()
    return {
        "total_requests": stats.total_requests,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "hit_rate": f"{stats.hit_rate:.1%}",
        "total_cost_saved": f"${stats.total_cost_saved:.2f}",
        "avg_similarity": stats.avg_similarity
    }


@app.post("/cache/invalidate")
async def invalidate_cache(pattern: str):
    """Invalidate cache entries matching pattern."""
    semantic_cache.invalidate(pattern)
    return {"status": "invalidated", "pattern": pattern}


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache entries."""
    semantic_cache.clear()
    return {"status": "cache_cleared"}


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION CALLING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# Register example functions
@function_registry.register(
    name="get_current_time",
    description="Get the current time",
    parameters={}
)
async def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@function_registry.register(
    name="calculate",
    description="Perform a mathematical calculation",
    parameters={
        "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate"
        }
    }
)
async def calculate(expression: str) -> float:
    """Safely evaluate mathematical expression."""
    try:
        # Simple safe evaluation (production should use safer methods)
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


@function_registry.register(
    name="search_web",
    description="Search the web for information",
    parameters={
        "query": {
            "type": "string",
            "description": "Search query"
        }
    }
)
async def search_web(query: str) -> str:
    """Simulate web search."""
    return f"Search results for: {query}\n[Results would appear here in production]"


@app.post("/run-with-functions")
async def run_with_functions(
    prompt: str,
    available_functions: Optional[List[str]] = None
):
    """
    Run request with function calling support.
    
    LLM can call registered functions to accomplish tasks.
    """
    result = await function_router.route_with_functions(
        prompt=prompt,
        available_functions=available_functions
    )
    
    return {
        "output": result.output,
        "models_used": result.models_used,
        "cost_usd": result.cost_usd
    }


@app.get("/functions/list")
async def list_functions():
    """List all registered functions."""
    functions = function_registry.list_functions()
    return {
        "functions": functions,
        "count": len(functions)
    }


@app.get("/functions/{function_name}")
async def get_function_definition(function_name: str):
    """Get function definition and schema."""
    func_def = function_registry.get_function(function_name)
    if not func_def:
        raise HTTPException(status_code=404, detail="Function not found")
    
    return {
        "name": func_def.name,
        "description": func_def.description,
        "parameters": [
            {
                "name": p.name,
                "type": p.type.value,
                "description": p.description,
                "required": p.required
            }
            for p in func_def.parameters
        ],
        "returns": func_def.returns
    }


@app.post("/functions/execute")
async def execute_function(
    function_name: str,
    arguments: Dict[str, Any]
):
    """Execute a function directly."""
    from core.functions import FunctionCall
    
    call = FunctionCall(name=function_name, arguments=arguments)
    result = await function_executor.execute(call)
    
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms
    }


@app.get("/functions/openai-format")
async def get_functions_openai_format():
    """Get all functions in OpenAI-compatible format."""
    return {
        "functions": function_registry.to_openai_format()
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT LIBRARY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/prompts/save")
async def save_prompt(
    name: str,
    template: str,
    description: str = "",
    category: str = "general",
    metadata: Optional[Dict] = None
):
    """
    Save a new prompt template.
    
    Example:
    {
        "name": "customer_email",
        "template": "Dear {name},\n\nRegarding {issue}...",
        "description": "Customer support email template",
        "category": "support"
    }
    """
    template_id = prompt_library.save(
        name=name,
        template=template,
        description=description,
        category=category,
        metadata=metadata
    )
    
    return {
        "status": "saved",
        "template_id": template_id,
        "name": name
    }


@app.get("/prompts/{name}")
async def get_prompt(name: str):
    """Get prompt template by name."""
    template = prompt_library.get(name)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {
        "id": template.id,
        "name": template.name,
        "template": template.template,
        "variables": template.variables,
        "description": template.description,
        "category": template.category,
        "version": template.version,
        "usage_count": template.usage_count,
        "avg_rating": template.avg_rating
    }


@app.post("/prompts/{name}/render")
async def render_prompt(
    name: str,
    variables: Dict[str, str]
):
    """
    Render a template with variables.
    
    Example:
    {
        "name": "Alice",
        "issue": "login problem"
    }
    """
    try:
        rendered = prompt_library.render(name, **variables)
        return {
            "rendered": rendered,
            "template": name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/prompts")
async def list_prompts(category: Optional[str] = None):
    """List all prompt templates."""
    templates = prompt_library.list_templates(category=category)
    
    return {
        "templates": [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "variables": t.variables,
                "version": t.version,
                "usage_count": t.usage_count,
                "avg_rating": t.avg_rating
            }
            for t in templates
        ],
        "count": len(templates)
    }


@app.put("/prompts/{name}")
async def update_prompt(
    name: str,
    template: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    change_notes: str = ""
):
    """Update a prompt template (creates new version)."""
    prompt_library.update(
        name=name,
        template=template,
        description=description,
        category=category,
        change_notes=change_notes
    )
    
    return {"status": "updated", "name": name}


@app.post("/prompts/{name}/rate")
async def rate_prompt(
    name: str,
    user_id: str,
    rating: int,
    comment: str = ""
):
    """Rate a prompt template (1-5 stars)."""
    try:
        prompt_library.rate(name, user_id, rating, comment)
        return {"status": "rated", "name": name, "rating": rating}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/prompts/{name}")
async def delete_prompt(name: str):
    """Delete a prompt template."""
    try:
        prompt_library.delete(name)
        return {"status": "deleted", "name": name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/prompts/{name}/run")
async def run_with_prompt_template(
    name: str,
    variables: Dict[str, str]
):
    """
    Render and execute a prompt template.
    
    Combines template rendering with intelligent routing.
    """
    # Render template
    rendered = prompt_library.render(name, **variables)
    
    # Execute with router
    result, ctx, reward = await router.route_with_metadata(rendered)
    
    trace = build_trace(
        features=ctx.features.tolist(),
        strategy_name=ctx.strategy_name,
        reason=f"Selected by {ctx.selection_source}",
        result=result,
        selection_source=ctx.selection_source,
        template_name=name,
        bandit_scores=None,
        reward=reward
    )
    
    return {
        "output": result.output,
        "template_used": name,
        "rendered_prompt": rendered,
        "trace": trace
    }


logger.info("✅ Phase 2 API endpoints registered")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 & 4 UPGRADES: WORKFLOWS + INTEGRATIONS + GUARDRAILS + PLUGINS
# ══════════════════════════════════════════════════════════════════════════════

from core.workflows import WorkflowEngine
from core.integrations.integration_manager import IntegrationManager
from core.guardrails.guardrails_manager import GuardrailsManager

# Initialize Phase 3 & 4 systems
workflow_engine = WorkflowEngine()
integration_manager = IntegrationManager()
guardrails_manager = GuardrailsManager()

logger.info("✅ Phase 3 & 4 upgrades loaded: Workflows, Integrations, Guardrails")


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW ENGINE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/workflows/create")
async def create_workflow(yaml_content: str):
    """Create workflow from YAML definition."""
    workflow = workflow_engine.load_workflow(yaml_content)
    workflow_engine.save_workflow(workflow)
    return {"status": "created", "name": workflow.name}


@app.post("/workflows/{workflow_name}/execute")
async def execute_workflow(workflow_name: str, input_data: Optional[Dict] = None):
    """Execute a workflow."""
    execution = await workflow_engine.execute_workflow(workflow_name, input_data)
    return {
        "execution_id": execution.id,
        "status": execution.status.value,
        "steps_completed": len([s for s in execution.steps.values() if s.status.value == "success"])
    }


@app.get("/workflows/{workflow_name}/executions")
async def list_workflow_executions(workflow_name: str, limit: int = 50):
    """List workflow executions."""
    return {"executions": workflow_engine.list_executions(workflow_name, limit)}


@app.get("/workflows/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution details."""
    execution = workflow_engine.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/integrations")
async def list_integrations():
    """List all available integrations."""
    return {"integrations": integration_manager.list_integrations()}


@app.post("/integrations/{integration_name}/configure")
async def configure_integration(integration_name: str, config: Dict[str, Any]):
    """Configure an integration."""
    integration_manager.configure(integration_name, config)
    return {"status": "configured", "integration": integration_name}


@app.post("/integrations/{integration_name}/execute")
async def execute_integration(
    integration_name: str,
    action: str,
    params: Dict[str, Any]
):
    """Execute an integration action."""
    result = await integration_manager.execute(integration_name, action, params)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error
    }


# ══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/guardrails/check")
async def check_content(text: str, check_type: str = "both"):
    """
    Check content safety.
    check_type: "input", "output", or "both"
    """
    if check_type in ["input", "both"]:
        result = await guardrails_manager.check_input(text)
        return {
            "safe": result.safe,
            "level": result.level.value,
            "score": result.score,
            "flags": result.flags,
            "pii_detected": len(result.pii_detected),
            "redacted_text": result.redacted_text
        }


@app.post("/run-with-guardrails")
async def run_with_guardrails(
    prompt: str,
    auto_redact: bool = True
):
    """Run request with guardrails enabled."""
    # Check input
    input_result = await guardrails_manager.check_input(prompt)
    
    if not guardrails_manager.is_allowed(input_result):
        return {
            "blocked": True,
            "reason": "Input failed safety check",
            "flags": input_result.flags,
            "score": input_result.score
        }
    
    # Process request
    result, ctx, reward = await router.route_with_metadata(prompt)
    
    # Check output
    output_result = await guardrails_manager.check_output(result.output or "")
    
    # Auto-redact if enabled
    final_output = result.output
    if auto_redact and output_result.redacted_text:
        final_output = output_result.redacted_text
    
    trace = build_trace(
        features=ctx.features.tolist(),
        strategy_name=ctx.strategy_name,
        reason=f"Selected by {ctx.selection_source}",
        result=result,
        selection_source=ctx.selection_source,
        template_name=ctx.template_name,
        bandit_scores=None,
        reward=reward
    )
    
    return {
        "output": final_output,
        "trace": trace,
        "safety": {
            "input_score": input_result.score,
            "output_score": output_result.score,
            "pii_redacted": auto_redact and output_result.redacted_text is not None
        }
    }


logger.info("✅ Phase 3 & 4 API endpoints registered")
# Complete Phase 3 & 4 Integration
# Add these endpoints to the existing app.py

from core.plugins.plugin_system import PluginManager
from core.collaboration.collaboration_manager import CollaborationManager
from core.voice.voice_manager import VoiceManager

# Initialize Phase 4 systems
plugin_manager = PluginManager(router=router, memory_manager=memory_manager, integration_manager=integration_manager)
collaboration_manager = CollaborationManager()
voice_manager = VoiceManager()
voice_manager.router = router

logger.info("✅ Phase 4 upgrades loaded: Plugins, Collaboration, Voice")


# ══════════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/plugins/install")
async def install_plugin(plugin_path: str):
    """Install a plugin from file."""
    try:
        plugin_name = await plugin_manager.install_plugin(plugin_path)
        return {"status": "installed", "name": plugin_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/plugins/{plugin_name}/load")
async def load_plugin(plugin_name: str, config: Optional[Dict] = None):
    """Load and initialize a plugin."""
    try:
        await plugin_manager.load_plugin(plugin_name, config)
        return {"status": "loaded", "name": plugin_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/plugins/{plugin_name}/execute")
async def execute_plugin(
    plugin_name: str,
    action: str,
    params: Dict[str, Any]
):
    """Execute a plugin action."""
    try:
        result = await plugin_manager.execute_plugin(plugin_name, action, params)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plugins")
async def list_plugins(include_disabled: bool = False):
    """List all plugins."""
    return {"plugins": plugin_manager.list_plugins(include_disabled)}


@app.post("/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str):
    """Enable a plugin."""
    plugin_manager.enable_plugin(plugin_name)
    return {"status": "enabled", "name": plugin_name}


@app.post("/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str):
    """Disable a plugin."""
    plugin_manager.disable_plugin(plugin_name)
    return {"status": "disabled", "name": plugin_name}


@app.get("/plugins/{plugin_name}/stats")
async def get_plugin_stats(plugin_name: str):
    """Get plugin statistics."""
    return plugin_manager.get_plugin_stats(plugin_name)


# ══════════════════════════════════════════════════════════════════════════════
# COLLABORATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/workspaces/create")
async def create_workspace(
    name: str,
    owner_id: str,
    description: str = ""
):
    """Create a new team workspace."""
    workspace_id = collaboration_manager.create_workspace(name, owner_id, description)
    return {"status": "created", "workspace_id": workspace_id, "name": name}


@app.post("/workspaces/{workspace_id}/members")
async def add_workspace_member(
    workspace_id: str,
    user_id: str,
    role: str = "member",
    added_by: str = None
):
    """Add a member to workspace."""
    from core.collaboration.collaboration_manager import Role
    collaboration_manager.add_member(workspace_id, user_id, Role(role), added_by)
    return {"status": "added", "user_id": user_id, "role": role}


@app.delete("/workspaces/{workspace_id}/members/{user_id}")
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    removed_by: str
):
    """Remove a member from workspace."""
    collaboration_manager.remove_member(workspace_id, user_id, removed_by)
    return {"status": "removed", "user_id": user_id}


@app.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    """Get workspace details."""
    workspace = collaboration_manager.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@app.get("/users/{user_id}/workspaces")
async def list_user_workspaces(user_id: str):
    """List user's workspaces."""
    return {"workspaces": collaboration_manager.list_user_workspaces(user_id)}


@app.post("/workspaces/{workspace_id}/data")
async def set_shared_data(
    workspace_id: str,
    key: str,
    value: Any,
    user_id: str
):
    """Store shared data in workspace."""
    collaboration_manager.set_shared_data(workspace_id, key, value, user_id)
    return {"status": "stored", "key": key}


@app.get("/workspaces/{workspace_id}/data/{key}")
async def get_shared_data(workspace_id: str, key: str):
    """Get shared data from workspace."""
    data = collaboration_manager.get_shared_data(workspace_id, key)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
    return {"key": key, "value": data}


@app.get("/workspaces/{workspace_id}/activity")
async def get_workspace_activity(workspace_id: str, limit: int = 50):
    """Get workspace activity feed."""
    activities = collaboration_manager.get_activity_feed(workspace_id, limit)
    return {"activities": [a.__dict__ for a in activities]}


# ══════════════════════════════════════════════════════════════════════════════
# VOICE INTERFACE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/voice/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    provider: Optional[str] = None,
    language: Optional[str] = None
):
    """Transcribe audio to text."""
    audio_data = await audio.read()
    
    from core.voice.voice_manager import VoiceProvider
    provider_enum = VoiceProvider(provider) if provider else None
    
    result = await voice_manager.stt.transcribe(audio_data, provider_enum, language)
    
    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "duration_ms": result.duration_ms,
        "provider": result.provider
    }


@app.post("/voice/synthesize")
async def synthesize_speech(
    text: str,
    provider: Optional[str] = None,
    voice: Optional[str] = None,
    speed: float = 1.0
):
    """Synthesize text to speech."""
    from core.voice.voice_manager import VoiceProvider
    provider_enum = VoiceProvider(provider) if provider else None
    
    result = await voice_manager.tts.synthesize(text, provider_enum, voice, speed)
    
    return {
        "audio_data": result.audio_data.decode('latin1'),  # For JSON serialization
        "format": result.format,
        "duration_ms": result.duration_ms,
        "provider": result.provider,
        "voice_id": result.voice_id
    }


@app.post("/voice/conversation")
async def voice_conversation(
    audio: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Process complete voice conversation.
    Transcribe -> AI Process -> Synthesize response.
    """
    audio_data = await audio.read()
    result = await voice_manager.process_voice_input(audio_data, language)
    
    return result


logger.info("✅ All Phase 3 & 4 endpoints registered")
