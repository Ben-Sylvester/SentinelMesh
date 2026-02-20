# Observability, Explainability & Agent Deployment Guide

## Table of Contents
1. [Explainability (XAI)](#explainability-xai)
2. [Observability](#observability)
3. [Agent Framework](#agent-framework)
4. [Comparison with Other Orchestrators](#comparison-with-other-orchestrators)
5. [Quick Start Examples](#quick-start-examples)

---

## Explainability (XAI)

### Overview

SentinelMesh provides **full explainability** for every routing decision through the `ExplainTrace` system. Every request returns a detailed trace showing exactly why decisions were made.

### ExplainTrace Structure

```python
@dataclass
class ExplainTrace(BaseModel):
    """Complete explanation of routing decision."""
    
    # What was analyzed
    features: List[float]          # Extracted features from prompt
    
    # What was chosen
    strategy: str                  # Selected strategy name
    models_used: List[str]         # Actual models called
    
    # Why it was chosen
    reason: str                    # Human-readable explanation
    selection_source: str          # "bandit", "rl", "world_model"
    bandit_scores: Optional[Dict]  # UCB scores for all strategies
    
    # Performance metrics
    cost_usd: float               # Actual cost incurred
    latency_ms: int               # End-to-end latency
    confidence: float             # Self-model confidence (if used)
    reward: float                 # Quality score
    
    # Context
    template: str                 # Prompt template used
```

### How It Works

**Every request automatically includes a trace:**

```python
# POST /run
{
    "prompt": "Explain quantum entanglement"
}

# Response includes both output AND trace
{
    "output": "Quantum entanglement is...",
    "trace": {
        "features": [2, 0, 0, 1, 1],  # [length_bucket, has_code, has_image, is_explanation, question_count]
        "strategy": "single_openai",
        "models_used": ["gpt-4"],
        "reason": "Selected by RL policy (Q=0.87) due to high expected reward for explanation tasks",
        "selection_source": "rl",
        "bandit_scores": {
            "fast_cheap": 0.72,
            "single_openai": 0.89,
            "parallel_ensemble": 0.75
        },
        "cost_usd": 0.002,
        "latency_ms": 850,
        "confidence": 0.92,
        "reward": 0.87,
        "template": "default"
    }
}
```

### Explainability Levels

#### 1. Decision Transparency
**What strategy was chosen and why:**

```python
trace.reason = "Selected by RL policy (Q=0.87) due to high expected reward"
trace.selection_source = "rl"  # or "bandit", "world_model"
```

#### 2. Alternative Analysis
**What other options were considered:**

```python
trace.bandit_scores = {
    "fast_cheap": 0.72,        # Would save money but lower quality
    "single_openai": 0.89,     # CHOSEN - best expected reward
    "parallel_ensemble": 0.75  # Higher quality but too expensive
}
```

#### 3. Performance Breakdown
**Exact cost, latency, quality:**

```python
trace.cost_usd = 0.002      # Actual cost
trace.latency_ms = 850      # Actual latency
trace.reward = 0.87         # Quality score (0-1)
trace.confidence = 0.92     # Model confidence
```

#### 4. Feature Attribution
**What characteristics influenced the decision:**

```python
trace.features = [2, 0, 0, 1, 1]
# Decoded:
# - length_bucket=2 (long prompt)
# - has_code=0 (no code)
# - has_image=0 (no image)
# - is_explanation=1 (explanation request)
# - question_count=1 (1 question)
```

### Live Observability

#### WebSocket Live Feed
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const trace = JSON.parse(event.data);
    console.log('Live trace:', {
        strategy: trace.strategy,
        cost: trace.cost_usd,
        latency: trace.latency_ms,
        reason: trace.reason
    });
};
```

#### Dashboard Visualization

**Streamlit dashboards show:**
- Strategy selection heatmap
- Belief evolution over time
- Cost/latency trends per strategy
- Reward distributions
- Model ROI analysis

```bash
# Real-time learning dashboard
streamlit run dashboard.py
```

### Audit Trail

**All traces are persisted in SQLite:**

```python
# Query historical decisions
import sqlite3
conn = sqlite3.connect('data/learning_state.db')

# Find all decisions for a specific strategy
traces = conn.execute("""
    SELECT timestamp, data 
    FROM traces 
    WHERE strategy = 'single_openai'
    ORDER BY timestamp DESC
    LIMIT 100
""").fetchall()

for ts, data in traces:
    trace = json.loads(data)
    print(f"{ts}: {trace['reason']} (reward={trace['reward']})")
```

### Explainability API Endpoints

```bash
# Get decision explanation
GET /metrics/beliefs
# Returns: World model beliefs (task signatures → strategy performance)

# Get strategy evolution
GET /metrics/strategy-drift
# Returns: How strategy preferences change over time

# Get model ROI
GET /metrics/roi
# Returns: Cost vs reward analysis per model

# Get RL policy state
GET /admin/rl-stats
# Returns: Q-values, epsilon, state space coverage
```

---

## Observability

### System Metrics

**Exposed via `/stats` endpoint:**

```json
{
    "strategies": ["single_openai", "fast_cheap", "parallel_ensemble"],
    "bandit": {
        "single_openai": {
            "pulls": 1523,
            "avg_reward": 0.87,
            "avg_cost": 0.002,
            "avg_latency": 850
        },
        "fast_cheap": {
            "pulls": 892,
            "avg_reward": 0.75,
            "avg_cost": 0.0005,
            "avg_latency": 200
        }
    },
    "rl": {
        "epsilon": 0.05,
        "total_updates": 2415,
        "state_space_size": 47,
        "action_space_size": 6
    },
    "world_model": {
        "signatures": 23,
        "beliefs": 138
    }
}
```

### Learning System Observability

```bash
GET /learning/stats
```

```json
{
    "continuous_learning": {
        "corpus_size_disk": 15420,
        "high_quality_kept": 12338,
        "quality_threshold": 0.7
    },
    "model_builder": {
        "active_models": 8,
        "avg_accuracy": 0.91,
        "total_size_mb": 960.5
    },
    "domain_adaptation": {
        "detected_industry": "healthcare",
        "detection_confidence": 0.78,
        "compliance_rules": ["PHI_redaction", "audit_logging"]
    },
    "independence": {
        "maturity_level": 2,
        "maturity_name": "COMPETENT",
        "actual_self_model_pct": 0.52,
        "target_self_model_pct": 0.50,
        "cost_savings": {
            "actual_cost_usd": 798.20,
            "baseline_cost_usd": 1423.00,
            "savings_usd": 624.80,
            "savings_pct": 43.9
        }
    }
}
```

### Logging

**Structured JSON logs:**

```json
{
    "timestamp": "2026-02-18T10:30:45Z",
    "level": "INFO",
    "event": "request_complete",
    "request_id": "req_abc123",
    "tenant_id": "acme-corp",
    "strategy": "single_openai",
    "latency_ms": 850,
    "cost_usd": 0.002,
    "reward": 0.87,
    "used_self_model": false,
    "escalated": false
}
```

---

## Agent Framework

### Overview

SentinelMesh includes a **built-in agent framework** that makes deploying AI agents **dramatically easier** than other orchestrators.

### Key Advantages

| Feature | SentinelMesh | LangChain | AutoGen | CrewAI |
|---------|--------------|-----------|---------|--------|
| **Routing Integration** | ✅ Native | ❌ Manual | ❌ Manual | ❌ Manual |
| **Cost Optimization** | ✅ Automatic | ❌ Manual | ❌ Manual | ❌ Manual |
| **Learning** | ✅ Built-in | ❌ None | ❌ None | ❌ None |
| **Tool Registry** | ✅ Auto-discovery | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| **Multi-tenancy** | ✅ Built-in | ❌ DIY | ❌ DIY | ❌ DIY |
| **Setup Lines of Code** | **~10 lines** | ~50 lines | ~40 lines | ~30 lines |

### Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│              BaseAgent (Abstract)                    │
│  ┌───────────────────────────────────────────────┐  │
│  │ • run(task) → StrategyResult                  │  │
│  │ • Uses Router for intelligent model selection │  │
│  │ • Automatic cost tracking                     │  │
│  │ • Built-in learning                           │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                        ↓
         ┌──────────────┴──────────────┐
         ↓                             ↓
┌──────────────────┐        ┌──────────────────────┐
│   ToolAgent      │        │   CustomAgent        │
│                  │        │   (Your Agent)       │
│ • Tool calling   │        │                      │
│ • Loop detection │        │ • Custom logic       │
│ • MAX_ROUNDS     │        │ • Domain-specific    │
└──────────────────┘        └──────────────────────┘
```

### Deploying Agents: Comparison

#### **SentinelMesh (10 lines)**

```python
from core.agents.tool_agent import ToolAgent
from core.tools.tool_registry import ToolRegistry

# 1. Define tools
registry = ToolRegistry()

@registry.register("calculate")
def calculate(expression: str) -> str:
    return str(eval(expression))  # Simplified

# 2. Create agent
agent = ToolAgent(
    name="math_assistant",
    router=router,  # Already configured!
    tool_registry=registry
)

# 3. Run
result = await agent.run("What is 15% of 240?")
print(result.output)
```

**Total: 10 lines of actual code**

#### **LangChain (~50 lines)**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

# 1. Setup LLM
llm = OpenAI(temperature=0, model="gpt-4")

# 2. Define tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for math calculations"
    )
]

# 3. Setup memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 4. Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 5. Manual cost tracking
with get_openai_callback() as cb:
    result = agent.run("What is 15% of 240?")
    print(f"Cost: ${cb.total_cost}")
    print(result)

# 6. No routing optimization (stuck with gpt-4)
# 7. No learning (same cost every time)
# 8. No multi-tenancy (DIY)
```

**Total: ~50 lines + manual setup**

#### **AutoGen (~40 lines)**

```python
import autogen

# 1. Configure LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.environ["OPENAI_API_KEY"]
    }
]

# 2. Create assistant agent
assistant = autogen.AssistantAgent(
    name="math_assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0
    }
)

# 3. Create user proxy with code execution
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# 4. Manual tool registration
@user_proxy.register_for_execution()
@assistant.register_for_llm(description="Calculator")
def calculate(expression: str) -> str:
    return str(eval(expression))

# 5. Run
user_proxy.initiate_chat(
    assistant,
    message="What is 15% of 240?"
)

# 6. No cost optimization
# 7. No learning
# 8. No multi-tenancy
```

**Total: ~40 lines + complex setup**

---

## SentinelMesh Agent Advantages

### 1. **Zero Configuration Routing**

```python
# Agent automatically uses optimal strategy
agent = ToolAgent(name="assistant", router=router, tool_registry=tools)

# Router handles:
# - Strategy selection (cheap vs accurate)
# - Model selection (GPT-4 vs GPT-3.5 vs self-model)
# - Cost optimization
# - Escalation on failures
# - Learning from outcomes
```

**Competitors:** Require manual model selection per agent

### 2. **Automatic Cost Tracking**

```python
result = await agent.run(task)

# Cost automatically tracked:
print(f"Cost: ${result.cost_usd}")
print(f"Latency: {result.latency_ms}ms")
print(f"Models used: {result.models_used}")
```

**Competitors:** Manual cost tracking via callbacks

### 3. **Built-in Learning**

```python
# Every agent call improves future routing:
# - Bandit learns which strategies work best
# - RL learns long-term optimization
# - World model builds task → strategy mapping
# - Domain adapter detects your use case
```

**Competitors:** No learning - same cost/performance forever

### 4. **Multi-Tenancy Built-in**

```python
# Per-tenant agents with isolation
result = await router.route_agent_task(
    agent_name="math_assistant",
    task=user_query
)

# Automatic:
# - Cost attribution to tenant
# - Rate limiting per tenant
# - Budget enforcement
```

**Competitors:** DIY multi-tenancy

### 5. **Tool Auto-Discovery**

```python
# Tools auto-register from decorators
@registry.register("search")
def search(query: str) -> str:
    return search_api(query)

# Agent automatically knows about all tools
```

**Competitors:** Manual tool list maintenance

### 6. **Loop Detection**

```python
class ToolAgent:
    MAX_TOOL_ROUNDS = 5  # Prevents infinite loops
    
    async def run(self, task: str):
        for round in range(MAX_TOOL_ROUNDS):
            # Safe execution
            ...
```

**Competitors:** Manual loop protection

---

## Quick Start Examples

### Example 1: Simple Assistant Agent

```python
from core.agents.tool_agent import ToolAgent
from core.tools.tool_registry import ToolRegistry

# Setup tools
tools = ToolRegistry()

@tools.register("get_weather")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

# Create agent
agent = ToolAgent("weather_assistant", router, tools)

# Run
result = await agent.run("What's the weather in San Francisco?")
print(result.output)  # "The weather in San Francisco is Sunny, 72°F"
```

### Example 2: Research Agent

```python
@tools.register("search")
def search(query: str) -> str:
    return web_search_api(query)

@tools.register("summarize")
def summarize(text: str) -> str:
    return summarization_api(text)

agent = ToolAgent("researcher", router, tools)

result = await agent.run(
    "Research the latest developments in quantum computing "
    "and summarize the top 3 breakthroughs"
)
```

### Example 3: Multi-Agent System

```python
# Sales agent
sales_tools = ToolRegistry()
@sales_tools.register("get_pricing")
def get_pricing(product: str) -> str:
    return database.get_price(product)

sales_agent = ToolAgent("sales", router, sales_tools)

# Support agent
support_tools = ToolRegistry()
@support_tools.register("create_ticket")
def create_ticket(issue: str) -> str:
    return helpdesk.create(issue)

support_agent = ToolAgent("support", router, support_tools)

# Route based on intent
if "price" in user_query:
    result = await sales_agent.run(user_query)
else:
    result = await support_agent.run(user_query)
```

### Example 4: Custom Agent

```python
from core.agents.base_agent import BaseAgent

class DataAnalystAgent(BaseAgent):
    def __init__(self, name: str, router, database):
        super().__init__(name, router)
        self.db = database
    
    async def run(self, task: str) -> StrategyResult:
        # 1. Understand query
        query_analysis = await self.router.route(
            f"Convert to SQL: {task}"
        )
        
        # 2. Execute query
        sql = query_analysis.output
        data = self.db.execute(sql)
        
        # 3. Analyze results
        analysis = await self.router.route(
            f"Analyze this data: {data}"
        )
        
        return analysis

# Use custom agent
analyst = DataAnalystAgent("data_analyst", router, postgres_db)
result = await analyst.run("Show sales trends for Q4")
```

---

## Comparison Summary

### Ease of Deployment: **SentinelMesh Wins**

**SentinelMesh:**
- ✅ 10 lines of code
- ✅ Zero configuration
- ✅ Automatic optimization
- ✅ Built-in learning
- ✅ Production-ready multi-tenancy

**Other Orchestrators:**
- ❌ 30-50+ lines of code
- ❌ Manual configuration
- ❌ No optimization
- ❌ No learning
- ❌ DIY multi-tenancy

### Why SentinelMesh is Easier

1. **Router is Pre-configured:** Don't choose models manually
2. **Learning is Automatic:** Gets better over time
3. **Cost is Tracked:** No callback hell
4. **Multi-tenancy Works:** No DIY auth/limits
5. **Tools Auto-register:** No manual lists
6. **Explainability Built-in:** Every decision explained

---

## Conclusion

**Explainability:** ✅ **Fully Intact**
- Every decision includes `ExplainTrace`
- Live WebSocket feed
- Historical audit trail
- Dashboard visualization

**Observability:** ✅ **Fully Intact**
- `/stats` endpoint
- `/learning/stats` endpoint
- Structured logging
- Prometheus-ready metrics

**Agent Framework:** ✅ **Fully Intact**
- BaseAgent + ToolAgent classes
- Tool registry
- Loop protection
- **10 lines vs 50+ lines** compared to competitors

**SentinelMesh agents are easier to deploy than any other orchestrator** because routing, learning, cost tracking, and multi-tenancy are **built-in**, not **bolt-on**.
