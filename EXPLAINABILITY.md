# Explainability & Observability (XAI)

## Overview

SentinelMesh provides **complete transparency** into every routing decision, learning update, and model prediction through a comprehensive Explainability and Observability system.

---

## 🔍 Core XAI Features

### 1. **Decision Tracing**
Every request generates an `ExplainTrace` that captures:

```python
@dataclass
class ExplainTrace:
    # Input Analysis
    features: np.ndarray           # Feature vector extracted from prompt
    
    # Decision Process
    strategy: str                  # Which strategy was selected
    selection_source: str          # How it was selected (bandit/RL/world_model)
    bandit_scores: Dict            # UCB scores for all strategies
    
    # Execution Details
    models_used: List[str]         # Which LLMs were called
    latency_ms: int               # Total execution time
    cost_usd: float               # Request cost
    
    # Learning Feedback
    reward: float                 # Computed quality score (0-1)
    confidence: float             # Model confidence
    
    # Reasoning
    reason: str                   # Human-readable explanation
    template: str                 # Which prompt template was used
```

### 2. **Real-Time Observability**

**WebSocket Live Feed** (`ws://localhost:8000/ws`)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const trace = JSON.parse(event.data);
  console.log({
    strategy: trace.strategy,
    reward: trace.reward,
    latency: trace.latency_ms,
    cost: trace.cost_usd,
    reason: trace.reason
  });
};
```

**Example Trace:**
```json
{
  "type": "trace",
  "payload": {
    "strategy": "single_openai",
    "models_used": ["gpt-4"],
    "selection_source": "rl",
    "reason": "Q-learning selected single_openai (Q-value: 0.87) over parallel_ensemble (0.72) based on task signature abc123 in 'balanced' mode",
    "bandit_scores": {
      "single_openai": 0.89,
      "fast_cheap": 0.65,
      "parallel_ensemble": 0.78
    },
    "reward": 0.92,
    "confidence": 0.88,
    "latency_ms": 823,
    "cost_usd": 0.0024
  }
}
```

### 3. **Metrics Endpoints**

#### GET `/metrics/beliefs`
World model's learned beliefs about task-strategy performance:

```json
{
  "task_sig_abc123": {
    "single_openai_no_retrieval": {
      "mean": 0.87,
      "variance": 0.02,
      "count": 152,
      "trend": 0.05,
      "last_update": "2026-02-19T10:30:00"
    },
    "parallel_ensemble_with_retrieval": {
      "mean": 0.91,
      "variance": 0.03,
      "count": 43,
      "trend": 0.12
    }
  }
}
```

**Interpretation:**
- High mean = Strategy performs well for this task type
- Low variance = Consistent performance
- Positive trend = Improving over time
- High count = Sufficient data to trust the estimate

#### GET `/metrics/strategy-drift`
Performance trends over time (detect concept drift):

```json
{
  "single_openai": [
    {"timestamp": "2026-02-18T09:00:00", "reward": 0.85},
    {"timestamp": "2026-02-18T10:00:00", "reward": 0.87},
    {"timestamp": "2026-02-18T11:00:00", "reward": 0.86}
  ],
  "fast_cheap": [...]
}
```

#### GET `/metrics/roi`
Return on investment for each model:

```json
{
  "gpt-4": {
    "requests": 1523,
    "total_cost": 3.68,
    "avg_reward": 0.92,
    "roi": 250.0  // reward per dollar
  },
  "gpt-3.5-turbo": {
    "requests": 892,
    "total_cost": 0.45,
    "avg_reward": 0.78,
    "roi": 173.3
  }
}
```

### 4. **Trace Persistence**

All traces stored in SQLite for historical analysis:

```sql
-- Get traces for last hour
SELECT * FROM traces 
WHERE timestamp > datetime('now', '-1 hour')
ORDER BY timestamp DESC;

-- Analyze strategy performance
SELECT 
    strategy,
    COUNT(*) as count,
    AVG(json_extract(data, '$.reward')) as avg_reward,
    AVG(json_extract(data, '$.cost_usd')) as avg_cost
FROM traces
GROUP BY strategy;
```

### 5. **Learning Progress Tracking**

#### GET `/learning/stats`
```json
{
  "continuous_learning": {
    "corpus_size_disk": 15420,
    "high_quality_kept": 12338,
    "buffer_size": 47,
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
    "compliance_rules": ["PHI_redaction", "audit_logging"],
    "vocabulary_size": 2341
  },
  "independence": {
    "maturity_level": 2,
    "maturity_name": "COMPETENT",
    "actual_self_model_pct": 0.52,
    "target_pct": 0.50,
    "savings_usd": 624.80
  },
  "evolution": {
    "current_generation": 5,
    "avg_fitness": 0.87,
    "best_fitness": 0.93
  }
}
```

---

## 📊 Dashboards

### 1. **Admin Dashboard** (`dashboard/app.py`)

```bash
streamlit run dashboard/app.py
```

**Features:**
- 🔥 **Belief Heatmap:** Visual grid of task signatures × strategies
- 📈 **Strategy Drift Chart:** Performance trends over time
- 💰 **Model ROI Table:** Cost vs reward analysis
- 🤖 **RL Policy Stats:** Q-values, epsilon decay
- ⚡ **Live Trace Feed:** Real-time request flow

**Screenshot (ASCII art):**
```
┌─────────────────────────────────────────────────┐
│ 🔥 Belief Heatmap (Reward by Task × Strategy)  │
├─────────────────────────────────────────────────┤
│            │ single_openai │ fast_cheap │ ...  │
│ task_abc   │     0.87      │    0.65    │      │
│ task_xyz   │     0.72      │    0.81    │      │
│ task_123   │     0.93      │    0.58    │      │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 📈 Strategy Confidence Drift                    │
├─────────────────────────────────────────────────┤
│ [Line chart showing reward over time]           │
│ single_openai: ↗ trending up                    │
│ fast_cheap: → stable                            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 💰 Model ROI                                    │
├─────────────────────────────────────────────────┤
│ gpt-4:          250.0 (1523 requests, $3.68)    │
│ gpt-3.5-turbo:  173.3 (892 requests, $0.45)     │
│ claude-haiku:   312.5 (421 requests, $0.21)     │
└─────────────────────────────────────────────────┘
```

### 2. **Customer Dashboard** (`dashboard/customer_app.py`)

Tenant-specific view with:
- Personal usage statistics
- Budget tracking
- Request history
- Cost breakdown

### 3. **Live Intelligence Dashboard** (`dashboard.py`)

Real-time learning metrics:
- Independence progress
- Domain detection
- Model evolution
- Training triggers

---

## 🧪 Explainability API

### POST `/run` (with full trace)

**Request:**
```json
{
  "prompt": "Explain quantum entanglement",
  "max_tokens": 500
}
```

**Response:**
```json
{
  "output": "Quantum entanglement is a phenomenon where...",
  "trace": {
    "features": [0.5, 1.0, 0.0, 1.0, 0.0],
    "strategy": "single_openai",
    "models_used": ["gpt-4"],
    "cost_usd": 0.0024,
    "latency_ms": 823,
    "confidence": 0.88,
    "reward": 0.92,
    "selection_source": "rl",
    "reason": "Q-learning selected single_openai (Q-value: 0.87) based on task signature and 'balanced' mode. High confidence (0.88) due to 152 similar requests with avg reward 0.87.",
    "template": "default",
    "bandit_scores": {
      "single_openai": 0.89,
      "fast_cheap": 0.65,
      "parallel_ensemble": 0.78
    },
    "retrieval_used": false,
    "escalated": false
  }
}
```

**Reason Explanation Breakdown:**
- **Why this strategy?** Q-learning found highest expected value
- **How confident?** Based on 152 similar requests
- **Performance history:** Average reward 0.87 for this task type
- **Mode:** "balanced" (cost/quality tradeoff)
- **Alternatives considered:** fast_cheap (0.65), parallel_ensemble (0.78)

### GET `/admin/rl-stats`

Detailed RL policy state:

```json
{
  "epsilon": 0.05,
  "total_updates": 2415,
  "state_count": 347,
  "top_q_values": [
    {
      "state": "('abc123', 'accurate')",
      "action": "('parallel_ensemble', true)",
      "q_value": 0.93,
      "count": 28
    },
    {
      "state": "('xyz789', 'cheap')",
      "action": "('fast_cheap', false)",
      "q_value": 0.81,
      "count": 156
    }
  ]
}
```

---

## 🔬 Debugging & Analysis Tools

### 1. **Feature Inspection**

```python
# Extract features from a prompt
from core.router import Router

features = router.extract_features("Your prompt here")
print(features)
# Output: [1.0, 0.0, 0.0, 1.0, 1.0]
#         [len, code?, image?, reasoning?, questions]
```

### 2. **Bandit Arm Analysis**

```python
from core.contextual_bandit import ContextualBandit

bandit = router.bandit
stats = bandit.stats()

for arm, metrics in stats.items():
    print(f"{arm}: {metrics['pulls']} pulls, avg reward {metrics['avg_reward']:.2f}")
```

### 3. **World Model Inspection**

```python
beliefs = router.world_model.stats()

for signature, strategies in beliefs.items():
    print(f"\nTask signature: {signature}")
    for strategy, perf in strategies.items():
        print(f"  {strategy}: {perf['mean']:.2f} (n={perf['count']})")
```

### 4. **Trace Export**

```python
from core.trace_store import TraceStore

store = TraceStore()
traces = store.last(n=100)

# Export to CSV
import pandas as pd
df = pd.DataFrame([
    {
        'timestamp': t.timestamp,
        'strategy': t.strategy,
        'reward': t.reward,
        'cost': t.cost_usd,
        'latency': t.latency_ms
    }
    for t in traces
])
df.to_csv('traces.csv', index=False)
```

---

## 📈 Performance Monitoring

### Key Metrics to Track

**Quality Metrics:**
- Average reward (target: > 0.8)
- Strategy distribution (should be diverse)
- Escalation rate (target: < 5%)

**Efficiency Metrics:**
- Average latency (target: < 1000ms)
- Cost per request (target: decreasing over time)
- Independence percentage (target: increasing)

**Learning Metrics:**
- Corpus growth rate
- Self-model accuracy
- Domain detection confidence

### Alerting Rules

```python
# Example monitoring setup
if avg_reward < 0.7:
    alert("Quality degradation detected")

if escalation_rate > 0.10:
    alert("High escalation rate - strategies may be failing")

if independence_pct > 0.8 and self_model_accuracy < 0.90:
    alert("Self-models being used but accuracy is low")
```

---

## 🎯 Interpretability Best Practices

### 1. **Always Check Traces**

```python
result, ctx, reward = await router.route_with_metadata(prompt)

print(f"Selected: {ctx.strategy_name}")
print(f"Reason: {ctx.selection_source}")
print(f"Confidence: {result.confidence}")
print(f"Reward: {reward}")
```

### 2. **Monitor Belief Convergence**

Beliefs should stabilize over time:
```python
# Check variance for mature task signatures
for sig, strategies in beliefs.items():
    for strategy, perf in strategies.items():
        if perf['count'] > 50 and perf['variance'] > 0.1:
            print(f"Warning: High variance for {sig} → {strategy}")
```

### 3. **Validate Self-Model Decisions**

```python
# For self-model requests, check confidence
if used_self_model and confidence < 0.75:
    print(f"Warning: Low confidence ({confidence:.2f}) - consider external fallback")
```

### 4. **Track Domain Drift**

```python
# Periodically check if detected industry changes
industry = router.domain_adapter.detect_industry()
if industry != previous_industry:
    print(f"Domain shift detected: {previous_industry} → {industry}")
```

---

## 🔐 Privacy & Compliance

### Trace Data Handling

**What's Stored:**
- Feature vectors (numeric, anonymized)
- Strategy selections
- Performance metrics
- Timestamps

**What's NOT Stored (by default):**
- Full prompt text (unless learning is enabled)
- User identifiers
- API responses (only quality scores)

**GDPR Compliance:**
```python
# Enable prompt anonymization
learner = ContinuousLearner(anonymize_prompts=True)

# Data retention policy
trace_store.delete_older_than(days=90)
```

**HIPAA Compliance:**
For healthcare deployments:
```python
# PHI redaction
domain_adapter.enable_compliance("PHI_redaction")

# Audit logging
trace_store.enable_audit_trail()
```

---

## 📊 Example Analysis Workflow

### Scenario: Debugging Low Performance

```python
# 1. Check overall metrics
stats = router.stats()
print(f"Average reward: {stats['bandit']['avg_reward']:.2f}")

# 2. Identify problematic strategies
for strategy, metrics in stats['bandit'].items():
    if metrics['avg_reward'] < 0.7:
        print(f"⚠️ {strategy}: {metrics['avg_reward']:.2f}")

# 3. Inspect recent traces for that strategy
traces = trace_store.get_by_strategy(strategy, limit=10)
for t in traces:
    print(f"  Reward: {t.reward:.2f}, Reason: {t.reason}")

# 4. Check if world model agrees
beliefs = world_model.stats()
# Look for low mean rewards

# 5. Take action
# - Deactivate underperforming strategy
# - Retrain self-model
# - Adjust prompt templates
```

---

## 🎓 Summary

SentinelMesh provides **enterprise-grade explainability** through:

✅ **Complete Decision Transparency** - Every routing decision explained  
✅ **Real-Time Observability** - WebSocket live feed + metrics  
✅ **Historical Analysis** - SQLite trace persistence  
✅ **Multi-Level Dashboards** - Admin, customer, live intelligence  
✅ **Programmatic Access** - Full API for custom analysis  
✅ **Privacy Controls** - Anonymization + compliance features  
✅ **Performance Monitoring** - Automated alerting capabilities  

All decisions are **auditable, interpretable, and actionable**.
