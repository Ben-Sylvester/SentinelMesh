# SentinelMesh System Design Document

**Version:** 2.0  
**Last Updated:** February 2026  
**Status:** Production Ready with Self-Learning Capabilities

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Self-Learning System](#self-learning-system)
6. [Data Flow](#data-flow)
7. [Persistence Layer](#persistence-layer)
8. [Multi-Tenancy](#multi-tenancy)
9. [Security](#security)
10. [Performance](#performance)
11. [Scalability](#scalability)
12. [Monitoring & Observability](#monitoring--observability)
13. [Deployment](#deployment)
14. [API Specification](#api-specification)
15. [Error Handling](#error-handling)
16. [Testing Strategy](#testing-strategy)
17. [Future Enhancements](#future-enhancements)

---

## Executive Summary

SentinelMesh is an **intelligent AI orchestration platform** that routes user requests to optimal language models while **progressively learning** to replace external LLM API calls with self-trained, distilled models. The system combines:

- **Adaptive routing** using multi-armed bandits and reinforcement learning
- **Continuous learning** from every user interaction
- **Domain adaptation** to deployment industry
- **Progressive independence** from external LLM providers (0% → 95% over 6-12 months)
- **Cost optimization** through model evolution and local inference

**Key Metrics:**
- **Latency:** 50-200ms (self-models) vs 500-2000ms (external APIs)
- **Cost:** $0.0001/request (local) vs $0.001-0.01/request (external)
- **Accuracy:** 95%+ parity with external LLMs after training
- **Independence:** 90%+ self-sufficiency achievable in 12 months

---

## System Overview

### Design Principles

1. **Learn from Everything:** Every request is a training opportunity
2. **Fail Gracefully:** Always maintain external LLM fallback
3. **Optimize Continuously:** Evolve models for accuracy/speed/size tradeoff
4. **Adapt to Context:** Detect industry and adjust behavior
5. **Transparent Operation:** Full observability into decision-making

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Client Applications                       │
│  (Web, Mobile, API Integrations, Third-party Services)       │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTPS/WebSocket
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Application Layer                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ /run       │  │ /query     │  │ /learning/ │            │
│  │ endpoint   │  │ endpoint   │  │ endpoints  │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
└─────────┼────────────────┼────────────────┼──────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│                  Router (Orchestration Core)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Authentication & Rate Limiting                    │   │
│  │  2. Feature Extraction                                │   │
│  │  3. Strategy Selection (Bandit/RL/World Model)        │   │
│  │  4. Independence Decision (Self vs External)          │   │
│  │  5. Retrieval Gate (Context Injection)                │   │
│  │  6. Prompt Templating                                 │   │
│  │  7. Execution (with Escalation)                       │   │
│  │  8. Learning Update (Reward Computation)              │   │
│  │  9. Training Data Collection                          │   │
│  │ 10. Persistence                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ↓                                 ↓
┌───────────────────┐          ┌─────────────────────┐
│   Self-Models     │          │   External LLMs     │
│   (Local GPU)     │          │   (API Calls)       │
├───────────────────┤          ├─────────────────────┤
│ • Task-specific   │          │ • OpenAI GPT-4      │
│ • Distilled       │          │ • Anthropic Claude  │
│ • 100M-1B params  │          │ • Google Gemini     │
│ • 50-200ms        │          │ • Mistral           │
│ • $0.0001/req     │          │ • $0.001-0.01/req   │
└─────────┬─────────┘          └──────────┬──────────┘
          │                               │
          └───────────────┬───────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│               Learning & Adaptation System                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Continuous Learner    → Collects training data       │  │
│  │  Model Builder         → Trains distilled models      │  │
│  │  Domain Adapter        → Detects industry             │  │
│  │  Independence Scheduler → Manages self-sufficiency    │  │
│  │  Evolution Engine      → Optimizes models             │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                    Persistence Layer                          │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ SQLite (WAL)     │  │ File System      │                 │
│  │ • Bandit state   │  │ • Model weights  │                 │
│  │ • RL Q-values    │  │ • Checkpoints    │                 │
│  │ • World beliefs  │  │ • Generations    │                 │
│  │ • Training corpus│  │                  │                 │
│  │ • Tenants        │  │                  │                 │
│  │ • Budgets        │  │                  │                 │
│  │ • Traces         │  │                  │                 │
│  └──────────────────┘  └──────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Layered Architecture

**Layer 1: API Gateway**
- FastAPI endpoints
- Authentication (API keys)
- Rate limiting (Redis-backed)
- Request validation (Pydantic)
- WebSocket server

**Layer 2: Orchestration**
- Router (decision engine)
- Strategy registry
- Feature extraction
- Prompt engineering
- Escalation logic

**Layer 3: Execution**
- Model adapters (OpenAI, Anthropic, etc.)
- Self-model inference
- Parallel execution (voting strategies)
- Error handling & retries

**Layer 4: Learning**
- Continuous data collection
- Task clustering
- Model training
- Domain adaptation
- Evolution engine

**Layer 5: Persistence**
- SQLite databases
- File system (model weights)
- Async-safe operations
- Transaction management

### Component Interaction Patterns

**Synchronous Path (Fast):**
```
Request → Router → Strategy → Adapter → Response
```

**Async Learning Path (Background):**
```
Response → Collect Training Data → (Background) → Train Model
```

**Independence Decision Path:**
```
Request → Scheduler.should_use_self_model()
    ├─ Yes → Self-Model
    └─ No  → External LLM
```

---

## Core Components

### 1. Router (`core/router.py`)

**Responsibilities:**
- Central orchestration hub
- Feature extraction from prompts
- Strategy selection (bandit/RL/world model)
- Retrieval decision (context injection)
- Execution coordination
- Learning updates
- Training data collection

**Key Methods:**
```python
class Router:
    async def route(self, task: str) -> StrategyResult
    async def route_with_metadata(self, task: str) -> Tuple[StrategyResult, RequestContext, float]
    async def route_agent_task(self, agent_name: str, task: str) -> StrategyResult
    
    def extract_features(self, prompt: str) -> np.ndarray
    def _select_action(self, features: np.ndarray) -> Tuple[RequestContext, list]
    def escalate(self, current_strategy: str) -> Optional[Strategy]
    
    def _compute_and_apply_update(self, ctx, result, latency) -> Tuple[float, dict]
    def _collect_training_data(self, ctx, result, reward: float)
    async def _persist(self, snapshot: dict)
```

**State Management:**
- Bandit arms with LinUCB statistics
- RL Q-table (state-action values)
- World model beliefs (task signatures → strategy performance)
- Meta-policy (mode selection: cheap/balanced/accurate)
- Retrieval gate (learned context injection)

### 2. Contextual Bandit (`core/contextual_bandit.py`)

**Algorithm:** LinUCB (Linear Upper Confidence Bound)

**Mathematical Foundation:**
```
For each arm (strategy) i:
  Expected reward: μ_i = θ_i^T x
  Uncertainty: σ_i = √(x^T A_i^{-1} x)
  UCB score: μ_i + α × σ_i
  
Where:
  θ_i = A_i^{-1} b_i  (parameter estimate)
  A_i = I + Σ x_t x_t^T  (covariance matrix)
  b_i = Σ r_t x_t  (reward vector)
  x = feature vector
  α = exploration parameter
```

**Key Operations:**
```python
class ContextualBandit:
    def select(self, features: np.ndarray) -> str
    def update_memory(self, name: str, features: np.ndarray, reward: float)
    def _ensure_linucb_keys(self, arm: Dict) -> Dict
```

**Persistence:**
- Arms stored as JSON in SQLite
- LinUCB matrices (A, b) serialized as nested lists
- Automatic migration from old UCB1 schema

### 3. RL Policy (`core/rl_policy.py`)

**Algorithm:** Q-Learning with ε-greedy exploration

**State Space:**
```python
state = (task_signature: str, mode: str)
# task_signature: hash of feature vector
# mode: "cheap" | "balanced" | "accurate"
```

**Action Space:**
```python
action = (strategy_name: str, retrieval_flag: bool)
# strategy_name: "fast_cheap", "single_openai", etc.
# retrieval_flag: whether to inject context
```

**Update Rule:**
```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]

Where:
  s = current state
  a = action taken
  r = reward received
  s' = next state
  α = learning rate (0.1)
  γ = discount factor (0.9)
```

**Key Features:**
- Epsilon-greedy exploration (ε decays from 0.1 → 0.02)
- Persistent Q-table in SQLite
- Handles novel state-action pairs gracefully

### 4. World Model (`core/world_model.py`)

**Purpose:** Maintain beliefs about task → strategy performance mappings

**Data Structure:**
```python
beliefs = {
    task_signature: {
        action_key: {
            "mean": float,     # Average reward
            "m2": float,       # Sum of squared differences (for variance)
            "count": int,      # Number of observations
            "trend": float,    # Recent trend (-1 to +1)
            "last_update": float  # Timestamp
        }
    }
}
```

**Key Operations:**
```python
class WorldModel:
    def task_signature(self, features: np.ndarray) -> str
    def update(self, signature: str, strategy: str, retrieval_flag: bool, reward: float)
    def predict(self, signature: str, action: Tuple[str, bool]) -> float
    def stats(self) -> Dict
```

**Decay Mechanism:**
- Beliefs decay over time (decay factor = 0.995)
- Prevents stale patterns from dominating
- Balances old knowledge with new observations

### 5. Strategies (`core/strategy.py`)

**Base Interface:**
```python
class Strategy(ABC):
    name: str
    
    @abstractmethod
    async def execute(self, prompt: str) -> StrategyResult
```

**Implementations:**

**SingleModelStrategy:**
- Direct call to one LLM
- Fastest, lowest cost
- Use for simple/routine tasks

**ParallelVoteStrategy:**
- Calls multiple LLMs in parallel
- Majority voting or consensus selection
- Use for high-stakes/complex tasks
- Higher accuracy, higher cost

**Custom Strategies:**
Users can implement domain-specific strategies (e.g., ChainOfThoughtStrategy, DebateStrategy)

### 6. Model Adapters (`adapters/`)

**Base Interface:**
```python
class ModelAdapter(ABC):
    name: str
    
    @abstractmethod
    async def run(self, prompt: str, context: dict) -> ModelResult
```

**Implementations:**
- `OpenAIAdapter` — GPT-4, GPT-3.5
- `AnthropicAdapter` — Claude 3 (Haiku, Sonnet, Opus)
- `GoogleAdapter` — Gemini 1.5 (Flash, Pro)
- `MistralAdapter` — Mistral Small/Medium/Large
- `LocalAdapter` — Ollama, vLLM
- `MockModel` — Testing/development

**Return Type:**
```python
@dataclass
class ModelResult:
    output: Optional[str]
    tokens: int
    latency_ms: int
    error: Optional[str] = None
```

**Error Handling:**
- Graceful degradation on API failures
- Automatic retry with exponential backoff
- Fallback to stub responses (local models)

---

## Self-Learning System

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   User Request → Response                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │  Continuous Learner  │
              │  (Data Collection)   │
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │   Training Corpus    │
              │   (SQLite)           │
              │   • 10k-1M examples  │
              └──────────┬───────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌─────────────┐  ┌───────────────┐
│ Task         │  │ Domain      │  │ Model         │
│ Clustering   │  │ Detection   │  │ Builder       │
│ (K-means)    │  │ (Keywords)  │  │ (Training)    │
└──────┬───────┘  └──────┬──────┘  └───────┬───────┘
       │                 │                  │
       └────────────────┬┴──────────────────┘
                        ↓
              ┌──────────────────────┐
              │  Self-Model Registry │
              │  (Active Models)     │
              └──────────┬───────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌─────────────┐  ┌───────────────┐
│ Independence │  │ Evolution   │  │ Confidence    │
│ Scheduler    │  │ Engine      │  │ Calibration   │
└──────────────┘  └─────────────┘  └───────────────┘
```

### 1. Continuous Learner

**Data Collection Pipeline:**
```python
For each request:
  1. Extract features (embedding)
  2. Record (input, output, model, reward)
  3. Filter by quality threshold (reward > 0.7)
  4. Buffer in memory (100 examples)
  5. Flush to SQLite periodically
```

**Training Corpus Schema:**
```sql
CREATE TABLE training_corpus (
    id              INTEGER PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    task_vector     BLOB NOT NULL,          -- 384-dim embedding
    input_text      TEXT NOT NULL,
    output_text     TEXT NOT NULL,
    external_model  TEXT NOT NULL,
    reward          REAL NOT NULL CHECK(reward >= 0 AND reward <= 1),
    cluster_id      INTEGER,
    used_for_training BOOLEAN DEFAULT 0,
    
    INDEX idx_reward (reward DESC),
    INDEX idx_cluster (cluster_id)
);
```

**Embedding Model:**
- `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional dense vectors
- Cosine similarity for clustering

**Quality Filtering:**
```python
def should_collect(reward: float) -> bool:
    return reward >= min_quality_threshold  # Default: 0.7
```

### 2. Model Builder

**Training Pipeline:**
```python
1. Cluster training data (K-means, k=20 clusters)
2. For each cluster with >100 examples:
    a. Split train/val/test (80/10/10)
    b. Initialize base model (e.g., FLAN-T5-base)
    c. Fine-tune with knowledge distillation
    d. Evaluate on validation set
    e. Save if accuracy > 90% vs external baseline
3. Register model in self-model registry
```

**Model Architecture:**
```python
@dataclass
class ModelConfig:
    architecture: str     # "seq2seq", "classification", "embedding"
    hidden_size: int      # 256-1024
    num_layers: int       # 4-12
    vocab_size: int       # 32000 (BPE)
    max_length: int       # 512
    task_clusters: List[int]
```

**Training Framework (Production):**
```python
# Using Hugging Face Transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

training_args = TrainingArguments(
    output_dir="./models/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**Model Registry:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "cluster_5_v1234567890",
      "config": {...},
      "weights_path": "models/checkpoints/cluster_5_v1234567890.pt",
      "accuracy": 0.92,
      "size_mb": 120.5,
      "avg_latency_ms": 85,
      "created_at": "2026-02-18T10:30:00",
      "active": true
    }
  ]
}
```

### 3. Domain Adapter

**Industry Detection Algorithm:**
```python
def detect_industry(request_history: List[str]) -> str:
    scores = {industry: 0 for industry in INDUSTRIES}
    
    for req in request_history[-1000:]:
        for industry, profile in INDUSTRIES.items():
            for keyword in profile.keywords:
                if keyword in req.lower():
                    scores[industry] += 1
    
    total = sum(scores.values())
    normalized = {k: v/total for k, v in scores.items()}
    
    best = max(normalized, key=normalized.get)
    confidence = normalized[best]
    
    return best if confidence > 0.3 else None
```

**Industry Profiles:**
```python
INDUSTRIES = {
    "healthcare": {
        "keywords": {"patient", "diagnosis", "medical", "HIPAA", ...},
        "compliance": ["PHI_redaction", "audit_logging"],
        "safety_multiplier": 1.5,  # Higher penalty for errors
    },
    "finance": {
        "keywords": {"portfolio", "trading", "SEC", "risk", ...},
        "compliance": ["PII_masking", "audit_trail"],
        "safety_multiplier": 1.3,
    },
    # ... 7 industries total
}
```

**Domain-Specific Reward Adjustment:**
```python
def compute_domain_reward(base_reward: float, industry: str) -> float:
    profile = INDUSTRIES[industry]
    adjusted = base_reward / profile.safety_multiplier
    
    if profile.custom_reward_fn:
        adjusted = profile.custom_reward_fn(result, adjusted)
    
    return adjusted
```

### 4. Independence Scheduler

**Maturity Model:**
```python
class MaturityLevel(IntEnum):
    BOOTSTRAP   = 0  # 0% self-model
    LEARNING    = 1  # 20% self-model
    COMPETENT   = 2  # 50% self-model
    PROFICIENT  = 3  # 80% self-model
    EXPERT      = 4  # 95% self-model
```

**Confidence Thresholds:**
```python
CONFIDENCE_THRESHOLDS = {
    MaturityLevel.BOOTSTRAP:  1.00,  # Never use self-model
    MaturityLevel.LEARNING:   0.95,  # Very high confidence only
    MaturityLevel.COMPETENT:  0.85,  # High confidence
    MaturityLevel.PROFICIENT: 0.75,  # Medium-high confidence
    MaturityLevel.EXPERT:     0.60,  # Medium confidence
}
```

**Decision Logic:**
```python
def should_use_self_model(confidence: float, cluster_id: int) -> bool:
    # 1. Check maturity level allows self-model
    if current_level == MaturityLevel.BOOTSTRAP:
        return False
    
    # 2. Check confidence threshold
    required_confidence = CONFIDENCE_THRESHOLDS[current_level]
    if confidence < required_confidence:
        return False
    
    # 3. Gradual rollout (maintain target %)
    target_pct = MATURITY_THRESHOLDS[current_level]
    current_pct = self_model_requests / total_requests
    
    if current_pct < target_pct:
        return True
    else:
        return random.random() < target_pct
```

**Maturity Advancement:**
```python
def evaluate_maturity_transition(
    corpus_size: int,
    avg_accuracy: float
) -> bool:
    required_corpus = min_corpus_size * (current_level + 1)
    required_accuracy = 0.85
    
    if corpus_size >= required_corpus and avg_accuracy >= required_accuracy:
        advance_to_next_level()
        return True
    return False
```

### 5. Evolution Engine

**Mutation Operators:**

**1. Weight Pruning:**
```python
def prune_weights(model, target_sparsity=0.3):
    """Remove low-magnitude weights."""
    for layer in model.layers:
        mask = create_magnitude_mask(layer.weights, sparsity=target_sparsity)
        layer.weights *= mask
    return model
```

**2. INT8 Quantization:**
```python
def quantize_int8(model):
    """Reduce precision from FP32 to INT8."""
    for param in model.parameters():
        param.data = quantize_tensor(param.data, dtype=torch.int8)
    return model
```

**3. Knowledge Merging:**
```python
def merge_models(models: List[Model], method="average"):
    """Combine multiple specialist models."""
    if method == "average":
        merged_weights = average_weights([m.state_dict() for m in models])
    elif method == "fisher":
        merged_weights = fisher_merging(models)
    
    merged_model.load_state_dict(merged_weights)
    return merged_model
```

**Fitness Function:**
```python
def compute_fitness(
    accuracy: float,
    latency_ms: int,
    size_mb: float,
    w_acc=0.6,
    w_speed=0.3,
    w_size=0.1
) -> float:
    acc_norm = accuracy
    speed_norm = max(0, 1 - latency_ms/1000)
    size_norm = max(0, 1 - size_mb/1000)
    
    return w_acc * acc_norm + w_speed * speed_norm + w_size * size_norm
```

**Evolutionary Cycle:**
```python
def evolution_cycle():
    # Every 10k requests or weekly
    1. Evaluate all active models
    2. Generate mutants (3-5 per model)
    3. Evaluate mutants on validation set
    4. Compute fitness scores
    5. Select best models (keep top 3)
    6. Archive generation
    7. Prune old generations (keep last 5)
    8. Deploy best models
```

---

## Data Flow

### Request Flow (Normal Operation)

```
1. Client Request
   ↓
2. FastAPI Endpoint (/run)
   ↓
3. Authentication & Rate Limiting
   ↓
4. Router.route_with_metadata()
   ├─ Extract features
   ├─ Select strategy (bandit/RL)
   ├─ Decide retrieval
   ├─ Format prompt
   └─ Execute
   ↓
5. Independence Scheduler
   ├─ Should use self-model?
   │  ├─ Yes → Self-Model Inference
   │  └─ No  → External LLM API
   ↓
6. Strategy.execute()
   ├─ Adapter.run()
   ├─ Escalation (if needed)
   └─ Return StrategyResult
   ↓
7. Learning Update (async)
   ├─ Compute reward
   ├─ Update bandit/RL/world model
   ├─ Collect training data
   └─ Persist state
   ↓
8. Response to Client
```

### Training Flow (Background)

```
1. Training Trigger
   ├─ Corpus size threshold (every 1000 examples)
   ├─ Manual trigger (/learning/trigger-training)
   └─ Scheduled job (daily/weekly)
   ↓
2. Task Clustering
   ├─ Load all task embeddings
   ├─ K-means (k=20)
   └─ Assign cluster IDs
   ↓
3. Model Training (per cluster)
   ├─ Load training batch (cluster_id)
   ├─ Initialize/load base model
   ├─ Fine-tune with distillation
   ├─ Evaluate on validation set
   └─ Save if accuracy > threshold
   ↓
4. Model Registration
   ├─ Add to self-model registry
   ├─ Mark as active
   └─ Update independence metrics
   ↓
5. Evolutionary Optimization (periodic)
   ├─ Generate mutants
   ├─ Evaluate fitness
   ├─ Select best
   └─ Archive generation
```

### Domain Adaptation Flow

```
1. Every Request
   ↓
2. Domain Adapter.ingest_request()
   ├─ Extract keywords
   ├─ Update vocabulary frequency
   └─ Add to request history
   ↓
3. Detection (every 100 requests)
   ├─ Score all industries
   ├─ Normalize scores
   ├─ Select best if confidence > 0.3
   └─ Update detected_industry
   ↓
4. Apply Industry Profile
   ├─ Adjust reward function
   ├─ Inject compliance rules
   ├─ Modify safety thresholds
   └─ Suggest vocabulary expansion
```

---

## Persistence Layer

### SQLite Databases

**1. `data/learning_state.db`**
```sql
-- Bandit state
CREATE TABLE bandit_state (
    arm        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,  -- JSON: {pulls, reward, A, b}
    updated_at TIMESTAMP
);

-- RL Q-values
CREATE TABLE rl_qvalues (
    state      TEXT NOT NULL,     -- JSON: ["signature", "mode"]
    action     TEXT NOT NULL,     -- JSON: ["strategy", true/false]
    value      REAL NOT NULL,
    count      INTEGER,
    updated_at TIMESTAMP,
    PRIMARY KEY (state, action)
);

-- World model beliefs
CREATE TABLE world_model (
    signature  TEXT PRIMARY KEY,
    data       TEXT NOT NULL,  -- JSON: {action_key: {mean, m2, count, ...}}
    updated_at TIMESTAMP
);

-- Meta policy
CREATE TABLE meta_policy (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,  -- JSON: mode values, retrieval gate weights
    updated_at TIMESTAMP
);

-- Traces
CREATE TABLE traces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    strategy   TEXT,
    data       TEXT NOT NULL  -- JSON: full trace
);
```

**2. `data/training_corpus.db`**
```sql
CREATE TABLE training_corpus (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    task_vector     BLOB NOT NULL,
    input_text      TEXT NOT NULL,
    output_text     TEXT NOT NULL,
    external_model  TEXT NOT NULL,
    reward          REAL NOT NULL,
    cluster_id      INTEGER,
    used_for_training BOOLEAN DEFAULT 0
);

CREATE INDEX idx_reward ON training_corpus(reward DESC);
CREATE INDEX idx_cluster ON training_corpus(cluster_id);
CREATE INDEX idx_timestamp ON training_corpus(timestamp DESC);
```

**3. `tenants.db`**
```sql
CREATE TABLE tenants (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    api_key             TEXT UNIQUE NOT NULL,
    daily_limit         REAL NOT NULL,
    requests_per_minute INTEGER NOT NULL,
    created_at          TEXT NOT NULL
);
```

**4. `tenant_budget.db`**
```sql
CREATE TABLE spend (
    tenant_id  TEXT NOT NULL,
    day        TEXT NOT NULL,
    total      REAL NOT NULL,
    PRIMARY KEY (tenant_id, day)
);
```

**5. Model Registry (`models/checkpoints/registry.json`)**
```json
{
  "models": [
    {
      "id": 1,
      "name": "cluster_5_v1234567890",
      "config": {...},
      "weights_path": "models/checkpoints/cluster_5_v1234567890.pt",
      "accuracy": 0.92,
      "size_mb": 120.5,
      "avg_latency_ms": 85,
      "created_at": "2026-02-18T10:30:00",
      "active": true
    }
  ]
}
```

### WAL Mode & Thread Safety

**SQLite Configuration:**
```python
conn = sqlite3.connect(db_path, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA cache_size=10000")
conn.execute("PRAGMA temp_store=MEMORY")
```

**Thread Safety:**
- Global `threading.Lock` for all DB writes
- WAL mode allows concurrent reads
- No long-running transactions (< 100ms)
- Automatic retry on `SQLITE_BUSY`

**Async Safety:**
- All I/O in `_persist_sync()` (sync def)
- Called via `asyncio.to_thread()` (thread pool)
- No blocking in async context

---

## Multi-Tenancy

### Tenant Model

```python
@dataclass
class Tenant:
    id: str              # UUID
    name: str            # Display name
    api_key: str         # sk-sentinel-...
    daily_limit: float   # USD
    requests_per_minute: int
    created_at: datetime
```

### Authentication Flow

```
1. Client sends request with header:
   X-API-Key: sk-sentinel-abc123...
   ↓
2. resolve_tenant(api_key)
   ├─ Query tenants DB
   ├─ Return Tenant object
   └─ Or raise 401 Unauthorized
   ↓
3. Rate Limiting
   ├─ Check RPM (Redis counter)
   └─ Or raise 429 Too Many Requests
   ↓
4. Budget Check
   ├─ Get today's spend
   ├─ Estimate request cost
   ├─ Check: spend + cost <= daily_limit
   └─ Or raise 402 Budget Exceeded
   ↓
5. Process Request
   ↓
6. Record Spend (atomic)
   ├─ Lock
   ├─ Read current spend
   ├─ Add request cost
   ├─ Write back
   └─ Unlock
```

### Rate Limiting

**Implementation: Redis-backed sliding window**

```python
class RateLimiter:
    def __init__(self, redis_url: Optional[str] = None):
        if redis_url:
            self.redis = redis.from_url(redis_url)
        else:
            self.redis = None  # Fallback to in-memory
    
    async def allow_async(self, tenant_id: str, rpm: int) -> bool:
        if self.redis:
            # Redis sliding window
            key = f"rate:{tenant_id}"
            now = time.time()
            window_start = now - 60
            
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, 60)
            results = pipe.execute()
            
            count = results[2]
            return count <= rpm
        else:
            # In-memory fallback
            return self._check_in_memory(tenant_id, rpm)
```

### Budget Enforcement

**Atomic Check-and-Record:**

```python
class TenantBudget:
    _LOCK = threading.Lock()
    
    def check_and_record(
        self,
        tenant_id: str,
        limit: float,
        cost: float
    ) -> bool:
        with self._LOCK:
            conn = sqlite3.connect(BUDGET_DB)
            
            # Read current spend
            today = date.today().isoformat()
            row = conn.execute(
                "SELECT total FROM spend WHERE tenant_id=? AND day=?",
                (tenant_id, today)
            ).fetchone()
            
            current_spend = row[0] if row else 0.0
            
            # Check budget
            if current_spend + cost > limit:
                conn.close()
                return False  # Would exceed limit
            
            # Record spend
            conn.execute("""
                INSERT INTO spend (tenant_id, day, total)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id, day) DO UPDATE SET
                    total = total + excluded.total
            """, (tenant_id, today, cost))
            
            conn.commit()
            conn.close()
            return True
```

---

## Security

### API Key Management

**Format:** `sk-sentinel-<32-hex-chars>`

**Generation:**
```python
import secrets
api_key = f"sk-sentinel-{secrets.token_hex(32)}"
```

**Storage:** SHA-256 hashed in database

**Admin Keys:**
- Separate admin API key for protected endpoints
- Format: `<64-hex-chars>`
- Set via `ADMIN_API_KEY` environment variable

### Input Validation

**Pydantic Schemas:**
```python
class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    daily_limit: float = Field(..., gt=0, le=1000)
    requests_per_minute: int = Field(..., ge=1, le=1000)
```

### SQL Injection Prevention

- All queries use parameterized statements
- No string concatenation in SQL
- Input sanitization via Pydantic

### Secrets Management

**Environment Variables:**
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ADMIN_API_KEY=<strong-random-key>
```

**Never committed to version control:**
- `.env` in `.gitignore`
- Use `.env.example` for template

### HTTPS/TLS

**Production deployment:**
```nginx
server {
    listen 443 ssl http2;
    server_name api.sentinelmesh.com;
    
    ssl_certificate /etc/letsencrypt/live/api.sentinelmesh.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.sentinelmesh.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Performance

### Latency Breakdown

**Typical Request (External LLM):**
```
Authentication:         1ms
Feature Extraction:     2ms
Strategy Selection:     3ms
Retrieval (if used):    50ms
Template Formatting:    1ms
External LLM API:       500-2000ms
Learning Update:        5ms
Persistence:            10ms (async)
Total:                  ~560-2070ms
```

**Self-Model Request:**
```
Authentication:         1ms
Feature Extraction:     2ms
Strategy Selection:     3ms
Confidence Check:       1ms
Self-Model Inference:   50-200ms
Learning Update:        5ms
Persistence:            10ms (async)
Total:                  ~70-220ms
```

### Throughput

**Single Instance (8GB RAM, 4 CPU):**
- External LLM: ~100 req/min (limited by API rate limits)
- Self-Model: ~500 req/min (CPU-bound)
- Mixed (50/50): ~300 req/min

**With GPU (NVIDIA T4):**
- Self-Model: ~2000 req/min (batch inference)

### Optimization Strategies

**1. Batch Inference (Self-Models):**
```python
# Group requests arriving within 50ms window
batch = collect_requests(timeout_ms=50, max_size=32)
outputs = model.generate(batch)
```

**2. Model Caching:**
- Keep active models in GPU memory
- LRU eviction when memory pressure
- Lazy loading on first request

**3. Connection Pooling:**
- Redis connection pool (size=10)
- HTTP client session reuse (aiohttp)

**4. Async Persistence:**
- All I/O in thread pool
- No blocking in event loop
- Batch writes (every 100 requests)

---

## Scalability

### Horizontal Scaling

**Load Balancer Configuration:**
```nginx
upstream sentinelmesh {
    least_conn;  # Route to least busy instance
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}
```

**Shared State Requirements:**
- Redis for rate limiting (shared across instances)
- Shared file system for model weights (NFS/EFS)
- SQLite databases can be read-only replicas

**Session Affinity:**
- Not required (stateless requests)
- WebSocket connections pinned to instance

### Vertical Scaling

**Memory Requirements:**
```
Base application:       500MB
Bandit/RL/World Model:  100MB
Self-Models (each):     100-500MB
Training corpus:        100MB per 100k examples
Connections:            ~10MB per 100 concurrent
```

**Recommended Specs:**
```
Small (100 req/min):    4GB RAM, 2 CPU
Medium (500 req/min):   8GB RAM, 4 CPU
Large (2000 req/min):   16GB RAM, 8 CPU, 1 GPU
```

### Database Scaling

**Read Replicas:**
```python
# Primary (read-write)
primary_conn = sqlite3.connect("data/learning_state.db")

# Replica (read-only)
replica_conn = sqlite3.connect("data/learning_state_replica.db", uri=True)
replica_conn.execute("PRAGMA query_only=ON")
```

**Sharding:**
- Shard by tenant_id (multi-tenant deployments)
- Separate DB per tenant for large customers

---

## Monitoring & Observability

### Key Metrics

**System Health:**
```python
# Requests
request_count_total
request_duration_seconds (histogram)
request_errors_total

# Independence
independence_level_gauge
self_model_requests_total
external_requests_total
cost_savings_usd_total

# Learning
training_corpus_size
active_models_count
avg_self_model_accuracy

# Tenants
tenant_count
tenant_budget_usage_pct
tenant_rate_limit_hits_total
```

### Logging

**Structured Logging:**
```python
logger.info(
    "request_complete",
    extra={
        "request_id": req_id,
        "tenant_id": tenant_id,
        "strategy": strategy_name,
        "latency_ms": latency,
        "cost_usd": cost,
        "reward": reward,
        "used_self_model": used_self,
    }
)
```

**Log Levels:**
- DEBUG: Feature vectors, strategy scores
- INFO: Request completion, training triggers
- WARNING: Escalations, budget warnings
- ERROR: API failures, DB errors

### Dashboards

**Streamlit Dashboards:**
1. `dashboard/app.py` — Admin view (system-wide)
2. `dashboard/customer_app.py` — Tenant view (per-tenant)
3. `dashboard.py` — Live intelligence (real-time learning)

**Grafana Integration:**
```python
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('requests_total', 'Total requests', ['strategy', 'tenant'])
independence = Gauge('independence_pct', 'Self-model usage percentage')
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  sentinelmesh:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  dashboard:
    build: .
    command: streamlit run dashboard/app.py
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentinelmesh
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentinelmesh
  template:
    metadata:
      labels:
        app: sentinelmesh
    spec:
      containers:
      - name: sentinelmesh
        image: sentinelmesh:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: sentinelmesh-data
      - name: models
        persistentVolumeClaim:
          claimName: sentinelmesh-models
```

---

## API Specification

See [API.md](API.md) for complete OpenAPI specification.

---

## Error Handling

### Error Hierarchy

```
AppError (base)
├── AuthenticationError (401)
├── AuthorizationError (403)
├── RateLimitError (429)
├── BudgetExceededError (402)
├── ValidationError (422)
├── ModelError (500)
│   ├── ModelTimeoutError
│   └── ModelAPIError
└── InternalError (500)
```

### Error Response Format

```json
{
  "error": {
    "type": "BudgetExceededError",
    "message": "Daily budget limit exceeded",
    "details": {
      "daily_limit": 10.0,
      "current_spend": 10.05,
      "request_cost": 0.002
    }
  }
}
```

### Retry Strategy

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ModelTimeoutError)
)
async def call_external_llm(prompt: str):
    ...
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_bandit.py
def test_linucb_cold_start():
    bandit = ContextualBandit(feature_dim=5)
    bandit.register_arm("strategy_a")
    bandit.register_arm("strategy_b")
    
    features = np.random.randn(5)
    
    # Should explore all arms before exploiting
    selections = [bandit.select(features) for _ in range(100)]
    assert "strategy_a" in selections
    assert "strategy_b" in selections
```

### Integration Tests

```python
# tests/test_router_integration.py
@pytest.mark.asyncio
async def test_full_routing_flow():
    router = Router({...}, retriever)
    result, ctx, reward = await router.route_with_metadata("Test prompt")
    
    assert result.output is not None
    assert 0 <= reward <= 1
    assert ctx.strategy_name in router.strategies
```

### Load Tests

```python
# tests/load_test.py
import locust

class SentinelMeshUser(HttpUser):
    @task
    def run_query(self):
        self.client.post("/run", json={"prompt": "Test"})
```

---

## Future Enhancements

### Roadmap

**Q2 2026:**
- [ ] Active learning (intelligently select which examples to label)
- [ ] Federated learning (privacy-preserving distributed training)
- [ ] Multi-modal self-models (vision + text)

**Q3 2026:**
- [ ] Neural Architecture Search (auto-discover optimal model structures)
- [ ] Continual learning (learn new tasks without forgetting)
- [ ] Model compression (further optimize size/speed)

**Q4 2026:**
- [ ] Full production deployment at scale
- [ ] 99.9% uptime SLA
- [ ] Multi-region deployment
- [ ] Enterprise features (SSO, RBAC, audit logs)

---

## Conclusion

SentinelMesh represents a **paradigm shift** from static LLM routing to **dynamic, self-improving AI orchestration**. The system learns from every interaction, progressively building its own capabilities while maintaining the safety net of external LLM fallback.

**Key Achievements:**
1. Adaptive routing with multi-armed bandits + RL  
2. Continuous learning from every request  
3. Progressive independence (0% → 95% in 12 months)  
4. Domain adaptation (7 industries supported)  
5. Cost reduction (up to 90% vs external APIs)  
6. Production-ready architecture  

The system is **ready for deployment** and will continuously improve its performance, efficiency, and independence over time.

---

**For questions or support, see [README.md](README.md)**
