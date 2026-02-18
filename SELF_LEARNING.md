# SentinelMesh Self-Learning System

## Overview

The Self-Learning System transforms SentinelMesh from an intelligent router into a **self-evolving AI platform** that progressively becomes independent from external LLM providers.

## Architecture

```
User Request
     ↓
┌────────────────────┐
│  Router (Gateway)  │
└────────┬───────────┘
         │
         ├→ [Independence Scheduler] ──→ Use Self-Model?
         │         ↓ Yes                       ↓ No
         │   ┌─────────────┐            ┌──────────────┐
         │   │ Self-Model  │            │ External LLM │
         │   │ (Local)     │            │ (API)        │
         │   └──────┬──────┘            └──────┬───────┘
         │          │                          │
         │          └──────────┬───────────────┘
         │                     ↓
         ├→ [Continuous Learner] ── Collect training data
         ├→ [Domain Adapter] ────── Detect industry
         ├→ [Model Builder] ────────Train new models
         └→ [Evolution Engine] ─────Improve models
```

## Core Components

### 1. Continuous Learner (`continuous_learner.py`)
**Purpose:** Collect training data from every interaction

**Features:**
- Knowledge distillation from external LLM outputs
- Quality filtering (only collect high-reward examples)
- Task embedding generation for clustering
- SQLite-based training corpus
- Automatic data buffering and flushing

**Usage:**
```python
learner = ContinuousLearner()
learner.collect(
    input_text=user_prompt,
    output_text=llm_response,
    external_model="gpt-4",
    reward=0.95
)
```

### 2. Model Builder (`model_builder.py`)
**Purpose:** Train distilled models on clustered task data

**Features:**
- Task clustering (group similar requests)
- Model training per cluster
- Architecture search (find optimal model size)
- Model registry and versioning
- Performance tracking

**Usage:**
```python
builder = ModelBuilder()
model = builder.train_cluster_model(
    cluster_id=5,
    training_examples=examples,
    hidden_size=256,
    epochs=3
)
```

### 3. Domain Adapter (`domain_adapter.py`)
**Purpose:** Automatically detect deployment industry and adapt

**Industries Supported:**
- Healthcare (HIPAA compliance, medical terminology)
- Finance (SEC compliance, trading vocabulary)
- Legal (case law, contracts, statutes)
- E-commerce (products, inventory, orders)
- Software (code, debugging, APIs)
- Education (FERPA compliance, pedagogy)
- Government (FedRAMP, FISMA, policy)

**Features:**
- Industry detection from request patterns
- Domain-specific vocabulary learning
- Compliance requirement injection
- Custom reward functions per industry

**Usage:**
```python
adapter = DomainAdapter()
adapter.ingest_request(user_prompt)
industry = adapter.detect_industry()  # Returns "healthcare", "finance", etc.
```

### 4. Independence Scheduler (`independence_scheduler.py`)
**Purpose:** Manage transition from external LLMs to self-models

**Maturity Levels:**
- **Level 0 (Bootstrap):** 100% external LLM
- **Level 1 (Learning):** 20% self-model (simple tasks)
- **Level 2 (Competent):** 50% self-model (common patterns)
- **Level 3 (Proficient):** 80% self-model (edge cases external)
- **Level 4 (Expert):** 95% self-model (fully independent)

**Features:**
- Confidence-based routing decisions
- Gradual rollout (A/B testing)
- Cost savings tracking
- Automatic maturity advancement

**Usage:**
```python
scheduler = IndependenceScheduler()
use_self_model = scheduler.should_use_self_model(
    confidence=0.92,
    cluster_id=5
)
```

### 5. Evolution Engine (`evolution.py`)
**Purpose:** Continuously improve models through evolution

**Mutation Operators:**
- **Prune Weights:** Remove low-importance neurons (20-30% size reduction)
- **Quantize INT8:** Reduce precision (75% size reduction)
- **Knowledge Merge:** Combine specialist models into generalist
- **Add Adapter:** Task-specific fine-tuning layers
- **Distill Smaller:** Train smaller student from large teacher

**Features:**
- Multi-objective fitness (accuracy × speed / size)
- Generational selection (keep best models)
- Automatic archiving and pruning
- Rollback safety

**Usage:**
```python
engine = EvolutionEngine()
mutants = engine.evolve(base_model, evaluate_fn, n_mutations=3)
best = engine.select_best(mutants, keep_top_n=3)
```

## API Endpoints

### Learning Statistics
```
GET /learning/stats
```
Returns comprehensive statistics for all learning components.

### Trigger Training
```
POST /learning/trigger-training
Header: x-admin-key: <admin_key>
```
Manually trigger model training on collected data.

### Evolve Models
```
POST /learning/evolve-models
Header: x-admin-key: <admin_key>
```
Apply evolutionary mutations to improve existing models.

### Independence Progress
```
GET /learning/independence-progress
```
Get detailed metrics on independence maturity and cost savings.

### Domain Detection
```
GET /learning/domain-detection
```
View detected industry and domain adaptation statistics.

## Migration Timeline

### Week 1-2: Foundation
- [x] Deploy data collection pipeline
- [x] Initialize continuous learner
- [x] Collect 10k-100k training examples
- [x] Train first task classifier
- **Target:** 5% independence (trivial queries)

### Month 1-3: Task Mastery
- [ ] Cluster tasks into 10-50 families
- [ ] Train specialist model per cluster
- [ ] Implement confidence-based routing
- **Target:** 30% independence (common patterns)

### Month 3-6: Domain Adaptation
- [ ] Auto-detect deployment industry
- [ ] Fine-tune on domain vocabulary
- [ ] Add industry-specific safety layers
- **Target:** 60% independence (domain expert)

### Month 6-12: Full Independence
- [ ] Train 1B-param distilled model on full corpus
- [ ] External LLM only for novelty/edge cases
- [ ] Continuous evolution running
- **Target:** 90%+ independence (self-sustaining)

## Cost Savings Projection

| Phase | Independence | Monthly Cost | Savings |
|-------|--------------|--------------|---------|
| Baseline | 0% | $2,000 | - |
| Phase 1 | 5% | $1,900 | 5% |
| Phase 2 | 30% | $1,400 | 30% |
| Phase 3 | 60% | $800 | 60% |
| Phase 4 | 90% | $200 | 90% |

**Annual Savings:** ~$22,000 after 12 months

## Technical Details

### Training Data Schema
```sql
CREATE TABLE training_corpus (
    id              INTEGER PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    task_vector     BLOB NOT NULL,        -- 384-dim embedding
    input_text      TEXT NOT NULL,
    output_text     TEXT NOT NULL,
    external_model  TEXT NOT NULL,
    reward          REAL NOT NULL,        -- 0.0 - 1.0
    cluster_id      INTEGER,
    used_for_training BOOLEAN DEFAULT 0
);
```

### Model Registry Schema
```json
{
  "models": [
    {
      "id": 1,
      "name": "cluster_5_v1234567890",
      "config": {
        "architecture": "seq2seq",
        "hidden_size": 256,
        "num_layers": 4,
        "task_clusters": [5, 12]
      },
      "weights_path": "models/checkpoints/cluster_5_v1234567890.pt",
      "accuracy": 0.92,
      "size_mb": 120.5,
      "avg_latency_ms": 85,
      "active": true
    }
  ]
}
```

### Production Training Pipeline

For production deployment, the model training would use:

```python
# 1. Load pre-trained base model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# 2. Create dataset from training corpus
from torch.utils.data import Dataset, DataLoader

class DistillationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = tokenizer(ex.input_text, return_tensors="pt", padding=True)
        outputs = tokenizer(ex.output_text, return_tensors="pt", padding=True)
        return {"input_ids": inputs.input_ids, "labels": outputs.input_ids}

# 3. Fine-tune with knowledge distillation
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./models/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## Monitoring & Observability

Key metrics to track:
- **Corpus Size:** Number of training examples collected
- **Model Count:** Active self-models in production
- **Independence %:** Requests handled by self-models
- **Accuracy:** Self-model accuracy vs external baseline
- **Cost Savings:** Monthly savings from local inference
- **Latency:** Self-model response time vs external APIs

Dashboard visualization in `/learning/stats` endpoint.

## Safety & Fallback

- **Confidence Thresholds:** Only use self-model when confidence > threshold
- **External Escape Hatch:** Always keep external LLM available for fallback
- **Quality Monitoring:** Continuous evaluation against external baseline
- **Gradual Rollout:** A/B testing before full deployment
- **Rollback:** Instant deactivation if accuracy drops

## Future Enhancements

- **Active Learning:** Intelligently select which examples to label
- **Federated Learning:** Train on distributed data (privacy-preserving)
- **Meta-Learning:** Fast adaptation to new tasks (few-shot)
- **Neural Architecture Search:** Automatically discover optimal architectures
- **Continual Learning:** Learn new tasks without forgetting old ones
