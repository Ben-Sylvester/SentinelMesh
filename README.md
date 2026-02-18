# SentinelMesh 🧠

**An intelligent, self-learning AI orchestration system that progressively becomes independent from external LLM providers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB+ RAM
- (Optional) CUDA-capable GPU for model training

### Installation

```bash
# 1. Clone or extract the repository
cd SentinelMeshFixed

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# 5. Start the server
uvicorn app:app --reload

# Server runs on: http://localhost:8000
```

### First Request

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the benefits of microservices?"}'
```

---
#Powershell
---bash
curl.exe -X POST http://localhost:8000/run `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"What are the benefits of microservices?\"}"
---

---



## ✨ Features

### Intelligent Routing & Learning
1. Multi-Armed Bandit (LinUCB) for adaptive strategy selection  
2. Reinforcement Learning (Q-learning) for long-term optimization  
3. World Model with task signature recognition  
4. Automatic escalation on failures  
5. Retrieval-Augmented Generation (RAG)  

### Self-Learning & Independence
🧠 **Continuous Learning** from every interaction  
🏗️ **Self-Model Building** via knowledge distillation  
🔄 **Model Evolution** through pruning/quantization/merging  
🔌 **Domain Adaptation** (healthcare, finance, legal, etc.)  
📈 **Progressive Independence:** 0% → 95% self-sufficiency  
💰 **Cost Savings:** Up to 90% reduction vs external APIs  

### Production-Ready
🔒 Multi-tenant with API key auth  
⚡ Rate limiting & budget controls  
💾 Thread-safe SQLite persistence  
🔄 Async-safe operations  
📊 Real-time WebSocket feed  
📈 Comprehensive metrics  

---

## 📚 Documentation

- **[README.md](README.md)** — Quick start & overview (this file)
- **[SELF_LEARNING.md](SELF_LEARNING.md)** — Complete self-learning system guide
- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** — Detailed architecture documentation

---

## 🏗️ Architecture Overview

```
User Request → Router → [Independence Scheduler]
                    ↓                      ↓
            Self-Model (Local)    External LLM (API)
                    ↓                      ↓
            Learning System (collects data, trains models)
                    ↓
            Persistence (SQLite)
```

---

## 🔧 Core Components

| Component | Purpose |
|-----------|---------|
| **Router** | Orchestrates routing decisions |
| **Contextual Bandit** | LinUCB for exploration/exploitation |
| **RL Policy** | Q-learning for long-term optimization |
| **World Model** | Task pattern recognition |
| **Continuous Learner** | Collects training data |
| **Model Builder** | Trains distilled models |
| **Domain Adapter** | Industry detection & adaptation |
| **Independence Scheduler** | Manages self-sufficiency transition |
| **Evolution Engine** | Model optimization |

---

## 📡 API Endpoints

### Core
- `POST /run` — Main orchestration endpoint
- `POST /query` — Simplified query
- `GET /stats` — System statistics

### Learning
- `GET /learning/stats` — Learning system status
- `POST /learning/trigger-training` — Manual training
- `POST /learning/evolve-models` — Evolutionary optimization
- `GET /learning/independence-progress` — Maturity metrics
- `GET /learning/domain-detection` — Industry detection

### Admin
- `POST /admin/create-tenant` — Create tenant
- `GET /admin/rl-stats` — RL policy statistics

### Metrics
- `GET /metrics/beliefs` — World model beliefs
- `GET /metrics/strategy-drift` — Strategy trends
- `GET /metrics/roi` — Model ROI

### WebSocket
- `WS /ws` — Real-time trace feed

---

## 🧠 Self-Learning System

### 5-Level Maturity Model

| Level | Name | Independence | Description |
|-------|------|--------------|-------------|
| 0 | Bootstrap | 0% | 100% external LLM |
| 1 | Learning | 20% | Simple tasks only |
| 2 | Competent | 50% | Common patterns mastered |
| 3 | Proficient | 80% | Edge cases external |
| 4 | Expert | 95% | Fully independent |

### Cost Savings Projection

| Phase | Duration | Independence | Monthly Cost | Savings |
|-------|----------|--------------|--------------|---------|
| Baseline | - | 0% | $2,000 | - |
| Phase 1 | Week 1-2 | 5% | $1,900 | 5% |
| Phase 2 | Month 1-3 | 30% | $1,400 | 30% |
| Phase 3 | Month 3-6 | 60% | $800 | 60% |
| Phase 4 | Month 6-12 | 90% | $200 | 90% |

**Annual Savings:** ~$22,000

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

```bash
docker build -t sentinelmesh .
docker run -p 8000:8000 --env-file .env sentinelmesh
```

### Production Checklist

- [ ] Set strong `ADMIN_API_KEY`
- [ ] Configure Redis for rate limiting
- [ ] Set up GPU for training (optional)
- [ ] Configure backup for `data/` directory
- [ ] Set up monitoring
- [ ] Enable HTTPS
- [ ] Configure log rotation

---

## 📊 Monitoring

### Key Metrics
- Request throughput & latency
- Independence percentage
- Cost savings
- Self-model accuracy
- Training corpus size

### Dashboards
```bash
# Admin dashboard
streamlit run dashboard/app.py

# Customer dashboard
streamlit run dashboard/customer_app.py

# Live intelligence
streamlit run dashboard.py
```

---

## 🐛 Troubleshooting

**Missing dependencies:**
```bash
pip install sentence-transformers scikit-learn torch transformers
```

**Database locked:**
```bash
# Verify WAL mode
sqlite3 data/learning_state.db "PRAGMA journal_mode;"
```

**Debug mode:**
```bash
export LOG_LEVEL=DEBUG
uvicorn app:app --log-level debug
```

---

## 🤝 Contributing

```bash
# Install dev dependencies
pip install pytest black mypy

# Run tests
pytest tests/

# Format code
black core/ adapters/ app.py
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

Built with FastAPI, PyTorch, sentence-transformers, scikit-learn, Streamlit

---

**See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for complete technical documentation.**
