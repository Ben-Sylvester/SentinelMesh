# SentinelMesh v3.0 - Easy Deployment Guide

**Deploy in 5 minutes or less!**

---

## 🎯 Quickest Options (Choose One)

### Option 1: One-Click Railway (EASIEST ⭐)

1. Click: [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)
2. Select "Deploy from GitHub repo"
3. Enter: `https://github.com/yourusername/sentinelmesh`
4. Add environment variable: `OPENAI_API_KEY=your-key`
5. Click "Deploy"
6. Done! Get your URL: `https://yourapp.railway.app`

**Time:** 3 minutes  
**Cost:** Free tier, then $5/month  
**Perfect for:** Testing, demos, small projects

---

### Option 2: Local Quick Start

```bash
# One command:
curl -fsSL https://raw.githubusercontent.com/yourusername/sentinelmesh/main/quickstart.sh | bash
```

This will:
- Install dependencies
- Setup configuration
- Start server on http://localhost:8000

**Time:** 5 minutes  
**Cost:** Free  
**Perfect for:** Development, learning

---

### Option 3: Docker (ONE COMMAND)

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  sentinelmesh/sentinelmesh:latest
```

**Time:** 2 minutes  
**Cost:** Free  
**Perfect for:** Production, any environment

---

## 🔥 Production Deployment

### Cloud Run (Google Cloud) - Recommended

```bash
gcloud run deploy sentinelmesh \
  --source . \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key
```

**Benefits:**
- Auto-scaling (0 to 1000+ instances)
- Pay per request
- Global CDN
- SSL included

**Cost:** ~$0.10 per 100K requests

---

### Fly.io (Developer Favorite)

```bash
# Install
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly secrets set OPENAI_API_KEY=your-key
fly deploy
```

**Benefits:**
- Global edge deployment
- Auto-scaling
- Zero-downtime deploys

**Cost:** Free 3 VMs, then $1.94/month each

---

## 🌐 Public Access Setup

### Make Your Instance Public

**Option 1: Use Ngrok (Testing)**
```bash
# Start server locally
uvicorn app:app

# In another terminal
ngrok http 8000

# Share URL: https://random-string.ngrok.io
```

**Option 2: Custom Domain**

Most cloud platforms provide:
1. Auto-generated URL: `https://yourapp.platform.com`
2. Custom domain support: Point your domain DNS

---

## 📱 For Developers

### GitHub Codespaces (Free 60 hours/month)

1. Go to: https://github.com/yourusername/sentinelmesh
2. Click: Code → Codespaces → New codespace
3. Wait 2 minutes for setup
4. Port 8000 auto-exposed
5. Start coding!

---

### Replit (Instant IDE)

1. Go to: https://replit.com
2. Import from GitHub
3. Set secrets (API keys)
4. Click "Run"
5. Get instant URL

---

## 🏢 Enterprise Options

### Kubernetes (Scale to Millions)

```bash
# Quick deploy
kubectl apply -f https://raw.githubusercontent.com/yourusername/sentinelmesh/main/k8s/all-in-one.yaml

# Set secrets
kubectl create secret generic sentinelmesh-secrets \
  --from-literal=openai-api-key=your-key
```

### AWS ECS (Managed Containers)

```bash
# Use AWS Copilot
copilot init --app sentinelmesh
copilot deploy
```

---

## ✅ Post-Deployment Checklist

After deploying, test these URLs:

```bash
# Health check
curl https://your-url.com/health

# API docs
https://your-url.com/docs

# Test request
curl -X POST https://your-url.com/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'
```

**Expected:** All return 200 OK

---

## 🎓 Configuration Examples

### Minimal (Quick Start)

```yaml
# config.yaml
core:
  environment: production
  port: 8000

api:
  openai_api_key: ${OPENAI_API_KEY}

features:
  enable_memory: true
  enable_cache: true
```

### Full Power (All Features)

```yaml
# config.yaml
features:
  enable_memory: true
  enable_streaming: true
  enable_vision: true
  enable_cache: true
  enable_functions: true
  enable_prompt_library: true
  enable_workflows: true
  enable_integrations: true
  enable_guardrails: true
```

---

## 🆘 Troubleshooting

### Common Issues

**"Port already in use"**
```bash
# Change port
uvicorn app:app --port 8001
```

**"Module not found"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**"API key invalid"**
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY
```

---

## 📞 Get Help

- **Quick questions:** https://discord.gg/sentinelmesh
- **Bug reports:** https://github.com/yourusername/sentinelmesh/issues
- **Documentation:** https://docs.sentinelmesh.ai

---

## 🚀 Recommended Path

**For Testing:**
1. Railway (one-click)
2. Test all features
3. Get feedback

**For Production:**
1. Docker + Cloud Run
2. Enable monitoring
3. Set up CI/CD
4. Scale as needed

---

**You're ready to deploy! Pick an option above and go!** 🎉
