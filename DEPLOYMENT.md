# SentinelMesh Deployment Guide

**Complete deployment guide for all platforms - from local to production**

---

## 🚀 Quick Start Options

Choose your deployment method:

| Method | Use Case | Setup Time | Difficulty |
|--------|----------|------------|------------|
| **Docker Compose** | Local development | 5 minutes | Easy |
| **Railway** | Quick public demo | 10 minutes | Easiest |
| **Render** | Production ready | 15 minutes | Easy |
| **AWS ECS** | Enterprise scale | 30 minutes | Medium |
| **Kubernetes** | Maximum scale | 60 minutes | Advanced |
| **Self-Hosted** | Full control | 20 minutes | Medium |

---

## 1. Docker Compose (Recommended for Development)

### Prerequisites
- Docker & Docker Compose installed
- At least one LLM API key

### Steps

```bash
# 1. Clone/extract SentinelMesh
cd SentinelMeshFixed

# 2. Create environment file
cat > .env << 'EOF'
# LLM Provider Keys (at least one required)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-google-key

# Database Password (change in production!)
POSTGRES_PASSWORD=your-secure-password-here

# Optional: Additional providers
# COHERE_API_KEY=
# HUGGINGFACE_API_KEY=

# Optional: Voice providers
# ELEVENLABS_API_KEY=
# AZURE_SPEECH_KEY=
EOF

# 3. Start all services
docker-compose up -d

# 4. Check logs
docker-compose logs -f sentinelmesh

# 5. Verify health
curl http://localhost:8000/health

# 6. Access API
curl http://localhost:8000/docs
```

### Access Points
- **API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Management Commands

```bash
# Stop services
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Update
git pull  # or re-extract
docker-compose build
docker-compose up -d

# Backup data
docker run --rm -v sentinelmeshfixed_sentinelmesh-data:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/sentinelmesh-backup.tar.gz /data
```

---

## 2. Railway (Easiest Public Deployment)

Railway provides **one-click deployment** with automatic HTTPS!

### Steps

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
cd SentinelMeshFixed
railway init

# 4. Add environment variables
railway variables set OPENAI_API_KEY=sk-your-key
railway variables set ANTHROPIC_API_KEY=sk-ant-your-key
railway variables set DB_TYPE=postgres
railway variables set REDIS_ENABLED=true

# 5. Deploy
railway up

# 6. Get public URL
railway domain
```

**Your app will be live at:** `https://your-app.railway.app`

### Features
- ✅ Automatic HTTPS
- ✅ Managed PostgreSQL database
- ✅ Managed Redis cache
- ✅ Auto-scaling
- ✅ Zero-downtime deployments

**Cost:** Free tier available, ~$20/month for production

---

## 3. Render (Production-Ready)

Render provides managed infrastructure with excellent reliability.

### Steps

1. **Push to GitHub/GitLab**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

2. **Create Render Account**
   - Go to https://render.com
   - Sign up / Login

3. **Create Services**
   
   **a) PostgreSQL Database:**
   - Click "New +" → "PostgreSQL"
   - Name: sentinelmesh-db
   - Plan: Free (or paid)
   - Create Database
   - **Save the connection string!**

   **b) Redis:**
   - Click "New +" → "Redis"
   - Name: sentinelmesh-redis
   - Create
   - **Save the connection string!**

   **c) Web Service:**
   - Click "New +" → "Web Service"
   - Connect your repository
   - Name: sentinelmesh
   - Environment: Docker
   - Plan: Starter ($7/mo) or higher
   - Add Environment Variables:
     ```
     OPENAI_API_KEY=sk-your-key
     ANTHROPIC_API_KEY=sk-ant-your-key
     DATABASE_URL=[postgres connection string]
     REDIS_URL=[redis connection string]
     DB_TYPE=postgres
     REDIS_ENABLED=true
     SENTINELMESH_ENV=production
     ```
   - Create Web Service

4. **Access Your App**
   - URL: `https://sentinelmesh.onrender.com`
   - Render provides automatic HTTPS!

### Features
- ✅ Automatic HTTPS/SSL
- ✅ Managed database
- ✅ Auto-deploy from git
- ✅ Health checks
- ✅ Built-in monitoring

**Cost:** ~$7/month (Starter) to $25/month (Standard)

---

## 4. AWS ECS (Enterprise Scale)

For enterprise deployments with auto-scaling and high availability.

### Prerequisites
- AWS Account
- AWS CLI installed
- Docker installed

### Quick Deploy Script

```bash
# Install AWS Copilot (easiest way)
brew install aws/tap/copilot-cli  # macOS
# or
curl -Lo copilot https://github.com/aws/copilot-cli/releases/latest/download/copilot-linux && chmod +x copilot

# Initialize application
copilot app init sentinelmesh

# Create environment
copilot env init --name production \
  --profile default \
  --default-config

# Create service
copilot svc init --name api \
  --svc-type "Load Balanced Web Service" \
  --dockerfile ./Dockerfile

# Add secrets
copilot secret init --name OPENAI_API_KEY
copilot secret init --name ANTHROPIC_API_KEY

# Deploy
copilot svc deploy --name api --env production
```

### Features
- ✅ Auto-scaling (0-100+ instances)
- ✅ Load balancing
- ✅ High availability (multi-AZ)
- ✅ Managed secrets
- ✅ CloudWatch monitoring

**Cost:** ~$50-200/month (depends on usage)

---

## 5. Google Cloud Run (Serverless)

Serverless deployment that scales to zero.

### Steps

```bash
# 1. Install gcloud CLI
# Follow: https://cloud.google.com/sdk/docs/install

# 2. Login
gcloud auth login

# 3. Set project
gcloud config set project YOUR_PROJECT_ID

# 4. Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/sentinelmesh

# 5. Deploy
gcloud run deploy sentinelmesh \
  --image gcr.io/YOUR_PROJECT_ID/sentinelmesh \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-your-key,SENTINELMESH_ENV=production

# 6. Get URL
gcloud run services describe sentinelmesh --region us-central1 --format 'value(status.url)'
```

### Features
- ✅ Auto-scaling (including to zero)
- ✅ Pay per request
- ✅ Automatic HTTPS
- ✅ Global CDN

**Cost:** Free tier, then ~$0.40 per million requests

---

## 6. Kubernetes (Maximum Scale)

For organizations needing ultimate control and scale.

### Prerequisites
- Kubernetes cluster (EKS, GKE, AKS, or self-hosted)
- kubectl installed
- Helm installed

### Deploy with Helm

```bash
# 1. Create namespace
kubectl create namespace sentinelmesh

# 2. Create secrets
kubectl create secret generic sentinelmesh-secrets \
  --from-literal=openai-api-key=sk-your-key \
  --from-literal=anthropic-api-key=sk-ant-your-key \
  -n sentinelmesh

# 3. Deploy PostgreSQL
helm install postgres bitnami/postgresql \
  --set auth.postgresPassword=your-password \
  --set auth.database=sentinelmesh \
  -n sentinelmesh

# 4. Deploy Redis
helm install redis bitnami/redis \
  -n sentinelmesh

# 5. Deploy SentinelMesh
cat > values.yaml << 'EOF'
replicaCount: 3
image:
  repository: your-registry/sentinelmesh
  tag: latest
service:
  type: LoadBalancer
  port: 80
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: sentinelmesh-secrets
        key: openai-api-key
  - name: DB_TYPE
    value: "postgres"
  - name: DB_HOST
    value: "postgres-postgresql"
EOF

helm install sentinelmesh ./helm -f values.yaml -n sentinelmesh

# 6. Get external IP
kubectl get svc sentinelmesh -n sentinelmesh
```

### Features
- ✅ Horizontal auto-scaling
- ✅ Self-healing
- ✅ Rolling updates
- ✅ Multi-region deployment
- ✅ Service mesh integration

---

## 7. Self-Hosted (VPS/Bare Metal)

For complete control on your own infrastructure.

### Prerequisites
- Ubuntu 22.04 or later
- 4GB RAM minimum (8GB recommended)
- Root access

### Installation Script

```bash
#!/bin/bash

# 1. Update system
apt-get update && apt-get upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 4. Clone SentinelMesh
git clone YOUR_REPO_URL /opt/sentinelmesh
cd /opt/sentinelmesh

# 5. Configure
cp .env.example .env
nano .env  # Edit API keys

# 6. Start services
docker-compose up -d

# 7. Setup Nginx (optional but recommended)
apt-get install -y nginx certbot python3-certbot-nginx

# 8. Configure Nginx
cat > /etc/nginx/sites-available/sentinelmesh << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

ln -s /etc/nginx/sites-available/sentinelmesh /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# 9. Setup SSL (Let's Encrypt)
certbot --nginx -d your-domain.com

# 10. Setup systemd service
cat > /etc/systemd/system/sentinelmesh.service << 'EOF'
[Unit]
Description=SentinelMesh AI Operating System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/sentinelmesh
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
EOF

systemctl enable sentinelmesh
systemctl start sentinelmesh

echo "✅ SentinelMesh installed and running!"
echo "Access at: https://your-domain.com"
```

---

## 8. Vercel/Netlify (API-only)

While these platforms are primarily for static sites, you can deploy SentinelMesh as serverless functions.

**Note:** Not recommended for full deployment due to cold starts and limitations.

---

## Configuration for Production

### Essential Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key
SENTINELMESH_ENV=production

# Database (recommended: PostgreSQL)
DB_TYPE=postgres
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=sentinelmesh
DB_USER=sentinelmesh
DB_PASSWORD=secure-password

# Redis (recommended)
REDIS_ENABLED=true
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# Security
JWT_SECRET=your-secret-key-min-32-chars
API_KEYS_ENABLED=true

# Performance
API_WORKERS=4
CACHE_SIMILARITY_THRESHOLD=0.95

# Safety
ENABLE_GUARDRAILS=true
AUTO_REDACT_PII=true
MIN_SAFETY_SCORE=0.7

# Logging
LOG_LEVEL=INFO
```

### Scaling Recommendations

| Users | API Workers | RAM | CPU | Database |
|-------|-------------|-----|-----|----------|
| <100 | 2 | 2GB | 1 core | SQLite |
| 100-1K | 4 | 4GB | 2 cores | Postgres |
| 1K-10K | 8 | 8GB | 4 cores | Postgres + Redis |
| 10K-100K | 16+ | 16GB+ | 8+ cores | Postgres (managed) + Redis |
| 100K+ | Auto-scale | 32GB+ | 16+ cores | Multi-region Postgres + Redis Cluster |

---

## Monitoring & Observability

### Health Checks

```bash
# Basic health
curl https://your-domain.com/health

# Detailed status
curl https://your-domain.com/status

# Metrics (if enabled)
curl https://your-domain.com/metrics
```

### Logging

SentinelMesh logs to:
- Console (stdout)
- File (`logs/sentinelmesh.log`)
- External systems (Datadog, Sentry, etc. via integrations)

### Monitoring Integration

**Datadog:**
```bash
DD_API_KEY=your-key docker-compose up -d
```

**Prometheus:**
```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

---

## Security Checklist

- [ ] Change default passwords
- [ ] Enable HTTPS/SSL
- [ ] Set strong JWT_SECRET
- [ ] Enable API key authentication
- [ ] Configure CORS appropriately
- [ ] Enable guardrails
- [ ] Set up firewall rules
- [ ] Regular backups
- [ ] Monitor logs
- [ ] Update regularly

---

## Backup & Disaster Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup database
docker exec sentinelmesh-postgres pg_dump -U sentinelmesh sentinelmesh | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup data volume
docker run --rm -v sentinelmeshfixed_sentinelmesh-data:/data \
  -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/data_$DATE.tar.gz /data

# Backup configuration
cp .env $BACKUP_DIR/env_$DATE

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

Add to crontab:
```bash
# Daily backup at 2 AM
0 2 * * * /opt/sentinelmesh/backup.sh
```

---

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Change port in docker-compose.yml or .env
API_PORT=8001
```

**Database connection failed:**
```bash
# Check database is running
docker-compose ps
# Check credentials
docker-compose logs postgres
```

**Out of memory:**
```bash
# Reduce workers
API_WORKERS=2
# Or increase Docker memory limit
```

---

## Support & Resources

- **Documentation:** See all `*_GUIDE.md` files
- **Issues:** GitHub Issues (if open source)
- **Community:** Discord/Slack (if available)
- **Enterprise Support:** contact@sentinelmesh.ai

---

## Next Steps

After deployment:

1. ✅ Test all endpoints: `/docs`
2. ✅ Configure integrations
3. ✅ Set up monitoring
4. ✅ Enable backups
5. ✅ Review security settings
6. ✅ Train your team
7. ✅ Launch! 🚀

**You now have a production-ready AI Operating System!**
