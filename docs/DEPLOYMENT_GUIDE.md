# Deployment Guide

## Supported Platforms

| Platform | Best For | Cost |
|----------|----------|------|
| **Vercel** (recommended) | Serverless API, zero-ops | Free tier available |
| **Railway** | Full-stack with Docker + DB | $5/month credit |
| **Render** | Docker with cron jobs | Free tier (750h/month) |
| **Docker (self-hosted)** | Full control, local dev | Infrastructure cost |

---

## Vercel Deployment

Vercel is the recommended platform. The app auto-detects serverless mode and disables filesystem-dependent features (SQLite, file logging).

### Prerequisites
- Node.js (for Vercel CLI)
- A [Twelve Data API key](https://twelvedata.com/register) (free: 800 calls/day)

### Steps

```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Login
vercel login

# 3. Deploy (preview)
vercel

# 4. Deploy to production
vercel --prod

# 5. Set environment variables
vercel env add TWELVE_DATA_API_KEY
```

### What happens on Vercel

- `requirements.txt` (slim, ~150MB) is installed — only API-serving dependencies
- The `VERCEL=1` env var is set automatically, enabling serverless mode:
  - SQLite database writes are skipped
  - File logging falls back to console-only
  - Prediction history served from in-memory deque
  - Infrastructure links (Grafana, Prometheus, etc.) hidden from dashboard
- The bundled model (`models/latest_model.pkl`, 87KB) is loaded on cold start
- Max lambda: 100MB, memory: 1024MB, timeout: 60s

### Vercel Configuration

The project includes `vercel.json`:

```json
{
  "builds": [{
    "src": "src/api/main.py",
    "use": "@vercel/python",
    "config": { "maxLambdaSize": "100mb" }
  }],
  "routes": [
    { "src": "/static/(.*)", "dest": "src/ui/static/$1" },
    { "src": "/(.*)", "dest": "src/api/main.py" }
  ],
  "functions": {
    "src/api/main.py": { "memory": 1024, "maxDuration": 60 }
  }
}
```

---

## Railway Deployment

Railway supports Docker and provides database hosting.

```bash
# 1. Install CLI
npm i -g @railway/cli

# 2. Login and initialize
railway login
railway init

# 3. Deploy
railway up

# 4. Set environment variables
railway variables set TWELVE_DATA_API_KEY=your_key
```

Railway uses the `Dockerfile` and `requirements-full.txt` (full dependencies).

**Config file:** `railway.json` is included in the repo.

---

## Render Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" > "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Environment:** Docker
   - **Plan:** Free or paid
5. Add environment variables from `.env.example`
6. Optionally add a Cron Job: `0 */2 * * *` > `python src/models/production_trainer.py`

**Config file:** `render.yaml` is included in the repo.

---

## Docker Deployment (Self-Hosted)

```bash
# Build (uses requirements-full.txt with all dependencies)
docker build -t usd-volatility-predictor .

# Run
docker run -d \
  --name usd-api \
  -p 8000:8000 \
  --env-file .env \
  usd-volatility-predictor

# Verify
curl http://localhost:8000/health
```

### Docker Compose (Full Stack)

Starts all 8 services: FastAPI, Airflow, MLflow, PostgreSQL, MinIO, Prometheus, Grafana, and the API.

```bash
docker-compose up -d
docker-compose ps      # Check status
docker-compose logs -f # View logs
```

| Service | Port | Credentials |
|---------|------|-------------|
| FastAPI | 8000 | -- |
| Airflow | 8080 | airflow / airflow |
| MLflow | 5000 | -- |
| Grafana | 3000 | admin / admin |
| Prometheus | 9090 | -- |
| MinIO | 9001 | minioadmin / minioadmin |

---

## Post-Deployment Verification

```bash
# 1. Health check
curl https://your-domain.com/health
# Expected: {"status": "healthy", "model_loaded": true, ...}

# 2. Test prediction
curl -X POST https://your-domain.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"close_lag_1": 1.0854, "close_rolling_mean_24": 1.085, "close_rolling_std_24": 0.0015, "log_return": 0.0002, "hour_sin": 0.5, "hour_cos": 0.866}}'

# 3. Dashboard
open https://your-domain.com/dashboard

# 4. Prometheus metrics
curl https://your-domain.com/metrics
```

---

## Automated Data Updates

The pipeline updates every 2 hours via GitHub Actions:

- **Workflow:** `.github/workflows/data-pipeline.yml`
- **Schedule:** `0 */2 * * *` (00:00, 02:00, ..., 22:00 UTC)
- **Steps:** Fetch data > Transform > Train > Commit model artifacts

**Manual trigger:** Actions tab > Data Pipeline > Run workflow

---

## Requirements Files

| File | Contents | Use Case |
|------|----------|----------|
| `requirements.txt` | 11 packages (fastapi, xgboost, pandas, etc.) | Vercel serverless |
| `requirements-full.txt` | 24 packages (adds mlflow, dvc, pytest, boto3, etc.) | Docker / local dev |

The Dockerfile references `requirements-full.txt`. Vercel reads `requirements.txt` from root.

---

## Troubleshooting

### Model not loading
```bash
# Check model exists
ls -la models/latest_model.pkl models/latest_metadata.json

# Pull from DVC (if using DVC)
dvc pull models/latest_model.pkl.dvc
```

### High cold start on Vercel
- Ensure `requirements.txt` stays slim (< 250MB installed)
- The current slim requirements install at ~150MB
- Model is 87KB — negligible impact

### API returns 503
- Model failed to load. Check logs for file path errors
- Verify `models/latest_model.pkl` is committed to git (not gitignored)

### Docker build fails
- Ensure `requirements-full.txt` exists (renamed from old `requirements.txt`)
- Check that `libgomp1` is installed (needed for XGBoost)

---

**Last Updated:** March 2026
