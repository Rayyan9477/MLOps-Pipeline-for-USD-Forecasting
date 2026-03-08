# Production Deployment

## GitHub Secrets Setup

Configure these in your repository: **Settings > Secrets and variables > Actions > New repository secret**

### Required
| Secret | Description | Source |
|--------|-------------|--------|
| `TWELVE_DATA_API_KEY` | Forex data API key | [twelvedata.com](https://twelvedata.com/register) (free: 800/day) |

### Optional (for full MLOps stack)
| Secret | Description |
|--------|-------------|
| `MLFLOW_TRACKING_URI` | MLflow server (e.g., `https://dagshub.com/user/repo.mlflow`) |
| `MLFLOW_TRACKING_USERNAME` | MLflow auth username |
| `MLFLOW_TRACKING_PASSWORD` | MLflow auth password / DagsHub token |
| `DAGSHUB_TOKEN` | DagsHub API token (for DVC remote) |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or access token |

---

## Deployment Options

### Vercel (Recommended)
Best for portfolio hosting. Zero-ops, automatic HTTPS, global CDN.

```bash
npm i -g vercel && vercel login && vercel --prod
vercel env add TWELVE_DATA_API_KEY
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details.

### Railway
Full Docker support with database hosting.

```bash
npm i -g @railway/cli && railway login && railway init && railway up
railway variables set TWELVE_DATA_API_KEY=your_key
```

### Docker (Self-Hosted)
```bash
docker build -t usd-volatility-predictor .
docker run -p 8000:8000 --env-file .env usd-volatility-predictor
```

---

## Automated Updates (Every 2 Hours)

**GitHub Actions Cron:**
- Workflow: `.github/workflows/data-pipeline.yml`
- Schedule: `0 */2 * * *`
- Steps: Fetch > Transform > Train ensemble > Commit artifacts

**Manual trigger:** Actions tab > Data Pipeline > Run workflow

---

## Model Performance Standards

### Current Model: XGBoost / Stacking Ensemble

**Architecture:**
- Single model: XGBoost with early stopping
- Ensemble: XGBoost + Random Forest + Gradient Boosting + Ridge meta-learner
- Cross-validation: `TimeSeriesSplit` (no future leakage)

**Performance targets:**
- R-squared > 0.65
- MAPE < 25%
- RMSE minimized for volatility scale

**Current metrics:** Check `models/latest_metadata.json`

---

## Health Monitoring

```bash
# API health (includes model status)
curl https://your-domain.com/health

# Prediction stats
curl https://your-domain.com/api/stats

# Prometheus metrics
curl https://your-domain.com/metrics

# Interactive dashboard
open https://your-domain.com/dashboard
```

---

## Security Checklist

- [x] No hardcoded credentials — all via env vars / GitHub Secrets
- [x] `.env` files gitignored
- [x] Model file committed (87KB, no secrets)
- [x] CORS middleware configured
- [x] Non-root user in Docker container
- [x] HTTPS enforced by deployment platforms
- [x] API key rotation supported via env var updates

---

## Maintenance

### Weekly
- Review model performance metrics
- Check GitHub Actions workflow logs
- Verify cron execution history

### Monthly
- Rotate API keys if needed
- Review dependency updates
- Check Twelve Data API usage against limits

---

**Last Updated:** March 2026
**Status:** Production Ready
