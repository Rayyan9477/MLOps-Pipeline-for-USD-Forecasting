# USD Volatility Prediction — Real-Time MLOps Pipeline

> Production-grade MLOps pipeline for EUR/USD forex volatility forecasting. Live API, automated retraining every 2 hours, full observability stack, and one-click deployment to Vercel.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Deploy](https://img.shields.io/badge/Deploy-Vercel%20%7C%20Railway%20%7C%20Render-blueviolet.svg)](docs/DEPLOYMENT_GUIDE.md)

---

## What This Does

Predicts the **next-hour volatility** of the EUR/USD forex pair using an XGBoost model trained on 30+ engineered features (lag prices, rolling statistics, cyclical time encodings, price dynamics). The system fetches live data from the Twelve Data API, retrains automatically, serves predictions via a FastAPI REST API, and monitors for data drift — all wired together with CI/CD, experiment tracking, and containerization.

**Live endpoints** (after deployment):
- `GET /health` — service health + model status
- `POST /predict` — volatility prediction with drift detection
- `GET /dashboard` — interactive monitoring UI
- `GET /api/stats` — model metrics and prediction stats
- `GET /metrics` — Prometheus scrape endpoint

---

## Architecture

```
Data Source          Pipeline              Model                 Serving & Monitoring
───────────         ──────────            ─────                 ────────────────────

Twelve Data    ──>  Extract    ──>   XGBoost Trainer  ──>   FastAPI API
(EUR/USD 1h)        (quality           (30+ features,         /predict
                     checks)            TimeSeriesSplit,       /health
                        |               early stopping)        /dashboard
                        v                     |                    |
                   Transform                  v                    v
                   (lag, rolling,        MLflow Registry      Prometheus
                    cyclical,           (experiment            + Grafana
                    price feats)         tracking)            (alerting)
                        |                     |                    |
                        v                     v                    v
                   DVC Versioning       Model Artifacts       Drift Detection
                   (DagsHub remote)     (latest_model.pkl)    (z-score based)
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **XGBoost over deep learning** | Faster training, better on tabular data with < 10K rows, easier to deploy (87KB model) |
| **Volatility as target** | More actionable than raw price prediction; models regime changes in risk |
| **Serverless + Docker dual-mode** | Vercel for zero-ops portfolio hosting; Docker Compose for full local dev stack |
| **Thread-safe API state** | `threading.Lock` protects shared deques and counters under concurrent requests |
| **Feature exclusion guards** | `volatility` excluded from features to prevent target leakage |

---

## Quick Start

### Option A: Local Development (Full Stack)

```bash
# Clone and setup
git clone https://github.com/Rayyan9477/Real-Time-MLOps-Pipeline-for-USD-Forecasting.git
cd Real-Time-MLOps-Pipeline-for-USD-Forecasting
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt

# Configure (set your Twelve Data API key)
cp .env.example .env
# Edit .env with your TWELVE_DATA_API_KEY

# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Open dashboard: http://localhost:8000/dashboard
# API docs:      http://localhost:8000/docs
```

### Option B: Docker Compose (All Services)

```bash
docker-compose up -d

# Services available:
# FastAPI     http://localhost:8000
# Airflow     http://localhost:8080  (airflow/airflow)
# MLflow      http://localhost:5000
# Grafana     http://localhost:3000  (admin/admin)
# Prometheus  http://localhost:9090
# MinIO       http://localhost:9001  (minioadmin/minioadmin)
```

### Option C: Deploy to Vercel (Production)

```bash
npm i -g vercel
vercel login
vercel --prod
# Set env vars: vercel env add TWELVE_DATA_API_KEY
```

The app runs in serverless mode on Vercel — SQLite and filesystem writes are disabled, the bundled model serves predictions from memory.

---

## Making Predictions

```bash
# Health check
curl http://localhost:8000/health

# Predict next-hour volatility
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "close_lag_1": 1.0854, "close_lag_2": 1.0851,
      "close_lag_3": 1.0849, "close_lag_4": 1.0847,
      "close_lag_6": 1.0843, "close_lag_8": 1.0840,
      "close_lag_12": 1.0835, "close_lag_24": 1.0820,
      "close_rolling_mean_4": 1.0850, "close_rolling_mean_8": 1.0848,
      "close_rolling_mean_24": 1.0840,
      "close_rolling_std_4": 0.0015, "close_rolling_std_8": 0.0018,
      "close_rolling_std_24": 0.0020,
      "close_rolling_min_4": 1.0840, "close_rolling_max_4": 1.0860,
      "close_rolling_min_8": 1.0835, "close_rolling_max_8": 1.0865,
      "close_rolling_min_24": 1.0810, "close_rolling_max_24": 1.0880,
      "log_return": 0.0002,
      "hour_sin": 0.5, "hour_cos": 0.866,
      "day_sin": 0.78, "day_cos": 0.62,
      "hour": 12.0, "day_of_week": 2.0,
      "day_of_month": 15.0, "month": 6.0,
      "price_range": 0.0012, "price_change": 0.0003,
      "price_change_pct": 0.0003, "avg_price": 1.0850
    }
  }'
```

**Response:**
```json
{
  "prediction": 0.001468,
  "risk_level": "Low",
  "model_version": "20260308_041826",
  "latency_ms": 11.0,
  "drift_detected": false,
  "timestamp": "2026-03-08T04:21:45Z"
}
```

---

## Feature Engineering (33 Features)

The transformation pipeline (`src/data/data_transformation.py`) creates:

| Category | Features | Count |
|----------|----------|-------|
| **Lag prices** | `close_lag_{1,2,3,4,6,8,12,24}` | 8 |
| **Rolling mean** | `close_rolling_mean_{4,8,24}` | 3 |
| **Rolling std** | `close_rolling_std_{4,8,24}` (volatility proxy) | 3 |
| **Rolling min/max** | `close_rolling_{min,max}_{4,8,24}` | 6 |
| **Time cyclical** | `hour_sin`, `hour_cos`, `day_sin`, `day_cos` | 4 |
| **Time raw** | `hour`, `day_of_week`, `day_of_month`, `month` | 4 |
| **Price dynamics** | `price_range`, `price_change`, `price_change_pct`, `avg_price` | 4 |
| **Returns** | `log_return` | 1 |
| **Target** | `target_volatility` = `volatility.shift(-1)` | -- |

Data quality: outlier removal via z-score (threshold=3.0, index-aligned), forward-fill for missing values, NaN target rows dropped.

---

## Model Training

Two trainer implementations:

| Trainer | File | Use Case |
|---------|------|----------|
| `ModelTrainer` | `src/models/trainer.py` | Single XGBoost with MLflow tracking, used in CI |
| `ProductionModelTrainer` | `src/models/production_trainer.py` | Stacking ensemble (XGBoost + RF + GBR + Ridge meta-learner) |

```bash
# Train single XGBoost model
python src/models/trainer.py --experiment "my_experiment"

# Train production ensemble
python src/models/production_trainer.py
```

Both trainers:
- Use `TimeSeriesSplit` cross-validation (no future leakage)
- Exclude `volatility` from features (target leakage prevention)
- Apply `early_stopping_rounds=10`
- Handle MAPE division-by-zero safely
- Log metrics to MLflow with dual key format support

**Typical performance (EUR/USD hourly):**

| Metric | Range |
|--------|-------|
| RMSE | 0.0004 - 0.0012 |
| MAE | 0.0003 - 0.0008 |
| R-squared | 0.65 - 0.85 |
| MAPE | 10 - 25% |

---

## Project Structure

```
├── src/
│   ├── api/
│   │   └── main.py                    # FastAPI app (serverless-compatible)
│   ├── data/
│   │   ├── data_extraction.py         # Twelve Data API client + quality checks
│   │   └── data_transformation.py     # Feature engineering + data cleaning
│   ├── models/
│   │   ├── trainer.py                 # XGBoost trainer with MLflow
│   │   ├── production_trainer.py      # Stacking ensemble trainer
│   │   └── mlflow_registry.py         # Model registry integration
│   ├── monitoring/
│   │   ├── drift.py                   # Distribution drift detection
│   │   └── alerts.py                  # Alert rules engine (13 rules)
│   ├── ui/
│   │   ├── templates/index.html       # Dashboard UI (Tailwind CSS)
│   │   └── static/js/dashboard.js     # Dashboard JavaScript
│   └── utils/
│       ├── logger.py                  # Logging (file + console, serverless-safe)
│       └── storage.py                 # MinIO client
├── config/
│   └── config.py                      # Centralized config from env vars
├── models/
│   ├── latest_model.pkl               # Bundled XGBoost model (87KB)
│   └── latest_metadata.json           # Model metadata + feature list
├── tests/
│   ├── unit/                          # 37 unit tests
│   └── integration/                   # Integration tests
├── infrastructure/
│   ├── docker/docker-compose.yml      # 8-service local dev stack
│   ├── kubernetes/                    # K8s deployment manifests
│   ├── prometheus/                    # Prometheus config + 13 alert rules
│   └── grafana/                       # Grafana dashboard + datasource
├── airflow/dags/etl_dag.py            # Airflow DAG (2h schedule)
├── .github/workflows/                 # 5 CI/CD workflows
├── Dockerfile                         # Multi-stage production image
├── docker-compose.yml                 # Full local dev stack
├── vercel.json                        # Vercel serverless config
├── requirements.txt                   # Slim deps for Vercel (~150MB)
└── requirements-full.txt              # Full deps for Docker/local dev
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data** | Twelve Data API | Live EUR/USD forex data (1h interval) |
| **Versioning** | DVC + DagsHub | Data and model versioning |
| **Training** | XGBoost, scikit-learn | Gradient boosting + ensemble stacking |
| **Tracking** | MLflow + DagsHub | Experiment tracking, model registry |
| **Serving** | FastAPI + Uvicorn | REST API with async support |
| **Monitoring** | Prometheus + Grafana | Metrics collection, dashboards, alerting |
| **Orchestration** | Apache Airflow | ETL pipeline scheduling |
| **Storage** | MinIO (S3-compatible) | Artifact and data storage |
| **Containers** | Docker + Docker Compose | 8-service local dev environment |
| **CI/CD** | GitHub Actions + CML | Automated testing, training, deployment |
| **Deployment** | Vercel / Railway / Render | Serverless and PaaS hosting |

---

## Monitoring

### Prometheus Metrics
- `predictions_total` — total predictions served
- `prediction_latency_seconds` — latency histogram (P50, P95, P99)
- `data_drift_ratio` — feature drift percentage
- `prediction_errors_total` — error counter

### Drift Detection
The API performs real-time drift detection on every prediction request. Features are checked against expected ranges; if >30% of features are out-of-range, drift is flagged in the response.

### Grafana Alerts (13 rules)
High latency (>500ms), high drift (>20%), error rate spikes, model staleness.

---

## CI/CD Workflows

| Workflow | Trigger | Actions |
|----------|---------|---------|
| `ci-cd.yml` | Push/PR to main | Lint (Black, Flake8, Pylint) + pytest |
| `data-pipeline.yml` | Cron `0 */2 * * *` | Fetch data, transform, train, commit |
| `train-cml.yml` | Push to main | Train model + CML metric report |
| `lint-test.yml` | Push/PR | Code quality checks |
| `deploy.yml` | Push to main | Docker build + push |

---

## Deployment

### Requirements Files

| File | Purpose | Installed Size |
|------|---------|----------------|
| `requirements.txt` | Vercel serverless (slim, API-only deps) | ~150MB |
| `requirements-full.txt` | Docker / local dev (all deps) | ~500MB |

### Vercel Specifics
- Serverless mode auto-detected via `VERCEL` env var
- SQLite writes disabled (ephemeral filesystem)
- Model served from bundled `models/latest_model.pkl`
- Infrastructure UI links (Grafana, Prometheus, etc.) hidden automatically
- Max lambda size: 100MB, memory: 1024MB, timeout: 60s

### Docker
```bash
docker build -t usd-volatility-predictor .
docker run -p 8000:8000 --env-file .env usd-volatility-predictor
```

For full deployment instructions, see **[docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)**.

---

## Testing

```bash
# Run all tests (37 tests)
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TWELVE_DATA_API_KEY` | Yes | -- | Forex data API key |
| `MLFLOW_TRACKING_URI` | No | `""` (file-based) | MLflow server URI |
| `MLFLOW_TRACKING_USERNAME` | No | `""` | MLflow auth username |
| `MLFLOW_TRACKING_PASSWORD` | No | `""` | MLflow auth password |
| `DAGSHUB_TOKEN` | No | `""` | DagsHub API token |
| `DOCKER_USERNAME` | No | `""` | Docker Hub username |
| `API_HOST` | No | `0.0.0.0` | API bind host |
| `API_PORT` | No | `8000` | API bind port |
| `API_WORKERS` | No | `4` | Uvicorn worker count |

See **[docs/GITHUB_SECRETS_SETUP.md](docs/GITHUB_SECRETS_SETUP.md)** for CI/CD secrets configuration.

---

## Documentation

- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) — Platform-specific deployment instructions
- [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md) — Production checklist and monitoring
- [GitHub Secrets Setup](docs/GITHUB_SECRETS_SETUP.md) — CI/CD secrets configuration
- [DVC Setup](docs/DVC_SETUP.md) — Data versioning with DagsHub

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/ -v`
4. Commit: `git commit -m 'Add your feature'`
5. Push: `git push origin feature/your-feature`
6. Open a Pull Request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Rayyan Ahmed** — [GitHub](https://github.com/Rayyan9477)

[Project Repository](https://github.com/Rayyan9477/Real-Time-MLOps-Pipeline-for-USD-Forecasting)
