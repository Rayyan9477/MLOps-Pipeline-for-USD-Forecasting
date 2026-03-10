"""
Model service layer for Streamlit — replicates src/api/main.py logic exactly.
Uses @st.cache_resource so the model is loaded once per server process.

Key functions mirrored from main.py:
  - load_local_model() → get_model_service()
  - interpret_prediction() → interpret_prediction()
  - detect_drift() → detect_drift_simple()
  - predict() → predict_single()
"""

import pickle
import json
import os
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import MODELS_DIR


# ---------------------------------------------------------------------------
# Interpretation helpers — exact copies from src/api/main.py
# ---------------------------------------------------------------------------

def interpret_prediction(value: float) -> Tuple[str, str, str]:
    """
    Convert raw prediction value into human-readable interpretation.
    Thresholds: < 0.005 = Low, < 0.015 = Medium, >= 0.015 = High.
    Exact copy of main.py:interpret_prediction().
    """
    pct = value * 100
    if value < 0.005:
        interpretation = (
            f"Low volatility expected ({value:.4f} or {pct:.2f}%). "
            "Market conditions appear stable with minimal price fluctuations."
        )
        risk_level = "Low"
        confidence_score = "High"
    elif value < 0.015:
        interpretation = (
            f"Moderate volatility expected ({value:.4f} or {pct:.2f}%). "
            "Normal market activity with typical price movements."
        )
        risk_level = "Medium"
        confidence_score = "Medium"
    else:
        interpretation = (
            f"High volatility expected ({value:.4f} or {pct:.2f}%). "
            "Market conditions are turbulent with significant price swings."
        )
        risk_level = "High"
        confidence_score = "High"

    return interpretation, risk_level, confidence_score


def detect_drift_simple(features: Dict[str, float]) -> Tuple[bool, float]:
    """
    Range-based drift detection — exact copy of main.py:detect_drift().
    Checks 6 key features against expected EUR/USD ranges.
    drift_detected = True if drift_ratio > 0.3.
    """
    expected_ranges = {
        "close_lag_1":          (0.9, 1.2),
        "close_rolling_mean_24":(0.9, 1.2),
        "close_rolling_std_24": (0.0, 0.05),
        "log_return":           (-0.1, 0.1),
        "hour_sin":             (-1.0, 1.0),
        "hour_cos":             (-1.0, 1.0),
    }

    out_of_range = 0
    total = len(features)

    for name, value in features.items():
        if name in expected_ranges:
            low, high = expected_ranges[name]
            if value < low or value > high:
                out_of_range += 1

    drift_ratio = out_of_range / total if total > 0 else 0.0
    drift_detected = drift_ratio > 0.3
    return drift_detected, drift_ratio


# ---------------------------------------------------------------------------
# Model loading — exact fallback chain from main.py:load_local_model()
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model from local storage...")
def get_model_service(_reload_count: int = 0) -> Optional[Dict]:
    """
    Load model + scaler from MODELS_DIR.
    _reload_count is a cache-busting parameter — increment session state
    "model_reload_count" to force a fresh load.

    Fallback chain (mirrors main.py exactly):
      1. xgboost_model_*.pkl  (sorted, newest)
      2. simple_api_model_*.pkl (sorted, newest)
      3. latest_model.pkl

    Returns dict with keys:
      model, scaler (or None), metadata, version, feature_names
    Or None if no model found.
    """
    from src.utils.logger import get_logger
    logger = get_logger("streamlit_model_service")

    try:
        logger.info("Loading model from local storage...")
        model_path = None
        metadata_path = None

        # 1. XGBoost timestamped models
        xgboost_files = sorted(MODELS_DIR.glob("xgboost_model_*.pkl"))
        if xgboost_files:
            model_path = xgboost_files[-1]
            ts = model_path.stem.replace("xgboost_model_", "")
            metadata_path = MODELS_DIR / f"model_metadata_{ts}.json"
        else:
            # 2. Simple API models
            simple_files = sorted(MODELS_DIR.glob("simple_api_model_*.pkl"))
            if simple_files:
                model_path = simple_files[-1]
                ts = model_path.stem.replace("simple_api_model_", "")
                metadata_path = MODELS_DIR / f"simple_api_metadata_{ts}.json"
            else:
                # 3. Fallback to latest_model.pkl
                model_path = MODELS_DIR / "latest_model.pkl"
                metadata_path = MODELS_DIR / "latest_metadata.json"

                if not model_path.exists():
                    # DVC pull if configured
                    dvc_pointer = MODELS_DIR / "latest_model.pkl.dvc"
                    if dvc_pointer.exists() and os.getenv("DVC_PULL_ON_STARTUP", "").lower() in {"1", "true", "yes"}:
                        logger.warning("Attempting DVC pull for model...")
                        try:
                            subprocess.run(["dvc", "pull", str(dvc_pointer)], check=True)
                        except Exception as e:
                            logger.error(f"DVC pull failed: {e}")

                if not model_path.exists():
                    logger.error("No model files found in models directory.")
                    return None

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {model_path}")

        # Load metadata
        metadata = {}
        version = "local"
        if metadata_path and metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            version = metadata.get("timestamp", "unknown")
            logger.info(f"Metadata loaded. Features: {len(metadata.get('feature_names', []))}")
        else:
            logger.warning("Model metadata not found.")

        # Load scaler (ProductionModelTrainer saves models/scaler.pkl separately)
        scaler = None
        scaler_path = MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded from models/scaler.pkl")

        feature_names = metadata.get("feature_names", [])

        return {
            "model": model,
            "scaler": scaler,
            "metadata": metadata,
            "version": version,
            "feature_names": feature_names,
            "model_path": str(model_path),
        }

    except Exception as e:
        from src.utils.logger import get_logger
        get_logger("streamlit_model_service").error(f"Failed to load model: {e}")
        return None


def reload_model():
    """
    Force a model reload by incrementing the cache-busting counter
    and clearing the cache resource.
    """
    st.cache_resource.clear()
    if "model_reload_count" in st.session_state:
        st.session_state["model_reload_count"] += 1


# ---------------------------------------------------------------------------
# Prediction — mirrors main.py:predict() logic
# ---------------------------------------------------------------------------

def predict_single(features_dict: Dict[str, float], svc: Dict) -> Dict:
    """
    Run a single prediction using the loaded model service.
    Replicates main.py predict() endpoint logic exactly.

    Args:
        features_dict: dict of feature_name → value (may be partial)
        svc: dict from get_model_service()

    Returns:
        dict with prediction, risk_level, drift_detected, latency_ms, etc.
    """
    start_time = time.perf_counter()

    model = svc["model"]
    scaler = svc.get("scaler")
    feature_names = svc.get("feature_names", list(features_dict.keys()))
    version = svc.get("version", "unknown")

    # Warn about missing features (fill with 0.0, same as main.py)
    missing = [f for f in feature_names if f not in features_dict]
    if missing:
        pass  # silently fill with 0.0 as per main.py behaviour

    # Build ordered feature vector
    feature_values = {f: features_dict.get(f, 0.0) for f in feature_names}
    feature_df = pd.DataFrame([feature_values])

    # Apply scaler if present (ProductionModelTrainer ensemble)
    if scaler is not None:
        try:
            feature_arr = scaler.transform(feature_df)
            prediction = float(model.predict(feature_arr)[0])
        except Exception:
            prediction = float(model.predict(feature_df)[0])
    else:
        prediction = float(model.predict(feature_df)[0])

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # Interpret and detect drift (exact main.py logic)
    interpretation, risk_level, confidence = interpret_prediction(prediction)
    drift_detected, drift_ratio = detect_drift_simple(features_dict)

    timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "timestamp": timestamp,
        "prediction": prediction,
        "prediction_interpretation": interpretation,
        "risk_level": risk_level,
        "confidence_score": confidence,
        "drift_detected": drift_detected,
        "drift_ratio": drift_ratio,
        "latency_ms": round(latency_ms, 2),
        "model_version": version,
    }

    # Append to session history
    if "prediction_history" in st.session_state:
        st.session_state["prediction_history"].append(result)
        # Check alert rules
        if "alert_manager" in st.session_state and st.session_state["alert_manager"]:
            st.session_state["alert_manager"].check_metrics({
                "prediction_latency_seconds": latency_ms / 1000,
                "drift_ratio": drift_ratio,
            })

    st.session_state["last_prediction"] = result
    return result


def predict_batch(rows: list, svc: Dict) -> list:
    """
    Run batch predictions. rows is a list of feature dicts.
    Returns list of result dicts.
    """
    results = []
    for features_dict in rows:
        try:
            result = predict_single(features_dict, svc)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return results


def get_session_stats() -> Dict:
    """
    Compute stats from session prediction history.
    Mirrors main.py:get_stats() but from session state.
    """
    history = st.session_state.get("prediction_history", [])
    if not history:
        return {
            "total_predictions": 0,
            "avg_latency_ms": 0.0,
            "drift_alerts": 0,
        }

    latencies = [p["latency_ms"] for p in history if "latency_ms" in p]
    drift_count = sum(1 for p in history if p.get("drift_detected", False))

    return {
        "total_predictions": len(history),
        "avg_latency_ms": round(np.mean(latencies), 2) if latencies else 0.0,
        "drift_alerts": drift_count,
    }
