"""
Script to register the locally trained model in MLflow tracking server.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.xgboost

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODELS_DIR  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("mlflow_register")


def register_model_to_mlflow(
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "usd-volatility-prediction",
):
    """Register the locally trained model to MLflow."""

    # Set S3/MinIO environment variables for artifact storage
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

    # Connect to MLflow tracking server
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"Connected to MLflow server: {mlflow_uri}")

    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"Experiment issue: {e}")
        experiment_id = "0"

    mlflow.set_experiment(experiment_name)

    # Load local model and metadata
    latest_model_path = MODELS_DIR / "latest_model.pkl"
    latest_metadata_path = MODELS_DIR / "latest_metadata.json"

    if not latest_model_path.exists():
        logger.error(f"Model not found: {latest_model_path}")
        return None

    # Load model
    with open(latest_model_path, "rb") as f:
        model = pickle.load(f)

    # Load metadata
    metadata = {}
    if latest_metadata_path.exists():
        with open(latest_metadata_path, "r") as f:
            metadata = json.load(f)

    metrics = metadata.get("metrics", {})
    feature_names = metadata.get("feature_names", [])

    # Start MLflow run
    with mlflow.start_run(
        run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:

        # Log parameters
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_features", metrics.get("n_features", len(feature_names)))
        mlflow.log_param("train_samples", metrics.get("train_samples", 0))
        mlflow.log_param("test_samples", metrics.get("test_samples", 0))

        # Log metrics
        if metrics:
            mlflow.log_metric("rmse", metrics.get("test_rmse", metrics.get("rmse", 0)))
            mlflow.log_metric("mae", metrics.get("test_mae", metrics.get("mae", 0)))
            mlflow.log_metric("r2", metrics.get("test_r2", metrics.get("r2", 0)))
            mlflow.log_metric("mape", metrics.get("test_mape", metrics.get("mape", 0)))

        # Log the model using sklearn flavor (more portable)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="usd_volatility_predictor",
        )

        # Log metadata as artifact
        if latest_metadata_path.exists():
            mlflow.log_artifact(str(latest_metadata_path), artifact_path="metadata")

        logger.info("Model registered successfully!")
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Experiment ID: {experiment_id}")

        return run.info.run_id


def main():
    """Main entry point."""
    run_id = register_model_to_mlflow()

    if run_id:
        print("\n✓ Model registered successfully!")
        print(f"  Run ID: {run_id}")
        print("  View at: http://localhost:5000")
    else:
        print("\n✗ Failed to register model")


if __name__ == "__main__":
    main()
