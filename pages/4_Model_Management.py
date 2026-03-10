"""
Model Management page — view model info, train XGBoost/Ensemble, register to MLflow.
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path

from app.css_theme import inject_css
from app.state import init_session_state
from app.model_service import get_model_service, reload_model
from app.charts import make_feature_importance_chart

st.set_page_config(
    page_title="Model Management | USD MLOps",
    page_icon="🧠",
    layout="wide",
)
inject_css()
init_session_state()

# ---------------------------------------------------------------------------
# StreamlitLogHandler (same pattern as Data Pipeline page)
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.lines = []
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record):
        self.lines.append(self.format(record))
        self.placeholder.code("\n".join(self.lines[-50:]), language=None)


def _attach(loggers, placeholder):
    h = StreamlitLogHandler(placeholder)
    for name in loggers:
        lg = logging.getLogger(name)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return h


def _detach(loggers, h):
    for name in loggers:
        logging.getLogger(name).removeHandler(h)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>🧠 Model Management</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;'>View model metadata, train new models, and register to MLflow.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Load current model
reload_count = st.session_state.get("model_reload_count", 0)
svc = get_model_service(reload_count)

# Check for processed data
from config import PROCESSED_DATA_DIR
proc_files = sorted(Path(PROCESSED_DATA_DIR).glob("*.parquet")) if Path(PROCESSED_DATA_DIR).exists() else []
has_processed_data = len(proc_files) > 0

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Model Info", "⚡ Train XGBoost", "🏗️ Train Ensemble", "🧪 Register to MLflow"]
)

# ===========================================================================
# TAB 1: Model Info
# ===========================================================================
with tab1:
    if not svc:
        st.error("⚠️ No model loaded.", icon="🚨")
    else:
        metadata = svc["metadata"]
        metrics_data = metadata.get("metrics", {})

        # Quick action
        if st.button("🔄 Reload Model from Disk", use_container_width=False):
            reload_model()
            st.success("Cache cleared. Model will reload on next page interaction.")
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        rmse = metrics_data.get("test_rmse", metrics_data.get("rmse", "—"))
        mae  = metrics_data.get("test_mae",  metrics_data.get("mae",  "—"))
        r2   = metrics_data.get("test_r2",   metrics_data.get("r2",   "—"))
        mape = metrics_data.get("test_mape", metrics_data.get("mape", "—"))

        with col1:
            st.metric("RMSE", f"{rmse:.6f}" if isinstance(rmse, float) else rmse)
        with col2:
            st.metric("MAE",  f"{mae:.6f}"  if isinstance(mae,  float) else mae)
        with col3:
            st.metric("R²",   f"{r2:.4f}"   if isinstance(r2,   float) else r2)
        with col4:
            st.metric("MAPE", f"{mape:.2f}%" if isinstance(mape, float) else mape)

        st.divider()

        col_meta, col_feat = st.columns([1, 1])
        with col_meta:
            st.markdown("#### Metadata")
            st.markdown(f"""
            | Property | Value |
            |---|---|
            | Model Type | `{metadata.get('model_type', 'XGBRegressor')}` |
            | Version | `{svc['version']}` |
            | Trained At | `{metadata.get('trained_at', '—')}` |
            | Training Samples | `{metadata.get('training_samples', '—')}` |
            | n_features | `{metadata.get('n_features', len(svc.get('feature_names', [])))}` |
            | Scaler Loaded | `{'Yes (RobustScaler)' if svc.get('scaler') else 'No'}` |
            | Model Path | `{svc.get('model_path', '—')}` |
            | Description | `{metadata.get('description', '—')}` |
            """)

            with st.expander("Raw Metadata JSON", expanded=False):
                st.json(metadata)

        with col_feat:
            st.markdown("#### Feature Names (33)")
            feature_names = svc.get("feature_names", [])
            if feature_names:
                feat_df = pd.DataFrame({
                    "#": range(1, len(feature_names) + 1),
                    "Feature": feature_names,
                })
                st.dataframe(feat_df, use_container_width=True, height=400)

        # Feature importance
        st.divider()
        st.markdown("#### Feature Importance")
        fig_imp = make_feature_importance_chart(svc["model"], feature_names)
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# TAB 2: Train XGBoost
# ===========================================================================
with tab2:
    st.markdown("#### Train XGBoost Model with MLflow Tracking")
    st.markdown(
        "<p style='color:#94a3b8;'>Trains a single XGBoost model using processed parquet data. "
        "Results are logged to MLflow (experiment: `usd_volatility_prediction`).</p>",
        unsafe_allow_html=True,
    )

    if not has_processed_data:
        st.warning(
            "⚠️ No processed parquet data found in `data/processed/`. "
            "Run the **Data Pipeline** page first to extract and transform data.",
            icon="⚠️",
        )
    else:
        st.success(
            f"✅ {len(proc_files)} processed data file(s) available: "
            f"`{proc_files[-1].name}`",
            icon="✅",
        )

    with st.expander("⚙️ Hyperparameters", expanded=True):
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
            max_depth    = st.slider("max_depth", 3, 10, 5, 1)
        with col_h2:
            learning_rate  = st.number_input("learning_rate", 0.01, 0.3, 0.1, 0.01, format="%.3f")
            subsample      = st.slider("subsample", 0.5, 1.0, 0.8, 0.05)

    xgb_log = st.empty()
    train_xgb_btn = st.button(
        "🚀 Start XGBoost Training",
        type="primary",
        disabled=not has_processed_data,
        use_container_width=True,
    )

    if train_xgb_btn:
        log_names = ["model_training", "mlflow"]
        h = _attach(log_names, xgb_log)

        try:
            from src.models.trainer import ModelTrainer
            from config import MODEL_CONFIG, MLFLOW_CONFIG
            import copy

            # Override config with UI hyperparameters
            config = copy.deepcopy(MODEL_CONFIG)
            config["xgboost_params"].update({
                "n_estimators":  n_estimators,
                "max_depth":     max_depth,
                "learning_rate": learning_rate,
                "subsample":     subsample,
            })

            # Patch config temporarily
            import config as cfg_module
            orig_params = cfg_module.MODEL_CONFIG["xgboost_params"].copy()
            cfg_module.MODEL_CONFIG["xgboost_params"].update(config["xgboost_params"])

            with st.spinner("Training XGBoost model..."):
                try:
                    trainer = ModelTrainer()
                    trainer.train_and_log()
                    run_id = getattr(trainer, "run_id", "—")
                    st.success(
                        f"✅ Training complete! MLflow run ID: `{run_id}`",
                        icon="✅",
                    )
                    # Reload model with fresh weights
                    reload_model()
                    st.info("Model cache cleared. Reload the page to see updated metrics.")
                except Exception as e:
                    st.error(f"Training failed: {e}", icon="❌")
                    st.exception(e)

            # Restore original config
            cfg_module.MODEL_CONFIG["xgboost_params"].update(orig_params)

        finally:
            _detach(log_names, h)


# ===========================================================================
# TAB 3: Train Ensemble
# ===========================================================================
with tab3:
    st.markdown("#### Train Production Ensemble Model")

    st.info(
        "**Stacking Architecture:**\n\n"
        "- **Base estimators:** XGBoost · RandomForestRegressor · GradientBoostingRegressor\n"
        "- **Meta-learner:** Ridge Regression\n"
        "- **Scaling:** RobustScaler (handles outliers)\n"
        "- **Output:** `models/latest_model.pkl` + `models/scaler.pkl`\n\n"
        "Note: This trainer does not use MLflow. Use the **Register to MLflow** tab after training.",
        icon="ℹ️",
    )

    if not has_processed_data:
        st.warning(
            "⚠️ No processed parquet data found. Run the **Data Pipeline** page first.",
            icon="⚠️",
        )

    ens_log = st.empty()
    train_ens_btn = st.button(
        "🏗️ Start Ensemble Training",
        type="primary",
        disabled=not has_processed_data,
        use_container_width=True,
    )

    if train_ens_btn:
        log_names = ["production_trainer", "model_training"]
        h = _attach(log_names, ens_log)

        try:
            from src.models.production_trainer import ProductionModelTrainer

            with st.spinner("Training ensemble (XGBoost + RF + GB + Ridge)..."):
                try:
                    trainer = ProductionModelTrainer()
                    results = trainer.run_training_pipeline()

                    if results:
                        col_r1, col_r2, col_r3 = st.columns(3)
                        test_r2   = results.get("test_r2",   results.get("r2",   "—"))
                        test_mape = results.get("test_mape", results.get("mape", "—"))
                        test_rmse = results.get("test_rmse", results.get("rmse", "—"))

                        with col_r1:
                            st.metric("Test R²",   f"{test_r2:.4f}"   if isinstance(test_r2, float)   else test_r2)
                        with col_r2:
                            st.metric("Test MAPE", f"{test_mape:.2f}%" if isinstance(test_mape, float) else test_mape)
                        with col_r3:
                            st.metric("Test RMSE", f"{test_rmse:.6f}"  if isinstance(test_rmse, float) else test_rmse)

                        st.success("✅ Ensemble training complete!", icon="✅")

                        if st.button("🔄 Use This Model (Reload)", use_container_width=True):
                            reload_model()
                            st.success("Model cache cleared.")
                            st.rerun()
                    else:
                        st.warning("Training finished but no metrics returned.", icon="⚠️")

                except Exception as e:
                    st.error(f"Ensemble training failed: {e}", icon="❌")
                    st.exception(e)
        finally:
            _detach(log_names, h)


# ===========================================================================
# TAB 4: Register to MLflow
# ===========================================================================
with tab4:
    st.markdown("#### Register Current Model to MLflow")
    st.markdown(
        "<p style='color:#94a3b8;'>Register the loaded model to MLflow Model Registry "
        "for versioning and stage management.</p>",
        unsafe_allow_html=True,
    )

    if not svc:
        st.error("No model loaded. Cannot register.", icon="🚨")
    else:
        from config import MLFLOW_CONFIG

        mlflow_uri = st.text_input(
            "MLflow Tracking URI",
            value=MLFLOW_CONFIG.get("tracking_uri", ""),
            placeholder="http://localhost:5000 or ./mlruns",
            help="Leave empty to use local ./mlruns directory",
        )
        experiment_name = st.text_input(
            "Experiment Name",
            value=MLFLOW_CONFIG.get("experiment_name", "usd_volatility_prediction"),
        )
        model_name = st.text_input(
            "Registered Model Name",
            value=MLFLOW_CONFIG.get("model_name", "usd_volatility_predictor"),
        )

        reg_log = st.empty()
        register_btn = st.button(
            "📦 Register Model to MLflow",
            type="primary",
            use_container_width=True,
        )

        if register_btn:
            h = _attach(["mlflow_registry", "mlflow"], reg_log)
            try:
                from src.models.mlflow_registry import register_model_to_mlflow

                with st.spinner("Registering model..."):
                    try:
                        result = register_model_to_mlflow(
                            model=svc["model"],
                            metadata=svc["metadata"],
                            tracking_uri=mlflow_uri or None,
                            experiment_name=experiment_name,
                            model_name=model_name,
                        )
                        run_id = result if isinstance(result, str) else str(result)
                        st.success(
                            f"✅ Model registered successfully!\n\nRun ID: `{run_id}`",
                            icon="✅",
                        )
                    except Exception as e:
                        st.error(f"Registration failed: {e}", icon="❌")
                        st.exception(e)
            finally:
                _detach(["mlflow_registry", "mlflow"], h)
