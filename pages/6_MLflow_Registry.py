"""
MLflow Registry page — browse experiments, runs, and registered models.
Falls back gracefully when MLflow is not configured or not installed.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

from app.css_theme import inject_css
from app.state import init_session_state
from app.charts import make_metrics_comparison_chart

st.set_page_config(
    page_title="MLflow Registry | USD MLOps",
    page_icon="🧪",
    layout="wide",
)
inject_css()
init_session_state()

# ---------------------------------------------------------------------------
# MLflow availability check
# ---------------------------------------------------------------------------
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

st.markdown("<h1 class='gradient-text'>🧪 MLflow Registry</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;'>Browse experiments, compare runs, and manage registered models.</p>",
    unsafe_allow_html=True,
)
st.divider()

if not MLFLOW_AVAILABLE:
    st.error(
        "⚠️ MLflow is not installed. Add `mlflow>=2.9.2` to requirements.txt and restart.",
        icon="❌",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------
from config import MLFLOW_CONFIG

col_uri, col_btn = st.columns([3, 1])
with col_uri:
    tracking_uri = st.text_input(
        "MLflow Tracking URI",
        value=os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_CONFIG.get("tracking_uri") or "./mlruns",
        placeholder="http://localhost:5000 or ./mlruns",
        help="Local path (./mlruns) or remote server URL",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    connect_btn = st.button("🔌 Connect", type="primary", use_container_width=True)

# Initialize connection in session state
if "mlflow_connected" not in st.session_state:
    st.session_state["mlflow_connected"] = False
if "mlflow_uri" not in st.session_state:
    st.session_state["mlflow_uri"] = tracking_uri

if connect_btn:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        st.session_state["mlflow_connected"] = True
        st.session_state["mlflow_uri"] = tracking_uri
        st.success(
            f"✅ Connected to MLflow at `{tracking_uri}` · "
            f"Found {len(experiments)} experiment(s).",
            icon="✅",
        )
    except Exception as e:
        st.session_state["mlflow_connected"] = False
        st.error(f"Connection failed: {e}", icon="❌")

# Auto-connect to default URI on first load
if not st.session_state["mlflow_connected"]:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        _ = client.search_experiments()  # test connection
        st.session_state["mlflow_connected"] = True
        st.session_state["mlflow_uri"] = tracking_uri
    except Exception:
        pass

st.divider()

# ---------------------------------------------------------------------------
# MLflow local directory listing (always shown as reference)
# ---------------------------------------------------------------------------
with st.expander("📁 Local MLflow Directory", expanded=False):
    local_mlruns = Path("./mlruns")
    if local_mlruns.exists():
        st.markdown(f"**`./mlruns` exists.** Contents:")
        dirs = sorted(local_mlruns.iterdir())
        for d in dirs[:20]:
            st.markdown(f"- `{d.name}/`")
        if len(dirs) > 20:
            st.markdown(f"_... and {len(dirs)-20} more_")
    else:
        st.info(
            "No local `./mlruns` directory found. MLflow will create it on the first "
            "training run when `MLFLOW_TRACKING_URI` is not set.",
            icon="ℹ️",
        )

# ---------------------------------------------------------------------------
# Main content: only shown when connected
# ---------------------------------------------------------------------------
if not st.session_state["mlflow_connected"]:
    st.warning(
        "Not connected to MLflow. Enter a tracking URI and click **Connect**.",
        icon="⚠️",
    )
    st.stop()

# Re-instantiate client with current URI
try:
    mlflow.set_tracking_uri(st.session_state["mlflow_uri"])
    client = MlflowClient(tracking_uri=st.session_state["mlflow_uri"])
except Exception as e:
    st.error(f"MLflow client error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Experiment selector
# ---------------------------------------------------------------------------
try:
    experiments = client.search_experiments()
except Exception as e:
    st.error(f"Failed to list experiments: {e}")
    st.stop()

if not experiments:
    st.info(
        "No experiments found. Train a model using the **Model Management** page "
        "to create the first experiment.",
        icon="ℹ️",
    )
    st.stop()

exp_names = [e.name for e in experiments]
selected_exp_name = st.selectbox(
    "Experiment",
    exp_names,
    index=0 if "usd_volatility_prediction" not in exp_names
    else exp_names.index("usd_volatility_prediction"),
)
selected_exp = next(e for e in experiments if e.name == selected_exp_name)

st.markdown(
    f"**Experiment ID:** `{selected_exp.experiment_id}` · "
    f"**Artifact Location:** `{selected_exp.artifact_location}`"
)
st.divider()

# ---------------------------------------------------------------------------
# Runs table
# ---------------------------------------------------------------------------
try:
    runs = client.search_runs(
        experiment_ids=[selected_exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=50,
    )
except Exception as e:
    st.error(f"Failed to fetch runs: {e}")
    st.stop()

st.markdown(f"### Runs ({len(runs)} found)")

if not runs:
    st.info("No runs in this experiment yet.", icon="ℹ️")
    st.stop()

# Build runs DataFrame
run_rows = []
for r in runs:
    metrics = r.data.metrics
    run_rows.append({
        "run_id":      r.info.run_id[:12],
        "run_id_full": r.info.run_id,
        "status":      r.info.status,
        "start_time":  pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M") if r.info.start_time else "—",
        "rmse":  round(metrics.get("test_rmse", metrics.get("rmse", float("nan"))), 6),
        "mae":   round(metrics.get("test_mae",  metrics.get("mae",  float("nan"))), 6),
        "r2":    round(metrics.get("test_r2",   metrics.get("r2",   float("nan"))), 4),
        "mape":  round(metrics.get("test_mape", metrics.get("mape", float("nan"))), 2),
    })

runs_df = pd.DataFrame(run_rows)

# Highlight best run (lowest RMSE)
def _highlight_best(row):
    if runs_df["rmse"].notna().any():
        best_rmse = runs_df["rmse"].min()
        if row["rmse"] == best_rmse:
            return ["background-color: rgba(16,185,129,0.15)"] * len(row)
    return [""] * len(row)

display_df = runs_df.drop(columns=["run_id_full"])

st.dataframe(
    display_df.style.apply(_highlight_best, axis=1),
    use_container_width=True,
    height=300,
)
st.markdown(
    "<p style='color:#64748b;font-size:0.75rem;'>🟢 Best RMSE highlighted</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Metrics comparison chart
# ---------------------------------------------------------------------------
st.markdown("#### Metrics Comparison Across Runs")
chart_df = runs_df[["run_id", "rmse", "r2"]].dropna(subset=["rmse"])
if not chart_df.empty:
    fig_comp = make_metrics_comparison_chart(chart_df)
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# Run detail expander
# ---------------------------------------------------------------------------
st.markdown("#### Run Details")
run_id_options = [r["run_id"] for r in run_rows]
selected_short = st.selectbox("Select Run (first 12 chars)", run_id_options)
selected_run_full = next(r["run_id_full"] for r in run_rows if r["run_id"] == selected_short)

try:
    run_detail = client.get_run(selected_run_full)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Parameters**")
        params = run_detail.data.params
        if params:
            params_df = pd.DataFrame(
                [{"Parameter": k, "Value": v} for k, v in params.items()]
            )
            st.dataframe(params_df, use_container_width=True)
        else:
            st.info("No parameters logged.")

    with col_d2:
        st.markdown("**Metrics**")
        metrics = run_detail.data.metrics
        if metrics:
            metrics_df = pd.DataFrame(
                [{"Metric": k, "Value": round(v, 6) if isinstance(v, float) else v}
                 for k, v in metrics.items()]
            )
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("No metrics logged.")

    # Tags
    tags = {k: v for k, v in run_detail.data.tags.items() if not k.startswith("mlflow.")}
    if tags:
        with st.expander("Tags", expanded=False):
            st.json(tags)

    # Artifacts
    try:
        artifacts = client.list_artifacts(selected_run_full)
        if artifacts:
            with st.expander(f"Artifacts ({len(artifacts)})", expanded=False):
                for a in artifacts:
                    st.markdown(f"- `{a.path}` ({a.file_size or 0:,} bytes)")
    except Exception:
        pass

except Exception as e:
    st.error(f"Failed to fetch run details: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------
with st.expander("📦 Registered Models", expanded=False):
    try:
        reg_models = client.search_registered_models()
        if not reg_models:
            st.info(
                "No registered models found. Use **Model Management → Register to MLflow** "
                "to register the current model.",
                icon="ℹ️",
            )
        else:
            for rm in reg_models:
                st.markdown(f"**{rm.name}**")
                versions = client.search_model_versions(f"name='{rm.name}'")
                ver_data = [
                    {
                        "Version": v.version,
                        "Stage": v.current_stage,
                        "Status": v.status,
                        "Created": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d") if v.creation_timestamp else "—",
                        "Run ID": v.run_id[:12] if v.run_id else "—",
                    }
                    for v in versions
                ]
                if ver_data:
                    st.dataframe(pd.DataFrame(ver_data), use_container_width=True)
                st.markdown("---")
    except Exception as e:
        st.warning(f"Could not list registered models: {e}", icon="⚠️")
