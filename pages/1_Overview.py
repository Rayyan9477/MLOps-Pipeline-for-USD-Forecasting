"""
Overview page — live KPI dashboard, session summary, recent predictions.
Auto-refreshes every 30 seconds.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.css_theme import inject_css
from app.state import init_session_state
from app.model_service import get_model_service, get_session_stats
from app.charts import make_volatility_timeseries, make_latency_bar

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(
    page_title="Overview | USD MLOps",
    page_icon="📊",
    layout="wide",
)
inject_css()
init_session_state()

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=30_000, key="overview_autorefresh")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
reload_count = st.session_state.get("model_reload_count", 0)
svc          = get_model_service(reload_count)
stats        = get_session_stats()
history      = st.session_state.get("prediction_history", [])
alert_manager = st.session_state.get("alert_manager")
alert_summary = alert_manager.get_alert_summary() if alert_manager else {"total": 0, "by_severity": {}}

metadata     = svc["metadata"] if svc else {}
metrics_data = metadata.get("metrics", {})

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>📊 Live Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;'>Session summary — all forecasts made since opening this app.</p>",
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄 Reload Model", use_container_width=True):
        from app.model_service import reload_model
        reload_model()
        st.success("Model cache cleared.")
        st.rerun()
    if not AUTOREFRESH_AVAILABLE:
        if st.button("↺ Refresh Now", use_container_width=True):
            st.rerun()
    st.caption("Auto-refreshes every 30 s when streamlit-autorefresh is installed.")

st.divider()

# ---------------------------------------------------------------------------
# Row 1: KPI cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Model Status", "🟢 Online" if svc else "🔴 Offline")

with c2:
    st.metric("Forecasts Made", stats["total_predictions"],
              help="Predictions made in this browser session")

with c3:
    lat = f"{stats['avg_latency_ms']:.0f} ms" if stats["avg_latency_ms"] else "—"
    st.metric("Avg Latency", lat)

with c4:
    d_alerts = stats["drift_alerts"]
    st.metric("Drift Flags", d_alerts,
              delta="⚠ check inputs" if d_alerts > 0 else None,
              delta_color="inverse",
              help="Times your input prices were outside expected EUR/USD ranges")

with c5:
    crit = alert_summary.get("by_severity", {}).get("critical", 0)
    warn = alert_summary.get("by_severity", {}).get("warning", 0)
    st.metric("Alerts", f"{warn}W  {crit}C",
              delta="⚠ critical!" if crit > 0 else None,
              delta_color="inverse",
              help="Warning / Critical alert counts this session")

st.divider()

# ---------------------------------------------------------------------------
# No history yet → CTA
# ---------------------------------------------------------------------------
if not history:
    st.info(
        "**No forecasts yet this session.**  \n"
        "Go to the **Home** page (or **🔮 Predictions** in the sidebar) "
        "and make your first EUR/USD forecast — it takes under 10 seconds.",
        icon="👈",
    )

    if svc:
        st.markdown("#### Model ready to use")
        rmse = metrics_data.get("test_rmse", metrics_data.get("rmse", 0))
        r2   = metrics_data.get("test_r2",   metrics_data.get("r2",   0))
        mape = metrics_data.get("test_mape", metrics_data.get("mape", 0))
        rmse_pips = round(float(rmse) * 10000, 1) if isinstance(rmse, (int, float)) else "—"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy (R²)", f"{r2:.3f}" if isinstance(r2, float) else r2,
                      help="1.0 = perfect. 0.85+ is good for forex volatility.")
        with m2:
            st.metric("Avg Error (RMSE)", f"±{rmse_pips} pips",
                      help="Average prediction error in pips")
        with m3:
            st.metric("MAPE", f"{mape:.1f}%" if isinstance(mape, float) else mape,
                      help="Mean Absolute Percentage Error")
        with m4:
            st.metric("Features", len(svc.get("feature_names", [])))
    st.stop()

# ---------------------------------------------------------------------------
# Row 2: Session signal distribution
# ---------------------------------------------------------------------------
st.markdown("#### Session Summary")

risk_counts = {"Low": 0, "Medium": 0, "High": 0}
for p in history:
    r = p.get("risk_level", "")
    if r in risk_counts:
        risk_counts[r] += 1
total = len(history)

d1, d2, d3, d4 = st.columns(4)
with d1:
    pct = round(risk_counts["Low"] / total * 100) if total else 0
    st.metric("🟢 Stable signals", risk_counts["Low"], delta=f"{pct}% of session")
with d2:
    pct = round(risk_counts["Medium"] / total * 100) if total else 0
    st.metric("🟡 Moderate signals", risk_counts["Medium"], delta=f"{pct}% of session")
with d3:
    pct = round(risk_counts["High"] / total * 100) if total else 0
    st.metric("🔴 High-risk signals", risk_counts["High"],
              delta="⚠ frequent" if risk_counts["High"] > total * 0.3 else f"{pct}% of session",
              delta_color="inverse" if risk_counts["High"] > total * 0.3 else "off")
with d4:
    avg_pips = round(sum(p["prediction"] for p in history) / total * 10000, 1)
    st.metric("Avg Expected Move", f"±{avg_pips} pips",
              help="Average predicted volatility across all session forecasts")

st.divider()

# ---------------------------------------------------------------------------
# Row 3: Charts
# ---------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.markdown("#### Volatility Forecast Timeline")
    fig_vol = make_volatility_timeseries(history)
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

with col_chart2:
    st.markdown("#### Latency Distribution")
    latencies = [p["latency_ms"] for p in history if "latency_ms" in p]
    fig_lat = make_latency_bar(latencies)
    st.plotly_chart(fig_lat, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# Row 4: Recent predictions table (pips-based, user-friendly)
# ---------------------------------------------------------------------------
st.markdown("#### Recent Forecasts")

display_data = []
for p in history[-25:][::-1]:
    risk  = p.get("risk_level", "—")
    drift = p.get("drift_detected", False)
    pips  = round(p["prediction"] * 10000, 1)
    risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
    display_data.append({
        "Time (UTC)":    pd.to_datetime(p["timestamp"]).strftime("%H:%M:%S"),
        "Signal":        f"{risk_icon} {risk}",
        "Expected Move": f"±{pips} pips",
        "Price Check":   "⚠ Unusual" if drift else "✓ Normal",
        "Latency":       f"{p.get('latency_ms', 0):.0f} ms",
        "Confidence":    p.get("confidence_score", "—"),
    })

df_display = pd.DataFrame(display_data)

def _highlight_risk(row):
    sig = str(row.get("Signal", ""))
    if "🔴" in sig:
        return ["background-color: rgba(239,68,68,0.12)"] * len(row)
    if "🟡" in sig:
        return ["background-color: rgba(245,158,11,0.08)"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_display.style.apply(_highlight_risk, axis=1),
    use_container_width=True,
    height=420,
    hide_index=True,
)

st.caption(
    "🔴 High-risk rows highlighted in red. ⚠ Unusual = input prices outside typical EUR/USD ranges."
)

st.divider()

# ---------------------------------------------------------------------------
# Model information expander
# ---------------------------------------------------------------------------
with st.expander("🧠 Model Information", expanded=False):
    if svc:
        rmse  = metrics_data.get("test_rmse", metrics_data.get("rmse", 0))
        mae   = metrics_data.get("test_mae",  metrics_data.get("mae",  0))
        r2    = metrics_data.get("test_r2",   metrics_data.get("r2",   0))
        mape  = metrics_data.get("test_mape", metrics_data.get("mape", 0))
        rmse_pips = round(float(rmse) * 10000, 1) if isinstance(rmse, (int, float)) else "—"
        mae_pips  = round(float(mae)  * 10000, 1) if isinstance(mae,  (int, float)) else "—"

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("RMSE", f"±{rmse_pips} pips",
                      help="Root Mean Squared Error — average prediction error in pips")
        with col_m2:
            st.metric("MAE", f"±{mae_pips} pips",
                      help="Mean Absolute Error in pips")
        with col_m3:
            st.metric("R²", f"{r2:.4f}" if isinstance(r2, float) else r2,
                      help="R-squared: proportion of variance explained. Higher is better.")
        with col_m4:
            st.metric("MAPE", f"{mape:.1f}%" if isinstance(mape, float) else mape,
                      help="Mean Absolute Percentage Error")

        st.markdown(
            f"| Property | Value |\n|---|---|\n"
            f"| Model type | `{metadata.get('model_type', 'XGBRegressor')}` |\n"
            f"| Version / timestamp | `{svc['version']}` |\n"
            f"| Trained at | `{metadata.get('trained_at', '—')}` |\n"
            f"| Training samples | `{metadata.get('training_samples', '—')}` |\n"
            f"| Number of features | `{len(svc.get('feature_names', []))}` |\n"
            f"| Scaler | `{'Yes (RobustScaler)' if svc.get('scaler') else 'No'}` |\n"
        )
    else:
        st.error("Model not loaded. Go to Model Management to train one.")
