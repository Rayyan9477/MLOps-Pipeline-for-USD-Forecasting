"""
Monitoring & Drift page — drift time-series, KS test analysis, alert history,
Prometheus metrics raw output.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from app.css_theme import inject_css
from app.state import init_session_state
from app.model_service import get_session_stats
from app.charts import (
    make_drift_timeseries,
    make_latency_bar,
    make_ks_heatmap,
)

st.set_page_config(
    page_title="Monitoring | USD MLOps",
    page_icon="📡",
    layout="wide",
)
inject_css()
init_session_state()

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>📡 Monitoring & Drift</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;'>Track data drift, alert history, and system health metrics.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Data from session state
history = st.session_state.get("prediction_history", [])
alert_manager = st.session_state.get("alert_manager")
alert_summary = alert_manager.get_alert_summary() if alert_manager else {"total": 0, "by_severity": {}}
stats = get_session_stats()

# ---------------------------------------------------------------------------
# Row 1: KPI metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Predictions", stats["total_predictions"])
with col2:
    st.metric(
        "Drift Detections",
        stats["drift_alerts"],
        delta="⚠ active" if stats["drift_alerts"] > 0 else None,
        delta_color="inverse",
    )
with col3:
    warn_count = alert_summary.get("by_severity", {}).get("warning", 0)
    st.metric("Warning Alerts", warn_count)
with col4:
    crit_count = alert_summary.get("by_severity", {}).get("critical", 0)
    st.metric(
        "Critical Alerts",
        crit_count,
        delta="⚠ critical!" if crit_count > 0 else None,
        delta_color="inverse",
    )

st.divider()

# ---------------------------------------------------------------------------
# Row 2: Drift + Latency charts
# ---------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.markdown("#### Data Drift Over Time")
    fig_drift = make_drift_timeseries(history)
    st.plotly_chart(fig_drift, use_container_width=True, config={"displayModeBar": False})

with col_chart2:
    st.markdown("#### Latency Distribution")
    latencies = [p["latency_ms"] for p in history if "latency_ms" in p]
    fig_lat = make_latency_bar(latencies)
    st.plotly_chart(fig_lat, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# Statistical Drift Analysis (KS Test)
# ---------------------------------------------------------------------------
with st.expander("🔬 Statistical Drift Analysis (KS Test)", expanded=False):
    from config import PROCESSED_DATA_DIR
    proc_files = sorted(Path(PROCESSED_DATA_DIR).glob("*.parquet")) if Path(PROCESSED_DATA_DIR).exists() else []

    if not proc_files:
        st.info(
            "ℹ️ No processed parquet data found in `data/processed/`. "
            "Run the **Data Pipeline** page to generate reference data for KS testing.",
            icon="ℹ️",
        )
    else:
        st.success(
            f"✅ Reference data available: `{proc_files[-1].name}`",
            icon="✅",
        )
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.85rem;'>The KS test compares the current "
            "session's prediction features against the training data distribution. "
            "p < 0.05 signals drift.</p>",
            unsafe_allow_html=True,
        )

        run_ks_btn = st.button("🔬 Run Full KS Drift Analysis", type="primary")

        if run_ks_btn:
            if len(history) < 10:
                st.warning(
                    "Need at least 10 predictions for KS test. "
                    f"Current session: {len(history)} predictions.",
                    icon="⚠️",
                )
            else:
                with st.spinner("Running Kolmogorov-Smirnov and Wasserstein analysis..."):
                    try:
                        from src.monitoring.drift import DriftDetector

                        # Load reference data (most recent processed parquet)
                        ref_df = pd.read_parquet(proc_files[-1])
                        ref_df = ref_df.select_dtypes(include=["number"])

                        # Fit detector on reference data
                        detector = DriftDetector(significance_level=0.05, z_score_threshold=3.0)
                        detector.fit(ref_df)
                        st.session_state["drift_detector"] = detector

                        # Build current data DataFrame from session history
                        current_rows = []
                        for p in history:
                            row = {
                                "prediction": p.get("prediction", 0),
                                "drift_ratio": p.get("drift_ratio", 0),
                                "latency_ms": p.get("latency_ms", 0),
                            }
                            current_rows.append(row)
                        current_df = pd.DataFrame(current_rows)

                        # KS test on available shared features
                        shared_features = [
                            f for f in ref_df.columns if f in current_df.columns
                        ]

                        ks_results = {}
                        if shared_features:
                            ks_results = detector.detect_ks_drift(
                                current_df[shared_features], features=shared_features
                            )

                        # Full drift metrics
                        metrics_result = {}
                        if shared_features:
                            metrics_result = detector.compute_drift_metrics(
                                current_df[shared_features]
                            )

                        # Display summary metrics
                        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                        with col_k1:
                            drift_ratio = metrics_result.get("drift_ratio", 0)
                            st.metric("Drift Ratio", f"{drift_ratio:.2%}")
                        with col_k2:
                            st.metric(
                                "Drifted Features",
                                f"{metrics_result.get('num_drifted_features', 0)}"
                                f" / {metrics_result.get('total_features', 0)}",
                            )
                        with col_k3:
                            wd = metrics_result.get("avg_wasserstein_distance", 0)
                            st.metric("Avg Wasserstein Dist", f"{wd:.4f}")
                        with col_k4:
                            max_wd = metrics_result.get("max_wasserstein_distance", 0)
                            st.metric("Max Wasserstein Dist", f"{max_wd:.4f}")

                        # KS heatmap
                        if ks_results:
                            st.markdown("#### KS Test p-values by Feature")
                            fig_ks = make_ks_heatmap(ks_results)
                            st.plotly_chart(
                                fig_ks, use_container_width=True,
                                config={"displayModeBar": False},
                            )

                            # Results table
                            ks_df = pd.DataFrame([
                                {
                                    "Feature": name,
                                    "p-value": f"{r.p_value:.4f}",
                                    "KS Statistic": f"{r.test_statistic:.4f}",
                                    "Drift Detected": "⚠ Yes" if r.is_drift else "✓ No",
                                }
                                for name, r in ks_results.items()
                            ])

                            def _highlight_drift(row):
                                if "⚠" in str(row.get("Drift Detected", "")):
                                    return ["background-color: rgba(239,68,68,0.15)"] * len(row)
                                return [""] * len(row)

                            st.dataframe(
                                ks_df.style.apply(_highlight_drift, axis=1),
                                use_container_width=True,
                                height=300,
                            )
                        else:
                            st.info("No shared features between reference and current data for KS test.")

                    except Exception as e:
                        st.error(f"KS analysis failed: {e}", icon="❌")
                        st.exception(e)

st.divider()

# ---------------------------------------------------------------------------
# Alert History
# ---------------------------------------------------------------------------
with st.expander("🔔 Alert History", expanded=False):
    if not alert_manager or not alert_manager.alert_history:
        st.info("No alerts triggered yet. Make predictions to generate alert data.", icon="ℹ️")
    else:
        col_filter, col_clear = st.columns([2, 1])
        with col_filter:
            severity_filter = st.selectbox(
                "Filter by severity",
                ["All", "critical", "warning", "info"],
                index=0,
            )
        with col_clear:
            if st.button("🗑️ Clear Alert History", use_container_width=True):
                alert_manager.clear_history()
                st.success("Alert history cleared.")
                st.rerun()

        # Get filtered alerts
        from src.monitoring.alerts import AlertSeverity
        severity_map = {
            "critical": AlertSeverity.CRITICAL,
            "warning":  AlertSeverity.WARNING,
            "info":     AlertSeverity.INFO,
        }
        sev = severity_map.get(severity_filter)
        alerts = alert_manager.get_alert_history(severity=sev, limit=100)

        alert_rows = [
            {
                "Time": a.timestamp.strftime("%H:%M:%S"),
                "Severity": a.severity.value.upper(),
                "Rule": a.name,
                "Message": a.message,
                "Metric": a.metric_name,
                "Value": f"{a.metric_value:.4f}",
                "Threshold": f"{a.threshold:.4f}",
            }
            for a in reversed(alerts)
        ]

        if alert_rows:
            df_alerts = pd.DataFrame(alert_rows)

            def _color_severity(row):
                sev = str(row.get("Severity", ""))
                if sev == "CRITICAL":
                    return ["background-color: rgba(239,68,68,0.2)"] * len(row)
                elif sev == "WARNING":
                    return ["background-color: rgba(245,158,11,0.15)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_alerts.style.apply(_color_severity, axis=1),
                use_container_width=True,
                height=350,
            )
        else:
            st.info(f"No {severity_filter} alerts found.")

st.divider()

# ---------------------------------------------------------------------------
# Alert Rules
# ---------------------------------------------------------------------------
with st.expander("📋 Alert Rules", expanded=False):
    if alert_manager:
        rules_data = [
            {
                "Rule Name": name,
                "Metric": rule.metric_name,
                "Comparison": rule.comparison,
                "Threshold": rule.threshold,
                "Severity": rule.severity.value.upper(),
                "Message Template": rule.message_template[:60] + "..." if len(rule.message_template) > 60 else rule.message_template,
            }
            for name, rule in alert_manager.rules.items()
        ]
        st.dataframe(pd.DataFrame(rules_data), use_container_width=True)
        st.markdown(
            "<p style='color:#64748b;font-size:0.8rem;'>Comparisons: "
            "<code>gt</code> = greater than, <code>lt</code> = less than, "
            "<code>gte/lte</code> = ≥/≤</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Alert manager not initialized.", icon="ℹ️")

st.divider()

# ---------------------------------------------------------------------------
# Prometheus Metrics (raw)
# ---------------------------------------------------------------------------
with st.expander("⚙️ Prometheus Metrics (Raw)", expanded=False):
    try:
        from prometheus_client import generate_latest
        raw_metrics = generate_latest().decode("utf-8")
        if raw_metrics.strip():
            st.code(raw_metrics, language=None)
        else:
            st.info("No Prometheus metrics collected yet. Metrics accumulate after predictions.", icon="ℹ️")
    except Exception as e:
        st.warning(f"Could not retrieve Prometheus metrics: {e}", icon="⚠️")
        st.markdown(
            "Note: Prometheus metrics are collected by the FastAPI app (`src/api/main.py`). "
            "The Streamlit app tracks session-level stats separately."
        )

st.divider()

# ---------------------------------------------------------------------------
# Session prediction table (last 24 for drift context)
# ---------------------------------------------------------------------------
if history:
    with st.expander("📊 Last 24 Predictions (Session)", expanded=False):
        rows = [
            {
                "Time": pd.to_datetime(p["timestamp"]).strftime("%H:%M:%S"),
                "Prediction": f"{p['prediction']:.6f}",
                "Risk": p.get("risk_level", "—"),
                "Drift Detected": "⚠" if p.get("drift_detected") else "✓",
                "Drift Ratio": f"{p.get('drift_ratio',0):.2%}",
                "Latency ms": f"{p.get('latency_ms',0):.1f}",
            }
            for p in history[-24:][::-1]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
