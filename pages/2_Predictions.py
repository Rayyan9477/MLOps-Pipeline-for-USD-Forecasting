"""
Predictions page — Simple Mode (3 plain-English inputs) and Advanced Mode (6 ML features).
Batch CSV upload for bulk predictions.
"""

import streamlit as st
import pandas as pd
import random
import io
from datetime import datetime, timezone
from math import sin, cos, pi, log

from app.css_theme import inject_css
from app.state import init_session_state
from app.model_service import get_model_service, predict_single, predict_batch

st.set_page_config(
    page_title="Predictions | USD MLOps",
    page_icon="🔮",
    layout="wide",
)
inject_css()
init_session_state()

# ---------------------------------------------------------------------------
# Feature names (exact 33)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4",
    "close_lag_6", "close_lag_8", "close_lag_12", "close_lag_24",
    "close_rolling_mean_4", "close_rolling_mean_8", "close_rolling_mean_24",
    "close_rolling_std_4", "close_rolling_std_8", "close_rolling_std_24",
    "close_rolling_min_4", "close_rolling_max_4",
    "close_rolling_min_8", "close_rolling_max_8",
    "close_rolling_min_24", "close_rolling_max_24",
    "log_return",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "hour", "day_of_week", "day_of_month", "month",
    "price_range", "price_change", "price_change_pct", "avg_price",
]

ACTIVITY_TO_STD = {
    "🟦 Calm — barely moving":       0.0005,
    "🟩 Normal — regular day":       0.0015,
    "🟨 Active — noticeable swings": 0.003,
    "🟥 Very Active — big moves":    0.006,
}


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _build_features_simple(current_price: float, prev_price: float, rolling_std: float) -> dict:
    """Derive all 33 features from 3 plain-English inputs + current time."""
    now = datetime.now(timezone.utc)
    h, dow, dom, mo = now.hour, now.weekday(), now.day, now.month

    lr = log(current_price / prev_price) if prev_price > 0 else 0.0
    mean_24 = (current_price + prev_price) / 2

    h_sin = round(sin(2 * pi * h / 24), 6)
    h_cos = round(cos(2 * pi * h / 24), 6)
    d_sin = round(sin(2 * pi * dow / 7), 6)
    d_cos = round(cos(2 * pi * dow / 7), 6)

    lags, p = {}, current_price
    for n in [2, 3, 4, 6, 8, 12, 24]:
        p = p * (1 + random.uniform(-0.0003, 0.0003))
        lags[n] = round(p, 6)
    lags[1] = current_price

    std_4  = rolling_std * random.uniform(0.7, 1.2)
    std_8  = rolling_std * random.uniform(0.85, 1.15)
    mean_4 = current_price * (1 + random.uniform(-0.0001, 0.0001))
    mean_8 = current_price * (1 + random.uniform(-0.0002, 0.0002))

    return {
        "close_lag_1": lags[1],     "close_lag_2": lags[2],
        "close_lag_3": lags[3],     "close_lag_4": lags[4],
        "close_lag_6": lags[6],     "close_lag_8": lags[8],
        "close_lag_12": lags[12],   "close_lag_24": lags[24],
        "close_rolling_mean_4":  round(mean_4, 6),
        "close_rolling_mean_8":  round(mean_8, 6),
        "close_rolling_mean_24": round(mean_24, 6),
        "close_rolling_std_4":   round(std_4, 6),
        "close_rolling_std_8":   round(std_8, 6),
        "close_rolling_std_24":  rolling_std,
        "close_rolling_min_4":   round(mean_4 - std_4 * 1.5, 6),
        "close_rolling_max_4":   round(mean_4 + std_4 * 1.5, 6),
        "close_rolling_min_8":   round(mean_8 - std_8 * 1.8, 6),
        "close_rolling_max_8":   round(mean_8 + std_8 * 1.8, 6),
        "close_rolling_min_24":  round(mean_24 - rolling_std * 2.0, 6),
        "close_rolling_max_24":  round(mean_24 + rolling_std * 2.0, 6),
        "log_return": round(lr, 6),
        "hour_sin": h_sin,     "hour_cos": h_cos,
        "day_sin":  d_sin,     "day_cos":  d_cos,
        "hour": float(h),      "day_of_week": float(dow),
        "day_of_month": float(dom), "month": float(mo),
        "price_range":     round(abs(rolling_std * 2.5), 6),
        "price_change":    round(lr * current_price, 6),
        "price_change_pct": round(lr * 100, 6),
        "avg_price":       round((current_price + lags[2]) / 2, 6),
    }


def _build_features_advanced(
    close_lag_1: float, rolling_mean_24: float, rolling_std_24: float,
    log_ret: float, hour_sin: float, hour_cos: float,
) -> dict:
    """Derive all 33 features from 6 technical ML inputs."""
    now = datetime.now(timezone.utc)
    h, dow, dom, mo = now.hour, now.weekday(), now.day, now.month

    d_sin = round(sin(2 * pi * dow / 7), 6)
    d_cos = round(cos(2 * pi * dow / 7), 6)

    lags, prev = {}, close_lag_1
    for n in [2, 3, 4, 6, 8, 12, 24]:
        prev = prev * (1 + random.uniform(-0.0003, 0.0003))
        lags[f"close_lag_{n}"] = round(prev, 6)

    mean_4 = close_lag_1 * (1 + random.uniform(-0.0001, 0.0001))
    mean_8 = close_lag_1 * (1 + random.uniform(-0.0002, 0.0002))
    std_4  = rolling_std_24 * random.uniform(0.7, 1.2)
    std_8  = rolling_std_24 * random.uniform(0.85, 1.15)

    return {
        "close_lag_1": close_lag_1, **lags,
        "close_rolling_mean_4":  round(mean_4, 6),
        "close_rolling_mean_8":  round(mean_8, 6),
        "close_rolling_mean_24": round(rolling_mean_24, 6),
        "close_rolling_std_4":   round(std_4, 6),
        "close_rolling_std_8":   round(std_8, 6),
        "close_rolling_std_24":  round(rolling_std_24, 6),
        "close_rolling_min_4":   round(mean_4 - std_4 * 1.5, 6),
        "close_rolling_max_4":   round(mean_4 + std_4 * 1.5, 6),
        "close_rolling_min_8":   round(mean_8 - std_8 * 1.8, 6),
        "close_rolling_max_8":   round(mean_8 + std_8 * 1.8, 6),
        "close_rolling_min_24":  round(rolling_mean_24 - rolling_std_24 * 2.0, 6),
        "close_rolling_max_24":  round(rolling_mean_24 + rolling_std_24 * 2.0, 6),
        "log_return": log_ret,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "day_sin": d_sin,     "day_cos": d_cos,
        "hour": float(h),     "day_of_week": float(dow),
        "day_of_month": float(dom), "month": float(mo),
        "price_range":     round(abs(rolling_std_24 * 2.5), 6),
        "price_change":    round(log_ret * close_lag_1, 6),
        "price_change_pct": round(log_ret * 100, 6),
        "avg_price":       round((close_lag_1 + lags["close_lag_2"]) / 2, 6),
    }


def _random_advanced() -> dict:
    """Generate realistic random EUR/USD values for advanced inputs."""
    close = random.uniform(1.050, 1.100)
    std   = random.uniform(0.001, 0.003)
    lr    = random.uniform(-0.002, 0.002)
    h     = random.randint(0, 23)
    return {
        "close_lag_1":           round(close, 6),
        "close_rolling_mean_24": round(close + random.uniform(-0.001, 0.001), 6),
        "close_rolling_std_24":  round(std, 6),
        "log_return":            round(lr, 6),
        "hour_sin":              round(sin(2 * pi * h / 24), 6),
        "hour_cos":              round(cos(2 * pi * h / 24), 6),
    }


def _render_result(res: dict, current_price: float | None = None) -> None:
    """Render prediction result using native Streamlit components."""
    pred  = res["prediction"]
    risk  = res["risk_level"]
    conf  = res["confidence_score"]
    drift = res["drift_detected"]
    d_rat = res["drift_ratio"]
    lat   = res["latency_ms"]

    pips_expected = round(pred * 10000, 1)

    signal_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    icon = signal_icons.get(risk, "⚪")

    # Headline signal
    if risk == "Low":
        st.success(f"{icon} **STABLE MARKET** — Confidence: {conf} · {lat:.0f} ms", icon=None)
    elif risk == "Medium":
        st.warning(f"{icon} **MODERATE MOVES** — Confidence: {conf} · {lat:.0f} ms", icon=None)
    else:
        st.error(f"{icon} **HIGH VOLATILITY** — Confidence: {conf} · {lat:.0f} ms", icon=None)

    # Price range (if current_price known)
    if current_price:
        price_low  = round(current_price - pred, 4)
        price_high = round(current_price + pred, 4)
        st.markdown("**Expected next-hour price range**")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Low end",  f"{price_low:.4f}",  delta=f"−{pips_expected} pips", delta_color="inverse")
        with r2:
            st.metric("Current",  f"{current_price:.4f}")
        with r3:
            st.metric("High end", f"{price_high:.4f}", delta=f"+{pips_expected} pips", delta_color="normal")
    else:
        st.metric("Predicted volatility (pips)", f"±{pips_expected}")
        st.metric("Raw volatility",  f"{pred:.6f}")

    st.divider()

    # Plain English guidance
    guidance = {
        "Low": (
            f"EUR/USD is expected to move only about **{pips_expected} pips** in the next hour. "
            "Calm conditions — good for precision entries and tight spreads."
        ),
        "Medium": (
            f"EUR/USD could move around **{pips_expected} pips** in the next hour — "
            "normal conditions. Watch for scheduled news events."
        ),
        "High": (
            f"EUR/USD may swing **{pips_expected} pips** or more. "
            "High-risk — consider wider stop-losses or reduced position sizes."
        ),
    }
    if risk == "Low":
        st.success(guidance[risk], icon="🟢")
    elif risk == "Medium":
        st.warning(guidance[risk], icon="🟡")
    else:
        st.error(guidance[risk], icon="🔴")

    if drift:
        st.warning(
            f"**Model note:** Your inputs look unusual compared to training data "
            f"({d_rat:.0%} of checks failed). Forecast may be less accurate.",
            icon="⚠️",
        )

    # Technical details collapsed
    with st.expander("🔬 Technical details", expanded=False):
        ts = pd.to_datetime(res["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(
            f"| | |\n|---|---|\n"
            f"| Raw predicted volatility | `{pred:.6f}` ({pred*100:.4f}%) |\n"
            f"| Equivalent in pips | `{pips_expected} pips` |\n"
            f"| Drift ratio | `{d_rat:.2%}` |\n"
            f"| Inference latency | `{lat:.1f} ms` |\n"
            f"| Model version | `{res['model_version']}` |\n"
            f"| Timestamp | `{ts}` |\n"
        )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>🔮 Predictions</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;'>Forecast EUR/USD next-hour volatility — "
    "choose Simple Mode for quick use or Advanced Mode for full control.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Load model
reload_count = st.session_state.get("model_reload_count", 0)
svc = get_model_service(reload_count)

if not svc:
    st.error(
        "No model loaded. Go to **Model Management** to train one, "
        "or place a `.pkl` file in the `models/` folder.",
        icon="🚨",
    )
    st.stop()

feature_names_actual = svc.get("feature_names", FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Mode toggle + tabs
# ---------------------------------------------------------------------------
mode_col, _ = st.columns([2, 3])
with mode_col:
    mode = st.radio(
        "Input mode",
        ["🟢 Simple Mode", "⚙️ Advanced Mode"],
        horizontal=True,
        help="Simple Mode: enter two prices and choose a market mood. "
             "Advanced Mode: enter 6 ML feature values directly.",
    )
simple_mode = mode == "🟢 Simple Mode"

st.divider()

tab_single, tab_batch = st.tabs(["🔮 Single Prediction", "📦 Batch Prediction"])

# ===========================================================================
# TAB 1: Single Prediction
# ===========================================================================
with tab_single:
    col_form, col_result = st.columns([1, 1], gap="large")

    # ── SIMPLE MODE ──────────────────────────────────────────────────────────
    if simple_mode:
        with col_form:
            st.markdown(
                '<p style="color:#06b6d4;font-weight:700;font-size:0.8rem;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.3rem;">Step 1</p>'
                '<p style="color:#f8fafc;font-weight:600;font-size:1rem;margin:0 0 0.75rem;">'
                "Enter current EUR/USD prices</p>",
                unsafe_allow_html=True,
            )

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                current_price = st.number_input(
                    "Current Price",
                    value=st.session_state.get("pred_current", 1.0850),
                    min_value=0.5, max_value=2.0,
                    step=0.0001, format="%.4f",
                    help="The EUR/USD exchange rate right now. Example: 1.0850",
                )
            with p_col2:
                prev_price = st.number_input(
                    "Previous Hour Price",
                    value=st.session_state.get("pred_prev", 1.0845),
                    min_value=0.5, max_value=2.0,
                    step=0.0001, format="%.4f",
                    help="The EUR/USD price from one hour ago.",
                )

            if current_price != prev_price:
                change = current_price - prev_price
                pips   = round(change * 10000, 1)
                arrow  = "▲" if change > 0 else "▼"
                pip_color = "#10b981" if change > 0 else "#ef4444"
                st.markdown(
                    f'<p style="color:{pip_color};font-size:0.82rem;margin:0.1rem 0 0.8rem;">'
                    f"{arrow} {abs(pips):.1f} pips since last hour</p>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#7c3aed;font-weight:700;font-size:0.8rem;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.3rem;">Step 2</p>'
                '<p style="color:#f8fafc;font-weight:600;font-size:1rem;margin:0 0 0.2rem;">'
                "How active has the market been today?</p>"
                '<p style="color:#64748b;font-size:0.83rem;margin:0 0 0.75rem;">'
                "This sets background volatility conditions for the model.</p>",
                unsafe_allow_html=True,
            )

            activity = st.radio(
                "Market activity",
                list(ACTIVITY_TO_STD.keys()),
                index=1,
                label_visibility="collapsed",
            )
            rolling_std = ACTIVITY_TO_STD[activity]

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("⚡ Get My Forecast", type="primary", use_container_width=True)

            with st.expander("💡 Not sure what prices to enter?", expanded=False):
                st.markdown(
                    "Find the current EUR/USD rate on:\n"
                    "- Google: search **EUR USD**\n"
                    "- TradingView, MT4, or your broker\n\n"
                    "**Typical range:** 1.00 – 1.20  \n"
                    "**What's a pip?** The 4th decimal place — "
                    "1.0850 → 1.0860 = 10 pips."
                )

        with col_result:
            if predict_btn:
                st.session_state["pred_current"] = current_price
                st.session_state["pred_prev"]    = prev_price
                features = _build_features_simple(current_price, prev_price, rolling_std)
                with st.spinner("Analysing market conditions..."):
                    try:
                        res = predict_single(features, svc)
                        st.session_state["pred_result"] = res
                        st.session_state["pred_result_price"] = current_price
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")

            res = st.session_state.get("pred_result")
            if res:
                saved_price = st.session_state.get("pred_result_price")
                _render_result(res, saved_price)
            else:
                st.markdown(
                    '<div style="border:2px dashed #334155;border-radius:14px;'
                    'padding:3rem 2rem;text-align:center;margin-top:1rem;">'
                    '<p style="font-size:2.5rem;margin:0 0 0.5rem;">🔮</p>'
                    '<p style="color:#f8fafc;font-weight:600;margin:0 0 0.3rem;">'
                    "Your forecast will appear here</p>"
                    '<p style="color:#64748b;font-size:0.87rem;margin:0;">'
                    "Enter two prices and click <b>⚡ Get My Forecast</b></p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── ADVANCED MODE ────────────────────────────────────────────────────────
    else:
        if "adv_inputs" not in st.session_state:
            st.session_state["adv_inputs"] = {
                "close_lag_1":           1.0850,
                "close_rolling_mean_24": 1.0830,
                "close_rolling_std_24":  0.0015,
                "log_return":            0.0002,
                "hour_sin":              0.5000,
                "hour_cos":              0.8660,
            }

        with col_form:
            st.markdown("#### Technical Feature Inputs")
            st.markdown(
                "<p style='color:#94a3b8;font-size:0.85rem;'>Enter 6 primary ML features. "
                "The remaining 27 are automatically derived.</p>",
                unsafe_allow_html=True,
            )

            if st.button("🎲 Random Fill", use_container_width=True):
                st.session_state["adv_inputs"] = _random_advanced()
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            inp = st.session_state["adv_inputs"]

            close_lag_1 = st.number_input(
                "close_lag_1 — Previous close price",
                value=float(inp["close_lag_1"]), step=0.0001, format="%.6f",
                help="Most recent hourly closing price. Typical: 0.90 – 1.20",
            )
            rolling_mean_24 = st.number_input(
                "close_rolling_mean_24 — 24h rolling mean",
                value=float(inp["close_rolling_mean_24"]), step=0.0001, format="%.6f",
            )
            rolling_std_24 = st.number_input(
                "close_rolling_std_24 — 24h rolling volatility",
                value=float(inp["close_rolling_std_24"]), step=0.0001, format="%.6f",
                min_value=0.0,
                help="24h rolling std. Typical: 0.0005 – 0.005",
            )
            log_return = st.number_input(
                "log_return — Logarithmic return",
                value=float(inp["log_return"]), step=0.00001, format="%.6f",
                help="log(close/prev_close). Typical: −0.01 to 0.01",
            )
            hour_sin = st.number_input(
                "hour_sin — Hour of day (sine encoding)",
                value=float(inp["hour_sin"]), step=0.001, format="%.6f",
                min_value=-1.0, max_value=1.0,
            )
            hour_cos = st.number_input(
                "hour_cos — Hour of day (cosine encoding)",
                value=float(inp["hour_cos"]), step=0.001, format="%.6f",
                min_value=-1.0, max_value=1.0,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn_adv = st.button("⚡ Predict Volatility", type="primary", use_container_width=True)

        with col_result:
            st.markdown("#### Prediction Result")

            if predict_btn_adv:
                features = _build_features_advanced(
                    close_lag_1, rolling_mean_24, rolling_std_24,
                    log_return, hour_sin, hour_cos,
                )
                with st.spinner("Running inference..."):
                    try:
                        res = predict_single(features, svc)
                        st.session_state["pred_result_adv"] = res
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

            res_adv = st.session_state.get("pred_result_adv")
            if res_adv:
                _render_result(res_adv, current_price=None)

                with st.expander("🔍 All 33 Feature Values", expanded=False):
                    if predict_btn_adv:
                        feat_display = _build_features_advanced(
                            close_lag_1, rolling_mean_24, rolling_std_24,
                            log_return, hour_sin, hour_cos,
                        )
                        df_feats = pd.DataFrame(
                            [{"Feature": k, "Value": f"{v:.6f}"} for k, v in feat_display.items()]
                        )
                        st.dataframe(df_feats, use_container_width=True, height=400)
            else:
                st.markdown(
                    '<div style="border:2px dashed #334155;border-radius:14px;'
                    'padding:3rem 2rem;text-align:center;margin-top:1rem;">'
                    '<p style="font-size:2rem;margin:0 0 0.5rem;">🔮</p>'
                    '<p style="color:#64748b;margin:0;">Fill in the inputs and click '
                    "<b>Predict Volatility</b></p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── Session history (both modes) ─────────────────────────────────────────
    history = st.session_state.get("prediction_history", [])
    if history:
        st.divider()
        st.markdown("#### Session Prediction History")
        hist_data = []
        for p in history[::-1][:50]:
            risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(p.get("risk_level", ""), "⚪")
            pips = round(p["prediction"] * 10000, 1)
            hist_data.append({
                "Time (UTC)": pd.to_datetime(p["timestamp"]).strftime("%H:%M:%S"),
                "Signal":     f"{risk_icon} {p.get('risk_level','—')}",
                "Pips ±":     f"{pips}",
                "Drift":      "⚠" if p.get("drift_detected") else "✓",
                "Latency":    f"{p.get('latency_ms',0):.1f} ms",
            })
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True, height=280, hide_index=True)


# ===========================================================================
# TAB 2: Batch Prediction
# ===========================================================================
with tab_batch:
    st.markdown("#### Batch Prediction via CSV Upload")
    st.markdown(
        "<p style='color:#94a3b8;'>Upload a CSV with all 33 feature columns. "
        "Download the template below to get started.</p>",
        unsafe_allow_html=True,
    )

    # Template generation
    template_row = _build_features_simple(1.085, 1.083, 0.0015)
    template_ordered = {f: template_row.get(f, 0.0) for f in feature_names_actual}
    template_df = pd.DataFrame([template_ordered])

    st.download_button(
        label="📥 Download Template CSV",
        data=template_df.to_csv(index=False),
        file_name="prediction_template.csv",
        mime="text/csv",
        help="Template CSV with the correct 33 feature columns and one example row",
    )

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload CSV for batch prediction",
        type=["csv"],
        help="CSV must contain all 33 feature columns. Use the template above.",
    )

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.markdown(f"**Uploaded:** {len(batch_df)} rows × {len(batch_df.columns)} columns")

        missing_cols = [f for f in feature_names_actual if f not in batch_df.columns]
        extra_cols   = [c for c in batch_df.columns if c not in feature_names_actual]

        if missing_cols:
            st.warning(
                f"⚠️ Missing {len(missing_cols)} columns: "
                f"`{', '.join(missing_cols[:10])}{'...' if len(missing_cols) > 10 else ''}`  \n"
                "Missing columns will be filled with 0.0."
            )
        if extra_cols:
            st.info(f"ℹ️ {len(extra_cols)} extra columns will be ignored.")

        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Predictions", type="primary", use_container_width=True):
            rows = batch_df.to_dict(orient="records")
            with st.spinner(f"Running {len(rows)} predictions..."):
                results = predict_batch(rows, svc)

            result_df = pd.DataFrame(results)
            display_cols = [
                "timestamp", "prediction", "risk_level", "confidence_score",
                "drift_detected", "drift_ratio", "latency_ms",
            ]
            display_cols = [c for c in display_cols if c in result_df.columns]

            # Pip-equivalent column
            if "prediction" in result_df.columns:
                result_df.insert(
                    result_df.columns.get_loc("prediction") + 1,
                    "pips_±",
                    result_df["prediction"].apply(lambda v: round(v * 10000, 1)),
                )
                display_cols.insert(display_cols.index("prediction") + 1, "pips_±")

            st.dataframe(result_df[display_cols], use_container_width=True, height=400)

            n_drift  = result_df["drift_detected"].sum() if "drift_detected" in result_df else 0
            avg_pips = result_df["pips_±"].mean()        if "pips_±" in result_df else 0

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total Rows",      len(results))
            with c2: st.metric("Avg Volatility",  f"±{avg_pips:.1f} pips")
            with c3: st.metric("Drift Detected",  f"{n_drift}/{len(results)}")

            st.download_button(
                label="📥 Download Results CSV",
                data=result_df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
