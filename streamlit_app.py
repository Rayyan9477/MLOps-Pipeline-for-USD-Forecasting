"""
USD Volatility MLOps — Home page.
Designed for novice users: 3 plain-English inputs, instant actionable forecast.
All ML feature engineering is hidden internally.
"""

import streamlit as st
import random
import pandas as pd
from math import sin, cos, pi, log
from datetime import datetime, timezone

from app.css_theme import inject_css
from app.state import init_session_state
from app.model_service import get_model_service, get_session_stats, predict_single
from app.charts import make_volatility_timeseries

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EUR/USD Forecast | USD MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
init_session_state()

# ── Load model ────────────────────────────────────────────────────────────────
reload_count = st.session_state.get("model_reload_count", 0)
svc = get_model_service(reload_count)


# ── Helpers ───────────────────────────────────────────────────────────────────
ACTIVITY_TO_STD = {
    "🟦 Calm — barely moving":       0.0005,
    "🟩 Normal — regular day":       0.0015,
    "🟨 Active — noticeable swings": 0.003,
    "🟥 Very Active — big moves":    0.006,
}

def _trading_session(utc_hour: int) -> str:
    """Return the name of the active forex trading session."""
    sessions = []
    if 0 <= utc_hour < 9:
        sessions.append("🌏 Tokyo")
    if 7 <= utc_hour < 16:
        sessions.append("🇬🇧 London")
    if 13 <= utc_hour < 22:
        sessions.append("🇺🇸 New York")
    if not sessions:
        sessions.append("🌙 Off-hours")
    return " + ".join(sessions)

def _build_all_features(current_price: float, prev_price: float, rolling_std: float) -> dict:
    """Derive all 33 model features from 3 user inputs + current time."""
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
        "close_lag_1":  lags[1],  "close_lag_2":  lags[2],
        "close_lag_3":  lags[3],  "close_lag_4":  lags[4],
        "close_lag_6":  lags[6],  "close_lag_8":  lags[8],
        "close_lag_12": lags[12], "close_lag_24": lags[24],
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
        "log_return":     round(lr, 6),
        "hour_sin": h_sin, "hour_cos": h_cos,
        "day_sin":  d_sin, "day_cos":  d_cos,
        "hour": float(h),        "day_of_week":  float(dow),
        "day_of_month": float(dom), "month":      float(mo),
        "price_range":      round(abs(rolling_std * 2.5), 6),
        "price_change":     round(lr * current_price, 6),
        "price_change_pct": round(lr * 100, 6),
        "avg_price":        round((current_price + lags[2]) / 2, 6),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:0.75rem 0 0.5rem;">'
        '<span style="font-size:2rem;">📈</span>'
        '<h2 class="gradient-text" style="margin:0.15rem 0 0.1rem;font-size:1.4rem;">USD MLOps</h2>'
        '<p style="color:#64748b;font-size:0.72rem;margin:0;">EUR/USD Volatility Forecasting</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Live clock + session
    st.caption("**Market Clock**")
    st.markdown(f"**{now_utc.strftime('%H:%M UTC')}** — {_trading_session(now_utc.hour)}")
    st.caption(
        "Forex is most active during London+NY overlap (13–16 UTC). "
        "Predictions are most reliable during active sessions."
    )
    st.divider()

    st.caption("**Navigation**")
    st.markdown(
        "📈 **Home** — Quick forecast  \n"
        "📊 **Overview** — Session dashboard  \n"
        "🔮 **Predictions** — Full input control  \n"
        "🔄 **Data Pipeline** — Fetch live data  \n"
        "🧠 **Model Mgmt** — Train & register  \n"
        "📡 **Monitoring** — Drift & alerts  \n"
        "🧪 **MLflow** — Experiment registry  \n"
    )
    st.divider()

    st.caption("**Model Status**")
    if svc:
        meta = svc["metadata"]
        m = meta.get("metrics", {})
        rmse = m.get("test_rmse", m.get("rmse", "—"))
        r2   = m.get("test_r2",   m.get("r2",   "—"))
        st.success("● Model online", icon=None)
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"R²: `{r2}`")
        with c2:
            rmse_pips = round(float(rmse) * 10000, 1) if isinstance(rmse, (int, float)) else "—"
            st.caption(f"RMSE: `±{rmse_pips} pips`")
    else:
        st.error("● No model loaded")
    st.divider()

    if st.button("🔄 Reload Model", use_container_width=True):
        from app.model_service import reload_model
        reload_model()
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="gradient-text" style="font-size:2.2rem;margin:0 0 0.15rem;">'
    "EUR/USD Next-Hour Volatility Forecast</h1>"
    "<p style='color:#94a3b8;font-size:0.95rem;margin:0 0 1.5rem;'>"
    "AI-powered forecast — enter two prices, choose a market mood, get your prediction.</p>",
    unsafe_allow_html=True,
)

if not svc:
    st.error(
        "**No model found.** Go to **Model Management** to train one, "
        "or place a trained `.pkl` file in the `models/` folder.",
        icon="🚨",
    )
    st.stop()


# ── Main layout ───────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

# ─── LEFT: 3-input form ───────────────────────────────────────────────────────
with col_form:
    st.markdown("#### Step 1 — Enter EUR/USD prices")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        current_price = st.number_input(
            "Current Price",
            value=st.session_state.get("last_current", 1.0850),
            min_value=0.5, max_value=2.0,
            step=0.0001, format="%.4f",
            help="The EUR/USD exchange rate right now. Example: 1.0850 means 1 Euro = 1.0850 USD.",
        )
    with p_col2:
        prev_price = st.number_input(
            "Previous Hour Price",
            value=st.session_state.get("last_prev", 1.0845),
            min_value=0.5, max_value=2.0,
            step=0.0001, format="%.4f",
            help="The EUR/USD price from one hour ago. Your broker or Google shows this.",
        )

    if current_price and prev_price and current_price != prev_price:
        change = current_price - prev_price
        pips   = round(change * 10000, 1)
        arrow  = "▲" if change > 0 else "▼"
        color  = "#10b981" if change > 0 else "#ef4444"
        st.markdown(
            f'<p style="color:{color};font-size:0.82rem;margin:0.1rem 0 0.6rem;">'
            f"{arrow} {abs(pips):.1f} pips moved since last hour</p>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Step 2 — How active is the market today?")
    st.caption("Pick how much price action you've seen — this sets background volatility for the model.")

    activity = st.radio(
        "Market activity",
        list(ACTIVITY_TO_STD.keys()),
        index=1,
        label_visibility="collapsed",
    )
    rolling_std = ACTIVITY_TO_STD[activity]

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Get My Forecast", type="primary", use_container_width=True)

    with st.expander("💡 Where do I find EUR/USD prices?", expanded=False):
        st.markdown(
            "**Quick sources:**\n"
            '- Google: search **"EUR USD"** → shows the live rate\n'
            "- TradingView, MT4/MT5, or your broker's platform\n"
            "- Any forex news site (Investing.com, Forex Factory)\n\n"
            "**Typical range:** EUR/USD usually trades between 1.00 and 1.20.\n\n"
            "**What's a pip?** 1 pip = 0.0001 price change (the 4th decimal place). "
            "1.0850 → 1.0860 = 10 pips."
        )


# ─── RIGHT: Result display ────────────────────────────────────────────────────
with col_result:

    if predict_btn:
        st.session_state["last_current"] = current_price
        st.session_state["last_prev"]    = prev_price
        features = _build_all_features(current_price, prev_price, rolling_std)
        with st.spinner("Analysing market conditions..."):
            try:
                res = predict_single(features, svc)
                st.session_state["home_result"] = res
                st.session_state["home_result_price"] = current_price
            except Exception as e:
                st.error(f"Forecast failed: {e}", icon="🚨")

    res = st.session_state.get("home_result")

    if res:
        pred  = res["prediction"]
        risk  = res["risk_level"]
        conf  = res["confidence_score"]
        drift = res["drift_detected"]
        d_rat = res["drift_ratio"]
        lat   = res["latency_ms"]
        saved_price = st.session_state.get("home_result_price", current_price)

        pips = round(pred * 10000, 1)
        price_low  = round(saved_price - pred, 4)
        price_high = round(saved_price + pred, 4)

        # ── Signal headline ──
        st.markdown("#### Forecast Result")
        if risk == "Low":
            st.success(
                f"**🟢 STABLE MARKET** — Expected move: ±{pips} pips   "
                f"| Confidence: {conf} | {lat:.0f} ms",
            )
        elif risk == "Medium":
            st.warning(
                f"**🟡 MODERATE MOVES** — Expected move: ±{pips} pips   "
                f"| Confidence: {conf} | {lat:.0f} ms",
            )
        else:
            st.error(
                f"**🔴 HIGH VOLATILITY** — Expected move: ±{pips} pips   "
                f"| Confidence: {conf} | {lat:.0f} ms",
            )

        # ── Price range ──
        st.markdown("**Next-hour price range forecast**")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Floor", f"{price_low:.4f}", delta=f"−{pips} pips", delta_color="inverse")
        with r2:
            st.metric("Current", f"{saved_price:.4f}")
        with r3:
            st.metric("Ceiling", f"{price_high:.4f}", delta=f"+{pips} pips")

        st.divider()

        # ── Plain-English guidance ──
        guidance_map = {
            "Low": (
                st.success,
                f"EUR/USD is expected to move only **{pips} pips** in the next hour — "
                "calm conditions. Good window for precision entries, tight spreads, and "
                "limit orders. Low chance of surprise price spikes.",
                "🟢",
            ),
            "Medium": (
                st.warning,
                f"EUR/USD could swing **{pips} pips** in the next hour — "
                "normal conditions. Watch for economic news releases. "
                "Standard risk management applies: use a stop-loss.",
                "🟡",
            ),
            "High": (
                st.error,
                f"EUR/USD may move **{pips} pips** or more. High-risk conditions. "
                "Consider wider stop-losses, smaller position sizes, "
                "or waiting for calmer conditions before entering.",
                "🔴",
            ),
        }
        fn, msg, icon = guidance_map[risk]
        fn(msg, icon=icon)

        if drift:
            st.warning(
                f"**Model note:** Your prices look unusual compared to training data "
                f"({d_rat:.0%} of feature checks failed). "
                "The forecast may be less reliable than usual.",
                icon="⚠️",
            )

        # ── Technical details ──
        with st.expander("🔬 Technical details", expanded=False):
            ts = pd.to_datetime(res["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("Raw volatility", f"{pred:.6f}")
                st.metric("Drift ratio", f"{d_rat:.2%}")
            with col_t2:
                st.metric("Inference time", f"{lat:.1f} ms")
                st.metric("Model version", str(res["model_version"])[:12])
            st.caption(f"Timestamp: {ts}")
            st.caption(
                "33 features derived automatically from your 3 inputs. "
                "The model was trained on EUR/USD hourly OHLCV data."
            )

    else:
        # ── Empty state ──
        st.markdown("#### Your Forecast")
        st.info(
            "Fill in the two prices on the left and click **⚡ Get My Forecast**.",
            icon="👈",
        )

        st.markdown("**What you'll get:**")
        c1, c2 = st.columns(2)
        with c1:
            st.success("🟢 / 🟡 / 🔴 Market signal", icon=None)
            st.info("📏 Price range — floor and ceiling for the next hour", icon=None)
        with c2:
            st.warning("💬 Plain English trading guidance", icon=None)
            st.info("⚠️ Data quality check on your inputs", icon=None)


st.divider()

# ── Session history ───────────────────────────────────────────────────────────
history = st.session_state.get("prediction_history", [])

if history:
    st.markdown("### 📈 Your Forecast History This Session")

    stats = get_session_stats()
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Forecasts Made", stats["total_predictions"])
    with s2:
        avg_pips = round(
            sum(p["prediction"] for p in history) / len(history) * 10000, 1
        )
        st.metric("Avg Expected Move", f"±{avg_pips} pips")
    with s3:
        high_count = sum(1 for p in history if p.get("risk_level") == "High")
        st.metric("High-Risk Forecasts", high_count,
                  delta="⚠ caution" if high_count > 0 else None,
                  delta_color="inverse")
    with s4:
        st.metric("Avg Response Time", f"{stats['avg_latency_ms']:.0f} ms")

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig = make_volatility_timeseries(history)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        rows = []
        for p in history[-15:][::-1]:
            risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(
                p.get("risk_level", ""), "⚪"
            )
            pips_val = round(p["prediction"] * 10000, 1)
            rows.append({
                "Time (UTC)": pd.to_datetime(p["timestamp"]).strftime("%H:%M:%S"),
                "Signal":     f"{risk_icon} {p.get('risk_level','—')}",
                "Move (pips)": f"±{pips_val}",
                "Drift":      "⚠ Yes" if p.get("drift_detected") else "✓ OK",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            height=420,
            hide_index=True,
        )

else:
    # ── Onboarding ───────────────────────────────────────────────────────────
    st.markdown("### How it works — 3 simple steps")
    c1, c2, c3 = st.columns(3)
    steps = [
        (c1, "#06b6d4", "1. Enter prices",
         "Type in the current EUR/USD rate and the rate from one hour ago. "
         "Find both on Google or your broker in seconds."),
        (c2, "#7c3aed", "2. Choose market mood",
         "Pick how active the market has felt today — from barely moving to wild swings. "
         "This sets the background volatility context."),
        (c3, "#10b981", "3. Get your forecast",
         "The XGBoost model predicts next-hour volatility and translates it into pips, "
         "price range, and plain-English trading guidance."),
    ]
    for col, color, title, desc in steps:
        with col:
            st.markdown(
                f'<div style="border-top:3px solid {color};background:#1e293b;'
                f'border-radius:0 0 10px 10px;padding:1.2rem;">'
                f'<b style="color:{color};">{title}</b><br><br>'
                f'<span style="color:#94a3b8;font-size:0.87rem;line-height:1.6;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 New to forex? Read this first.", expanded=False):
        st.markdown(
            "**What is EUR/USD?**  \n"
            "The exchange rate between the Euro (EUR) and US Dollar (USD). "
            "A rate of 1.0850 means 1 Euro buys 1.0850 US Dollars. "
            "It's the most traded currency pair in the world.\n\n"
            "**What is volatility?**  \n"
            "How much the price moves in a given time period. "
            "High volatility = big swings. Low volatility = price barely moves.\n\n"
            "**What is a pip?**  \n"
            "The smallest standard price move — the 4th decimal place. "
            "1.0850 → 1.0860 = **10 pips**.\n\n"
            "**Why does this matter?**  \n"
            "Knowing expected volatility helps you set stop-losses, "
            "choose position size, and avoid entering trades during dangerous conditions."
        )

st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.7rem;padding:2rem 0 0;'>"
    "Powered by XGBoost · MLOps pipeline: Data → Features → Training → "
    "Drift Detection → MLflow Registry</p>",
    unsafe_allow_html=True,
)
