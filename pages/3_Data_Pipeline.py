"""
Data Pipeline page — extract raw EUR/USD data and run feature engineering.
Requires TWELVE_DATA_API_KEY environment variable or Streamlit secret.
"""

import streamlit as st
import pandas as pd
import os
import io
import logging
from pathlib import Path

from app.css_theme import inject_css
from app.state import init_session_state
from app.charts import make_feature_distribution

st.set_page_config(
    page_title="Data Pipeline | USD MLOps",
    page_icon="🔄",
    layout="wide",
)
inject_css()
init_session_state()

# ---------------------------------------------------------------------------
# StreamlitLogHandler — captures logging output live into a code block
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    """Redirect log records into a Streamlit st.empty() code block."""

    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.lines = []
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record):
        self.lines.append(self.format(record))
        # Show last 40 lines
        self.placeholder.code("\n".join(self.lines[-40:]), language=None)


def attach_log_handler(loggers: list, placeholder) -> StreamlitLogHandler:
    handler = StreamlitLogHandler(placeholder)
    for name in loggers:
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    return handler


def detach_log_handler(loggers: list, handler: StreamlitLogHandler):
    for name in loggers:
        logging.getLogger(name).removeHandler(handler)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>🔄 Data Pipeline</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;'>Extract EUR/USD OHLCV data from Twelve Data API "
    "and engineer the 33 features required by the model.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
api_key = os.getenv("TWELVE_DATA_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        api_key = ""

if not api_key:
    st.warning(
        "⚠️ **TWELVE_DATA_API_KEY not configured.** Data extraction is disabled.\n\n"
        "Set the key as an environment variable or in `.streamlit/secrets.toml`:\n"
        "```toml\nTWELVE_DATA_API_KEY = \"your_key_here\"\n```",
        icon="🔑",
    )
else:
    st.success("✅ API key configured.", icon="🔑")

extraction_enabled = bool(api_key)

# ---------------------------------------------------------------------------
# Configuration expander
# ---------------------------------------------------------------------------
with st.expander("⚙️ Pipeline Configuration", expanded=True):
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        symbol = st.text_input("Symbol", value="EUR/USD", disabled=True)
        interval = st.selectbox("Interval", ["1h", "4h", "1day"], index=0)
    with col_c2:
        outputsize = st.slider(
            "Data Points (outputsize)",
            min_value=48, max_value=500, value=168, step=24,
            help="168 = 1 week of hourly data",
        )
    with col_c3:
        save_to_disk = st.checkbox(
            "Save files to disk",
            value=False,
            help="When enabled, saves raw CSV and processed parquet to data/ directory. "
                 "Disable for Streamlit Cloud (read-only filesystem).",
        )

st.divider()

# ---------------------------------------------------------------------------
# Step 1: Extract raw data
# ---------------------------------------------------------------------------
st.markdown("### Step 1 — Extract Raw Data")
col_btn1, col_info1 = st.columns([1, 3])
with col_btn1:
    extract_btn = st.button(
        "📡 Extract Data",
        type="primary",
        disabled=not extraction_enabled,
        use_container_width=True,
        help="Fetch EUR/USD data from Twelve Data API",
    )

if extract_btn:
    log_placeholder = st.empty()
    log_names = ["data_extraction", "twelve_data_client", "data_quality_checker"]
    handler = attach_log_handler(log_names, log_placeholder)

    try:
        from src.data.data_extraction import extract_forex_data

        with st.spinner("Fetching data from Twelve Data API..."):
            raw_df = extract_forex_data(
                api_key=api_key,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                save_raw=save_to_disk,
            )

        st.session_state["raw_data"] = raw_df
        st.success(f"✅ Extracted {len(raw_df)} rows of {symbol} {interval} data.", icon="✅")

    except Exception as e:
        st.error(f"Extraction failed: {e}", icon="❌")
    finally:
        detach_log_handler(log_names, handler)

# Show raw data if available
raw_df = st.session_state.get("raw_data")
if raw_df is not None:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Rows", len(raw_df))
    with col_m2:
        st.metric("Columns", len(raw_df.columns))
    with col_m3:
        null_pct = raw_df.isnull().sum().sum() / (len(raw_df) * len(raw_df.columns)) * 100
        st.metric("Null %", f"{null_pct:.2f}%")
    with col_m4:
        if "datetime" in raw_df.columns or raw_df.index.name == "datetime":
            dt_col = raw_df.index if raw_df.index.name == "datetime" else raw_df["datetime"]
            try:
                ts_range = f"{pd.to_datetime(dt_col).min().date()} → {pd.to_datetime(dt_col).max().date()}"
            except Exception:
                ts_range = "—"
        else:
            ts_range = "—"
        st.metric("Date Range", ts_range)

    st.dataframe(raw_df.head(50), use_container_width=True, height=300)

st.divider()

# ---------------------------------------------------------------------------
# Step 2: Transform data
# ---------------------------------------------------------------------------
st.markdown("### Step 2 — Feature Engineering")
col_btn2, _ = st.columns([1, 3])
with col_btn2:
    transform_btn = st.button(
        "⚙️ Transform Data",
        type="primary",
        disabled=(raw_df is None),
        use_container_width=True,
        help="Run feature engineering on extracted data",
    )

if raw_df is None:
    st.info("ℹ️ Run Step 1 first to extract raw data.", icon="ℹ️")

if transform_btn and raw_df is not None:
    log_placeholder2 = st.empty()
    log_names2 = ["data_transformation", "feature_engineer", "data_cleaner"]
    handler2 = attach_log_handler(log_names2, log_placeholder2)

    try:
        from src.data.data_transformation import transform_data

        with st.spinner("Engineering 33 features..."):
            processed_df = transform_data(raw_df, save_processed=save_to_disk)

        st.session_state["processed_data"] = processed_df
        st.success(
            f"✅ Feature engineering complete: {len(processed_df)} rows × {len(processed_df.columns)} features.",
            icon="✅",
        )

    except Exception as e:
        st.error(f"Transformation failed: {e}", icon="❌")
    finally:
        detach_log_handler(log_names2, handler2)

# Show processed data if available
processed_df = st.session_state.get("processed_data")
if processed_df is not None:
    col_pm1, col_pm2, col_pm3, col_pm4 = st.columns(4)
    with col_pm1:
        st.metric("Output Rows", len(processed_df))
    with col_pm2:
        st.metric("Features", len(processed_df.columns))
    with col_pm3:
        null_count = processed_df.isnull().sum().sum()
        st.metric("Nulls Remaining", null_count)
    with col_pm4:
        raw_rows = len(raw_df) if raw_df is not None else 0
        dropped = raw_rows - len(processed_df)
        st.metric("Rows Dropped", dropped, help="Dropped during cleaning/outlier removal")

    # Feature distributions chart
    st.markdown("#### Feature Distributions")
    fig_dist = make_feature_distribution(processed_df)
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    # Preview table
    st.markdown("#### Processed Data Preview (first 100 rows)")
    st.dataframe(processed_df.head(100), use_container_width=True, height=350)

    # Download button
    parquet_buf = io.BytesIO()
    processed_df.to_parquet(parquet_buf, index=False)
    parquet_buf.seek(0)
    st.download_button(
        label="📥 Download Processed Parquet",
        data=parquet_buf,
        file_name="processed_data.parquet",
        mime="application/octet-stream",
        help="Download the feature-engineered dataset for offline use",
    )

    # Feature stats expander
    with st.expander("📊 Feature Statistics", expanded=False):
        numeric_df = processed_df.select_dtypes(include=["number"])
        stats_df = numeric_df.describe().T
        stats_df.index.name = "Feature"
        st.dataframe(stats_df.round(6), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Existing local data files
# ---------------------------------------------------------------------------
with st.expander("📁 Local Data Files", expanded=False):
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    st.markdown("**Raw data files:**")
    raw_files = sorted(Path(RAW_DATA_DIR).glob("*.csv")) if Path(RAW_DATA_DIR).exists() else []
    if raw_files:
        for f in raw_files[-10:]:
            size_kb = f.stat().st_size / 1024
            st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")
    else:
        st.markdown("_No raw data files found._")

    st.markdown("**Processed data files:**")
    proc_files = sorted(Path(PROCESSED_DATA_DIR).glob("*.parquet")) if Path(PROCESSED_DATA_DIR).exists() else []
    if proc_files:
        for f in proc_files[-10:]:
            size_kb = f.stat().st_size / 1024
            st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")
    else:
        st.markdown("_No processed parquet files found._")
