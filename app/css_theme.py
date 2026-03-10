"""
Dark glassmorphic CSS theme.

Palette (single source of truth — matches index.html, charts.py P{} dict):
  Cyan      #06b6d4   primary accent
  Purple    #7c3aed   secondary accent
  Blue      #1e40af   tertiary / buttons
  Green     #10b981   success / Low risk
  Amber     #f59e0b   warning / Medium risk
  Red       #ef4444   error / High risk
  BG        #0f172a   page background
  Card      #1e293b   card / input background
  Border    #334155   subtle borders / grid
  Text      #94a3b8   muted labels
  White     #f8fafc   primary text
"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── App background ────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #f8fafc;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }
[data-testid="stSidebar"] p { color: #94a3b8; font-size: 0.85rem; }

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, rgba(30,41,59,0.85), rgba(15,23,42,0.92));
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.25rem 1.5rem !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(6,182,212,0.12);
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Plotly chart wrapper ───────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.85));
    border: 1px solid #334155;
    border-radius: 12px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(6,182,212,0.08);
    padding: 0.25rem;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
/* Primary */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #1e40af, #7c3aed) !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.025em;
    padding: 0.5rem 1.5rem;
    transition: all 0.25s ease;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #2563eb, #8b5cf6) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.35) !important;
}
/* Secondary / all other buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #7c3aed);
    color: #f8fafc;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.025em;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #8b5cf6);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.35);
    border: none;
}
.stButton > button:active { transform: translateY(0); }

/* Download button — cyan outline style */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #06b6d4 !important;
    border: 1.5px solid #06b6d4 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.25s ease;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(6,182,212,0.12) !important;
    box-shadow: 0 4px 16px rgba(6,182,212,0.25) !important;
    transform: translateY(-1px);
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
    color: #94a3b8;
    font-weight: 500;
    border-radius: 0;
    padding-bottom: 0.7rem;
    transition: color 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #06b6d4 !important;
    border-bottom: 2px solid #06b6d4 !important;
    font-weight: 600;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #f8fafc; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 1rem; }

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stNumberInput input,
.stTextInput  input,
.stTextArea   textarea {
    background: rgba(30,41,59,0.85) !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stNumberInput input:focus,
.stTextInput  input:focus,
.stTextArea   textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.22) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    background: rgba(30,41,59,0.85) !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* ── Radio buttons ──────────────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: #94a3b8 !important;
    transition: color 0.15s;
}
[data-testid="stRadio"] label:hover { color: #f8fafc !important; }
[data-testid="stRadio"] [data-baseweb="radio"] div:first-child {
    border-color: #334155 !important;
}
[data-testid="stRadio"] [data-checked="true"] [data-baseweb="radio"] div:first-child,
[data-testid="stRadio"] input:checked + div { border-color: #06b6d4 !important; }
[data-testid="stRadio"] [data-checked="true"] [data-baseweb="radio"] div:first-child div {
    background: #06b6d4 !important;
}

/* ── Checkboxes ─────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: #94a3b8 !important; }
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div {
    border-color: #334155 !important;
    background: rgba(30,41,59,0.8) !important;
    border-radius: 4px !important;
}
[data-testid="stCheckbox"] [data-checked="true"] [data-baseweb="checkbox"] div {
    background: #06b6d4 !important;
    border-color: #06b6d4 !important;
}

/* ── Sliders ────────────────────────────────────────────────────────────── */
.stSlider label { color: #94a3b8 !important; }
[data-baseweb="slider"] [data-testid="stThumbValue"] { color: #06b6d4 !important; }
[data-baseweb="slider"] div[role="slider"] { background: #06b6d4 !important; }

/* ── Alert / notification boxes ────────────────────────────────────────── */
/* Success — green palette */
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"],
.stSuccess > div,
div[data-testid="stAlertContentSuccess"] {
    background: rgba(16,185,129,0.12) !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}
/* Warning — amber palette */
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
.stWarning > div,
div[data-testid="stAlertContentWarning"] {
    background: rgba(245,158,11,0.12) !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}
/* Error — red palette */
[data-testid="stAlert"][data-baseweb="notification"][kind="negative"],
.stError > div,
div[data-testid="stAlertContentError"] {
    background: rgba(239,68,68,0.12) !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}
/* Info — cyan palette */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
.stInfo > div,
div[data-testid="stAlertContentInfo"] {
    background: rgba(6,182,212,0.10) !important;
    border-left: 4px solid #06b6d4 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}
/* Catch-all alert text */
[data-testid="stAlert"] p,
[data-testid="stAlert"] li { color: #f8fafc !important; }

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: rgba(15,23,42,0.75) !important;
    border-radius: 10px;
    border: 1px solid #334155;
}
[data-testid="stDataFrame"] th {
    background: rgba(30,41,59,0.9) !important;
    color: #94a3b8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stDataFrame"] td { color: #f8fafc !important; }

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: linear-gradient(145deg, rgba(30,41,59,0.6), rgba(15,23,42,0.7));
    border: 1px solid #334155 !important;
    border-radius: 12px;
    transition: border-color 0.2s;
}
[data-testid="stExpander"]:hover { border-color: rgba(6,182,212,0.3) !important; }
[data-testid="stExpander"] summary { color: #94a3b8; font-weight: 500; }
[data-testid="stExpander"] summary:hover { color: #f8fafc; }

/* ── Code blocks ────────────────────────────────────────────────────────── */
[data-testid="stCode"] {
    background: rgba(15,23,42,0.92) !important;
    border: 1px solid #334155;
    border-radius: 8px;
    font-size: 0.82rem;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(30,41,59,0.5);
    border: 2px dashed #334155;
    border-radius: 12px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #06b6d4; }

/* ── Spinner ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: #06b6d4 !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #1e293b; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #64748b; }

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr { border-color: #334155 !important; opacity: 0.6; }

/* ── Utility classes ────────────────────────────────────────────────────── */
.gradient-text {
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}
.tag-low    { display:inline-block; background:rgba(16,185,129,0.18);
              color:#10b981; border:1px solid rgba(16,185,129,0.4);
              padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.tag-medium { display:inline-block; background:rgba(245,158,11,0.18);
              color:#f59e0b; border:1px solid rgba(245,158,11,0.4);
              padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.tag-high   { display:inline-block; background:rgba(239,68,68,0.18);
              color:#ef4444; border:1px solid rgba(239,68,68,0.4);
              padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(DARK_CSS, unsafe_allow_html=True)
