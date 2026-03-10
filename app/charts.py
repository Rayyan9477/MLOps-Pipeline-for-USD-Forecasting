"""
Plotly chart factory — unified colour palette.

Palette (matches index.html + css_theme.py exactly):
  Cyan   #06b6d4  — primary accent / main data line
  Purple #7c3aed  — secondary accent
  Blue   #1e40af  — tertiary / headers
  Green  #10b981  — Low / stable / success
  Amber  #f59e0b  — Medium / caution / warning
  Red    #ef4444  — High / danger / error
  BG     #0f172a  — paper background
  Card   #1e293b  — plot area
  Grid   #334155  — grid lines / borders
  Text   #94a3b8  — axis labels / captions
  White  #f8fafc  — titles / values
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional

# ── Palette constants ─────────────────────────────────────────────────────────
P = dict(
    bg      = "#0f172a",   # paper / page background
    card    = "#1e293b",   # plot area background
    grid    = "#334155",   # grid lines
    border  = "#475569",   # borders / dividers
    text    = "#94a3b8",   # muted labels
    white   = "#f8fafc",   # primary text / values
    cyan    = "#06b6d4",   # primary accent
    purple  = "#7c3aed",   # secondary accent
    blue    = "#1e40af",   # tertiary
    green   = "#10b981",   # Low risk / success
    amber   = "#f59e0b",   # Medium risk / warning
    red     = "#ef4444",   # High risk / error
    # semi-transparent fills (zone bands, area fills)
    green_fill  = "rgba(16,185,129,0.18)",
    amber_fill  = "rgba(245,158,11,0.15)",
    red_fill    = "rgba(239,68,68,0.15)",
    cyan_fill   = "rgba(6,182,212,0.14)",
    purple_fill = "rgba(124,58,237,0.12)",
)

# Axis defaults applied to every chart
_AX = dict(
    gridcolor    = P["grid"],
    color        = P["text"],
    linecolor    = P["grid"],
    zerolinecolor= P["grid"],
    tickfont     = dict(color=P["text"], size=11),
    title_font   = dict(color=P["text"], size=12),
)

_BASE = dict(
    paper_bgcolor = P["bg"],
    plot_bgcolor  = P["card"],
    font          = dict(color=P["text"], family="Inter, sans-serif", size=12),
    margin        = dict(l=55, r=30, t=50, b=50),
    legend        = dict(
        bgcolor     = P["bg"],
        bordercolor = P["grid"],
        borderwidth = 1,
        font        = dict(color=P["text"], size=11),
    ),
    hoverlabel = dict(
        bgcolor    = P["bg"],
        bordercolor= P["cyan"],
        font       = dict(color=P["white"], size=12),
    ),
)


def _apply(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
    layout = dict(**_BASE, height=height)
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(color=P["white"], size=13, family="Inter, sans-serif"),
            x=0.01, xanchor="left",
        )
    fig.update_layout(**layout)
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX)
    return fig


def _empty(msg: str, title: str = "", height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=P["text"], size=13),
    )
    return _apply(fig, title, height)


# ── Volatility / pips timeline ────────────────────────────────────────────────
def make_volatility_timeseries(predictions: List[dict]) -> go.Figure:
    """
    Pip-based forecast timeline.
      • Y-axis  : expected move in pips (prediction × 10 000)
      • Zones   : green 0–50 / amber 50–150 / red >150
      • Markers : colour-coded green/amber/red by risk level
      • Line    : cyan primary accent
    Single go.Figure (no make_subplots) so plot_bgcolor always renders correctly.
    """
    if not predictions:
        return _empty(
            "Make a forecast to see your history here.",
            "Forecast History — Expected Move (pips)",
        )

    df = pd.DataFrame(predictions)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pips"] = (df["prediction"] * 10_000).round(1)

    # X-axis — sequential when all points within 5 s to avoid "18:05:55.711" clutter
    span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    if span < 5 or len(df) == 1:
        xs      = [f"#{i+1}" for i in range(len(df))]
        ht_time = [ts.strftime("%H:%M:%S") for ts in df["timestamp"]]
        x_title = "Forecast #"
    else:
        xs      = df["timestamp"].dt.strftime("%H:%M:%S")
        ht_time = None
        x_title = "Time (UTC)"

    y_max = max(df["pips"].max() * 1.45, 210)
    fig = go.Figure()

    # ── Background risk zones ─────────────────────────────────────────────────
    fig.add_hrect(y0=0,   y1=50,    fillcolor=P["green_fill"], layer="below", line_width=0)
    fig.add_hrect(y0=50,  y1=150,   fillcolor=P["amber_fill"], layer="below", line_width=0)
    fig.add_hrect(y0=150, y1=y_max, fillcolor=P["red_fill"],   layer="below", line_width=0)

    # ── Threshold lines ───────────────────────────────────────────────────────
    fig.add_hline(
        y=50, line_dash="dot", line_color=P["amber"], line_width=1.8, opacity=0.85,
        annotation_text="Moderate  50 pips",
        annotation_font_color=P["amber"], annotation_font_size=10,
        annotation_position="top right",
    )
    fig.add_hline(
        y=150, line_dash="dot", line_color=P["red"], line_width=1.8, opacity=0.85,
        annotation_text="High  150 pips",
        annotation_font_color=P["red"], annotation_font_size=10,
        annotation_position="top right",
    )

    # ── Area fill under line ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=df["pips"],
        mode="none", fill="tozeroy",
        fillcolor=P["cyan_fill"],
        hoverinfo="skip", showlegend=False,
    ))

    # ── Main line ─────────────────────────────────────────────────────────────
    risk_color = {"Low": P["green"], "Medium": P["amber"], "High": P["red"]}
    mcolors = [risk_color.get(p.get("risk_level", "Low"), P["cyan"]) for p in predictions]

    htmpl = (
        "<b>%{customdata}</b><br>Expected move: <b>±%{y:.1f} pips</b><extra></extra>"
        if ht_time
        else "<b>%{x}</b><br>Expected move: <b>±%{y:.1f} pips</b><extra></extra>"
    )
    fig.add_trace(go.Scatter(
        x=xs, y=df["pips"],
        mode="lines+markers",
        name="Expected move",
        line=dict(color=P["cyan"], width=3, shape="spline", smoothing=0.4),
        marker=dict(
            size=11, color=mcolors,
            line=dict(width=2.5, color=P["bg"]),
            symbol="circle",
        ),
        customdata=ht_time,
        hovertemplate=htmpl,
    ))

    _apply(fig, "Forecast History — Expected Move (pips)")
    fig.update_yaxes(title_text="Pips (±)", rangemode="tozero", range=[0, y_max], **_AX)
    fig.update_xaxes(title_text=x_title, **_AX)
    return fig


# ── Latency bar ───────────────────────────────────────────────────────────────
def make_latency_bar(latencies: List[float]) -> go.Figure:
    """
    Latency distribution, colour-coded green → red by severity bucket.
    """
    if not latencies:
        return _empty("No latency data yet.", "Latency Distribution")

    labels  = ["<10 ms", "10–25 ms", "25–50 ms", "50–100 ms", ">100 ms"]
    colours = [P["green"], "#84cc16", P["amber"], "#f97316", P["red"]]
    counts  = [0, 0, 0, 0, 0]
    for lat in latencies:
        if   lat < 10:  counts[0] += 1
        elif lat < 25:  counts[1] += 1
        elif lat < 50:  counts[2] += 1
        elif lat < 100: counts[3] += 1
        else:           counts[4] += 1

    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker=dict(
            color=colours,
            line=dict(color=P["bg"], width=1.5),
        ),
        text=[str(c) if c else "" for c in counts],
        textposition="outside",
        textfont=dict(color=P["text"], size=11),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))

    if counts[4] > 0:
        fig.add_annotation(
            text="⚠ High latency detected",
            xref="paper", yref="paper", x=0.97, y=0.94,
            showarrow=False,
            font=dict(color=P["red"], size=10),
            bgcolor="rgba(239,68,68,0.12)",
            bordercolor=P["red"], borderpad=5,
        )

    _apply(fig, "Prediction Latency")
    fig.update_yaxes(title_text="Count")
    return fig


# ── Drift timeseries ──────────────────────────────────────────────────────────
def make_drift_timeseries(predictions: List[dict]) -> go.Figure:
    """
    Drift ratio timeline (0–100%) with warning / critical thresholds.
    """
    if not predictions:
        return _empty("No predictions yet.", "Data Drift Over Time")

    df = pd.DataFrame(predictions)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    drift_pct = (df.get("drift_ratio", pd.Series([0.0] * len(df))) * 100)

    span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    if span < 5 or len(df) == 1:
        xs, x_title = [f"#{i+1}" for i in range(len(df))], "Forecast #"
    else:
        xs, x_title = df["timestamp"].dt.strftime("%H:%M:%S"), "Time (UTC)"

    fig = go.Figure()

    # zone fills
    fig.add_hrect(y0=0,  y1=20,  fillcolor=P["green_fill"], layer="below", line_width=0)
    fig.add_hrect(y0=20, y1=50,  fillcolor=P["amber_fill"], layer="below", line_width=0)
    fig.add_hrect(y0=50, y1=110, fillcolor=P["red_fill"],   layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=xs, y=drift_pct,
        name="Drift %",
        fill="tozeroy", fillcolor=P["cyan_fill"],
        line=dict(color=P["cyan"], width=2.5),
        mode="lines+markers",
        marker=dict(size=7, color=P["cyan"], line=dict(width=2, color=P["bg"])),
        hovertemplate="<b>%{x}</b><br>Drift: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=20, line_dash="dash", line_color=P["amber"], line_width=1.8, opacity=0.85,
                  annotation_text="Warning  20%",
                  annotation_font_color=P["amber"], annotation_font_size=10)
    fig.add_hline(y=50, line_dash="dash", line_color=P["red"],   line_width=1.8, opacity=0.85,
                  annotation_text="Critical  50%",
                  annotation_font_color=P["red"],   annotation_font_size=10)

    y_max = max(float(drift_pct.max()) * 1.3, 65) if len(drift_pct) else 65
    _apply(fig, "Data Drift Ratio Over Time")
    fig.update_yaxes(title_text="Drift (%)", range=[0, y_max], **_AX)
    fig.update_xaxes(title_text=x_title, **_AX)
    return fig


# ── Feature importance ────────────────────────────────────────────────────────
def make_feature_importance_chart(model, feature_names: List[str]) -> go.Figure:
    """
    Horizontal bar chart of top-20 feature importances.
    Bar colour scales from purple (low) to cyan (high).
    """
    importances = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_estimators_") and "xgb" in model.named_estimators_:
            xgb = model.named_estimators_["xgb"]
            if hasattr(xgb, "feature_importances_"):
                importances = xgb.feature_importances_
        elif hasattr(model, "estimators_"):
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances = est.feature_importances_
                    break
    except Exception:
        pass

    if importances is None or len(importances) == 0:
        return _empty("Feature importances not available.", "Feature Importance", height=420)

    n = min(len(importances), len(feature_names))
    pairs = sorted(zip(feature_names[:n], importances[:n]),
                   key=lambda x: x[1], reverse=True)[:20]
    names, vals = zip(*pairs)

    # Gradient: low importance → purple, high → cyan
    mx = max(vals)
    t  = [v / mx for v in vals]
    def _lerp_colour(t):
        # purple #7c3aed  →  cyan #06b6d4
        r = int(124 + (6   - 124) * t)
        g = int(58  + (182 - 58 ) * t)
        b = int(237 + (212 - 237) * t)
        return f"rgb({r},{g},{b})"
    colours = [_lerp_colour(ti) for ti in t]

    fig = go.Figure(go.Bar(
        x=list(vals), y=list(names),
        orientation="h",
        marker=dict(
            color=list(reversed(colours)),
            line=dict(color=P["bg"], width=1),
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    _apply(fig, "Top Feature Importances", height=500)
    fig.update_layout(yaxis=dict(autorange="reversed", **_AX))
    fig.update_xaxes(title_text="Importance Score")
    return fig


# ── KS-test p-value heatmap ───────────────────────────────────────────────────
def make_ks_heatmap(ks_results: dict) -> go.Figure:
    """
    Heatmap of KS-test p-values per feature.
    Red = drift (p < 0.05), green = no drift.
    """
    if not ks_results:
        return _empty("No KS results available.", "KS Test p-values", height=220)

    features = list(ks_results.keys())
    p_values = [ks_results[f].p_value for f in features]

    fig = go.Figure(go.Heatmap(
        z=[p_values], x=features, y=["p-value"],
        colorscale=[
            [0.00, P["red"]],
            [0.05, P["amber"]],
            [0.12, "#84cc16"],
            [1.00, P["green"]],
        ],
        zmin=0, zmax=1,
        text=[[f"{p:.3f}" for p in p_values]],
        texttemplate="%{text}",
        textfont=dict(size=9, color=P["white"]),
        hovertemplate="<b>%{x}</b><br>p-value: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title="p-value",
            titlefont=dict(color=P["text"]),
            tickfont=dict(color=P["text"]),
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor=P["grid"],
        ),
    ))
    fig.add_annotation(
        text="p < 0.05 → drift detected",
        xref="paper", yref="paper", x=0.0, y=-0.28,
        showarrow=False, font=dict(color=P["amber"], size=10),
    )
    _apply(fig, "KS Test p-values  (red = drift detected)", height=230)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9, color=P["text"]))
    return fig


# ── MLflow metrics comparison ─────────────────────────────────────────────────
def make_metrics_comparison_chart(runs_df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis line chart: RMSE in pips (red, left) and R² (cyan, right).
    Uses manual yaxis2 overlay — no make_subplots — so bg colour is correct.
    """
    if runs_df is None or runs_df.empty:
        return _empty("No MLflow runs to display.", "Model Metrics Across Runs")

    run_labels = runs_df.get("run_id", runs_df.index).astype(str).str[:8]
    fig = go.Figure()

    if "rmse" in runs_df.columns:
        rmse_pips = (runs_df["rmse"] * 10_000).round(2)
        fig.add_trace(go.Scatter(
            x=run_labels, y=rmse_pips,
            name="RMSE (pips)", yaxis="y1",
            line=dict(color=P["red"], width=2.5),
            mode="lines+markers",
            marker=dict(size=8, color=P["red"], line=dict(width=2, color=P["bg"])),
            hovertemplate="Run: %{x}<br>RMSE: ±%{y:.2f} pips<extra></extra>",
        ))
    if "r2" in runs_df.columns:
        fig.add_trace(go.Scatter(
            x=run_labels, y=runs_df["r2"],
            name="R²", yaxis="y2",
            line=dict(color=P["cyan"], width=2.5),
            mode="lines+markers",
            marker=dict(size=8, color=P["cyan"], line=dict(width=2, color=P["bg"])),
            hovertemplate="Run: %{x}<br>R²: %{y:.4f}<extra></extra>",
        ))

    _apply(fig, "Model Metrics Across Runs")
    fig.update_layout(
        yaxis =dict(title_text="RMSE (pips)", **_AX),
        yaxis2=dict(title_text="R²", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", **_AX),
    )
    fig.update_xaxes(title_text="Run ID (truncated)")
    return fig


# ── Feature distributions ─────────────────────────────────────────────────────
def make_feature_distribution(df: pd.DataFrame,
                               features: Optional[List[str]] = None) -> go.Figure:
    """2×3 grid of histograms. Uses per-axis bgcolor patch for dark theme."""
    key = features or [
        "close_lag_1", "log_return", "close_rolling_std_24",
        "close_rolling_mean_24", "hour", "price_change_pct",
    ]
    key = [f for f in key if f in df.columns][:6]
    if not key:
        return _empty("No matching features in data.", "Feature Distributions", height=400)

    n_cols = 3
    n_rows = (len(key) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=key,
        horizontal_spacing=0.1, vertical_spacing=0.18,
    )
    pal = [P["cyan"], P["purple"], P["green"], P["amber"], P["red"], "#8b5cf6"]

    for i, feat in enumerate(key):
        row, col = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(
            go.Histogram(
                x=df[feat].dropna(), nbinsx=30,
                marker=dict(
                    color=pal[i % len(pal)],
                    opacity=0.85,
                    line=dict(color=P["bg"], width=0.5),
                ),
                name=feat, showlegend=False,
                hovertemplate=f"<b>{feat}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=row, col=col,
        )

    # Patch all subplot axis backgrounds after layout is built
    fig.update_layout(
        paper_bgcolor=P["bg"],
        plot_bgcolor =P["card"],
        height=360 * n_rows,
        font=dict(color=P["text"], family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor=P["bg"], bordercolor=P["cyan"],
                        font=dict(color=P["white"])),
        margin=dict(l=40, r=20, t=55, b=40),
    )
    for key_name in list(fig.layout):
        if key_name.startswith(("xaxis", "yaxis")):
            fig.layout[key_name].update(
                gridcolor=P["grid"], color=P["text"],
                linecolor=P["grid"], zerolinecolor=P["grid"],
            )
    fig.update_annotations(font_color=P["text"])
    return fig
