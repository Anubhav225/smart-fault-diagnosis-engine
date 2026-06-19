"""
visualizations.py - Plotly charts with light theme.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from utils import health_score_color, severity_color

_L = dict(
    template      = "plotly_white",
    paper_bgcolor = "#FFFFFF",
    plot_bgcolor  = "#F8FAFC",
    font          = dict(family="Inter, sans-serif", color="#1E293B", size=12),
    margin        = dict(l=40, r=20, t=50, b=40),
)


def sensor_trend_chart(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    fig = go.Figure()
    palette = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
               "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4F46E5"]
    x = _time_col(df)
    for i, col in enumerate(columns[:10]):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=x[:len(s)], y=s, name=col, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{col}</b>: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(**_L, title="Sensor Trend Analysis",
                      xaxis_title="Sample / Time", yaxis_title="Value",
                      legend=dict(orientation="h", y=-0.25), height=400)
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    return fig


def anomaly_highlight_chart(df: pd.DataFrame, col: str, z_thresh: float = 2.5) -> go.Figure:
    s    = df[col].dropna()
    mean = s.mean()
    std  = s.std() or 1
    z    = (s - mean) / std
    out  = z.abs() > z_thresh
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=col,
                             line=dict(color="#2563EB", width=1.5)))
    fig.add_trace(go.Scatter(x=s[out].index, y=s[out], mode="markers",
                             name="Anomaly", marker=dict(color="#DC2626", size=9, symbol="x-thin",
                             line=dict(width=2, color="#DC2626"))))
    fig.add_hline(y=mean + z_thresh * std, line_dash="dot", line_color="#F59E0B",
                  annotation_text=f"+{z_thresh} sigma", annotation_font_color="#F59E0B")
    fig.add_hline(y=mean - z_thresh * std, line_dash="dot", line_color="#F59E0B",
                  annotation_text=f"-{z_thresh} sigma", annotation_font_color="#F59E0B")
    fig.update_layout(**_L, title=f"Anomaly Detection - {col}", height=360)
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    return fig


def health_gauge(score: int) -> go.Figure:
    color = health_score_color(score)
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": "Machine Health Score", "font": {"size": 16, "color": "#1E293B"}},
        number= {"font": {"size": 48, "color": color}, "suffix": "/100"},
        gauge = {
            "axis" : {"range": [0, 100], "tickcolor": "#94A3B8", "tickfont": {"color": "#64748B"}},
            "bar"  : {"color": color, "thickness": 0.25},
            "bgcolor": "#F1F5F9",
            "steps": [
                {"range": [0,  40], "color": "#FEE2E2"},
                {"range": [40, 60], "color": "#FEF3C7"},
                {"range": [60, 80], "color": "#D1FAE5"},
                {"range": [80,100], "color": "#DCFCE7"},
            ],
            "threshold": {"line": {"color": "#DC2626", "width": 3}, "value": 40},
        },
    ))
    fig.update_layout(**_L, height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def fault_pie(faults: list[dict]) -> go.Figure:
    if not faults:
        return _empty("No faults to display")
    labels = [f.get("fault_name", "Unknown") for f in faults]
    confs  = [f.get("confidence", "Low")     for f in faults]
    wmap   = {"High": 3, "Medium": 2, "Low": 1}
    values = [wmap.get(c, 1) for c in confs]
    colors = ["#DC2626" if c=="High" else "#D97706" if c=="Medium" else "#16A34A" for c in confs]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5, marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Confidence weight: %{value}<extra></extra>",
    ))
    fig.update_layout(**_L, title="Fault Distribution", height=320,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def priority_bar(fixes: list[dict]) -> go.Figure:
    if not fixes:
        return _empty("No fixes")
    pmap = {"Immediate": "#DC2626", "Within 24h": "#EA580C",
            "Within 1 week": "#D97706", "Scheduled": "#16A34A"}
    actions    = [f.get("action","")[:55] for f in fixes]
    priorities = [f.get("priority","Scheduled") for f in fixes]
    colors     = [pmap.get(p, "#6B7280") for p in priorities]
    fig = go.Figure(go.Bar(
        y=actions, x=[1]*len(actions), orientation="h",
        marker_color=colors, text=priorities, textposition="inside",
        textfont=dict(color="#fff", size=11),
        hovertemplate="<b>%{y}</b><br>Priority: %{text}<extra></extra>",
    ))
    fig.update_layout(**_L, title="Corrective Actions by Priority",
                      xaxis=dict(showticklabels=False, showgrid=False),
                      height=max(220, 48 * len(fixes)))
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return _empty("Need 2+ numeric columns")
    corr = num.corr().round(2)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values.round(2), texttemplate="%{text}",
        hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.2f}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(**_L, title="Sensor Correlation Matrix", height=440)
    return fig


def rolling_stats_chart(df: pd.DataFrame, col: str, window: int = 10) -> go.Figure:
    s  = df[col].dropna()
    rm = s.rolling(window).mean()
    rs = s.rolling(window).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index,  y=s,  mode="lines", name="Raw",
                             line=dict(color="#94A3B8", width=1)))
    fig.add_trace(go.Scatter(x=rm.index, y=rm, mode="lines", name="Rolling Mean",
                             line=dict(color="#2563EB", width=2.5)))
    fig.add_trace(go.Scatter(
        x=list(rm.index) + list(rm.index[::-1]),
        y=list(rm + rs) + list((rm - rs)[::-1]),
        fill="toself", name="+/-1 Std",
        fillcolor="rgba(37,99,235,0.1)", line=dict(color="rgba(0,0,0,0)"),
    ))
    fig.update_layout(**_L, title=f"Rolling Statistics - {col} (window={window})", height=340)
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    return fig


def fault_history_chart(history: list[dict]) -> go.Figure:
    if not history:
        return _empty("No history yet")
    dates   = [h["timestamp"]   for h in history]
    scores  = [h["health_score"] for h in history]
    sevs    = [h["severity"]     for h in history]
    colors  = [severity_color(s) for s in sevs]
    fig = go.Figure(go.Bar(
        x=dates, y=scores, marker_color=colors,
        text=sevs, textposition="outside",
        hovertemplate="<b>%{x}</b><br>Health: %{y}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(**_L, title="Health Score History", yaxis_range=[0, 110],
                      yaxis_title="Health Score", height=320)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    return fig


def _time_col(df: pd.DataFrame):
    for col in df.columns:
        if any(k in col.lower() for k in ("time","date","timestamp","ts")):
            return df[col]
    return df.index

def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#94A3B8"))
    fig.update_layout(**_L, height=220)
    return fig
