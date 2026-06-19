"""
utils.py - Shared helpers: colours, formatting, session state, JSON parsing.
"""

from __future__ import annotations
import json, re
from datetime import datetime
import pandas as pd
import streamlit as st

# -- Colour maps -----------------------------------------------------------
SEVERITY_COLORS = {
    "Critical": "#DC2626", "High": "#EA580C",
    "Medium":   "#D97706", "Low":  "#16A34A", "Normal": "#16A34A",
}
SEVERITY_ICONS = {
    "Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Normal": "✅",
}

def severity_color(s: str) -> str:
    return SEVERITY_COLORS.get(s, "#6B7280")

def severity_icon(s: str) -> str:
    return SEVERITY_ICONS.get(s, "⚪")

def health_score_color(score: int) -> str:
    if score >= 80: return "#16A34A"
    if score >= 60: return "#D97706"
    if score >= 40: return "#EA580C"
    return "#DC2626"

def health_score_label(score: int) -> str:
    if score >= 80: return "Healthy"
    if score >= 60: return "Fair"
    if score >= 40: return "Poor"
    return "Critical"

# -- JSON helpers ------------------------------------------------------------
def safe_parse_json(text: str) -> dict | None:
    """Extract and parse JSON from LLM response robustly."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None

# -- Datetime -----------------------------------------------------------------
def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)

def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# -- Session state --------------------------------------------------------------
def init_session_defaults(defaults: dict) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -- Data helpers -----------------------------------------------------------------
def df_to_text_summary(df: pd.DataFrame, max_rows: int = 25) -> str:
    lines = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        "",
        df.head(max_rows).to_string(index=False),
    ]
    return "\n".join(lines)

def stats_to_text(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return "No numeric columns."
    return numeric.describe().round(4).to_string()

# -- UI helpers -----------------------------------------------------------------
def badge(label: str, bg: str, fg: str = "#fff") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:20px;font-size:0.75rem;font-weight:600">{label}</span>'
    )
