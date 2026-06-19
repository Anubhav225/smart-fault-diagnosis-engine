"""
data_processor.py - File ingestion, preprocessing, anomaly detection.
"""

from __future__ import annotations
import json
from typing import Optional
import pandas as pd


def load_file(uploaded_file) -> tuple[Optional[pd.DataFrame], str, str]:
    """Load uploaded file. Returns (df, raw_text, file_type)."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return _load_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return _load_excel(uploaded_file)
    elif name.endswith(".json"):
        return _load_json(uploaded_file)
    else:
        return _load_text(uploaded_file)


def _load_csv(file):
    try:
        df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, encoding="latin-1", on_bad_lines="skip")
    return df, df.to_string(index=False), "csv"


def _load_excel(file):
    df = pd.read_excel(file, engine="openpyxl")
    return df, df.to_string(index=False), "excel"


def _load_json(file):
    raw = file.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.json_normalize(data)
        else:
            return None, raw, "json"
        return df, raw, "json"
    except json.JSONDecodeError:
        return None, raw, "json"


def _load_text(file):
    raw = file.read().decode("utf-8", errors="replace")
    return None, raw, "text"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, coerce numeric columns, drop empty rows."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.dropna(how="all", inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= 0.6:
                df[col] = converted
    return df


def detect_anomalies(df: pd.DataFrame, z_thresh: float = 2.5) -> dict:
    """Z-score anomaly detection on numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {"anomaly_columns": [], "anomaly_details": [], "anomaly_summary": "No numeric columns."}

    anomaly_columns, anomaly_details = [], []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 5:
            continue
        mean, std = series.mean(), series.std()
        if std == 0 or pd.isna(std):
            continue
        z = (series - mean) / std
        n_out = int((z.abs() > z_thresh).sum())
        if n_out > 0:
            anomaly_columns.append(col)
            anomaly_details.append({
                "column": col,
                "n_outliers": n_out,
                "pct_outliers": round(n_out / len(series) * 100, 1),
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "max_z_score": round(float(z.abs().max()), 2),
            })

    summary = "\n".join(
        f"- {d['column']}: {d['n_outliers']} outliers ({d['pct_outliers']}%), "
        f"mean={d['mean']}, std={d['std']}, max_z={d['max_z_score']}"
        for d in anomaly_details
    ) or "No significant anomalies detected."

    return {"anomaly_columns": anomaly_columns, "anomaly_details": anomaly_details, "anomaly_summary": summary}


def compute_column_stats(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number")
    return numeric.describe().round(4).to_string() if not numeric.empty else "No numeric columns."


def build_data_summary(df: Optional[pd.DataFrame], raw_text: str, file_type: str) -> str:
    if df is not None:
        return (
            f"Type: {file_type.upper()} | Shape: {df.shape[0]}x{df.shape[1]}\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
            f"Sample data:\n{df.head(30).to_string(index=False)}"
        )
    return f"Type: {file_type.upper()}\n\n{raw_text[:3000]}"
