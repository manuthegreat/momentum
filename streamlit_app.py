import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Momentum Strategy Dashboard",
    layout="wide",
)

ARTIFACTS = Path("artifacts")

SIGNALS_PATH = ARTIFACTS / "backtest_signals.parquet"
EQUITY_PATH  = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_PATH  = ARTIFACTS / "backtest_trades_C.parquet"
STATS_PATH   = ARTIFACTS / "backtest_stats_C.json"

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_parquet(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def clean_bucket_c(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()

    # Always start with position size
    out = (
        df.groupby("Ticker", as_index=False)["Position_Size"]
        .sum()
    )

    # Safely attach scores if they exist
    score_cols = [
        "Weighted_Score",
        "Momentum Score",
        "Early Momentum Score",
        "Consistency",
    ]

    for col in score_cols:
        if col in df.columns:
            scores = (
                df.groupby("Ticker")[col]
                .max()
                .reset_index()
            )
            out = out.merge(scores, on="Ticker", how="left")

    return out.sort_values("Position_Size", ascending=False).reset_index(drop=True)

# ============================================================
# LOAD DATA
# ============================================================

signals = load_parquet(SIGNALS_PATH)
equity  = load_parquet(EQUITY_PATH)
trades  = load_parquet(TRADES_PATH)
stats   = load_json(STATS_PATH)

st.title("📈 Momentum Strategy Dashboard")

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.subheader("Performance Summary (Bucket C)")

if stats:
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, round(v, 2) if isinstance(v, (int, float)) else v)
else:
    st.info("No performance stats available")

# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")

if not equity.empty and "Date" in equity.columns:
    equity = equity.sort_values("Date")

    value_col = (
        "Equity"
        if "Equity" in equity.columns
        else "Portfolio Value"
        if "Portfolio Value" in equity.columns
        else None
    )

    if value_col:
        st.line_chart(
            equity.set_index("Date")[value_col],
            width="stretch"
        )
    else:
        st.info("No equity value column found")
else:
    st.info("Equity data not available")

# ============================================================
# CURRENT PORTFOLIO — BUCKET C
# ============================================================

st.subheader("Current Portfolio — Bucket C")

if signals.empty or "Date" not in signals.columns:
    st.info("No signal data available")
else:
    latest_date = signals["Date"].max()
    today = signals[
        (signals["Bucket"] == "C") &
        (signals["Date"] == latest_date)
    ]

    if today.empty:
        st.info("No Bucket C positions for latest date")
    else:
        clean_c = clean_bucket_c(today)

        # Display only columns that exist
        preferred_cols = [
            "Ticker",
            "Position_Size",
            "Weighted_Score",
            "Momentum Score",
            "Early Momentum Score",
            "Consistency",
        ]

        display_cols = [c for c in preferred_cols if c in clean_c.columns]

        st.dataframe(
            clean_c[display_cols],
            width="stretch"
        )

# ============================================================
# ARTIFACTS EXPLORER (MEANINGFUL RAW VIEW)
# ============================================================

st.subheader("📦 Artifacts Explorer")

tab1, tab2, tab3 = st.tabs(["Signals", "Equity", "Trades"])

with tab1:
    if signals.empty:
        st.info("No signals available")
    else:
        sort_cols = [c for c in ["Date", "Weighted_Score"] if c in signals.columns]
        st.dataframe(
            signals.sort_values(sort_cols, ascending=False),
            width="stretch"
        )

with tab2:
    if equity.empty:
        st.info("No equity data available")
    else:
        st.dataframe(
            equity.sort_values("Date", ascending=False),
            width="stretch"
        )

with tab3:
    if trades.empty:
        st.info("No trades available")
    else:
        if "Date" in trades.columns:
            trades = trades.sort_values("Date", ascending=False)
        st.dataframe(trades, width="stretch")
