import json
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")

SIGNALS_PATH = ARTIFACTS / "backtest_signals.parquet"

EQUITY_PATH = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_PATH = ARTIFACTS / "backtest_trades_C.parquet"
STATS_PATH  = ARTIFACTS / "backtest_stats_C.json"

st.set_page_config(
    page_title="Momentum Dashboard",
    layout="wide"
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path)

@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Momentum Dashboard")

view = st.sidebar.radio(
    "View",
    ["Overview", "Signals", "Backtests"]
)

bucket = st.sidebar.selectbox(
    "Select Bucket",
    ["A", "B", "C"]
)

# ============================================================
# LOAD DATA
# ============================================================

signals = load_parquet(SIGNALS_PATH)

# ============================================================
# OVERVIEW
# ============================================================

if view == "Overview":

    st.title("Momentum Strategy — Overview")

    latest_date = signals["Date"].max()
    today = signals[signals["Date"] == latest_date]

    col1, col2, col3 = st.columns(3)

    col1.metric("Bucket A", len(today[today["Bucket"] == "A"]))
    col2.metric("Bucket B", len(today[today["Bucket"] == "B"]))
    col3.metric("Bucket C", len(today[today["Bucket"] == "C"]))

    st.subheader("Unified Portfolio (Bucket C)")

    st.dataframe(
        today[today["Bucket"] == "C"]
        .sort_values("Position_Size", ascending=False)
        .reset_index(drop=True)
    )

# ============================================================
# SIGNALS
# ============================================================

elif view == "Signals":

    st.title(f"Today's Signals — Bucket {bucket}")

    latest_date = signals["Date"].max()

    df = (
        signals[
            (signals["Bucket"] == bucket) &
            (signals["Date"] == latest_date)
        ]
        .sort_values("Position_Size", ascending=False)
    )

    if df.empty:
        st.warning("No signals available.")
    else:
        st.dataframe(df.reset_index(drop=True))

# ============================================================
# BACKTESTS
# ============================================================

elif view == "Backtests":

    st.title("Backtests — Bucket C")

    if not EQUITY_PATH.exists():
        st.warning("Backtest not available.")
        st.stop()

    equity = load_parquet(EQUITY_PATH)
    trades = load_parquet(TRADES_PATH)
    stats  = load_json(STATS_PATH)

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Return (%)", round(stats.get("Total Return (%)", 0), 2))
    c2.metric("CAGR (%)", round(stats.get("CAGR (%)", 0), 2))
    c3.metric("Sharpe", round(stats.get("Sharpe Ratio", 0), 2))
    c4.metric("Max Drawdown (%)", round(stats.get("Max Drawdown (%)", 0), 2))

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    st.subheader("Equity Curve")

    st.line_chart(
        equity.set_index("Date")["Equity"]
    )

    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    st.subheader("Trade Blotter")

    st.dataframe(
        trades.sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Momentum system • Deterministic • Artifact-driven • Production-grade"
)
