# streamlit_app.py

import json
from pathlib import Path

import pandas as pd
import streamlit as st

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

def load_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Missing artifact: {path.name}")
        return pd.DataFrame()
    return pd.read_parquet(path)

def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        st.warning(f"Missing artifact: {path.name}")
        return {}
    with open(path) as f:
        return json.load(f)

# ============================================================
# HEADER
# ============================================================

st.title("📈 Momentum Strategy Dashboard")
st.caption("Deterministic • Artifact-driven • Production-grade")

# ============================================================
# LOAD DATA
# ============================================================

signals = load_parquet(SIGNALS_PATH, "Signals")
equity  = load_parquet(EQUITY_PATH, "Equity")
trades  = load_parquet(TRADES_PATH, "Trades")
stats   = load_json(STATS_PATH, "Stats")

# Normalize equity column for UI
if not equity.empty and "Portfolio Value" in equity.columns:
    equity = equity.rename(columns={"Portfolio Value": "Equity"})

# ============================================================
# TODAY'S SIGNALS
# ============================================================

st.header("🟢 Today’s Signals")

if signals.empty:
    st.info("No signals available")
else:
    latest_date = signals["Date"].max()
    today = signals[signals["Date"] == latest_date].copy()

    st.caption(f"As of {latest_date.date()}")

    # Bucket selector
    bucket = st.selectbox(
        "Select Bucket",
        sorted(today["Bucket"].unique())
    )

    df = today[today["Bucket"] == bucket]

    # Safe sort
    if "Position_Size" in df.columns:
        df = df.sort_values("Position_Size", ascending=False)

    st.dataframe(
        df.reset_index(drop=True),
        use_container_width=True
    )

# ============================================================
# BACKTEST RESULTS — BUCKET C
# ============================================================

st.header("📊 Backtest Results — Bucket C")

if equity.empty:
    st.info("No backtest equity available")
else:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Equity Curve")
        st.line_chart(
            equity.set_index("Date")["Equity"]
        )

    with col2:
        st.subheader("Summary Stats")
        if stats:
            stats_df = (
                pd.DataFrame(stats.items(), columns=["Metric", "Value"])
                .sort_values("Metric")
            )
            st.dataframe(stats_df, hide_index=True)
        else:
            st.info("No performance stats available")

# ============================================================
# TRADES
# ============================================================

st.header("🧾 Trade History")

if trades.empty:
    st.info("No trades available")
else:
    st.dataframe(
        trades.sort_values("Date", ascending=False),
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Momentum system • Absolute + Relative regimes • "
    "Persistence-scored • Unified portfolio construction"
)
