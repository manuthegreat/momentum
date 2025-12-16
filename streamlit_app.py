import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Momentum Strategy — Execution View",
    layout="wide",
)

ARTIFACTS = Path("artifacts")

SIGNALS_PATH = ARTIFACTS / "backtest_signals.parquet"
EQUITY_PATH  = ARTIFACTS / "backtest_equity_C.parquet"
STATS_PATH   = ARTIFACTS / "backtest_stats_C.json"

# ============================================================
# LOADERS
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

signals = load_parquet(SIGNALS_PATH)
equity  = load_parquet(EQUITY_PATH)
stats   = load_json(STATS_PATH)

# ============================================================
# TITLE
# ============================================================

st.title("📈 Momentum Strategy — Daily Execution")

# ============================================================
# BACKTEST SUMMARY
# ============================================================

st.subheader("Backtest Summary (Bucket C)")

if stats:
    # Derive CAGR if equity exists
    cagr = None
    if not equity.empty and {"Date", "Equity"}.issubset(equity.columns):
        equity = equity.sort_values("Date")
        start_val = equity["Equity"].iloc[0]
        end_val   = equity["Equity"].iloc[-1]
        days = (equity["Date"].iloc[-1] - equity["Date"].iloc[0]).days
        if days > 0:
            cagr = (end_val / start_val) ** (365 / days) - 1

    cols = st.columns(4)

    cols[0].metric("Total Return (%)", round(stats.get("Total Return (%)", 0), 2))
    cols[1].metric("Sharpe", round(stats.get("Sharpe", 0), 2))
    cols[2].metric("Max Drawdown (%)", round(stats.get("Max Drawdown (%)", 0), 2))
    cols[3].metric(
        "CAGR (%)",
        round(cagr * 100, 2) if cagr is not None else "—"
    )

    if not equity.empty and "Date" in equity.columns:
        st.caption(f"Last equity date: {equity['Date'].max().date()}")
else:
    st.info("Backtest stats not available")

# ============================================================
# TODAY'S PORTFOLIO (EXECUTION VIEW)
# ============================================================

st.subheader("Today's Portfolio — Bucket C")

if signals.empty or "Date" not in signals.columns:
    st.info("No signal data available")
else:
    latest_date = signals["Date"].max()

    today = signals[
        (signals["Bucket"] == "C") &
        (signals["Date"] == latest_date)
    ]

    if today.empty:
        st.info("No positions for latest rebalance date")
    else:
        display_cols = [
            c for c in [
                "Ticker",
                "Position_Size",
                "Weighted_Score",
                "Momentum Score",
                "Early Momentum Score",
                "Consistency",
            ]
            if c in today.columns
        ]

        st.caption(f"Rebalance date: {latest_date.date()}")
        st.dataframe(
            today[display_cols]
            .sort_values("Position_Size", ascending=False),
            width="stretch"
        )

# ============================================================
# REBALANCE TIMELINE
# ============================================================

st.subheader("Rebalance Timeline")

if equity.empty or "Date" not in equity.columns:
    st.info("No equity history available")
else:
    eq = equity.sort_values("Date").copy()

    eq["Period_PnL"] = eq["Equity"].diff()
    eq["Num_Names"] = np.nan

    if not signals.empty:
        counts = (
            signals[signals["Bucket"] == "C"]
            .groupby("Date")["Ticker"]
            .count()
        )
        eq["Num_Names"] = eq["Date"].map(counts)

    timeline = eq[["Date", "Equity", "Period_PnL", "Num_Names"]].copy()
    timeline = timeline.rename(columns={
        "Equity": "Portfolio Equity",
        "Period_PnL": "Period PnL",
        "Num_Names": "# Names",
    })

    st.dataframe(
        timeline.sort_values("Date", ascending=False),
        width="stretch"
    )
