import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Momentum Strategy — Daily Execution",
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
    cols[3].metric("CAGR (%)", round(cagr * 100, 2) if cagr else "—")

    if not equity.empty and "Date" in equity.columns:
        st.caption(f"Last equity date: {equity['Date'].max().date()}")
else:
    st.info("Backtest stats not available")

# ============================================================
# TODAY'S PORTFOLIO — EXECUTION VIEW
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
        st.caption(f"Rebalance date: {latest_date.date()}")

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

        # 🔑 Safe sorting logic
        if "Position_Size" in today.columns:
            today = today.sort_values("Position_Size", ascending=False)
        elif "Weighted_Score" in today.columns:
            today = today.sort_values("Weighted_Score", ascending=False)

        st.dataframe(
            today[display_cols],
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

    # 🔑 Detect equity column safely
    value_col = None
    if "Equity" in eq.columns:
        value_col = "Equity"
    elif "Portfolio Value" in eq.columns:
        value_col = "Portfolio Value"

    if value_col is None:
        st.info("Equity value column not found")
    else:
        eq = eq.rename(columns={value_col: "Equity"})

        eq["Period PnL"] = eq["Equity"].diff()

        # Optional: number of names held per rebalance
        if not signals.empty and {"Date", "Bucket", "Ticker"}.issubset(signals.columns):
            counts = (
                signals[signals["Bucket"] == "C"]
                .groupby("Date")["Ticker"]
                .count()
            )
            eq["# Names"] = eq["Date"].map(counts)

        timeline_cols = ["Date", "Equity", "Period PnL"]
        if "# Names" in eq.columns:
            timeline_cols.append("# Names")

        st.dataframe(
            eq[timeline_cols]
            .sort_values("Date", ascending=False),
            width="stretch"
        )
