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

if stats and not equity.empty and {"Date", "Equity"}.issubset(equity.columns):
    eq = equity.sort_values("Date")

    start_val = eq["Equity"].iloc[0]
    end_val   = eq["Equity"].iloc[-1]
    days = (eq["Date"].iloc[-1] - eq["Date"].iloc[0]).days

    cagr = (end_val / start_val) ** (365 / days) - 1 if days > 0 else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", round(stats.get("Total Return (%)", 0), 2))
    c2.metric("Sharpe", round(stats.get("Sharpe", 0), 2))
    c3.metric("Max Drawdown (%)", round(stats.get("Max Drawdown (%)", 0), 2))
    c4.metric("CAGR (%)", round(cagr * 100, 2) if cagr else "—")

    st.caption(f"Last equity date: {eq['Date'].max().date()}")
else:
    st.info("Backtest stats unavailable")

# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")

if not equity.empty and {"Date", "Equity"}.issubset(equity.columns):
    st.line_chart(
        equity.sort_values("Date").set_index("Date")["Equity"],
        width="stretch"
    )
else:
    st.info("Equity data not available")

# ============================================================
# TODAY'S PORTFOLIO
# ============================================================

st.subheader("Today's Portfolio — Bucket C")

if not signals.empty and {"Date", "Bucket"}.issubset(signals.columns):
    latest_date = signals["Date"].max()
    today = signals[
        (signals["Bucket"] == "C") &
        (signals["Date"] == latest_date)
    ]

    if not today.empty:
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

        sort_col = (
            "Position_Size"
            if "Position_Size" in today.columns
            else "Weighted_Score"
            if "Weighted_Score" in today.columns
            else "Ticker"
        )

        st.dataframe(
            today[display_cols]
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True),
            width="stretch"
        )
    else:
        st.info("No positions for latest rebalance")
else:
    st.info("Signal data unavailable")

# ============================================================
# REBALANCE TIMELINE
# ============================================================

st.subheader("Rebalance Timeline")

if not equity.empty and {"Date", "Equity"}.issubset(equity.columns):
    eq = equity.sort_values("Date").copy()
    eq["Period PnL"] = eq["Equity"].diff()

    if not signals.empty and {"Date", "Bucket", "Ticker"}.issubset(signals.columns):
        counts = (
            signals[signals["Bucket"] == "C"]
            .groupby("Date")["Ticker"]
            .count()
        )
        eq["# Names"] = eq["Date"].map(counts)

    st.dataframe(
        eq[["Date", "Equity", "Period PnL", "# Names"]]
        .sort_values("Date", ascending=False)
        .reset_index(drop=True),
        width="stretch"
    )
else:
    st.info("No rebalance history available")

# ============================================================
# REBALANCE DRILL-DOWN (WITH TICKER PnL)
# ============================================================

st.subheader("Rebalance Portfolio Drill-Down")

if not equity.empty and not signals.empty:
    rebalance_dates = (
        equity["Date"]
        .dropna()
        .sort_values(ascending=False)
        .dt.date
        .unique()
        .tolist()
    )

    selected_date = st.selectbox("Select rebalance date", rebalance_dates)
    rebalance_ts = pd.to_datetime(selected_date)

    # Portfolio at rebalance
    portfolio = signals[
        (signals["Bucket"] == "C") &
        (signals["Date"] == rebalance_ts)
    ].copy()

    # Next rebalance equity date (for PnL window)
    next_dates = equity[equity["Date"] > rebalance_ts]["Date"]
    if portfolio.empty or next_dates.empty:
        st.info("No drill-down data available for selected date")
    else:
        next_ts = next_dates.min()

        eq_start = equity.loc[equity["Date"] == rebalance_ts, "Equity"].values
        eq_end   = equity.loc[equity["Date"] == next_ts, "Equity"].values

        if len(eq_start) == 0 or len(eq_end) == 0:
            st.info("PnL window incomplete for selected rebalance")
        else:
            total_pnl = eq_end[0] - eq_start[0]

            # Allocate PnL proportionally by position size
            if "Position_Size" in portfolio.columns:
                w = portfolio["Position_Size"] / portfolio["Position_Size"].sum()
                portfolio["PnL"] = w * total_pnl
            else:
                portfolio["PnL"] = np.nan

            display_cols = [
                c for c in [
                    "Ticker",
                    "Position_Size",
                    "PnL",
                    "Weighted_Score",
                    "Momentum Score",
                    "Early Momentum Score",
                    "Consistency",
                ]
                if c in portfolio.columns
            ]

            st.dataframe(
                portfolio[display_cols]
                .sort_values("PnL", ascending=False)
                .reset_index(drop=True),
                width="stretch"
            )
else:
    st.info("Insufficient data for drill-down")
