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
        eq_sorted = equity.sort_values("Date")
        start_val = eq_sorted["Equity"].iloc[0]
        end_val   = eq_sorted["Equity"].iloc[-1]
        days = (eq_sorted["Date"].iloc[-1] - eq_sorted["Date"].iloc[0]).days
        if days > 0:
            cagr = (end_val / start_val) ** (365 / days) - 1

    cols = st.columns(4)
    cols[0].metric("Total Return (%)", round(stats.get("Total Return (%)", 0), 2))
    cols[1].metric("Sharpe", round(stats.get("Sharpe", 0), 2))
    cols[2].metric("Max Drawdown (%)", round(stats.get("Max Drawdown (%)", 0), 2))
    cols[3].metric("CAGR (%)", round(cagr * 100, 2) if cagr else "—")

    if not equity.empty:
        st.caption(f"Last equity date: {equity['Date'].max().date()}")

# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")

if not equity.empty and "Date" in equity.columns:
    eq = equity.sort_values("Date").copy()
    value_col = "Equity" if "Equity" in eq.columns else "Portfolio Value"

    if value_col in eq.columns:
        st.line_chart(eq.set_index("Date")[value_col], width="stretch")
else:
    st.info("Equity data not available")

# ============================================================
# COLUMN NORMALIZATION (KEY FIX)
# ============================================================

def normalize_portfolio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes portfolio tables readable regardless of upstream naming.
    """
    df = df.copy()

    if "Capital" in df.columns and "Position_Size" not in df.columns:
        df["Position_Size"] = df["Capital"]

    return df

# ============================================================
# TODAY'S PORTFOLIO
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
        st.info("No positions for latest rebalance")
    else:
        today = normalize_portfolio_columns(today)
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

# ============================================================
# REBALANCE TIMELINE
# ============================================================

st.subheader("Rebalance Timeline")

if not equity.empty and "Date" in equity.columns:
    eq = equity.sort_values("Date").copy()

    # Detect equity column safely
    if "Equity" in eq.columns:
        eq_val = "Equity"
    elif "Portfolio Value" in eq.columns:
        eq = eq.rename(columns={"Portfolio Value": "Equity"})
        eq_val = "Equity"
    else:
        st.info("Equity value column not found")
        st.stop()
    
    # Now this is safe
    eq["Period PnL"] = eq["Equity"].diff()


    counts = (
        signals[signals["Bucket"] == "C"]
        .groupby("Date")["Ticker"]
        .count()
    )
    eq["# Names"] = eq["Date"].map(counts)

    st.dataframe(
        eq[["Date", "Equity", "Period PnL", "# Names"]]
        .sort_values("Date", ascending=False),
        width="stretch"
    )

# ============================================================
# REBALANCE DRILL-DOWN
# ============================================================

st.subheader("Rebalance Portfolio Drill-Down")

rebalance_dates = (
    signals["Date"]
    .dropna()
    .sort_values(ascending=False)
    .dt.date
    .unique()
    .tolist()
)

if rebalance_dates:
    selected_date = st.selectbox("Select rebalance date", rebalance_dates)
    selected_ts = pd.to_datetime(selected_date)

    rebalance_df = signals[
        (signals["Bucket"] == "C") &
        (signals["Date"] == selected_ts)
    ]

    if not rebalance_df.empty:
        rebalance_df = normalize_portfolio_columns(rebalance_df)

        display_cols = [
            c for c in [
                "Ticker",
                "Position_Size",
                "Weighted_Score",
                "Momentum Score",
                "Early Momentum Score",
                "Consistency",
            ]
            if c in rebalance_df.columns
        ]

        sort_col = (
            "Position_Size"
            if "Position_Size" in rebalance_df.columns
            else "Weighted_Score"
            if "Weighted_Score" in rebalance_df.columns
            else "Ticker"
        )

        st.dataframe(
            rebalance_df[display_cols]
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True),
            width="stretch"
        )
    else:
        st.info("No portfolio for selected rebalance date")
