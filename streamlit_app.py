import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Momentum Strategy — Bucket C",
    layout="wide",
)

ARTIFACTS = Path("artifacts")

STATS_PATH   = ARTIFACTS / "backtest_stats_C.json"
TRADES_PATH  = ARTIFACTS / "backtest_trades_C.parquet"
TODAY_PATH   = ARTIFACTS / "today_C.parquet"
EQUITY_PATH  = ARTIFACTS / "backtest_equity_C.parquet"

# ============================================================
# LOADERS (SAFE)
# ============================================================

@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def normalize_dates(df: pd.DataFrame, col="Date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.dropna(subset=[col])

def pick_equity_col(df: pd.DataFrame) -> str | None:
    if "Equity" in df.columns:
        return "Equity"
    if "Portfolio Value" in df.columns:
        return "Portfolio Value"
    return None

# ============================================================
# LOAD DATA
# ============================================================

stats   = load_json(STATS_PATH)
trades  = load_parquet(TRADES_PATH)
today   = load_parquet(TODAY_PATH)
equity  = load_parquet(EQUITY_PATH)

equity = normalize_dates(equity)
today  = normalize_dates(today)
trades = normalize_dates(trades)

# ============================================================
# UI
# ============================================================

st.title("📈 Momentum Strategy — Bucket C")

# ============================================================
# BACKTEST PERFORMANCE
# ============================================================

st.subheader("Backtest Performance")

if stats:
    cols = st.columns(5)

    cols[0].metric("Total Return (%)", round(stats.get("Total Return (%)", 0), 2))
    cols[1].metric("CAGR (%)", round(stats.get("CAGR (%)", 0), 2))
    cols[2].metric("Sharpe Ratio", round(stats.get("Sharpe Ratio", stats.get("Sharpe", 0)), 2))
    cols[3].metric("Sortino Ratio", round(stats.get("Sortino Ratio", stats.get("Sortino", 0)), 2))
    cols[4].metric("Max Drawdown (%)", round(stats.get("Max Drawdown (%)", 0), 2))
else:
    st.info("Backtest stats not available")

# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")

if equity.empty:
    st.info("Equity data not available")
else:
    value_col = pick_equity_col(equity)
    if value_col is None:
        st.info("Equity column not found")
    else:
        equity = equity.sort_values("Date")
        st.line_chart(
            equity.set_index("Date")[value_col],
            width="stretch"
        )
        st.caption(f"Last equity date: {equity['Date'].max().date()}")

# ============================================================
# TRADE STATISTICS
# ============================================================

st.subheader("Trade Statistics")

if trades.empty or "PnL" not in trades.columns:
    st.info("Trade statistics not available")
else:
    pnl = pd.to_numeric(trades["PnL"], errors="coerce")

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Number of Trades", len(pnl.dropna()))
    c2.metric("Win Rate (%)", round((wins.count() / pnl.count()) * 100, 2) if pnl.count() else "—")
    c3.metric("Average Win ($)", round(wins.mean(), 2) if not wins.empty else "—")
    c4.metric("Average Loss ($)", round(losses.mean(), 2) if not losses.empty else "—")
    c5.metric("Profit Factor", round(abs(wins.sum() / losses.sum()), 2) if losses.sum() != 0 else "—")

# ============================================================
# TODAY'S PORTFOLIO
# ============================================================

st.subheader("Today's Portfolio (Final Allocation)")

if today.empty:
    st.info("No portfolio for today")
else:
    if "Date" in today.columns:
        rebalance_date = today["Date"].max().date()
        st.caption(f"Rebalance date: {rebalance_date}")

    display_cols = [
        c for c in [
            "Ticker",
            "Position_Size",
            "Action",
            "Weighted_Score",
            "Momentum_Score",
            "Early_Momentum_Score",
            "Consistency",
        ]
        if c in today.columns
    ]

    sort_col = (
        "Position_Size" if "Position_Size" in today.columns
        else "Weighted_Score" if "Weighted_Score" in today.columns
        else "Ticker"
    )

    st.dataframe(
        today[display_cols]
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True),
        width="stretch"
    )

# ============================================================
# OPTIONAL DEBUG
# ============================================================

with st.expander("🔍 Raw Artifacts (debug)", expanded=False):
    st.write("today_C.parquet", today.head(10))
    st.write("backtest_trades_C.parquet", trades.head(10))
    st.write("backtest_equity_C.parquet", equity.head(10))
