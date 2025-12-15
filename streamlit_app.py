import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Momentum Dashboard",
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
    """
    Clean Bucket C:
    - One row per ticker
    - Sum Position_Size
    - Take MAX of available score columns
    - Never assume schema
    """
    if df.empty:
        return df

    agg = {}

    if "Position_Size" in df.columns:
        agg["Position_Size"] = "sum"

    if "Date" in df.columns:
        agg["Date"] = "max"

    # Score columns — only include if present
    for col in [
        "Momentum Score",
        "Early Momentum Score",
        "Consistency",
        "Weighted_Score",
    ]:
        if col in df.columns:
            agg[col] = "max"

    if not agg:
        return df

    return (
        df.groupby("Ticker", as_index=False)
        .agg(agg)
        .sort_values(
            "Position_Size" if "Position_Size" in agg else df.columns[0],
            ascending=False
        )
        .reset_index(drop=True)
    )

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

if not equity.empty and "Equity" in equity.columns:
    equity = equity.sort_values("Date")
    st.line_chart(
        equity.set_index("Date")["Equity"],
        width="stretch"
    )
else:
    st.info("Equity data not available")

# ============================================================
# ROLLING RETURNS
# ============================================================

st.subheader("Rolling Returns")

if not equity.empty and "Equity" in equity.columns:
    eq = equity.sort_values("Date").copy()
    eq["Return"] = eq["Equity"].pct_change()

    roll_20 = (1 + eq["Return"]).rolling(20).apply(np.prod, raw=True) - 1
    roll_60 = (1 + eq["Return"]).rolling(60).apply(np.prod, raw=True) - 1

    st.line_chart(
        pd.DataFrame({
            "20D Rolling Return": roll_20,
            "60D Rolling Return": roll_60,
        }).dropna(),
        width="stretch"
    )
else:
    st.info("Not enough equity history for rolling returns")

# ============================================================
# DRAWDOWNS
# ============================================================

st.subheader("Drawdowns")

if not equity.empty and "Equity" in equity.columns:
    roll_max = equity["Equity"].cummax()
    drawdown = equity["Equity"] / roll_max - 1

    st.area_chart(drawdown, width="stretch")
else:
    st.info("Drawdown data unavailable")

# ============================================================
# CURRENT PORTFOLIO — BUCKET C
# ============================================================

st.subheader("Current Portfolio — Bucket C")

if signals.empty:
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
        st.dataframe(clean_c, width="stretch")

        if "Weighted_Score" in clean_c.columns:
            st.subheader("Signal Strength (Weighted Score)")
            st.bar_chart(
                clean_c.set_index("Ticker")["Weighted_Score"],
                width="stretch"
            )

# ============================================================
# TRADE ANALYTICS
# ============================================================

st.subheader("Trade Diagnostics")

if trades.empty:
    st.info("No trades available")
else:
    if "Date" in trades.columns:
        trades = trades.sort_values("Date", ascending=False)

    if "Action" in trades.columns and "PnL" in trades.columns:
        sells = trades[trades["Action"] == "Sell"]

        if not sells.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Closed Trades", len(sells))
            col2.metric("Win Rate (%)", round((sells["PnL"] > 0).mean() * 100, 1))
            col3.metric("Avg PnL ($)", round(sells["PnL"].mean(), 2))

            st.dataframe(
                sells.sort_values("PnL", ascending=False),
                width="stretch"
            )
        else:
            st.info("No closed trades yet")
    else:
        st.dataframe(trades, width="stretch")

# ============================================================
# RAW DEBUG
# ============================================================

with st.expander("🔍 Raw Artifacts"):
    st.write("Signals (head)")
    st.dataframe(signals.head(20))
    st.write("Equity (head)")
    st.dataframe(equity.head(20))
    st.write("Trades (head)")
    st.dataframe(trades.head(20))
