import json
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
# LOADERS (SAFE)
# ============================================================

@st.cache_data
def load_parquet(path: Path):
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data
def load_json(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

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

    value_col = None
    if "Equity" in equity.columns:
        value_col = "Equity"
    elif "Portfolio Value" in equity.columns:
        value_col = "Portfolio Value"

    if value_col:
        st.line_chart(
            equity.set_index("Date")[value_col],
            width="stretch"
        )
    else:
        st.info("Equity value column not found")
else:
    st.info("Equity data not available")

# ============================================================
# CURRENT PORTFOLIO — BUCKET C (RAW, ALL 20 ROWS)
# ============================================================

st.subheader("Current Portfolio — Bucket C (Raw)")

if signals.empty or "Date" not in signals.columns:
    st.info("No signal data available")
else:
    latest_date = signals["Date"].max()

    today = signals[
        (signals.get("Bucket") == "C") &
        (signals["Date"] == latest_date)
    ]

    if today.empty:
        st.info("No Bucket C rows for latest date")
    else:
        st.dataframe(
            today.reset_index(drop=True),
            width="stretch"
        )

# ============================================================
# ARTIFACTS EXPLORER (RAW & HONEST)
# ============================================================

st.subheader("📦 Artifacts Explorer")

tab1, tab2, tab3 = st.tabs(["Signals", "Equity", "Trades"])

with tab1:
    if signals.empty:
        st.info("No signals available")
    else:
        st.dataframe(
            signals.sort_values(
                "Date" if "Date" in signals.columns else signals.columns[0],
                ascending=False
            ),
            width="stretch"
        )

with tab2:
    if equity.empty:
        st.info("No equity data available")
    else:
        st.dataframe(
            equity.sort_values(
                "Date" if "Date" in equity.columns else equity.columns[0],
                ascending=False
            ),
            width="stretch"
        )

with tab3:
    if trades.empty:
        st.info("No trades available")
    else:
        st.dataframe(
            trades.sort_values(
                "Date" if "Date" in trades.columns else trades.columns[0],
                ascending=False
            ),
            width="stretch"
        )
