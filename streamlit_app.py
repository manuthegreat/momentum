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

# Bucket artifacts
STATS_A = ARTIFACTS / "backtest_stats_A.json"
STATS_B = ARTIFACTS / "backtest_stats_B.json"
STATS_C = ARTIFACTS / "backtest_stats_C.json"

TODAY_A = ARTIFACTS / "today_A.parquet"
TODAY_B = ARTIFACTS / "today_B.parquet"
TODAY_C = ARTIFACTS / "today_C.parquet"

# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

stats_A = load_json(STATS_A)
stats_B = load_json(STATS_B)
stats_C = load_json(STATS_C)

today_A = load_parquet(TODAY_A)
today_B = load_parquet(TODAY_B)
today_C = load_parquet(TODAY_C)

# ============================================================
# UI HELPERS
# ============================================================

def render_stats(title: str, stats: dict):
    st.subheader(title)

    if not stats:
        st.info("No backtest statistics available")
        return

    cols = st.columns(5)
    cols[0].metric("Total Return (%)", round(stats.get("Total Return (%)", 0), 2))
    cols[1].metric("CAGR (%)", round(stats.get("CAGR (%)", 0), 2))
    cols[2].metric("Sharpe Ratio", round(stats.get("Sharpe Ratio", 0), 2))
    cols[3].metric("Sortino Ratio", round(stats.get("Sortino Ratio", 0), 2))
    cols[4].metric("Max Drawdown (%)", round(stats.get("Max Drawdown (%)", 0), 2))

def render_today(title: str, df: pd.DataFrame, sort_col: str | None = None):
    st.subheader(title)

    if df.empty:
        st.info("No trades available")
        return

    display_cols = [
        c for c in [
            "Ticker",
            "Action",
            "Position_Size",
            "Weighted_Score",
            "Momentum_Score",
            "Early_Momentum_Score",
            "Consistency",
        ]
        if c in df.columns
    ]

    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    st.dataframe(
        df[display_cols].reset_index(drop=True),
        width="stretch"
    )

# ============================================================
# MAIN PAGE
# ============================================================

st.title("📈 Momentum Strategy Dashboard")

# ============================
# BUCKET A — ABSOLUTE MOMENTUM
# ============================

render_stats(
    "Bucket A — Absolute Momentum (Backtest Performance)",
    stats_A
)

render_today(
    "Bucket A — Today's Trades",
    today_A,
    sort_col="Weighted_Score"
)

st.divider()

# ============================
# BUCKET B — RELATIVE MOMENTUM
# ============================

render_stats(
    "Bucket B — Relative Momentum (Backtest Performance)",
    stats_B
)

render_today(
    "Bucket B — Today's Trades",
    today_B,
    sort_col="Weighted_Score"
)

st.divider()

# ============================
# BUCKET C — COMBINED PORTFOLIO
# ============================

render_stats(
    "Bucket C — Combined Portfolio (Backtest Performance)",
    stats_C
)

render_today(
    "Bucket C — Today's Trades (Final Allocation)",
    today_C,
    sort_col="Position_Size"
)
