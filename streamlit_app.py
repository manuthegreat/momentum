import json
import pandas as pd
import streamlit as st
from pathlib import Path
import math

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")
ARTIFACTS = Path("artifacts")

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def fmt(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{float(x):.2f}"
    except Exception:
        return "—"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force canonical column names so Streamlit never guesses.
    """
    if df.empty:
        return df

    rename_map = {
        "Weighted Score": "Weighted_Score",
        "Momentum Score": "Momentum_Score",
        "Early Momentum Score": "Early_Momentum_Score",
        "Position Size": "Position_Size",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# -------------------------------------------------
# UI Blocks
# -------------------------------------------------

def metric_row(title: str, stats: dict):
    st.subheader(title)
    cols = st.columns(5)
    cols[0].metric("Total Return (%)", fmt(stats.get("Total Return (%)")))
    cols[1].metric("CAGR (%)", fmt(stats.get("CAGR (%)")))
    cols[2].metric("Sharpe Ratio", fmt(stats.get("Sharpe Ratio")))
    cols[3].metric("Sortino Ratio", fmt(stats.get("Sortino Ratio")))
    cols[4].metric("Max Drawdown (%)", fmt(stats.get("Max Drawdown (%)")))

def trade_stats_row(title: str, tstats: dict):
    st.subheader(title)
    cols = st.columns(5)
    cols[0].metric("Number of Trades", int(tstats.get("Number of Trades", 0)))
    cols[1].metric("Win Rate (%)", fmt(tstats.get("Win Rate (%)")))
    cols[2].metric("Average Win ($)", fmt(tstats.get("Average Win ($)")))
    cols[3].metric("Average Loss ($)", fmt(tstats.get("Average Loss ($)")))
    cols[4].metric("Profit Factor", fmt(tstats.get("Profit Factor")))

def show_today_table(title: str, df: pd.DataFrame, preferred_cols: list[str]):
    st.subheader(title)

    if df.empty:
        st.info("No rows available.")
        return

    df = normalize_columns(df)

    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        st.dataframe(df, width="stretch")
        return

    # deterministic sorting
    for sort_col in ["Position_Size", "Weighted_Score", "Momentum_Score"]:
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)
            break

    st.dataframe(df[cols].reset_index(drop=True), width="stretch")

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------

statsA = load_json(ARTIFACTS / "backtest_stats_A.json")
statsB = load_json(ARTIFACTS / "backtest_stats_B.json")
statsC = load_json(ARTIFACTS / "backtest_stats_C.json")

tstatsA = load_json(ARTIFACTS / "trade_stats_A.json")
tstatsB = load_json(ARTIFACTS / "trade_stats_B.json")
tstatsC = load_json(ARTIFACTS / "trade_stats_C.json")

todayA = load_parquet(ARTIFACTS / "today_A.parquet")
todayB = load_parquet(ARTIFACTS / "today_B.parquet")
todayC = load_parquet(ARTIFACTS / "today_C.parquet")

# -------------------------------------------------
# Page
# -------------------------------------------------

st.title("📈 Momentum Strategy Dashboard")
st.caption("If you just ran GitHub Actions and numbers look stale/wrong, hard refresh your browser tab.")

# ---------------- Bucket A ----------------
st.markdown("## ================= BUCKET A — ABSOLUTE MOMENTUM ================")
metric_row("BUCKET A — BACKTEST PERFORMANCE", statsA)
trade_stats_row("BUCKET A — TRADE STATISTICS", tstatsA)
show_today_table(
    "BUCKET A — TODAY'S TRADES (FINAL SELECTION)",
    todayA,
    ["Ticker", "Action", "Weighted_Score", "Momentum_Score", "Early_Momentum_Score", "Consistency"]
)

# ---------------- Bucket B ----------------
st.markdown("## ================= BUCKET B — RELATIVE MOMENTUM ================")
metric_row("BUCKET B — BACKTEST PERFORMANCE", statsB)
trade_stats_row("BUCKET B — TRADE STATISTICS", tstatsB)
show_today_table(
    "BUCKET B — TODAY'S TRADES (FINAL SELECTION)",
    todayB,
    ["Ticker", "Action", "Weighted_Score", "Momentum_Score", "Early_Momentum_Score", "Consistency"]
)

# ---------------- Bucket C ----------------
st.markdown("## ================= BUCKET C — COMBINED PORTFOLIO ================")
metric_row("BUCKET C — BACKTEST PERFORMANCE", statsC)
trade_stats_row("BUCKET C — TRADE STATISTICS", tstatsC)
show_today_table(
    "BUCKET C — TODAY'S TRADES (FINAL, 80/20 ALLOCATION)",
    todayC,
    ["Ticker", "Position_Size", "Action"]
)
