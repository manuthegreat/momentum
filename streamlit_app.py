import json
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")
ARTIFACTS = Path("artifacts")

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def metric_row(title: str, stats: dict):
    st.subheader(title)
    cols = st.columns(5)
    cols[0].metric("Total Return (%)", f'{stats.get("Total Return (%)", 0):.2f}')
    cols[1].metric("CAGR (%)", f'{stats.get("CAGR (%)", 0):.2f}')
    cols[2].metric("Sharpe Ratio", f'{stats.get("Sharpe Ratio", 0):.2f}')
    cols[3].metric("Sortino Ratio", f'{stats.get("Sortino Ratio", 0):.2f}')
    cols[4].metric("Max Drawdown (%)", f'{stats.get("Max Drawdown (%)", 0):.2f}')

def trade_stats_row(title: str, tstats: dict):
    st.subheader(title)
    cols = st.columns(5)
    cols[0].metric("Number of Trades", int(tstats.get("Number of Trades", 0)))
    cols[1].metric("Win Rate (%)", f'{tstats.get("Win Rate (%)", 0):.2f}')
    cols[2].metric("Average Win ($)", f'{tstats.get("Average Win ($)", 0):.2f}')
    cols[3].metric("Average Loss ($)", f'{tstats.get("Average Loss ($)", 0):.2f}')
    cols[4].metric("Profit Factor", f'{tstats.get("Profit Factor", 0):.2f}')

def show_today_table(title: str, df: pd.DataFrame, preferred_cols: list[str]):
    st.subheader(title)
    if df.empty:
        st.info("No rows available.")
        return

    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        # fallback: show everything
        st.dataframe(df, width="stretch")
        return

    # nice sort
    sort_col = None
    for c in ["Position_Size", "Weighted_Score", "Weighted Score", "Momentum_Score", "Momentum Score"]:
        if c in df.columns:
            sort_col = c
            break
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)

    st.dataframe(df[cols].reset_index(drop=True), width="stretch")

# -------------------------------
# Load artifacts
# -------------------------------
statsA = load_json(ARTIFACTS / "backtest_stats_A.json")
statsB = load_json(ARTIFACTS / "backtest_stats_B.json")
statsC = load_json(ARTIFACTS / "backtest_stats_C.json")

tstatsA = load_json(ARTIFACTS / "trade_stats_A.json")
tstatsB = load_json(ARTIFACTS / "trade_stats_B.json")
tstatsC = load_json(ARTIFACTS / "trade_stats_C.json")

todayA = load_parquet(ARTIFACTS / "today_A.parquet")
todayB = load_parquet(ARTIFACTS / "today_B.parquet")
todayC = load_parquet(ARTIFACTS / "today_C.parquet")

st.title("📈 Momentum Strategy Dashboard")
st.caption("If you just ran GitHub Actions and numbers look stale/wrong, hard refresh your browser tab.")

# -------------------------------
# Bucket A
# -------------------------------
st.markdown("## ================= BUCKET A — ABSOLUTE MOMENTUM ================")
metric_row("BUCKET A — BACKTEST PERFORMANCE", statsA)
trade_stats_row("BUCKET A — TRADE STATISTICS", tstatsA)
show_today_table(
    "BUCKET A — TODAY'S TRADES (FINAL SELECTION)",
    todayA,
    preferred_cols=["Ticker", "Action", "Weighted_Score", "Momentum_Score", "Early_Momentum_Score", "Consistency",
                    "Weighted Score", "Momentum Score", "Early Momentum Score"]
)

# -------------------------------
# Bucket B
# -------------------------------
st.markdown("## ================= BUCKET B — RELATIVE MOMENTUM ================")
metric_row("BUCKET B — BACKTEST PERFORMANCE", statsB)
trade_stats_row("BUCKET B — TRADE STATISTICS", tstatsB)
show_today_table(
    "BUCKET B — TODAY'S TRADES (FINAL SELECTION)",
    todayB,
    preferred_cols=["Ticker", "Action", "Weighted_Score", "Momentum_Score", "Early_Momentum_Score", "Consistency",
                    "Weighted Score", "Momentum Score", "Early Momentum Score"]
)

# -------------------------------
# Bucket C
# -------------------------------
st.markdown("## ================= BUCKET C — COMBINED PORTFOLIO ================")
metric_row("BUCKET C — BACKTEST PERFORMANCE", statsC)
trade_stats_row("BUCKET C — TRADE STATISTICS", tstatsC)
show_today_table(
    "BUCKET C — TODAY'S TRADES (FINAL, 80/20 ALLOCATION)",
    todayC,
    preferred_cols=["Ticker", "Position_Size", "Action", "Capital", "Weighted_Score",
                    "Position Size", "Weighted Score"]
)
