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
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def coerce_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if not df.empty and col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def pick_equity_value_col(equity: pd.DataFrame) -> str | None:
    """Return the best equity column name from known variants."""
    if equity.empty:
        return None
    for c in ["Equity", "Portfolio Value", "portfolio_value", "equity", "value"]:
        if c in equity.columns:
            return c
    return None

def clean_bucket_c(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Bucket C:
    - One row per ticker
    - Sum Position_Size (if present)
    - Take MAX of available score columns (if present)
    - Never assume schema beyond having 'Ticker'
    """
    if df.empty or "Ticker" not in df.columns:
        return df

    df = df.copy()

    # Build aggregation dict only for columns that exist
    agg: dict[str, str] = {}

    if "Position_Size" in df.columns:
        agg["Position_Size"] = "sum"

    # Keep the most recent date seen for that ticker (helps display)
    for date_col in ["Date", "AsOf", "As_Of", "SignalDate"]:
        if date_col in df.columns:
            agg[date_col] = "max"
            break

    # Score columns — take max if present
    score_candidates = [
        "Momentum Score",
        "Early Momentum Score",
        "Consistency",
        "Weighted_Score",
        "Weighted Score",
        "Score",
        "Final Score",
    ]
    for col in score_candidates:
        if col in df.columns:
            agg[col] = "max"

    if not agg:
        # If we can't aggregate meaningfully, just dedupe
        return df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    out = (
        df.groupby("Ticker", as_index=False)
          .agg(agg)
    )

    # Sort preference: Position_Size if present, else Weighted score-ish, else first col
    sort_col = None
    if "Position_Size" in out.columns:
        sort_col = "Position_Size"
    else:
        for c in ["Weighted_Score", "Weighted Score", "Score", "Final Score", "Momentum Score"]:
            if c in out.columns:
                sort_col = c
                break

    if sort_col:
        out = out.sort_values(sort_col, ascending=False)

    return out.reset_index(drop=True)

def safe_metric_value(v):
    if isinstance(v, (int, float, np.number)) and np.isfinite(v):
        return round(float(v), 2)
    return v

# ============================================================
# LOAD DATA
# ============================================================

signals = coerce_datetime(load_parquet(SIGNALS_PATH), "Date")
equity  = coerce_datetime(load_parquet(EQUITY_PATH), "Date")
trades  = coerce_datetime(load_parquet(TRADES_PATH), "Date")
stats   = load_json(STATS_PATH)

st.title("📈 Momentum Strategy Dashboard")

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.subheader("Performance Summary (Bucket C)")

if isinstance(stats, dict) and len(stats) > 0:
    keys = list(stats.keys())
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        col.metric(k, safe_metric_value(stats.get(k)))
else:
    st.info("No performance stats available")

# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")

value_col = pick_equity_value_col(equity)

if not equity.empty and "Date" in equity.columns and value_col:
    eq = equity.dropna(subset=["Date", value_col]).sort_values("Date").copy()
    if not eq.empty:
        st.line_chart(eq.set_index("Date")[value_col], width="stretch")
    else:
        st.info("Equity data exists but is empty after cleaning")
else:
    st.info("Equity data not available (missing file or expected columns)")

# ============================================================
# ROLLING RETURNS
# ============================================================

st.subheader("Rolling Returns")

if not equity.empty and "Date" in equity.columns and value_col:
    eq = equity.dropna(subset=["Date", value_col]).sort_values("Date").copy()

    if len(eq) >= 25:
        eq["Return"] = eq[value_col].pct_change()

        roll_20 = (1 + eq["Return"]).rolling(20).apply(np.prod, raw=True) - 1
        roll_60 = (1 + eq["Return"]).rolling(60).apply(np.prod, raw=True) - 1

        rr = pd.DataFrame(
            {"20D Rolling Return": roll_20, "60D Rolling Return": roll_60},
            index=eq["Date"],
        ).dropna()

        if not rr.empty:
            st.line_chart(rr, width="stretch")
        else:
            st.info("Rolling return series is empty after dropna()")
    else:
        st.info("Not enough equity history for rolling returns (need ~25+ points)")
else:
    st.info("Not enough equity history for rolling returns")

# ============================================================
# DRAWDOWNS
# ============================================================

st.subheader("Drawdowns")

if not equity.empty and "Date" in equity.columns and value_col:
    eq = equity.dropna(subset=["Date", value_col]).sort_values("Date").copy()
    if not eq.empty:
        roll_max = eq[value_col].cummax()
        drawdown = (eq[value_col] / roll_max) - 1
        dd = pd.DataFrame({"Drawdown": drawdown.values}, index=eq["Date"])
        st.area_chart(dd, width="stretch")
    else:
        st.info("Drawdown data unavailable (empty after cleaning)")
else:
    st.info("Drawdown data unavailable")

# ============================================================
# CURRENT PORTFOLIO — BUCKET C
# ============================================================

st.subheader("Current Portfolio — Bucket C")

if signals.empty:
    st.info("No signal data available")
else:
    # Ensure required columns exist
    if "Bucket" not in signals.columns:
        st.info("Signals exist but missing 'Bucket' column")
    elif "Date" not in signals.columns:
        st.info("Signals exist but missing 'Date' column")
    elif "Ticker" not in signals.columns:
        st.info("Signals exist but missing 'Ticker' column")
    else:
        bucket_c = signals[signals["Bucket"] == "C"].copy()

        if bucket_c.empty:
            st.info("No Bucket C signals found")
        else:
            latest_date = bucket_c["Date"].max()
            today = bucket_c[bucket_c["Date"] == latest_date].copy()

            if today.empty:
                st.info("No Bucket C positions for latest Bucket C date")
            else:
                clean_c = clean_bucket_c(today)

                # Display: enforce one row per ticker and show useful columns first (if present)
                preferred_order = [
                    "Ticker",
                    "Position_Size",
                    "Weighted_Score",
                    "Weighted Score",
                    "Momentum Score",
                    "Early Momentum Score",
                    "Consistency",
                    "Date",
                ]
                cols_present = [c for c in preferred_order if c in clean_c.columns]
                remaining = [c for c in clean_c.columns if c not in cols_present]
                clean_c = clean_c[cols_present + remaining] if cols_present else clean_c

                st.caption(f"Latest Bucket C date: {latest_date.date() if pd.notna(latest_date) else latest_date}")
                st.dataframe(clean_c, use_container_width=True)

                score_col = None
                for c in ["Weighted_Score", "Weighted Score", "Score", "Final Score", "Momentum Score"]:
                    if c in clean_c.columns:
                        score_col = c
                        break

                if score_col:
                    st.subheader(f"Signal Strength ({score_col})")
                    chart_df = clean_c.set_index("Ticker")[score_col].dropna()
                    if not chart_df.empty:
                        st.bar_chart(chart_df, width="stretch")
                    else:
                        st.info("Score column exists but has no numeric values")

# ============================================================
# TRADE ANALYTICS
# ============================================================

st.subheader("Trade Diagnostics")

if trades.empty:
    st.info("No trades available")
else:
    # If Date exists, show recent first
    if "Date" in trades.columns:
        trades = trades.sort_values("Date", ascending=False)

    # Flexible naming for pnl/action
    pnl_col = None
    for c in ["PnL", "Pnl", "pnl", "Profit", "profit"]:
        if c in trades.columns:
            pnl_col = c
            break

    action_col = None
    for c in ["Action", "action", "Side", "side", "Event", "event"]:
        if c in trades.columns:
            action_col = c
            break

    # Try to compute diagnostics on "closed" trades if we can identify them
    sells = pd.DataFrame()
    if action_col and pnl_col:
        # Common patterns for sells/exit
        sells = trades[trades[action_col].astype(str).str.lower().isin(["sell", "exit", "close"])].copy()

    if not sells.empty and pnl_col:
        col1, col2, col3 = st.columns(3)
        col1.metric("Closed Trades", len(sells))
        col2.metric("Win Rate (%)", round((pd.to_numeric(sells[pnl_col], errors="coerce") > 0).mean() * 100, 1))
        col3.metric("Avg PnL", round(pd.to_numeric(sells[pnl_col], errors="coerce").mean(), 2))

        # Show best -> worst
        sells_sorted = sells.copy()
        sells_sorted[pnl_col] = pd.to_numeric(sells_sorted[pnl_col], errors="coerce")
        sells_sorted = sells_sorted.sort_values(pnl_col, ascending=False)
        st.dataframe(sells_sorted, use_container_width=True)
    else:
        st.dataframe(trades, use_container_width=True)

# ============================================================
# RAW DEBUG
# ============================================================

with st.expander("🔍 Raw Artifacts"):
    st.write("Signals (head)")
    st.dataframe(signals.head(20), use_container_width=True)

    st.write("Equity (head)")
    st.dataframe(equity.head(20), use_container_width=True)

    st.write("Trades (head)")
    st.dataframe(trades.head(20), use_container_width=True)
