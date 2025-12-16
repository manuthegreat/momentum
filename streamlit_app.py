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
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

signals = load_parquet(SIGNALS_PATH)
equity_raw = load_parquet(EQUITY_PATH)
stats = load_json(STATS_PATH)

# ============================================================
# HELPERS
# ============================================================

def normalize_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a normalized equity DF with columns: Date, Equity
    Accepts either 'Equity' or 'Portfolio Value'.
    """
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    value_col = None
    if "Equity" in out.columns:
        value_col = "Equity"
    elif "Portfolio Value" in out.columns:
        value_col = "Portfolio Value"

    if value_col is None:
        return pd.DataFrame()

    out = out[["Date", value_col]].rename(columns={value_col: "Equity"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out

def safe_sort(df: pd.DataFrame, preferred: list[str], ascending=False) -> pd.DataFrame:
    for c in preferred:
        if c in df.columns:
            return df.sort_values(c, ascending=ascending)
    return df

def portfolio_for_date(signals_df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    """Return Bucket C portfolio rows for an exact timestamp match, with tolerant date handling."""
    if signals_df.empty or not {"Date", "Bucket"}.issubset(signals_df.columns):
        return pd.DataFrame()

    s = signals_df.copy()
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    s = s.dropna(subset=["Date"])

    # match exact timestamp if present; otherwise match by date()
    exact = s[(s["Bucket"] == "C") & (s["Date"] == dt)]
    if not exact.empty:
        return exact

    return s[(s["Bucket"] == "C") & (s["Date"].dt.date == dt.date())]

def add_ticker_pnl_attribution(portfolio: pd.DataFrame, total_pnl: float) -> pd.DataFrame:
    """
    Adds a 'PnL' column by allocating total_pnl proportionally by Position_Size if present.
    If Position_Size missing, PnL will be NaN.
    """
    out = portfolio.copy()
    if "Position_Size" in out.columns:
        ps = pd.to_numeric(out["Position_Size"], errors="coerce")
        denom = ps.sum(skipna=True)
        if denom and denom != 0:
            out["PnL"] = (ps / denom) * total_pnl
        else:
            out["PnL"] = np.nan
    else:
        out["PnL"] = np.nan
    return out

# ============================================================
# NORMALIZE EQUITY
# ============================================================

equity = normalize_equity_df(equity_raw)

# ============================================================
# UI
# ============================================================

st.title("📈 Momentum Strategy — Daily Execution")

# ------------------------------------------------------------
# BACKTEST SUMMARY
# ------------------------------------------------------------

st.subheader("Backtest Summary (Bucket C)")

if stats:
    # CAGR from normalized equity if available
    cagr = None
    last_eq_date = None

    if not equity.empty:
        start_val = float(equity["Equity"].iloc[0])
        end_val   = float(equity["Equity"].iloc[-1])
        days = (equity["Date"].iloc[-1] - equity["Date"].iloc[0]).days
        if days > 0 and start_val > 0:
            cagr = (end_val / start_val) ** (365 / days) - 1
        last_eq_date = equity["Date"].iloc[-1].date()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", round(float(stats.get("Total Return (%)", 0.0)), 2))
    c2.metric("Sharpe", round(float(stats.get("Sharpe", 0.0)), 2))
    c3.metric("Max Drawdown (%)", round(float(stats.get("Max Drawdown (%)", 0.0)), 2))
    c4.metric("CAGR (%)", round(cagr * 100, 2) if isinstance(cagr, (int, float)) else "—")

    if last_eq_date:
        st.caption(f"Last equity date: {last_eq_date}")
else:
    st.info("Backtest stats unavailable")

# ------------------------------------------------------------
# EQUITY CURVE
# ------------------------------------------------------------

st.subheader("Equity Curve")

if not equity.empty:
    st.line_chart(
        equity.set_index("Date")["Equity"],
        width="stretch"
    )
else:
    st.info("Equity data not available")

# ------------------------------------------------------------
# TODAY'S PORTFOLIO
# ------------------------------------------------------------

st.subheader("Today's Portfolio — Bucket C")

if signals.empty or "Date" not in signals.columns:
    st.info("No signal data available")
else:
    s = signals.copy()
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    s = s.dropna(subset=["Date"])

    latest_signal_dt = s["Date"].max()
    today = portfolio_for_date(s, latest_signal_dt)

    if today.empty:
        st.info("No positions for latest rebalance date")
    else:
        st.caption(f"Rebalance date: {latest_signal_dt.date()}")

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

        today = safe_sort(today, ["Position_Size", "Weighted_Score", "Ticker"], ascending=False)

        st.dataframe(
            today[display_cols].reset_index(drop=True),
            width="stretch"
        )

# ------------------------------------------------------------
# REBALANCE TIMELINE
# ------------------------------------------------------------

st.subheader("Rebalance Timeline")

if equity.empty:
    st.info("No rebalance history available")
else:
    eq = equity.copy()
    eq["Period PnL"] = eq["Equity"].diff()

    # Optional: number of names per rebalance
    if not signals.empty and {"Date", "Bucket", "Ticker"}.issubset(signals.columns):
        s = signals.copy()
        s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
        s = s.dropna(subset=["Date"])
        counts = (
            s[s["Bucket"] == "C"]
            .groupby(s["Date"].dt.date)["Ticker"]
            .count()
        )
        eq["# Names"] = eq["Date"].dt.date.map(counts)

    st.dataframe(
        eq[["Date", "Equity", "Period PnL"] + (["# Names"] if "# Names" in eq.columns else [])]
        .sort_values("Date", ascending=False)
        .reset_index(drop=True),
        width="stretch"
    )

# ------------------------------------------------------------
# REBALANCE DRILL-DOWN (Ticker-level PnL)
# ------------------------------------------------------------

st.subheader("Rebalance Portfolio Drill-Down")

if equity.empty or signals.empty:
    st.info("No drill-down data available")
else:
    # use equity dates as the rebalance schedule (since the backtest writes those)
    rebalance_dates = equity["Date"].dt.date.unique().tolist()
    rebalance_dates = sorted(rebalance_dates, reverse=True)

    selected_date = st.selectbox("Select rebalance date", rebalance_dates, index=0)
    selected_ts = pd.to_datetime(selected_date)

    # next rebalance date (to compute period pnl)
    future = equity[equity["Date"] > selected_ts]["Date"]
    if future.empty:
        st.info("Selected date is the last rebalance in equity history (no forward PnL window).")
    else:
        next_ts = future.min()

        eq0 = equity.loc[equity["Date"].dt.date == selected_ts.date(), "Equity"]
        eq1 = equity.loc[equity["Date"].dt.date == next_ts.date(), "Equity"]

        if eq0.empty or eq1.empty:
            st.info("PnL window incomplete for the selected rebalance date.")
        else:
            period_pnl = float(eq1.iloc[0] - eq0.iloc[0])

            port = portfolio_for_date(signals, selected_ts)
            if port.empty:
                st.info("No portfolio rows found in signals for the selected rebalance date.")
            else:
                port = add_ticker_pnl_attribution(port, period_pnl)

                st.caption(f"PnL window: {selected_ts.date()} → {next_ts.date()} | Period PnL: {period_pnl:,.2f}")

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
                    if c in port.columns
                ]

                port = safe_sort(port, ["PnL", "Position_Size", "Weighted_Score", "Ticker"], ascending=False)

                st.dataframe(
                    port[display_cols].reset_index(drop=True),
                    width="stretch"
                )
