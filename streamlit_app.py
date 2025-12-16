import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Momentum Strategy — Daily Execution", layout="wide")

ARTIFACTS = Path("artifacts")
SIGNALS_PATH = ARTIFACTS / "backtest_signals.parquet"
EQUITY_PATH = ARTIFACTS / "backtest_equity_C.parquet"
STATS_PATH = ARTIFACTS / "backtest_stats_C.json"

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

signals_raw = load_parquet(SIGNALS_PATH)
equity_raw = load_parquet(EQUITY_PATH)
stats = load_json(STATS_PATH)

# ============================================================
# NORMALIZERS (robust against schema drift)
# ============================================================

def normalize_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns normalized equity DF: columns = [Date, Equity]
    Accepts either 'Equity' or 'Portfolio Value' as the value column.
    """
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    value_col = None
    if "Equity" in out.columns:
        value_col = "Equity"
    elif "Portfolio Value" in out.columns:
        value_col = "Portfolio Value"

    if value_col is None:
        return pd.DataFrame()

    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])

    out = out[["Date", value_col]].rename(columns={value_col: "Equity"})
    out = out.sort_values("Date").reset_index(drop=True)
    return out

def normalize_signals_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns normalized signals DF with:
      - Date (datetime)
      - Bucket (string)
      - Ticker (string)
      - Position_Size (numeric) where possible (derived from common alternatives)

    This avoids you having to guess whether the artifact uses Position_Size vs Capital, etc.
    """
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Required columns
    if "Date" not in out.columns or "Ticker" not in out.columns:
        return pd.DataFrame()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    if "Bucket" in out.columns:
        out["Bucket"] = out["Bucket"].astype(str)
    else:
        out["Bucket"] = "C"  # fallback, but ideally bucket exists

    out["Ticker"] = out["Ticker"].astype(str)

    # Unify position sizing column if possible
    # Try these in priority order:
    candidate_size_cols = [
        "Position_Size",
        "Capital",
        "Allocation",
        "Alloc",
        "Dollar_Allocation",
        "Dollars",
        "Notional",
        "Position",
    ]

    size_col = next((c for c in candidate_size_cols if c in out.columns), None)
    if size_col is not None:
        out["Position_Size"] = pd.to_numeric(out[size_col], errors="coerce")
    else:
        # if nothing exists, create it so downstream code doesn't break
        out["Position_Size"] = np.nan

    # Unify score columns if there are slight naming variations
    rename_map = {}
    if "Weighted Score" in out.columns and "Weighted_Score" not in out.columns:
        rename_map["Weighted Score"] = "Weighted_Score"
    if rename_map:
        out = out.rename(columns=rename_map)

    return out

equity = normalize_equity_df(equity_raw)
signals = normalize_signals_df(signals_raw)

# ============================================================
# HELPERS
# ============================================================

def pick_existing_cols(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [c for c in preferred if c in df.columns]

def safe_sort(df: pd.DataFrame, preferred_cols: list[str], ascending=False) -> pd.DataFrame:
    for c in preferred_cols:
        if c in df.columns:
            # only sort if there is at least one non-null
            if df[c].notna().any():
                return df.sort_values(c, ascending=ascending)
    return df

def build_rebalance_timeline(eq: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """
    Timeline based on equity rows (rebalance dates in your backtest).
    Adds:
      - Period PnL (diff)
      - Period Return (pct_change)
      - # Names (from signals, if available)
      - Rebalance Date (date)
    """
    if eq.empty:
        return pd.DataFrame()

    out = eq.copy().sort_values("Date").reset_index(drop=True)
    out["Period PnL"] = out["Equity"].diff()
    out["Period Return"] = out["Equity"].pct_change()

    if not sig.empty and {"Date", "Bucket", "Ticker"}.issubset(sig.columns):
        tmp = sig.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["Date"])
        counts = (
            tmp[tmp["Bucket"] == "C"]
            .groupby(tmp["Date"].dt.date)["Ticker"]
            .count()
        )
        out["# Names"] = out["Date"].dt.date.map(counts)

    return out

timeline = build_rebalance_timeline(equity, signals)

def portfolio_for_timestamp(sig: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch portfolio rows for bucket C for a given rebalance timestamp.
    Matches by exact timestamp if possible, else by date().
    """
    if sig.empty:
        return pd.DataFrame()

    s = sig.copy()
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    s = s.dropna(subset=["Date"])

    exact = s[(s["Bucket"] == "C") & (s["Date"] == dt)]
    if not exact.empty:
        return exact

    return s[(s["Bucket"] == "C") & (s["Date"].dt.date == dt.date())]

def allocate_ticker_pnl(portfolio: pd.DataFrame, period_pnl: float) -> pd.DataFrame:
    """
    Allocate portfolio period PnL to tickers proportional to Position_Size.
    If Position_Size missing/unusable => PnL = NaN.
    """
    out = portfolio.copy()

    if "Position_Size" not in out.columns:
        out["PnL"] = np.nan
        return out

    ps = pd.to_numeric(out["Position_Size"], errors="coerce")
    denom = ps.sum(skipna=True)

    if denom is None or denom == 0 or np.isnan(denom):
        out["PnL"] = np.nan
        return out

    out["PnL"] = (ps / denom) * float(period_pnl)
    return out

# ============================================================
# UI
# ============================================================

st.title("📈 Momentum Strategy — Daily Execution")

# ------------------------------------------------------------
# BACKTEST SUMMARY
# ------------------------------------------------------------

st.subheader("Backtest Summary (Bucket C)")

if stats:
    cagr = None
    last_eq_date = None

    if not equity.empty:
        start_val = float(equity["Equity"].iloc[0])
        end_val = float(equity["Equity"].iloc[-1])
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

if equity.empty:
    st.info("Equity data not available")
else:
    st.line_chart(equity.set_index("Date")["Equity"], width="stretch")

# ------------------------------------------------------------
# TODAY'S PORTFOLIO (show size + score if available)
# ------------------------------------------------------------

st.subheader("Today's Portfolio — Bucket C")

if signals.empty or "Date" not in signals.columns:
    st.info("No signal data available")
else:
    latest_dt = signals["Date"].max()
    today = portfolio_for_timestamp(signals, latest_dt)

    if today.empty:
        st.info("No positions for latest rebalance date")
    else:
        st.caption(f"Rebalance date: {latest_dt.date()}")

        display_cols = pick_existing_cols(
            today,
            [
                "Ticker",
                "Position_Size",
                "Weighted_Score",
                "Momentum Score",
                "Early Momentum Score",
                "Consistency",
            ],
        )

        # If Position_Size is all NaN, don't hide it—still show it (useful debugging),
        # but sort by Weighted_Score if available.
        today_sorted = safe_sort(today[display_cols], ["Position_Size", "Weighted_Score"], ascending=False)

        st.dataframe(today_sorted.reset_index(drop=True), width="stretch")

        # Helpful hint if sizing isn't available in the artifact
        if "Position_Size" in today.columns and not today["Position_Size"].notna().any():
            st.info(
                "Note: Position sizing is not present in this signals artifact for today "
                "(Position_Size/Capital not found or all null). If you want ticker-level PnL attribution, "
                "we’ll need sizing written into backtest_signals.parquet."
            )

# ------------------------------------------------------------
# REBALANCE TIMELINE (use next row for period pnl)
# ------------------------------------------------------------

st.subheader("Rebalance Timeline")

if timeline.empty:
    st.info("No rebalance history available")
else:
    show_cols = ["Date", "Equity", "Period PnL", "Period Return"]
    if "# Names" in timeline.columns:
        show_cols.append("# Names")

    tl = timeline.copy().sort_values("Date", ascending=False).reset_index(drop=True)
    st.dataframe(tl[show_cols], width="stretch")

# ------------------------------------------------------------
# REBALANCE DRILL-DOWN (ticker pnl attribution)
# ------------------------------------------------------------

st.subheader("Rebalance Portfolio Drill-Down")

if timeline.empty or signals.empty:
    st.info("No drill-down data available")
else:
    # Select among equity (rebalance) dates only
    rebalance_dates = (
        timeline["Date"]
        .dropna()
        .sort_values(ascending=False)
        .dt.date
        .unique()
        .tolist()
    )

    selected_date = st.selectbox("Select rebalance date", rebalance_dates, index=0)
    selected_ts = pd.to_datetime(selected_date)

    # Find the timeline row for this selected date
    tl_match = timeline[timeline["Date"].dt.date == selected_date].sort_values("Date")
    period_pnl = None
    period_ret = None

    if not tl_match.empty:
        # Period PnL is already computed as diff vs previous row in the equity table.
        # For drill-down we want the PnL *realized over the NEXT period*.
        # So we locate the selected row in the sorted timeline and take next row difference.
        t_sorted = timeline.sort_values("Date").reset_index(drop=True)
        idxs = t_sorted.index[t_sorted["Date"].dt.date == selected_date].tolist()
        if idxs:
            i = idxs[-1]
            if i + 1 < len(t_sorted):
                period_pnl = float(t_sorted.loc[i + 1, "Equity"] - t_sorted.loc[i, "Equity"])
                period_ret = float((t_sorted.loc[i + 1, "Equity"] / t_sorted.loc[i, "Equity"]) - 1)

    if period_pnl is not None:
        st.caption(f"Next-period PnL from this rebalance: {period_pnl:,.2f}  |  Return: {period_ret*100:,.2f}%")
    else:
        st.caption("Next-period PnL from this rebalance: unavailable (no next equity point)")

    port = portfolio_for_timestamp(signals, selected_ts)
    if port.empty:
        st.info("No portfolio found for selected rebalance date")
    else:
        display_cols = pick_existing_cols(
            port,
            [
                "Ticker",
                "Position_Size",
                "Weighted_Score",
                "Momentum Score",
                "Early Momentum Score",
                "Consistency",
            ],
        )

        df = port[display_cols].copy()

        # Add ticker-level pnl attribution (only meaningful if we have both period_pnl and Position_Size)
        if period_pnl is not None:
            df = allocate_ticker_pnl(df, period_pnl)
        else:
            df["PnL"] = np.nan

        # Sort: prefer Position_Size, else Weighted_Score, else Ticker
        df = safe_sort(df, ["Position_Size", "Weighted_Score"], ascending=False)

        # Put PnL near the front if present
        if "PnL" in df.columns:
            cols_order = ["Ticker"]
            if "Position_Size" in df.columns:
                cols_order.append("Position_Size")
            cols_order.append("PnL")
            for c in ["Weighted_Score", "Momentum Score", "Early Momentum Score", "Consistency"]:
                if c in df.columns:
                    cols_order.append(c)

            # Keep only columns that exist
            cols_order = [c for c in cols_order if c in df.columns]
            df = df[cols_order]

        st.dataframe(df.reset_index(drop=True), width="stretch")

        if "Position_Size" in df.columns and df["Position_Size"].isna().all():
            st.info(
                "Ticker-level PnL attribution is blank because position sizing isn't present in the signals "
                "artifact for this rebalance date."
            )
