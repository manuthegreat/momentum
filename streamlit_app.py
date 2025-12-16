import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")

ARTIFACTS = Path("artifacts")

# Stats artifacts (A/B/C)
STATS_PATHS = {
    "A": ARTIFACTS / "backtest_stats_A.json",
    "B": ARTIFACTS / "backtest_stats_B.json",
    "C": ARTIFACTS / "backtest_stats_C.json",
}

# "Today" artifacts (may or may not be correct schema depending on your pipeline stage)
TODAY_PATHS = {
    "A": ARTIFACTS / "today_A.parquet",
    "B": ARTIFACTS / "today_B.parquet",
    "C": ARTIFACTS / "today_C.parquet",
}

# Canonical signals artifact that should contain bucket selections
BACKTEST_SIGNALS_PATH = ARTIFACTS / "backtest_signals.parquet"
DAILY_SIGNALS_PATH = ARTIFACTS / "daily_signals.parquet"  # optional fallback


# ============================================================
# LOADERS (with cache)
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


def hard_refresh_button():
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🔄 Hard refresh"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        st.caption("If you just ran GitHub Actions and numbers look stale/wrong, hit Hard refresh.")


# ============================================================
# NORMALIZERS
# ============================================================

def _first_present(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_stats(stats: dict) -> dict:
    """
    Normalize stats from differing key conventions.
    Supports:
      - Sharpe vs Sharpe Ratio
      - Sortino vs Sortino Ratio
      - CAGR (%) might be absent
    """
    total_return = _first_present(stats, ["Total Return (%)", "Total Return", "total_return_pct"])
    cagr = _first_present(stats, ["CAGR (%)", "CAGR", "cagr_pct"])
    sharpe = _first_present(stats, ["Sharpe Ratio", "Sharpe", "sharpe"])
    sortino = _first_present(stats, ["Sortino Ratio", "Sortino", "sortino"])
    max_dd = _first_present(stats, ["Max Drawdown (%)", "Max Drawdown", "max_drawdown_pct"])

    def fmt(x):
        if isinstance(x, (int, float)):
            return float(x)
        return None

    return {
        "Total Return (%)": fmt(total_return),
        "CAGR (%)": fmt(cagr),
        "Sharpe Ratio": fmt(sharpe),
        "Sortino Ratio": fmt(sortino),
        "Max Drawdown (%)": fmt(max_dd),
    }


def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.dropna(subset=[col])


def get_latest_bucket_portfolio_from_signals(signals_df: pd.DataFrame, bucket: str) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """
    From a signals dataframe, pull latest date rows for a bucket.
    Requires columns: Date, Bucket, Ticker (others optional).
    """
    if signals_df.empty:
        return pd.DataFrame(), None

    needed = {"Date", "Bucket", "Ticker"}
    if not needed.issubset(signals_df.columns):
        return pd.DataFrame(), None

    s = ensure_datetime(signals_df, "Date")
    s = s[s["Bucket"] == bucket]
    if s.empty:
        return pd.DataFrame(), None

    latest = s["Date"].max()
    out = s[s["Date"] == latest].copy()
    return out, latest


def choose_today_table(bucket: str) -> tuple[pd.DataFrame, pd.Timestamp | None, str]:
    """
    Prefer today_X.parquet IF it looks like a portfolio (has Ticker).
    Otherwise fallback to backtest_signals.parquet (latest date for bucket).
    Finally fallback to daily_signals.parquet.
    Returns (df, as_of_date, source_label)
    """
    # 1) Try today_X.parquet
    df_today = load_parquet(TODAY_PATHS[bucket])
    if not df_today.empty and "Ticker" in df_today.columns:
        # If it also has Date, show it; else date unknown.
        dt = None
        if "Date" in df_today.columns:
            dft = ensure_datetime(df_today, "Date")
            if not dft.empty:
                dt = dft["Date"].max()
                df_today = dft[dft["Date"] == dt].copy() if "Bucket" not in dft.columns else dft.copy()
        return df_today, dt, f"{TODAY_PATHS[bucket].name}"

    # 2) Fallback to backtest_signals.parquet
    backtest_signals = load_parquet(BACKTEST_SIGNALS_PATH)
    df_sig, dt = get_latest_bucket_portfolio_from_signals(backtest_signals, bucket)
    if not df_sig.empty:
        return df_sig, dt, BACKTEST_SIGNALS_PATH.name

    # 3) Fallback to daily_signals.parquet (optional)
    daily_signals = load_parquet(DAILY_SIGNALS_PATH)
    df_sig2, dt2 = get_latest_bucket_portfolio_from_signals(daily_signals, bucket)
    if not df_sig2.empty:
        return df_sig2, dt2, DAILY_SIGNALS_PATH.name

    return pd.DataFrame(), None, "none"


def render_today(df: pd.DataFrame, bucket: str, as_of: pd.Timestamp | None, source: str):
    st.subheader(f"Bucket {bucket} — TODAY'S TRADES (Final Selection)")

    if as_of is not None:
        st.caption(f"As of: {as_of.date()} • Source: {source}")
    else:
        st.caption(f"Source: {source}")

    if df.empty:
        st.info("No rows found for this bucket.")
        return

    # Prefer baseline columns (your CLI)
    preferred = [
        "Ticker",
        "Action",
        "Position_Size",
        "Weighted_Score",
        "Momentum_Score",
        "Early_Momentum_Score",
        "Consistency",
    ]
    display_cols = [c for c in preferred if c in df.columns]

    # If your artifacts use spaces like "Momentum Score", normalize on-the-fly (display only)
    rename_map = {}
    if "Momentum Score" in df.columns and "Momentum_Score" not in df.columns:
        rename_map["Momentum Score"] = "Momentum_Score"
    if "Early Momentum Score" in df.columns and "Early_Momentum_Score" not in df.columns:
        rename_map["Early Momentum Score"] = "Early_Momentum_Score"
    if "Weighted Score" in df.columns and "Weighted_Score" not in df.columns:
        rename_map["Weighted Score"] = "Weighted_Score"

    if rename_map:
        df = df.rename(columns=rename_map)
        display_cols = [c for c in preferred if c in df.columns]

    # Sort like you expect
    sort_col = None
    if "Position_Size" in df.columns:
        sort_col = "Position_Size"
    elif "Weighted_Score" in df.columns:
        sort_col = "Weighted_Score"

    if sort_col:
        df = df.sort_values(sort_col, ascending=False)

    # Ensure we ALWAYS show something useful
    if not display_cols and "Ticker" in df.columns:
        display_cols = ["Ticker"]

    st.dataframe(df[display_cols].reset_index(drop=True), width="stretch")


def render_stats(bucket: str, stats_raw: dict):
    st.subheader(f"Bucket {bucket} — BACKTEST PERFORMANCE")

    if not stats_raw:
        st.info("No backtest stats found.")
        return

    s = normalize_stats(stats_raw)

    cols = st.columns(5)
    cols[0].metric("Total Return (%)", "—" if s["Total Return (%)"] is None else f'{s["Total Return (%)"]:.2f}')
    cols[1].metric("CAGR (%)", "—" if s["CAGR (%)"] is None else f'{s["CAGR (%)"]:.2f}')
    cols[2].metric("Sharpe Ratio", "—" if s["Sharpe Ratio"] is None else f'{s["Sharpe Ratio"]:.2f}')
    cols[3].metric("Sortino Ratio", "—" if s["Sortino Ratio"] is None else f'{s["Sortino Ratio"]:.2f}')
    cols[4].metric("Max Drawdown (%)", "—" if s["Max Drawdown (%)"] is None else f'{s["Max Drawdown (%)"]:.2f}')

    # Debug-friendly: show which keys exist (collapsed)
    with st.expander("Stats debug (keys)"):
        st.write(stats_raw)


# ============================================================
# MAIN
# ============================================================

st.title("📈 Momentum Strategy Dashboard")
hard_refresh_button()

for bucket in ["A", "B", "C"]:
    render_stats(bucket, load_json(STATS_PATHS[bucket]))

    df_today, as_of, source = choose_today_table(bucket)
    render_today(df_today, bucket, as_of, source)

    st.divider()
