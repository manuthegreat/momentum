import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")

ARTIFACTS = Path("artifacts")

SIGNALS_PATH = ARTIFACTS / "backtest_signals.parquet"
EQUITY_PATH  = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_PATH  = ARTIFACTS / "backtest_trades_C.parquet"
STATS_PATH   = ARTIFACTS / "backtest_stats_C.json"

DEFAULT_BUCKET = "C"

# ============================================================
# HELPERS (robust)
# ============================================================

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Normalize Date columns if present
    for c in ["Date", "Entry_Date", "Exit_Date", "Rebalance_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing col in candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def normalize_weights(pos: pd.Series) -> pd.Series:
    pos = to_numeric_safe(pos).fillna(0.0)
    total = pos.sum()
    if total <= 0:
        # equal weights fallback
        n = len(pos)
        return pd.Series(np.ones(n) / n, index=pos.index) if n else pos
    return pos / total

def clean_bucket_c_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return RAW (no aggregation) but only the most relevant columns if present.
    Keeps all rows (e.g., 20 rows), no KeyErrors.
    """
    if df.empty:
        return df

    preferred = [
        "Ticker",
        "Momentum Score",
        "Early Momentum Score",
        "Consistency",
        "Weighted_Score",
        "Position_Size",
        "Capital",
        "Source",
        "Appearances",
        "Date",
        "Entry_Date",
        "Exit_Date",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        return df

    out = df[cols].copy()
    # nice numeric formatting hints (Streamlit will still render raw)
    return out

def clean_bucket_c_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per ticker:
      - sum Position_Size (or Capital)
      - max scores
    Never raises KeyError.
    """
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()

    g = df.groupby("Ticker", dropna=False)

    out = pd.DataFrame({"Ticker": g.size().index.astype(str)})

    pos_col = safe_col(df, ["Position_Size", "Capital"])
    if pos_col:
        pos = g[pos_col].sum(min_count=1).reset_index()
        pos.columns = ["Ticker", "Position_Size"]
        out = out.merge(pos, on="Ticker", how="left")
    else:
        out["Position_Size"] = np.nan

    score_cols = ["Weighted_Score", "Momentum Score", "Early Momentum Score", "Consistency"]
    for col in score_cols:
        if col in df.columns:
            mx = g[col].max().reset_index()
            out = out.merge(mx, on="Ticker", how="left")

    out = out.sort_values(
        "Position_Size" if "Position_Size" in out.columns else "Ticker",
        ascending=False
    ).reset_index(drop=True)

    return out

def compute_equity_metrics(equity: pd.DataFrame) -> dict:
    if equity.empty or "Date" not in equity.columns:
        return {}

    value_col = safe_col(equity, ["Equity", "Portfolio Value"])
    if not value_col:
        return {}

    e = equity.dropna(subset=["Date"]).sort_values("Date").copy()
    e[value_col] = to_numeric_safe(e[value_col])
    e = e.dropna(subset=[value_col])
    if len(e) < 2:
        return {}

    start_date = e["Date"].iloc[0]
    end_date   = e["Date"].iloc[-1]
    start_val  = float(e[value_col].iloc[0])
    end_val    = float(e[value_col].iloc[-1])

    # CAGR (calendar)
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (end_val / start_val) ** (1 / years) - 1 if years and years > 0 and start_val > 0 else np.nan

    peak = float(e[value_col].cummax().iloc[-1])
    below_ath = (end_val / peak - 1) if peak > 0 else np.nan

    ret = e[value_col].pct_change().dropna()
    vol = ret.std() * np.sqrt(252) if len(ret) > 1 else np.nan
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if (len(ret) > 1 and ret.std() not in [0, np.nan]) else np.nan

    return {
        "Start": start_date.date().isoformat(),
        "End": end_date.date().isoformat(),
        "Start Value": start_val,
        "End Value": end_val,
        "CAGR (%)": (cagr * 100) if pd.notna(cagr) else np.nan,
        "Below ATH (%)": (below_ath * 100) if pd.notna(below_ath) else np.nan,
        "Ann. Vol (%)": (vol * 100) if pd.notna(vol) else np.nan,
        "Sharpe (from equity)": sharpe if pd.notna(sharpe) else np.nan,
    }

def rolling_return_series(equity: pd.DataFrame, window: int) -> pd.Series:
    value_col = safe_col(equity, ["Equity", "Portfolio Value"])
    if equity.empty or "Date" not in equity.columns or not value_col:
        return pd.Series(dtype=float)

    e = equity.dropna(subset=["Date"]).sort_values("Date").copy()
    e[value_col] = to_numeric_safe(e[value_col])
    e = e.dropna(subset=[value_col])
    if len(e) < window + 2:
        return pd.Series(dtype=float)

    r = e[value_col].pct_change()
    roll = (1 + r).rolling(window).apply(np.prod, raw=True) - 1
    roll.index = e["Date"]
    return roll.dropna()

def compute_turnover(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> float | None:
    """
    Turnover vs previous rebalance:
      0.5 * sum(|w_new - w_old|)
    Uses Position_Size if present; else equal-weight fallback.
    """
    if prev_df.empty or curr_df.empty or "Ticker" not in prev_df.columns or "Ticker" not in curr_df.columns:
        return None

    prev = prev_df.copy()
    curr = curr_df.copy()

    pos_col_prev = safe_col(prev, ["Position_Size", "Capital"])
    pos_col_curr = safe_col(curr, ["Position_Size", "Capital"])

    prev = prev.groupby("Ticker", dropna=False)[pos_col_prev].sum() if pos_col_prev else prev.groupby("Ticker").size()
    curr = curr.groupby("Ticker", dropna=False)[pos_col_curr].sum() if pos_col_curr else curr.groupby("Ticker").size()

    prev = prev.astype(float)
    curr = curr.astype(float)

    all_tickers = sorted(set(prev.index.astype(str)).union(set(curr.index.astype(str))))
    prev = prev.reindex(all_tickers).fillna(0.0)
    curr = curr.reindex(all_tickers).fillna(0.0)

    w0 = normalize_weights(prev)
    w1 = normalize_weights(curr)

    return float(0.5 * np.abs(w1 - w0).sum())

def signal_persistence(signals: pd.DataFrame, bucket: str, lookback_days: int) -> pd.DataFrame:
    """
    For each ticker: appearances in last N days + avg daily rank (by Weighted_Score if available, else by Momentum Score)
    """
    if signals.empty or "Date" not in signals.columns or "Ticker" not in signals.columns:
        return pd.DataFrame()

    df = signals.copy()
    if "Bucket" in df.columns:
        df = df[df["Bucket"] == bucket]

    latest = df["Date"].max()
    if pd.isna(latest):
        return pd.DataFrame()

    cutoff = latest - pd.Timedelta(days=lookback_days)
    df = df[(df["Date"] >= cutoff) & (df["Date"] <= latest)].copy()
    if df.empty:
        return pd.DataFrame()

    score_col = safe_col(df, ["Weighted_Score", "Momentum Score", "Early Momentum Score"])
    if score_col:
        df[score_col] = to_numeric_safe(df[score_col])
        df["Rank"] = df.groupby("Date")[score_col].rank(ascending=False, method="average")
    else:
        df["Rank"] = np.nan

    out = df.groupby("Ticker", dropna=False).agg(
        Days_Appeared=("Date", "nunique"),
        Avg_Rank=("Rank", "mean"),
    ).reset_index()

    out = out.sort_values(["Days_Appeared", "Avg_Rank"], ascending=[False, True]).reset_index(drop=True)
    return out

def regime_summary(today: pd.DataFrame) -> dict:
    """
    Pulls whatever exists — shows signal health. No assumptions.
    """
    if today.empty:
        return {}

    out = {}

    # Common score cols
    for col in ["Weighted_Score", "Momentum Score", "Early Momentum Score", "Consistency"]:
        if col in today.columns:
            s = to_numeric_safe(today[col])
            out[f"Avg {col}"] = float(s.mean()) if s.notna().any() else np.nan
            out[f"Median {col}"] = float(s.median()) if s.notna().any() else np.nan

    # Any "Regime" columns if present
    regime_cols = [c for c in today.columns if "regime" in c.lower()]
    for c in regime_cols[:6]:  # cap to avoid spam
        s = to_numeric_safe(today[c]) if pd.api.types.is_numeric_dtype(today[c]) else today[c]
        if pd.api.types.is_numeric_dtype(s):
            out[f"Avg {c}"] = float(pd.to_numeric(s, errors="coerce").mean())
        else:
            # categorical-ish
            vc = s.astype(str).value_counts().head(1)
            if len(vc):
                out[f"Top {c}"] = f"{vc.index[0]} ({int(vc.iloc[0])})"

    # Acceleration-type columns
    accel_cols = [c for c in today.columns if "accel" in c.lower()]
    for c in accel_cols[:3]:
        s = to_numeric_safe(today[c])
        if s.notna().any():
            out[f"% {c} > 0"] = float((s > 0).mean() * 100)

    return out

# ============================================================
# LOAD DATA
# ============================================================

signals = load_parquet(SIGNALS_PATH)
equity  = load_parquet(EQUITY_PATH)
trades  = load_parquet(TRADES_PATH)
stats   = load_json(STATS_PATH)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Controls")

bucket = DEFAULT_BUCKET
if not signals.empty and "Bucket" in signals.columns:
    buckets = sorted([b for b in signals["Bucket"].dropna().unique().tolist()])
    bucket = st.sidebar.selectbox("Bucket", options=buckets, index=buckets.index(DEFAULT_BUCKET) if DEFAULT_BUCKET in buckets else 0)

portfolio_mode = st.sidebar.radio("Current Portfolio View", ["Raw (show all rows)", "Aggregate by Ticker"], index=0)
max_rows = st.sidebar.slider("Max rows in tables", min_value=20, max_value=500, value=200, step=20)

persist_days = st.sidebar.slider("Signal persistence lookback (days)", 5, 120, 30, 5)

st.title("📈 Momentum Strategy Dashboard")

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.subheader(f"Performance Summary (Bucket {bucket})")

left, right = st.columns([2, 3])

with left:
    if stats:
        # Keep your existing stats as-is
        cols = st.columns(min(len(stats), 4))
        items = list(stats.items())
        for i, (k, v) in enumerate(items[:4]):
            cols[i].metric(k, round(v, 2) if isinstance(v, (int, float)) else v)

        # If more than 4 stats, show the rest in a small table
        if len(items) > 4:
            more = pd.DataFrame(items[4:], columns=["Metric", "Value"])
            st.dataframe(more, width="stretch", height=220)
    else:
        st.info("No performance stats available")

with right:
    eq_metrics = compute_equity_metrics(equity)
    if eq_metrics:
        cols = st.columns(4)
        cols[0].metric("Start", eq_metrics["Start"])
        cols[1].metric("End", eq_metrics["End"])
        cols[2].metric("CAGR (%)", f'{eq_metrics["CAGR (%)"]:.2f}' if pd.notna(eq_metrics["CAGR (%)"]) else "—")
        cols[3].metric("Below ATH (%)", f'{eq_metrics["Below ATH (%)"]:.2f}' if pd.notna(eq_metrics["Below ATH (%)"]) else "—")

        cols2 = st.columns(4)
        cols2[0].metric("Start Value", f'{eq_metrics["Start Value"]:.0f}')
        cols2[1].metric("End Value", f'{eq_metrics["End Value"]:.0f}')
        cols2[2].metric("Ann. Vol (%)", f'{eq_metrics["Ann. Vol (%)"]:.2f}' if pd.notna(eq_metrics["Ann. Vol (%)"]) else "—")
        cols2[3].metric("Sharpe (from equity)", f'{eq_metrics["Sharpe (from equity)"]:.2f}' if pd.notna(eq_metrics["Sharpe (from equity)"]) else "—")
    else:
        st.info("Equity context metrics unavailable (missing Date/Equity columns).")

# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")

value_col = safe_col(equity, ["Equity", "Portfolio Value"])
if not equity.empty and "Date" in equity.columns and value_col:
    eplot = equity.dropna(subset=["Date"]).sort_values("Date").copy()
    eplot[value_col] = to_numeric_safe(eplot[value_col])
    eplot = eplot.dropna(subset=[value_col])
    if not eplot.empty:
        st.line_chart(eplot.set_index("Date")[value_col], width="stretch")
    else:
        st.info("Equity curve: no numeric values to plot.")
else:
    st.info("Equity data not available.")

# ============================================================
# ROLLING RETURNS (only if not empty)
# ============================================================

st.subheader("Rolling Returns")

r20 = rolling_return_series(equity, 20)
r60 = rolling_return_series(equity, 60)

roll_df = pd.DataFrame()
if not r20.empty:
    roll_df["20D Rolling Return"] = r20
if not r60.empty:
    roll_df["60D Rolling Return"] = r60

if not roll_df.empty:
    st.line_chart(roll_df, width="stretch")
else:
    st.info("Rolling returns not shown (insufficient equity history for the selected windows).")

# ============================================================
# CURRENT PORTFOLIO + REGIME + DRIFT
# ============================================================

st.subheader(f"Current Portfolio — Bucket {bucket}")

if signals.empty or "Date" not in signals.columns:
    st.info("No signal data available.")
else:
    latest_date = signals["Date"].max()
    if pd.isna(latest_date):
        st.info("Signals have no valid dates.")
    else:
        today = signals[signals["Date"] == latest_date].copy()
        if "Bucket" in today.columns:
            today = today[today["Bucket"] == bucket].copy()

        if today.empty:
            st.info(f"No Bucket {bucket} positions for latest date.")
        else:
            # ---- Portfolio view
            if portfolio_mode == "Aggregate by Ticker":
                port = clean_bucket_c_agg(today)
            else:
                port = clean_bucket_c_raw(today)

            # Sort nicely if possible
            sort_pref = safe_col(port, ["Position_Size", "Capital", "Weighted_Score", "Momentum Score", "Ticker"])
            if sort_pref:
                asc = False if sort_pref != "Ticker" else True
                try:
                    port = port.sort_values(sort_pref, ascending=asc)
                except Exception:
                    pass

            st.dataframe(port.head(max_rows), width="stretch")

            # ---- Regime / signal health
            st.subheader("Signal / Regime Health (what exists in your artifacts)")
            reg = regime_summary(today)
            if reg:
                keys = list(reg.keys())
                cols = st.columns(min(4, len(keys)))
                for i, k in enumerate(keys[:4]):
                    v = reg[k]
                    if isinstance(v, (int, float, np.floating)) and pd.notna(v):
                        cols[i].metric(k, f"{v:.3f}" if abs(v) < 100 else f"{v:.2f}")
                    else:
                        cols[i].metric(k, str(v))

                if len(keys) > 4:
                    tail = pd.DataFrame([(k, reg[k]) for k in keys[4:]], columns=["Metric", "Value"])
                    st.dataframe(tail, width="stretch", height=220)
            else:
                st.info("No regime/signal-health columns detected (this is fine).")

            # ---- Drift monitor: days since rebalance + turnover vs previous rebalance date
            st.subheader("Portfolio Drift Monitor")

            # rebalance dates inferred from unique signal dates for the bucket
            bdf = signals.copy()
            if "Bucket" in bdf.columns:
                bdf = bdf[bdf["Bucket"] == bucket]
            dates = sorted([d for d in bdf["Date"].dropna().unique()])
            if len(dates) >= 2:
                prev_date = dates[-2]
                curr_date = dates[-1]
                days_since = (pd.Timestamp(curr_date) - pd.Timestamp(prev_date)).days

                prev = bdf[bdf["Date"] == prev_date]
                curr = bdf[bdf["Date"] == curr_date]
                turn = compute_turnover(prev, curr)

                c1, c2, c3 = st.columns(3)
                c1.metric("Latest signal date", pd.Timestamp(curr_date).date().isoformat())
                c2.metric("Days since last rebalance", str(days_since))
                c3.metric("Turnover vs last rebalance", f"{turn:.2%}" if turn is not None else "—")
            else:
                st.info("Not enough distinct signal dates to compute drift/turnover.")

# ============================================================
# SIGNAL PERSISTENCE (Upgrade 4)
# ============================================================

st.subheader("Signal Persistence")

persist = signal_persistence(signals, bucket=bucket, lookback_days=persist_days)
if persist.empty:
    st.info("No persistence data available for the chosen window.")
else:
    st.dataframe(persist.head(max_rows), width="stretch")

# ============================================================
# ARTIFACTS EXPLORER (meaningful)
# ============================================================

st.subheader("📦 Artifacts Explorer")

tab1, tab2, tab3 = st.tabs(["Signals", "Equity", "Trades"])

with tab1:
    if signals.empty:
        st.info("No signals available.")
    else:
        df = signals.copy()

        # Filters
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if "Bucket" in df.columns:
                bopts = ["(All)"] + sorted([b for b in df["Bucket"].dropna().unique().tolist()])
                bsel = st.selectbox("Filter Bucket", options=bopts, index=(bopts.index(bucket) if bucket in bopts else 0))
                if bsel != "(All)":
                    df = df[df["Bucket"] == bsel]
        with c2:
            if "Date" in df.columns:
                latest = df["Date"].max()
                days_back = st.number_input("Days back", min_value=1, max_value=3650, value=60, step=5)
                cutoff = latest - pd.Timedelta(days=int(days_back)) if pd.notna(latest) else None
                if cutoff is not None:
                    df = df[df["Date"] >= cutoff]
        with c3:
            q = st.text_input("Search ticker (contains)", value="")
            if q and "Ticker" in df.columns:
                df = df[df["Ticker"].astype(str).str.contains(q, case=False, na=False)]

        # Sort
        sort_cols = [c for c in ["Date", "Weighted_Score", "Momentum Score"] if c in df.columns]
        if sort_cols:
            try:
                df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            except Exception:
                pass

        st.dataframe(df.head(max_rows), width="stretch")

with tab2:
    if equity.empty:
        st.info("No equity data available.")
    else:
        df = equity.copy()
        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=False)
        st.dataframe(df.head(max_rows), width="stretch")

with tab3:
    if trades.empty:
        st.info("No trades available.")
    else:
        df = trades.copy()
        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=False)

        # optional small summary if possible (doesn't replace raw view)
        pnl_col = safe_col(df, ["PnL", "pnl", "Profit"])
        if pnl_col:
            s = to_numeric_safe(df[pnl_col])
            c1, c2, c3 = st.columns(3)
            c1.metric("Trades", str(len(df)))
            c2.metric("Win rate", f"{(s.gt(0).mean()*100):.1f}%" if s.notna().any() else "—")
            c3.metric("Avg PnL", f"{s.mean():.2f}" if s.notna().any() else "—")

        st.dataframe(df.head(max_rows), width="stretch")
