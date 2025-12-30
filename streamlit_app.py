# streamlit_app.py
# CLEAN 2-TAB VERSION (C ONLY)
# Tab 1: Signals (Bucket C unified / Option A) + click ticker -> chart updates below
# Tab 2: Backtest Results (Bucket C overview)
#
# ✅ Strategy math unchanged (same A/B feature + scoring blocks)
# ✅ Monthly rebalance schedule remains month-end
# ✅ Signals table shows Consistency score

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time
import yfinance as yf

@st.cache_data(show_spinner=False, ttl=60*60*24*7)
def fetch_yahoo_metadata(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []

    for t in tickers:
        name = sector = industry = None
        try:
            tk = yf.Ticker(t)

            # 1) Try fast_info (very reliable for name)
            fi = getattr(tk, "fast_info", {}) or {}
            name = fi.get("shortName") or fi.get("longName")

            # 2) Try full info ONLY if needed
            if sector is None or industry is None:
                info = tk.get_info() or {}
                name = name or info.get("shortName") or info.get("longName")
                sector = info.get("sector")
                industry = info.get("industry")

        except Exception:
            pass

        rows.append({
            "Ticker": t,
            "Name": name,
            "Sector": sector,
            "Industry": industry,
        })

    return pd.DataFrame(rows)



st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")


# ============================================================
# 1) DATA LOADING
# ============================================================

def load_price_data_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' found in parquet.")

    keep = ["Ticker", "Date", "Price", "Index"] + ohlc_cols
    missing = [c for c in ["Ticker", "Date", "Price", "Index"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in price parquet: {missing}")

    df = df[keep].drop_duplicates(subset=["Ticker", "Date"], keep="last")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_index_returns_parquet(path: str) -> pd.DataFrame:
    idx = pd.read_parquet(path)
    idx.columns = idx.columns.str.lower().str.replace(" ", "_")
    idx["date"] = pd.to_datetime(idx["date"])

    if "index_name" in idx.columns:
        idx = idx.rename(columns={"index_name": "index"})

    required = {"date", "close", "index"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(f"Missing required columns in index parquet: {sorted(missing)}")

    frames = []
    for _, g in idx.groupby("index"):
        g = g.sort_values("date").copy()
        close = g["close"]

        g["idx_ret_1d"] = close.pct_change()
        g["idx_ret_20d"] = close.pct_change(20)
        g["idx_ret_60d"] = close.pct_change(60)
        g["idx_uptrend"] = (g["idx_ret_60d"] > 0).astype(int)

        frames.append(g[["date", "index", "idx_ret_1d", "idx_ret_20d", "idx_ret_60d", "idx_uptrend"]])

    return pd.concat(frames, ignore_index=True)


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    return df[df["Index"] == index_name].reset_index(drop=True)


# ============================================================
# 2) RETURN DEFINITIONS
# ============================================================

def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def compute_index_momentum(idx: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    idx = idx.copy().sort_values(["index", "date"])
    idx["idx_1d"] = 1.0 + idx["idx_ret_1d"]

    for w in windows:
        idx[f"idx_{w}D"] = (
            idx.groupby("index")["idx_1d"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )
    return idx


# ============================================================
# 3) MOMENTUM FEATURE ENGINE
# ============================================================

def calculate_momentum_features(df: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    df = df.copy()

    for w in windows:
        r = f"{w}D Return"
        z = f"{w}D zscore"
        dz = f"{w}D zscore change"

        df[r] = (
            df.groupby("Ticker")["1D Return"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )

        mean = df.groupby("Date")[r].transform("mean")
        std = df.groupby("Date")[r].transform("std").replace(0, np.nan)
        df[z] = ((df[r] - mean) / std)

        df[dz] = (
            df.groupby("Ticker")[z]
            .diff()
            .ewm(span=w, adjust=False)
            .mean()
        )

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)
    return df


def add_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Momentum_Fast"] = (0.6 * df["5D zscore"] + 0.4 * df["10D zscore"])
    df["Momentum_Mid"]  = (0.5 * df["30D zscore"] + 0.5 * df["45D zscore"])
    df["Momentum_Slow"] = (0.5 * df["60D zscore"] + 0.5 * df["90D zscore"])

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )
    return df.fillna(0.0)


def add_regime_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Accel_Fast"] = df.groupby("Ticker")["Momentum_Fast"].diff()
    df["Accel_Mid"]  = df.groupby("Ticker")["Momentum_Mid"].diff()
    df["Accel_Slow"] = df.groupby("Ticker")["Momentum_Slow"].diff()

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    df["Accel_Fast_z"] = df.groupby("Date")["Accel_Fast"].transform(zscore_safe)
    df["Accel_Mid_z"]  = df.groupby("Date")["Accel_Mid"].transform(zscore_safe)
    df["Accel_Slow_z"] = df.groupby("Date")["Accel_Slow"].transform(zscore_safe)

    df["Acceleration Score"] = (
        0.5 * df["Accel_Fast_z"] +
        0.3 * df["Accel_Mid_z"] +
        0.2 * df["Accel_Slow_z"]
    )
    return df.fillna(0.0)


def add_regime_residual_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Residual_Momentum"] = (
        df["Momentum_Fast"] -
        df.groupby("Ticker")["Momentum_Slow"].transform("mean")
    )

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    df["Residual_Momentum_z"] = df.groupby("Date")["Residual_Momentum"].transform(zscore_safe)
    return df.fillna(0.0)


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Early_Fast"] = (0.6 * df["Accel_Fast_z"] + 0.4 * df["Momentum_Fast"])
    df["Early_Mid"]  = (0.5 * df["Accel_Mid_z"] + 0.5 * df["Momentum_Mid"])
    df["Early_Slow"] = (0.5 * df["Accel_Slow_z"] + 0.5 * df["Momentum_Slow"])

    df["Early Momentum Score"] = (
        0.5 * df["Early_Slow"] +
        0.3 * df["Early_Mid"] +
        0.2 * df["Early_Fast"]
    )
    return df.fillna(0.0)


# ============================================================
# 3B) RELATIVE REGIME MOMENTUM (Bucket B core)
# ============================================================

def add_relative_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses df's existing stock returns features (5D/10D/30D/45D/60D/90D Return)
    and index momentum columns (idx_5D..idx_90D) to build relative regimes,
    then sets Momentum_* and Momentum Score from relative z-scores.
    """
    df = df.copy()

    # Relative regime components
    df["Rel_Slow"] = (
        0.5 * df["60D Return"] + 0.5 * df["90D Return"]
    ) - (
        0.5 * df["idx_60D"] + 0.5 * df["idx_90D"]
    )

    df["Rel_Mid"] = (
        0.5 * df["30D Return"] + 0.5 * df["45D Return"]
    ) - (
        0.5 * df["idx_30D"] + 0.5 * df["idx_45D"]
    )

    df["Rel_Fast"] = (
        0.6 * df["5D Return"] + 0.4 * df["10D Return"]
    ) - (
        0.6 * df["idx_5D"] + 0.4 * df["idx_10D"]
    )

    # Cross-sectional z-scores AFTER benchmark subtraction
    for col in ["Rel_Slow", "Rel_Mid", "Rel_Fast"]:
        mean = df.groupby("Date")[col].transform("mean")
        std  = df.groupby("Date")[col].transform("std").replace(0, np.nan)
        df[col + "_z"] = ((df[col] - mean) / std).fillna(0.0)

    # Define regime scaffolding using RELATIVE z-scores
    df["Momentum_Slow"] = df["Rel_Slow_z"]
    df["Momentum_Mid"]  = df["Rel_Mid_z"]
    df["Momentum_Fast"] = df["Rel_Fast_z"]

    # Momentum Score built from relative regimes
    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )

    return df.fillna(0.0)


# ============================================================
# 4) DAILY LISTS + FINAL (PERSISTENCE) SELECTION
# ============================================================

def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    records = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d].sort_values("Momentum Score", ascending=False).head(top_n).copy()
        if snap.empty:
            continue

        # rank 1..top_n (1 = best)
        snap["Rank"] = np.arange(1, len(snap) + 1)

        for _, r in snap.iterrows():
            records.append({
                "Date": d,
                "Ticker": r["Ticker"],
                "Rank": int(r["Rank"]),
                "Momentum Score": float(r["Momentum Score"]),
                "Early Momentum Score": float(r["Early Momentum Score"]),
            })
    return pd.DataFrame(records)



def final_selection_from_daily(
    daily_df: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    as_of_date=None,
    top_n: int = 10
) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    if as_of_date is None:
        as_of_date = daily_df["Date"].max()

    dates = sorted(
        daily_df.loc[daily_df["Date"] <= as_of_date, "Date"].unique(),
        reverse=True
    )[:lookback_days]

    if not dates:
        return pd.DataFrame()

    window = daily_df[daily_df["Date"].isin(dates)]
    if window.empty:
        return pd.DataFrame()

    agg = (
        window.groupby("Ticker")
        .agg(
            Momentum_Score=("Momentum Score", "mean"),
            Early_Momentum_Score=("Early Momentum Score", "mean"),
            Appearances=("Date", "count"),
            Rank_Mean=("Rank", "mean"),
            Rank_Std=("Rank", "std"),
        )
        .reset_index()
    )
    
    # std can be NaN when only 1 appearance; treat that as perfectly stable (std=0)
    agg["Rank_Std"] = agg["Rank_Std"].fillna(0.0)


    agg["Consistency"] = agg["Appearances"] / len(dates)
    agg["Weighted_Score"] = (
        w_momentum * agg["Momentum_Score"] +
        w_early * agg["Early_Momentum_Score"] +
        w_consistency * agg["Consistency"]
    )

    # Consistency already 0..1
    agg["ConsistencyScore"] = agg["Consistency"] * 100.0
    
    # Rank stability from Rank_Std (lower is better)
    # Normalize: std >= (top_n/2) is considered "very unstable" => score goes to 0
    max_std = max(1.0, top_n / 2.0)
    agg["RankStabilityScore"] = (1.0 - (agg["Rank_Std"] / max_std)).clip(0.0, 1.0) * 100.0
    
    # 50/50 confidence
    agg["Signal_Confidence"] = 0.5 * agg["ConsistencyScore"] + 0.5 * agg["RankStabilityScore"]


    return (
        agg.sort_values("Weighted_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 5) BUCKET C SIGNALS (UNIFIED TARGET) + BACKTEST
# ============================================================

def monthly_rebalance_dates(df_prices: pd.DataFrame) -> list:
    dates = pd.to_datetime(df_prices["Date"]).dropna().sort_values().unique()
    if len(dates) == 0:
        return []
    tmp = pd.DataFrame({"Date": dates})
    tmp["Month"] = tmp["Date"].dt.to_period("M")
    return tmp.groupby("Month")["Date"].max().tolist()


def get_last_price(price_table: pd.DataFrame, ticker: str, date) -> float | None:
    try:
        s = price_table.loc[:date, ticker].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def build_bucket_c_signals(
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    as_of_date,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital,
    weight_A=0.20,
    weight_B=0.80,
) -> pd.DataFrame:
    """
    Bucket C = combine final selection from A and B, allocate dollars by weights,
    and aggregate diagnostics per ticker (showing Consistency etc.).
    """
    selA = final_selection_from_daily(
        dailyA,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    ).copy()

    selB = final_selection_from_daily(
        dailyB,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    ).copy()

    frames = []

    if not selA.empty:
        selA["Bucket"] = "A"
        selA["Position_Size"] = (total_capital * weight_A) / len(selA)
        frames.append(selA)

    if not selB.empty:
        selB["Bucket"] = "B"
        selB["Position_Size"] = (total_capital * weight_B) / len(selB)
        frames.append(selB)

    if not frames:
        return pd.DataFrame()

    combo = pd.concat(frames, ignore_index=True)

    # Aggregate per ticker
    out = (
        combo.groupby("Ticker", as_index=False)
        .agg(
            Position_Size=("Position_Size", "sum"),
            Weighted_Score=("Weighted_Score", "max"),
            Momentum_Score=("Momentum_Score", "max"),
            Early_Momentum_Score=("Early_Momentum_Score", "max"),
            Consistency=("Consistency", "max"),
    
            # ✅ add these
            Rank_Std=("Rank_Std", "min"),                 # smaller = more stable
            RankStabilityScore=("RankStabilityScore", "max"),
            Signal_Confidence=("Signal_Confidence", "max"),
    
            Bucket_Source=("Bucket", lambda x: "+".join(sorted(set(x))))
        )
        .sort_values(["Position_Size", "Signal_Confidence"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Normalize source formatting
    out["Bucket_Source"] = out["Bucket_Source"].replace({"A+B": "A+B", "A": "A", "B": "B"})
    return out


def simulate_unified_portfolio(
    df_prices: pd.DataFrame,
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    top_n: int = 10,
    total_capital: float = 100_000.0,
    weight_A: float = 0.20,
    weight_B: float = 0.80,
):
    rebalance_dates = monthly_rebalance_dates(df_prices)

    cash = total_capital
    portfolio = {}   # ticker -> shares
    cost_basis = {}  # ticker -> dollars invested

    history = []
    trades = []

    for d in rebalance_dates:
        target_df = build_bucket_c_signals(
            dailyA=dailyA,
            dailyB=dailyB,
            as_of_date=d,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=total_capital,
            weight_A=weight_A,
            weight_B=weight_B
        )
        if target_df.empty:
            continue

        target_sizes = dict(zip(target_df["Ticker"], target_df["Position_Size"]))
        target = set(target_sizes.keys())
        held = set(portfolio.keys())

        # SELL
        for t in list(held - target):
            px = get_last_price(price_table, t, d)
            if px is None or px <= 0:
                continue
            shares = portfolio[t]
            proceeds = shares * px
            cash += proceeds
            pnl = proceeds - cost_basis.get(t, 0.0)
            trades.append({"Date": d, "Ticker": t, "Action": "Sell", "Price": px, "PnL": pnl})
            portfolio.pop(t, None)
            cost_basis.pop(t, None)

        # BUY / RESIZE
        for t, target_dollars in target_sizes.items():
            px = get_last_price(price_table, t, d)
            if px is None or px <= 0:
                continue

            current_shares = portfolio.get(t, 0.0)
            current_value = current_shares * px
            delta_value = target_dollars - current_value

            if delta_value > 0:
                if delta_value > cash:
                    continue
                delta_shares = delta_value / px
                portfolio[t] = current_shares + delta_shares
                cost_basis[t] = cost_basis.get(t, 0.0) + delta_value
                cash -= delta_value
                trades.append({"Date": d, "Ticker": t, "Action": "Buy", "Price": px, "PnL": 0.0})

            elif delta_value < 0:
                trim_value = -delta_value
                trim_shares = trim_value / px
                portfolio[t] = current_shares - trim_shares
                cost_basis[t] = cost_basis.get(t, 0.0) - trim_value
                cash += trim_value
                trades.append({"Date": d, "Ticker": t, "Action": "Resize", "Price": px, "PnL": 0.0})

        holdings_value = sum(
            get_last_price(price_table, t, d) * s
            for t, s in portfolio.items()
            if get_last_price(price_table, t, d) is not None
        )
        nav = cash + holdings_value
        history.append({"Date": d, "Portfolio Value": nav})

    return pd.DataFrame(history), pd.DataFrame(trades)


# ============================================================
# 6) PERFORMANCE + PLOTS
# ============================================================

def compute_performance_stats(history_df: pd.DataFrame) -> dict:
    df = history_df.sort_values("Date").copy()
    if df.empty or len(df) < 2:
        return {"Message": "Not enough history"}

    start = float(df["Portfolio Value"].iloc[0])
    end = float(df["Portfolio Value"].iloc[-1])
    if start <= 0:
        return {"Message": "Invalid start value"}

    total_return = (end / start - 1) * 100
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25

    cagr = ((end / start) ** (1 / years) - 1) * 100 if (end > 0 and years > 0) else np.nan

    ret = df["Portfolio Value"].pct_change().dropna()

    std = ret.std()
    sharpe = np.sqrt(252) * ret.mean() / std if (std is not None and std > 0) else np.nan

    downside = ret[ret < 0]
    dstd = downside.std()
    sortino = np.sqrt(252) * ret.mean() / dstd if (dstd is not None and dstd > 0) else np.nan

    rolling_max = df["Portfolio Value"].cummax()
    drawdown = (df["Portfolio Value"] - rolling_max) / rolling_max

    return {
        "Total Return (%)": float(total_return),
        "CAGR (%)": float(cagr) if pd.notna(cagr) else np.nan,
        "Sharpe Ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Sortino Ratio": float(sortino) if pd.notna(sortino) else np.nan,
        "Max Drawdown (%)": float(drawdown.min() * 100),
    }


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {"Message": "No trades"}
    sells = trades_df[trades_df["Action"] == "Sell"].copy()
    if sells.empty:
        return {"Message": "No closed trades"}

    wins = sells[sells["PnL"] > 0]
    losses = sells[sells["PnL"] < 0]

    total_win = wins["PnL"].sum()
    total_loss = abs(losses["PnL"].sum())

    return {
        "Number of Trades": int(len(sells)),
        "Win Rate (%)": float(len(wins) / len(sells) * 100),
        "Average Win ($)": float(wins["PnL"].mean()) if not wins.empty else 0.0,
        "Average Loss ($)": float(losses["PnL"].mean()) if not losses.empty else 0.0,
        "Profit Factor": float(total_win / total_loss) if total_loss > 0 else np.nan,
    }


def plot_equity_and_drawdown(history_df: pd.DataFrame, title: str):
    if history_df is None or history_df.empty:
        st.info("No equity history to display.")
        return

    df = history_df.sort_values("Date").copy()
    df["Rolling_Max"] = df["Portfolio Value"].cummax()
    df["Drawdown"] = (df["Portfolio Value"] - df["Rolling_Max"]) / df["Rolling_Max"]

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=df["Date"], y=df["Portfolio Value"], mode="lines", name="Portfolio Value"))
    fig_eq.update_layout(
        title=f"{title} — Equity Curve",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Portfolio Value", tickformat=",.0f"),
        template="plotly_white",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df["Date"], y=df["Drawdown"], mode="lines", name="Drawdown"))
    fig_dd.update_layout(
        title=f"{title} — Drawdown",
        height=220,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Drawdown", tickformat=".0%"),
        template="plotly_white",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


def pick_single_row(df: pd.DataFrame, key: str, label_col: str = "Ticker", column_config=None):
    if df is None or df.empty or label_col not in df.columns:
        st.dataframe(df, use_container_width=True)
        return None

    view = df.reset_index(drop=True)
    event = st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
    )

    rows = getattr(event, "selection", {}).get("rows", []) if hasattr(event, "selection") else []
    if not rows:
        return None

    val = str(view.iloc[rows[0]][label_col])
    if val == "TOTAL":
        return None
    return val


def _trade_markers_for_ticker(trades_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["Date", "Action", "Price"])
    x = trades_df[trades_df["Ticker"] == ticker].copy()
    if x.empty:
        return x
    x["Date"] = pd.to_datetime(x["Date"])
    return x.sort_values("Date")


def plot_ticker_price_with_trades_and_momentum(
    base_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    score_df: pd.DataFrame,
    ticker: str,
    lookback_days: int = 500
):
    if not ticker:
        return

    px = base_df[base_df["Ticker"] == ticker].copy()
    if px.empty:
        st.warning(f"No price history for {ticker}")
        return

    px = px.sort_values("Date")
    if lookback_days and len(px) > lookback_days:
        px = px.tail(lookback_days)

    sc = score_df[score_df["Ticker"] == ticker].copy() if score_df is not None else pd.DataFrame()
    if not sc.empty:
        sc["Date"] = pd.to_datetime(sc["Date"])
        sc = sc.sort_values("Date")
        if lookback_days and len(sc) > lookback_days:
            sc = sc.tail(lookback_days)

    tdf = _trade_markers_for_ticker(trades_df, ticker)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
    )

    has_ohlc = all(c in px.columns for c in ["Open", "High", "Low", "Close"])

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=px["Date"], open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"],
                name="Price",
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(go.Scatter(x=px["Date"], y=px["Price"], mode="lines", name="Price"), row=1, col=1)

    if not tdf.empty:
        if "Price" not in tdf.columns or tdf["Price"].isna().all():
            s = px.set_index("Date")["Price"]
            tdf["Price"] = [float(s.loc[:d].iloc[-1]) if len(s.loc[:d]) else np.nan for d in tdf["Date"]]

        def add_marker(action, symbol):
            sub = tdf[tdf["Action"].str.upper() == action].copy()
            if sub.empty:
                return
            fig.add_trace(
                go.Scatter(
                    x=sub["Date"], y=sub["Price"],
                    mode="markers+text",
                    text=[action] * len(sub),
                    textposition="top center",
                    name=action,
                    marker=dict(size=11, symbol=symbol),
                ),
                row=1, col=1
            )

        add_marker("BUY", "triangle-up")
        add_marker("SELL", "triangle-down")
        add_marker("RESIZE", "diamond")

    if not sc.empty:
        fig.add_trace(go.Scatter(x=sc["Date"], y=sc["Momentum Score"], mode="lines", name="Momentum Score"), row=2, col=1)
        fig.add_trace(go.Scatter(x=sc["Date"], y=sc["Early Momentum Score"], mode="lines", name="Early Momentum"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Momentum Score"), row=2, col=1)

    fig.update_layout(
        height=720,
        margin=dict(l=30, r=20, t=40, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        title=f"{ticker} — Price + Trades + Momentum"
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PIPELINE
# ============================================================

@st.cache_data(show_spinner=False)
def run_full_pipeline():
    parquet_path = "artifacts/index_constituents_5yr.parquet"
    index_path = "artifacts/index_returns_5y.parquet"

    DAILY_TOP_N = 10
    FINAL_TOP_N = 10
    LOOKBACK_DAYS = 10
    WINDOWS = (5, 10, 30, 45, 60, 90)

    W_MOM = 0.50
    W_EARLY = 0.30
    W_CONS = 0.20

    base = load_price_data_parquet(parquet_path)
    base = filter_by_index(base, "SP500")
    idx = load_index_returns_parquet(index_path)

    # ---------------- Bucket A (absolute) ----------------
    dfA = calculate_momentum_features(add_absolute_returns(base), windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)

    priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    # ---------------- Bucket B (relative) ----------------
    dfB = base.copy()
    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
        validate="many_to_one"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

    # Use same return feature factory (absolute 1D return) to build 5/10/30... stock return features
    dfB = calculate_momentum_features(add_absolute_returns(dfB), windows=WINDOWS)

    # Merge index momentum features idx_5D..idx_90D
    idx_mom = compute_index_momentum(idx, windows=WINDOWS)
    dfB = dfB.merge(
        idx_mom[["date", "index", "idx_5D", "idx_10D", "idx_30D", "idx_45D", "idx_60D", "idx_90D"]],
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left"
    ).drop(columns=["date", "index"], errors="ignore")

    # Relative regime scoring (sets Momentum_* and Momentum Score)
    dfB = add_relative_regime_momentum_score(dfB)

    # Filters (as in your original long version)
    dfB = dfB[dfB["Momentum_Slow"] > 1].copy()
    dfB = dfB[dfB["Momentum_Mid"] > 0.5].copy()
    dfB = dfB[dfB["Momentum_Fast"] > 1].copy()

    # Early momentum scaffolding (same functions, now applied to relative regimes)
    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

    # ---------------- Bucket C (unified) backtest ----------------
    histU, tradesU = simulate_unified_portfolio(
        df_prices=base,
        price_table=priceA,          # same as your existing version
        dailyA=dailyA,
        dailyB=dailyB,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000.0,
        weight_A=0.20,
        weight_B=0.80
    )

    statsU = compute_performance_stats(histU)
    trade_statsU = compute_trade_stats(tradesU)

    scoreA = dfA[["Date", "Ticker", "Momentum Score", "Early Momentum Score"]].copy()
    scoreB = dfB[["Date", "Ticker", "Momentum Score", "Early Momentum Score"]].copy()

    return {
        "params": {
            "DAILY_TOP_N": DAILY_TOP_N,
            "FINAL_TOP_N": FINAL_TOP_N,
            "LOOKBACK_DAYS": LOOKBACK_DAYS,
            "WINDOWS": WINDOWS,
            "W_MOM": W_MOM,
            "W_EARLY": W_EARLY,
            "W_CONS": W_CONS,
        },
        "BucketC": {"history": histU, "trades": tradesU, "stats": statsU, "trade_stats": trade_statsU},
        "internals": {
            "base": base,
            "dailyA": dailyA,
            "dailyB": dailyB,
            "scoreA": scoreA,
            "scoreB": scoreB,
            "priceA": priceA
        }
    }


def build_bucket_c_signal_preview(
    dailyA,
    dailyB,
    as_of_date,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital,
    weight_A=0.20,
    weight_B=0.80,
):
    tgtC = build_bucket_c_signals(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        top_n=top_n,
        total_capital=total_capital,
        weight_A=weight_A,
        weight_B=weight_B,
    )

    if tgtC is None or tgtC.empty:
        return pd.DataFrame()

    out = tgtC.copy()
    out = out.rename(columns={"Position_Size": "Target_$"})
    total = float(out["Target_$"].sum()) if "Target_$" in out.columns else 0.0
    out["Weight_%"] = (out["Target_$"] / total * 100.0) if total > 0 else np.nan

    # Order / keep columns for the Signals tab (includes Consistency)
    cols = [
        "Ticker", "Name",
        "Target_$", "Weight_%",
        "Signal_Confidence",
        "Weighted_Score", "Momentum_Score", "Early_Momentum_Score",
        "Sector", "Industry",
    ]


    out["Consistency"] = out["Consistency"] * 100.0   # convert to %


    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(["Singal_Confidence", "Weighted_Score"], ascending=[False, False]).reset_index(drop=True)

    return out


# ============================================================
# UI (2 TABS)
# ============================================================

st.title("📈 Momentum Portfolio")

with st.spinner("Running full pipeline from artifacts…"):
    out = run_full_pipeline()

params = out["params"]
bucketC = out["BucketC"]
internals = out["internals"]

base = internals["base"]
dailyA = internals["dailyA"]
dailyB = internals["dailyB"]
scoreA = internals["scoreA"]
scoreB = internals["scoreB"]

hist = bucketC["history"]
trades = bucketC["trades"]
stats = bucketC["stats"]
trade_stats = bucketC["trade_stats"]

common_dates = sorted(set(dailyA["Date"]).intersection(set(dailyB["Date"])))
signal_date = common_dates[-1] if common_dates else None

if signal_date is None:
    st.error("No overlapping signal dates between Bucket A and Bucket B (cannot build Bucket C signals).")
    st.stop()

tab_signals, tab_backtest = st.tabs(["Signals", "Backtest Results"])

with tab_signals:
    st.markdown("### Next Rebalance Preview (Option A)")

    st.caption(
        f"Signals as of **{signal_date.date()}** · "
        f"Lookback: last **{params['LOOKBACK_DAYS']}** trading days · "
        f"Rebalance: Month-end"
    )

    # Optional: choose which score series to show in the chart (does not affect signals)
    chart_source = st.radio(
        "Chart momentum source (visual only)",
        options=["Absolute (A)", "Relative (B)"],
        horizontal=True,
        index=0
    )
    score_for_chart = scoreA if chart_source.startswith("Absolute") else scoreB

    preview = build_bucket_c_signal_preview(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=signal_date,
        lookback_days=params["LOOKBACK_DAYS"],
        w_momentum=params["W_MOM"],
        w_early=params["W_EARLY"],
        w_consistency=params["W_CONS"],
        top_n=params["FINAL_TOP_N"],
        total_capital=100_000.0,
        weight_A=0.20,
        weight_B=0.80
    )

    tickers = tuple(preview["Ticker"].astype(str).unique())
    meta = fetch_yahoo_metadata(tickers)
    preview = preview.merge(meta, on="Ticker", how="left")

    st.markdown("### Sector exposure")

    sector_df = preview.copy()
    sector_df["Sector"] = sector_df["Sector"].fillna("Unknown")
    
    # Use Weight_% if present; fallback to Target_$ if you prefer
    if "Weight_%" in sector_df.columns and sector_df["Weight_%"].notna().any():
        expo = sector_df.groupby("Sector", as_index=False)["Weight_%"].sum()
        values_col = "Weight_%"
        title = "Sector Exposure (by weight %)"
    else:
        expo = sector_df.groupby("Sector", as_index=False)["Target_$"].sum()
        values_col = "Target_$"
        title = "Sector Exposure (by $)"
    
    fig_sector = go.Figure(
        data=[go.Pie(labels=expo["Sector"], values=expo[values_col], hole=0.45)]
    )
    fig_sector.update_layout(
        title=title,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white"
    )
    st.plotly_chart(fig_sector, use_container_width=True)



    if preview.empty:
        st.info("No signals available.")
    else:
        st.markdown("#### Target portfolio (click a row to update chart)")
        battery_cfg = {
            "Signal_Confidence": st.column_config.ProgressColumn(
                "Signal Strength",
                min_value=0,
                max_value=100,
                format="%d%%",
            )
        }

        selected = pick_single_row(
            preview,
            key="signal_preview",
            label_col="Ticker",
            column_config=battery_cfg
        )

        if selected:
            st.markdown("#### Selected ticker chart")
            plot_ticker_price_with_trades_and_momentum(
                base_df=base,
                trades_df=trades,
                score_df=score_for_chart,
                ticker=selected,
                lookback_days=500
            )

with tab_backtest:
    st.markdown("### Backtest Results (Bucket C)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", f"{stats.get('Total Return (%)', np.nan):.2f}")
    c2.metric("CAGR (%)", f"{stats.get('CAGR (%)', np.nan):.2f}")
    c3.metric("Sharpe", f"{stats.get('Sharpe Ratio', np.nan):.2f}")
    c4.metric("Max Drawdown (%)", f"{stats.get('Max Drawdown (%)', np.nan):.2f}")

    st.markdown("### Equity & Drawdown")
    plot_equity_and_drawdown(hist, title="Momentum Portfolio")

    st.markdown("### Trade Statistics")
    st.dataframe(
        pd.DataFrame({"Metric": list(trade_stats.keys()), "Value": list(trade_stats.values())}),
        use_container_width=True
    )

    with st.expander("Show full backtest trades"):
        if trades is None or trades.empty:
            st.write("No trades.")
        else:
            x = trades.copy()
            x["Date"] = pd.to_datetime(x["Date"])
            st.dataframe(x.sort_values("Date"), use_container_width=True)
