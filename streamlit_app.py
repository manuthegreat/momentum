# streamlit_app.py
# FULL, 1:1 PIPELINE MIRROR (same functions, same order, same math)
# Only differences (additive + schedule-only):
#   1) Rebalance schedule is MONTHLY (first trading day of each month) instead of every N trading days
#      -> implemented by using monthly_rebalance_dates() in the two simulators.
#   2) Current Portfolio tab marks-to-market from last monthly rebalance date
#      to the latest available price date (so P&L is not forced to 0).
#   3) Click-a-row (radio-like) selection on Current Portfolio + Monthly Rebalance tables
#      -> plots historic price and highlights BUY/SELL dates from the Bucket C trades blotter.
# All momentum / scoring / filters are untouched.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")

# ============================================================
# 1) DATA LOADING
# ============================================================

def load_price_data_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' found in parquet.")

    keep = ["Ticker", "Date", "Price", "Index"]
    missing = [c for c in keep if c not in df.columns]
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
# 2) RETURN DEFINITIONS (ONLY DIFFERENCE BETWEEN A & B)
# ============================================================

def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def add_relative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative daily growth factor = (1 + stock_ret) / (1 + index_ret)
    This compounds correctly over time when you later do rolling product.
    """
    df = df.copy()
    if "idx_ret_1d" not in df.columns:
        raise ValueError("idx_ret_1d missing. Merge index returns before calling add_relative_returns().")

    stock_ret = df.groupby("Ticker")["Price"].pct_change()
    idx_ret = df["idx_ret_1d"]

    denom = (1.0 + idx_ret).replace(0.0, np.nan)  # guard divide-by-zero
    df["1D Return"] = (1.0 + stock_ret) / denom
    return df


def compute_index_momentum(idx: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    idx = idx.copy()
    idx = idx.sort_values(["index", "date"])

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
# 3) MOMENTUM FEATURE ENGINE (SHARED)
# ============================================================

def calculate_momentum_features(
    df: pd.DataFrame,
    windows=(5, 10, 30, 45, 60, 90)
) -> pd.DataFrame:
    """
    Core feature factory:
    - N-day compounded returns
    - Cross-sectional z-scores
    - Smoothed z-score changes
    """
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

    df["Momentum_Fast"] = (
        0.6 * df["5D zscore"] +
        0.4 * df["10D zscore"]
    )

    df["Momentum_Mid"] = (
        0.5 * df["30D zscore"] +
        0.5 * df["45D zscore"]
    )

    df["Momentum_Slow"] = (
        0.5 * df["60D zscore"] +
        0.5 * df["90D zscore"]
    )

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )

    return df.fillna(0.0)


def add_regime_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acceleration = change in regime momentum, not raw z-score noise.
    """
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
    """
    Residual = fast regime vs stock’s own slow baseline.
    """
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

    df["Residual_Momentum_z"] = (
        df.groupby("Date")["Residual_Momentum"]
        .transform(zscore_safe)
    )

    return df.fillna(0.0)


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Early_Fast"] = (
        0.6 * df["Accel_Fast_z"] +
        0.4 * df["Momentum_Fast"]
    )

    df["Early_Mid"] = (
        0.5 * df["Accel_Mid_z"] +
        0.5 * df["Momentum_Mid"]
    )

    df["Early_Slow"] = (
        0.5 * df["Accel_Slow_z"] +
        0.5 * df["Momentum_Slow"]
    )

    df["Early Momentum Score"] = (
        0.5 * df["Early_Slow"] +
        0.3 * df["Early_Mid"] +
        0.2 * df["Early_Fast"]
    )

    return df.fillna(0.0)


def add_relative_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Rel_Slow"] = (
        0.5 * df["60D Return"] +
        0.5 * df["90D Return"]
    ) - (
        0.5 * df["idx_60D"] +
        0.5 * df["idx_90D"]
    )

    df["Rel_Mid"] = (
        0.5 * df["30D Return"] +
        0.5 * df["45D Return"]
    ) - (
        0.5 * df["idx_30D"] +
        0.5 * df["idx_45D"]
    )

    df["Rel_Fast"] = (
        0.6 * df["5D Return"] +
        0.4 * df["10D Return"]
    ) - (
        0.6 * df["idx_5D"] +
        0.4 * df["idx_10D"]
    )

    for col in ["Rel_Slow", "Rel_Mid", "Rel_Fast"]:
        mean = df.groupby("Date")[col].transform("mean")
        std  = df.groupby("Date")[col].transform("std").replace(0, np.nan)
        df[col + "_z"] = ((df[col] - mean) / std).fillna(0.0)

    df["Momentum_Slow"] = df["Rel_Slow_z"]
    df["Momentum_Mid"]  = df["Rel_Mid_z"]
    df["Momentum_Fast"] = df["Rel_Fast_z"]

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )

    return df.fillna(0.0)


# ============================================================
# 4) DAILY LISTS (BASE BUY-LIST PER DAY)
# ============================================================

def get_daily_momentum_topn(df_date: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df_date.empty:
        return pd.DataFrame()
    return df_date.sort_values("Momentum Score", ascending=False).head(top_n)


def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    For each date, store the base daily buy list (top_n by Momentum Score).
    This is the input to the *persistence* / *consistency* scoring layer.
    """
    records = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d]
        picks = get_daily_momentum_topn(snap, top_n=top_n)
        if picks.empty:
            continue
        for _, r in picks.iterrows():
            records.append({
                "Date": d,
                "Ticker": r["Ticker"],
                "Momentum Score": float(r["Momentum Score"]),
                "Early Momentum Score": float(r["Early Momentum Score"]),
            })
    return pd.DataFrame(records)


# ============================================================
# 4B) PERSISTENCE / CONSISTENCY SCORING (RESTORED)
# ============================================================

def final_selection_from_daily(
    daily_df: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    as_of_date=None,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Restore your original logic:
      - look back N recent dates
      - compute average Momentum/Early scores over those dates
      - compute Consistency = appearances / N
      - Weighted_Score = blend
    """
    if daily_df.empty:
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
            Appearances=("Date", "count")
        )
        .reset_index()
    )

    agg["Consistency"] = agg["Appearances"] / len(dates)
    agg["Weighted_Score"] = (
        w_momentum * agg["Momentum_Score"] +
        w_early * agg["Early_Momentum_Score"] +
        w_consistency * agg["Consistency"]
    )

    return (
        agg.sort_values("Weighted_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 5) BACKTEST ENGINE HELPERS
# ============================================================

def get_last_price(price_table: pd.DataFrame, ticker: str, date) -> float | None:
    try:
        s = price_table.loc[:date, ticker].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


# ============================================================
# 6) PERFORMANCE + TRADE STATS
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

    if start <= 0 or end <= 0 or years <= 0:
        cagr = np.nan
    else:
        cagr = ((end / start) ** (1 / years) - 1) * 100

    ret = df["Portfolio Value"].pct_change().dropna()
    sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() and ret.std() > 0 else np.nan
    sortino = np.nan
    downside = ret[ret < 0]
    if downside.std() and downside.std() > 0:
        sortino = np.sqrt(252) * ret.mean() / downside.std()

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


# ============================================================
# 7) TODAY'S REBALANCE DECISION (BUY/HOLD/SELL) USING FINAL SELECTION
# ============================================================

def build_unified_target(
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
    weight_B=0.80
) -> pd.DataFrame:

    selA = final_selection_from_daily(
        dailyA,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    )

    selB = final_selection_from_daily(
        dailyB,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    )

    frames = []

    if not selA.empty:
        dollars_per_name_A = (total_capital * weight_A) / len(selA)
        tmpA = selA[["Ticker"]].copy()
        tmpA["Position_Size"] = dollars_per_name_A
        frames.append(tmpA)

    if not selB.empty:
        dollars_per_name_B = (total_capital * weight_B) / len(selB)
        tmpB = selB[["Ticker"]].copy()
        tmpB["Position_Size"] = dollars_per_name_B
        frames.append(tmpB)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    return (
        combined
        .groupby("Ticker", as_index=False)["Position_Size"]
        .sum()
    )


def get_today_trades_bucket_c(
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
    weight_B=0.80
) -> pd.DataFrame:
    """
    Final BUY / HOLD / SELL list for unified Bucket C
    """
    all_dates = sorted(set(dailyA["Date"]).intersection(set(dailyB["Date"])))
    if len(all_dates) < 2:
        return pd.DataFrame()

    today = all_dates[-1]
    prev = all_dates[-2]

    tgt_today = build_unified_target(
        dailyA, dailyB,
        as_of_date=today,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        top_n=top_n,
        total_capital=total_capital,
        weight_A=weight_A,
        weight_B=weight_B
    )

    tgt_prev = build_unified_target(
        dailyA, dailyB,
        as_of_date=prev,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        top_n=top_n,
        total_capital=total_capital,
        weight_A=weight_A,
        weight_B=weight_B
    )

    if tgt_today.empty:
        return pd.DataFrame()

    today_set = set(tgt_today["Ticker"])
    prev_set = set(tgt_prev["Ticker"]) if not tgt_prev.empty else set()

    out = tgt_today.copy()
    out["Action"] = "BUY"
    out.loc[out["Ticker"].isin(prev_set), "Action"] = "HOLD"

    sells = prev_set - today_set
    if sells:
        sell_df = pd.DataFrame({
            "Ticker": list(sells),
            "Position_Size": 0.0,
            "Action": "SELL"
        })
        out = pd.concat([out, sell_df], ignore_index=True)

    return out.sort_values(
        ["Action", "Position_Size"],
        ascending=[True, False]
    ).reset_index(drop=True)


# ============================================================
# ✅ MONTHLY REBALANCE SCHEDULE (FIRST TRADING DAY OF MONTH)
# ============================================================

def monthly_rebalance_dates(df_prices: pd.DataFrame) -> list:
    """Monthly rebalance dates = FIRST available trading day in each month."""
    dates = pd.to_datetime(df_prices["Date"]).dropna().sort_values().unique()
    if len(dates) == 0:
        return []
    tmp = pd.DataFrame({"Date": dates})
    tmp["Month"] = tmp["Date"].dt.to_period("M")
    return tmp.groupby("Month")["Date"].min().tolist()


# ============================================================
# BACKTEST ENGINES (ONLY CHANGE = REBALANCE DATES)
# ============================================================

def simulate_unified_portfolio(
    df_prices: pd.DataFrame,
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    rebalance_interval: int = 10,  # kept for signature compatibility; unused in monthly mode
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    top_n: int = 10,
    total_capital: float = 100_000.0
):
    """
    Unified portfolio:
    - Fixed total capital
    - $2,500 per bucket per name
    - Name in both A & B gets $5,000
    - Cash-constrained, no leverage

    ✅ Rebalance schedule: MONTHLY (first trading day)
    """
    rebalance_dates = monthly_rebalance_dates(df_prices)

    cash = total_capital
    portfolio = {}   # ticker -> shares
    cost_basis = {}  # ticker -> dollars invested

    history = []
    trades = []

    for d in rebalance_dates:
        target_df = build_unified_target(
            dailyA, dailyB,
            as_of_date=d,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=total_capital,
            weight_A=0.20,
            weight_B=0.80
        )
        if target_df.empty:
            continue

        target_sizes = dict(zip(target_df["Ticker"], target_df["Position_Size"]))
        target = set(target_sizes.keys())
        held = set(portfolio.keys())

        # SELL
        for t in held - target:
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


def simulate_single_bucket_as_unified(
    df_prices,
    price_table,
    daily_df,
    rebalance_interval,  # kept for signature compatibility; unused in monthly mode
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital=100_000.0,
    dollars_per_name=5_000.0
):
    """
    Runs Bucket A or B using the SAME capital mechanics as Bucket C.
    ✅ Rebalance schedule: MONTHLY (first trading day)
    """
    def build_single_target(as_of_date):
        sel = final_selection_from_daily(
            daily_df,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            as_of_date=as_of_date,
            top_n=top_n
        )
        if sel.empty:
            return pd.DataFrame()
        sel = sel.copy()
        sel["Position_Size"] = dollars_per_name
        return sel[["Ticker", "Position_Size"]]

    rebalance_dates = monthly_rebalance_dates(df_prices)

    cash = total_capital
    portfolio = {}
    cost_basis = {}

    history, trades = [], []

    for d in rebalance_dates:
        tgt = build_single_target(d)
        if tgt.empty:
            continue

        target_sizes = dict(zip(tgt["Ticker"], tgt["Position_Size"]))
        target = set(target_sizes)
        held = set(portfolio)

        # SELL
        for t in held - target:
            px = get_last_price(price_table, t, d)
            if px:
                shares = portfolio[t]
                proceeds = shares * px
                cash += proceeds
                pnl = proceeds - cost_basis[t]
                trades.append({"Date": d, "Ticker": t, "Action": "Sell", "Price": px, "PnL": pnl})

            portfolio.pop(t, None)
            cost_basis.pop(t, None)

        # BUY / RESIZE
        for t, dollars in target_sizes.items():
            px = get_last_price(price_table, t, d)
            if not px:
                continue

            current_shares = portfolio.get(t, 0.0)
            current_value = current_shares * px
            delta = dollars - current_value

            if delta > 0 and delta <= cash:
                shares = delta / px
                portfolio[t] = current_shares + shares
                cost_basis[t] = cost_basis.get(t, 0.0) + delta
                cash -= delta
                trades.append({"Date": d, "Ticker": t, "Action": "Buy", "Price": px, "PnL": 0.0})

            elif delta < 0:
                trim = -delta / px
                portfolio[t] = current_shares - trim
                cost_basis[t] -= -delta
                cash += -delta
                trades.append({"Date": d, "Ticker": t, "Action": "Resize", "Price": px, "PnL": 0.0})

        nav = cash + sum(
            get_last_price(price_table, t, d) * s
            for t, s in portfolio.items()
            if get_last_price(price_table, t, d) is not None
        )

        history.append({"Date": d, "Portfolio Value": nav})

    return pd.DataFrame(history), pd.DataFrame(trades)


# ============================================================
# MONTHLY DATES / TARGETS / MARK-TO-MARKET TABLES
# ============================================================

def month_end_dates(dates: pd.Series) -> list:
    """
    (Name kept for compatibility) — returns FIRST available trading day in each month.
    """
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    if len(d) == 0:
        return []
    df = pd.DataFrame({"Date": sorted(d)})
    df["Month"] = df["Date"].dt.to_period("M")
    return df.groupby("Month")["Date"].min().tolist()


def compute_targets_over_dates(
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    dates: list,
    lookback_days: int,
    w_momentum: float,
    w_early: float,
    w_consistency: float,
    top_n: int,
    total_capital: float,
    weight_A: float = 0.20,
    weight_B: float = 0.80
) -> dict:
    targets = {}
    actions = {}

    prev_set = set()
    for d in dates:
        tgt = build_unified_target(
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

        if tgt is None or tgt.empty:
            targets[d] = pd.DataFrame(columns=["Ticker", "Position_Size"])
            actions[d] = pd.DataFrame(columns=["Ticker", "Position_Size", "Action"])
            continue

        today_set = set(tgt["Ticker"])
        out = tgt.copy()
        out["Action"] = "BUY"
        out.loc[out["Ticker"].isin(prev_set), "Action"] = "HOLD"

        sells = prev_set - today_set
        if sells:
            sell_df = pd.DataFrame({"Ticker": list(sells)})
            sell_df["Position_Size"] = 0.0
            sell_df["Action"] = "SELL"
            out = pd.concat([out, sell_df], ignore_index=True)

        out = out.sort_values(["Action", "Position_Size"], ascending=[True, False]).reset_index(drop=True)

        targets[d] = tgt.sort_values("Position_Size", ascending=False).reset_index(drop=True)
        actions[d] = out
        prev_set = today_set

    return {"targets": targets, "actions": actions}


def trade_diagnostics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {"Message": "No trades"}

    df = trades_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    sells = df[df["Action"] == "Sell"].copy()
    realized = float(sells["PnL"].sum()) if not sells.empty else 0.0

    buys = int((df["Action"] == "Buy").sum())
    sells_n = int((df["Action"] == "Sell").sum())
    resz = int((df["Action"] == "Resize").sum())

    monthly = (
        df.groupby(["Month", "Action"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    return {
        "Realized_PnL_$": realized,
        "Buys": buys,
        "Sells": sells_n,
        "Resizes": resz,
        "monthly_table": monthly
    }


def portfolio_mark_to_market(
    price_table: pd.DataFrame,
    target_df: pd.DataFrame,
    entry_date,
    mark_date
) -> pd.DataFrame:
    if target_df is None or target_df.empty:
        return pd.DataFrame()

    entry_date = pd.to_datetime(entry_date)
    mark_date = pd.to_datetime(mark_date)

    rows = []
    total = float(target_df["Position_Size"].sum())

    for _, r in target_df.iterrows():
        t = r["Ticker"]
        dollars = float(r["Position_Size"])

        entry_px = get_last_price(price_table, t, entry_date)
        mark_px  = get_last_price(price_table, t, mark_date)

        if entry_px is None or entry_px <= 0:
            shares = np.nan
            pnl = np.nan
            pnl_pct = np.nan
        else:
            shares = dollars / entry_px
            if mark_px is None or mark_px <= 0:
                pnl = np.nan
                pnl_pct = np.nan
            else:
                pnl = shares * (mark_px - entry_px)
                pnl_pct = (mark_px / entry_px - 1.0) * 100.0

        rows.append({
            "Ticker": t,
            "Target_$": dollars,
            "Weight_%": (dollars / total * 100.0) if total > 0 else np.nan,
            "Entry_Date": entry_date.date(),
            "Entry_Price": entry_px,
            "Current_Date": mark_date.date(),
            "Current_Price": mark_px,
            "Shares": shares,
            "PnL_$": pnl,
            "PnL_%": pnl_pct,
        })

    df = pd.DataFrame(rows).sort_values("Target_$", ascending=False).reset_index(drop=True)

    if not df.empty:
        total_row = {
            "Ticker": "TOTAL",
            "Target_$": df["Target_$"].sum(),
            "Weight_%": df["Weight_%"].sum(),
            "Entry_Date": "",
            "Entry_Price": np.nan,
            "Current_Date": "",
            "Current_Price": np.nan,
            "Shares": np.nan,
            "PnL_$": df["PnL_$"].sum(skipna=True),
            "PnL_%": np.nan,
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df


def monthly_pnl_table(
    price_table: pd.DataFrame,
    targets_by_month: dict,
    month_end_dates_list: list
) -> pd.DataFrame:
    """
    For each monthly rebalance portfolio target, compute P&L to the next rebalance date.
    """
    if not month_end_dates_list or len(month_end_dates_list) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(month_end_dates_list) - 1):
        d0 = pd.to_datetime(month_end_dates_list[i])
        d1 = pd.to_datetime(month_end_dates_list[i + 1])

        tgt = targets_by_month.get(d0)
        if tgt is None or tgt.empty:
            continue

        mtm = portfolio_mark_to_market(price_table, tgt, entry_date=d0, mark_date=d1)
        per = mtm[mtm["Ticker"] != "TOTAL"].copy()

        month_pnl = per["PnL_$"].sum(skipna=True)
        month_cap = per["Target_$"].sum()
        month_ret = (month_pnl / month_cap * 100.0) if month_cap > 0 else np.nan

        rows.append({
            "Month": d0.strftime("%Y-%m"),
            "Entry_Date": d0.date(),
            "Exit_Date": d1.date(),
            "Capital_$": month_cap,
            "PnL_$": month_pnl,
            "Return_%": month_ret
        })

    return pd.DataFrame(rows).sort_values("Entry_Date").reset_index(drop=True)


# ============================================================
# ✅ RADIO-LIKE TABLE + PRICE PLOT WITH BUY/SELL MARKERS (ADDITIVE)
# ============================================================

def radio_pick_table(df: pd.DataFrame, key: str, label_col: str = "Ticker"):
    """
    "Radio button in table" behavior using a Pick checkbox column + single-selection enforcement.
    Returns selected label (or None).
    """
    if df is None or df.empty or label_col not in df.columns:
        return None

    view = df.copy()
    pick_col = "__Pick__"
    if pick_col not in view.columns:
        view.insert(0, pick_col, False)

    # Restore previous selection state if present
    state_key = f"{key}_state"
    if state_key in st.session_state:
        prev = st.session_state[state_key]
        if isinstance(prev, pd.DataFrame) and len(prev) == len(view) and pick_col in prev.columns:
            view[pick_col] = prev[pick_col].values

    edited = st.data_editor(
        view,
        key=key,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in view.columns if c != pick_col],
        column_config={
            pick_col: st.column_config.CheckboxColumn(
                "Pick",
                help="Select one row (radio-like)",
                default=False
            )
        }
    )

    # Enforce single selection; never allow selecting TOTAL
    picked = edited.index[(edited[pick_col] == True) & (edited[label_col].astype(str) != "TOTAL")].tolist()
    selected = None
    if picked:
        keep = picked[0]
        edited.loc[(edited.index != keep), pick_col] = False
        selected = str(edited.loc[keep, label_col])

    # Ensure TOTAL can't remain selected
    edited.loc[edited[label_col].astype(str) == "TOTAL", pick_col] = False

    st.session_state[state_key] = edited.copy()
    return selected


def plot_price_with_trades(price_table: pd.DataFrame, trades_df: pd.DataFrame, ticker: str, title_prefix: str = ""):
    if price_table is None or price_table.empty or ticker not in price_table.columns:
        st.info(f"No price history found for {ticker}.")
        return

    px = price_table[[ticker]].dropna().reset_index().rename(columns={ticker: "Price"})
    if px.empty:
        st.info(f"No price history found for {ticker}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px["Date"], y=px["Price"], mode="lines", name=ticker))

    if trades_df is not None and not trades_df.empty:
        t = trades_df.copy()
        t["Date"] = pd.to_datetime(t["Date"])
        t = t[t["Ticker"] == ticker].sort_values("Date")

        buys = t[t["Action"].str.lower() == "buy"]
        sells = t[t["Action"].str.lower() == "sell"]

        if not buys.empty:
            bpx = pd.merge_asof(
                buys[["Date"]].sort_values("Date"),
                px.sort_values("Date"),
                on="Date",
                direction="backward"
            )
            fig.add_trace(go.Scatter(
                x=bpx["Date"], y=bpx["Price"],
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=11)
            ))

        if not sells.empty:
            spx = pd.merge_asof(
                sells[["Date"]].sort_values("Date"),
                px.sort_values("Date"),
                on="Date",
                direction="backward"
            )
            fig.add_trace(go.Scatter(
                x=spx["Date"], y=spx["Price"],
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=11)
            ))

    fig.update_layout(
        title=f"{title_prefix}{ticker} — Price with Trades",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Price"),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_equity_and_drawdown(history_df: pd.DataFrame, title: str):
    if history_df.empty:
        st.info("No equity history to display.")
        return

    df = history_df.sort_values("Date").copy()
    df["Rolling_Max"] = df["Portfolio Value"].cummax()
    df["Drawdown"] = (df["Portfolio Value"] - df["Rolling_Max"]) / df["Rolling_Max"]

    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Portfolio Value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(width=2),
        )
    )
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
    fig_dd.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(width=1.5, color="firebrick"),
        )
    )
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


# ============================================================
# STREAMLIT UI (RUNS THE SAME "MAIN" PIPELINE)
# ============================================================

@st.cache_data(show_spinner=False)
def run_full_pipeline():
    parquet_path = "artifacts/index_constituents_5yr.parquet"
    index_path = "artifacts/index_returns_5y.parquet"

    # Strategy knobs (momentum logic untouched)
    REBALANCE_INTERVAL = 10  # kept for parity but monthly schedule is enforced in simulators
    DAILY_TOP_N = 10
    FINAL_TOP_N = 10
    LOOKBACK_DAYS = 10
    CAPITAL_PER_TRADE = 5000
    WINDOWS = (5, 10, 30, 45, 60, 90)

    W_MOM = 0.50
    W_EARLY = 0.30
    W_CONS = 0.20

    base = load_price_data_parquet(parquet_path)
    base = filter_by_index(base, "SP500")
    idx = load_index_returns_parquet(index_path)

    # ---------------- BUCKET A ----------------
    dfA = calculate_momentum_features(
        add_absolute_returns(base),
        windows=WINDOWS
    )
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)
    priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    histA, tradesA = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=priceA,
        daily_df=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000,
        dollars_per_name=5_000
    )
    statsA = compute_performance_stats(histA)
    trade_statsA = compute_trade_stats(tradesA)

    # ---------------- BUCKET B ----------------
    dfB = base.copy()

    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
        validate="many_to_one"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

    dfB = calculate_momentum_features(
        add_absolute_returns(dfB),
        windows=WINDOWS
    )

    idx_mom = compute_index_momentum(idx, windows=WINDOWS)

    dfB = dfB.merge(
        idx_mom[
            [
                "date", "index",
                "idx_5D", "idx_10D",
                "idx_30D", "idx_45D",
                "idx_60D", "idx_90D",
            ]
        ],
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)

    dfB["Momentum Score"] = (
        0.5 * dfB["Rel_Slow_z"] +
        0.3 * dfB["Rel_Mid_z"] +
        0.2 * dfB["Rel_Fast_z"]
    )

    dfB = dfB[dfB["Momentum_Slow"] > 1].copy()
    dfB = dfB[dfB["Momentum_Mid"] > 0.5].copy()
    dfB = dfB[dfB["Momentum_Fast"] > 1].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    priceB = dfB.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

    histB, tradesB = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=priceB,
        daily_df=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000,
        dollars_per_name=5_000
    )

    statsB = compute_performance_stats(histB)
    trade_statsB = compute_trade_stats(tradesB)

    # ---------------- BUCKET C (UNIFIED) ----------------
    histU, tradesU = simulate_unified_portfolio(
        df_prices=base,
        price_table=priceA,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N
    )

    statsU = compute_performance_stats(histU)
    trade_statsU = compute_trade_stats(tradesU)

    todayC = get_today_trades_bucket_c(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=None,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000,
        weight_A=0.20,
        weight_B=0.80
    )

    return {
        "params": {
            "REBALANCE_INTERVAL": REBALANCE_INTERVAL,
            "DAILY_TOP_N": DAILY_TOP_N,
            "FINAL_TOP_N": FINAL_TOP_N,
            "LOOKBACK_DAYS": LOOKBACK_DAYS,
            "CAPITAL_PER_TRADE": CAPITAL_PER_TRADE,
            "WINDOWS": WINDOWS,
            "W_MOM": W_MOM,
            "W_EARLY": W_EARLY,
            "W_CONS": W_CONS,
        },
        "BucketA": {"history": histA, "trades": tradesA, "stats": statsA, "trade_stats": trade_statsA},
        "BucketB": {"history": histB, "trades": tradesB, "stats": statsB, "trade_stats": trade_statsB},
        "BucketC": {"history": histU, "trades": tradesU, "stats": statsU, "trade_stats": trade_statsU, "today": todayC},
        "internals": {
            "base": base,
            "dailyA": dailyA,
            "dailyB": dailyB,
            "priceA": priceA,
        }
    }


def _stats_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"Metric": list(d.keys()), "Value": list(d.values())})


# ============================================================
# UI
# ============================================================

st.title("📈 Momentum Portfolio")
st.caption(
    "This strategy seeks to capture persistent equity momentum by systematically ranking stocks on both absolute price strength and alpha-based momentum relative to the broader market. Momentum candidates are selected using a disciplined, multi-horizon process that emphasizes consistency and trend persistence rather than short-term price moves. The portfolio is rebalanced monthly on the first trading day, rotating capital into the strongest opportunities while exiting positions where momentum has deteriorated. Positions are held as long as momentum persists, with no fixed take-profit targets—allowing winners to compound while enforcing objective exits through the rebalance process."
)

with st.spinner("Running full pipeline from artifacts…"):
    out = run_full_pipeline()

params = out["params"]
bucketC = out["BucketC"]
internals = out["internals"]

base = internals["base"]
dailyA = internals["dailyA"]
dailyB = internals["dailyB"]
priceA = internals["priceA"]

hist = bucketC["history"]
trades = bucketC["trades"]
stats = bucketC["stats"]
trade_stats = bucketC["trade_stats"]

# --- common dates across A and B ---
all_dates = sorted(set(dailyA["Date"]).intersection(set(dailyB["Date"])))
today = all_dates[-1] if all_dates else None

# --- monthly rebalance dates (first trading day each month) ---
m_dates = month_end_dates(all_dates)

# --- targets/actions for each monthly rebalance date ---
bundle = compute_targets_over_dates(
    dailyA=dailyA,
    dailyB=dailyB,
    dates=m_dates,
    lookback_days=params["LOOKBACK_DAYS"],
    w_momentum=params["W_MOM"],
    w_early=params["W_EARLY"],
    w_consistency=params["W_CONS"],
    top_n=params["FINAL_TOP_N"],
    total_capital=100_000.0,
    weight_A=0.20,
    weight_B=0.80
)
targets_by_month = bundle["targets"]
actions_by_month = bundle["actions"]

# Current portfolio: entry at LAST monthly rebalance date; mark to latest available price date.
entry_date = m_dates[-1] if m_dates else today
latest_price_date = priceA.index.max() if len(priceA.index) else entry_date

current_target = targets_by_month.get(entry_date, pd.DataFrame())
current_port = portfolio_mark_to_market(
    price_table=priceA,
    target_df=current_target,
    entry_date=entry_date,
    mark_date=latest_price_date
)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Current Portfolio", "Monthly Rebalances", "Diagnostics"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", f"{stats.get('Total Return (%)', np.nan):.2f}")
    c2.metric("CAGR (%)", f"{stats.get('CAGR (%)', np.nan):.2f}")
    c3.metric("Sharpe", f"{stats.get('Sharpe Ratio', np.nan):.2f}")
    c4.metric("Max Drawdown (%)", f"{stats.get('Max Drawdown (%)', np.nan):.2f}")

    st.markdown("### Monthly Performance (rebalance to next rebalance)")
    monthly_summary = monthly_pnl_table(
        price_table=priceA,
        targets_by_month=targets_by_month,
        month_end_dates_list=m_dates
    )

    if monthly_summary is None or monthly_summary.empty:
        st.info("Not enough data to compute monthly performance.")
    else:
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly_summary["Month"], y=monthly_summary["Return_%"], name="Monthly Return (%)"))
        fig_m.update_layout(
            height=320,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(title="Month"),
            yaxis=dict(title="Return (%)"),
            template="plotly_white",
        )
        st.plotly_chart(fig_m, use_container_width=True)

        with st.expander("Show monthly performance table"):
            st.dataframe(monthly_summary, use_container_width=True)

    st.markdown("### Equity & Drawdown")
    plot_equity_and_drawdown(hist, title="Momentum Portfolio")

    st.markdown("### Trade Statistics")
    st.dataframe(_stats_to_df(trade_stats), use_container_width=True)


with tab2:
    st.markdown(
        f"### Current Portfolio (rebalance: {pd.to_datetime(entry_date).date() if entry_date is not None else 'N/A'} → "
        f"marked: {pd.to_datetime(latest_price_date).date() if latest_price_date is not None else 'N/A'})"
    )

    if current_port is None or current_port.empty:
        st.info("No portfolio could be constructed.")
    else:
        selected = radio_pick_table(current_port, key="cp_table", label_col="Ticker")
        st.markdown("#### Price chart (pick a row above)")
        if selected:
            plot_price_with_trades(priceA, trades, selected, title_prefix="Current Portfolio: ")
        else:
            st.info("Pick a ticker to plot its chart (BUY/SELL markers come from the backtest trades).")

        dfw = current_port[current_port["Ticker"] != "TOTAL"].copy()
        top5 = float(dfw["Weight_%"].head(5).sum()) if not dfw.empty else 0.0
        st.metric("Top 5 weight (%)", f"{top5:.1f}")

        try:
            total_row = current_port[current_port["Ticker"] == "TOTAL"].iloc[0]
            st.metric("Portfolio P&L ($)", f"{float(total_row['PnL_$']):,.0f}")
        except Exception:
            pass


with tab3:
    st.markdown("### Monthly Rebalance History")

    if not m_dates:
        st.info("Not enough data to compute monthly rebalances.")
    else:
        month_labels = [pd.to_datetime(d).strftime("%Y-%m") for d in m_dates]
        choice = st.selectbox("Select month", options=month_labels, index=len(month_labels) - 1)
        chosen_date = m_dates[month_labels.index(choice)]

        st.markdown(f"#### Target portfolio — {choice} (rebalance date: {pd.to_datetime(chosen_date).date()})")

        idx_choice = month_labels.index(choice)
        if idx_choice < len(m_dates) - 1:
            exit_date = m_dates[idx_choice + 1]
        else:
            exit_date = priceA.index.max() if len(priceA.index) else chosen_date

        tgt = targets_by_month.get(chosen_date, pd.DataFrame())
        mtm_month = portfolio_mark_to_market(
            price_table=priceA,
            target_df=tgt,
            entry_date=chosen_date,
            mark_date=exit_date
        )

        selected_m = radio_pick_table(mtm_month, key="m_table", label_col="Ticker")
        st.markdown("#### Price chart (pick a row above)")
        if selected_m:
            plot_price_with_trades(priceA, trades, selected_m, title_prefix=f"{choice}: ")
        else:
            st.info("Pick a ticker to plot its chart (BUY/SELL markers come from the backtest trades).")

        st.markdown("#### Actions vs previous month")
        st.dataframe(actions_by_month[chosen_date], use_container_width=True)


with tab4:
    st.markdown("### Diagnostics")

    diag = trade_diagnostics(trades)
    if "Message" in diag:
        st.info(diag["Message"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Realized PnL ($)", f"{diag['Realized_PnL_$']:,.0f}")
        c2.metric("Buys", f"{diag['Buys']}")
        c3.metric("Sells", f"{diag['Sells']}")
        c4.metric("Resizes", f"{diag['Resizes']}")

        st.markdown("#### Monthly trade activity")
        st.dataframe(diag["monthly_table"], use_container_width=True)

    with st.expander("Show full backtest trades"):
        if trades is None or trades.empty:
            st.write("No trades.")
        else:
            st.dataframe(trades.sort_values("Date"), use_container_width=True)
