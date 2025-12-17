# updater/run_backtest_from_signals.py
# Self-contained "pipeline-exact" implementation (no core.* imports)

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")
PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N_DAILY = 10
TOP_N_FINAL = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

# --- Outputs (same naming convention you used) ---
EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
TRADES_A_OUT = ARTIFACTS / "backtest_trades_A.parquet"
STATS_A_OUT = ARTIFACTS / "backtest_stats_A.json"
TRADE_STATS_A_OUT = ARTIFACTS / "trade_stats_A.json"
TODAY_A_OUT = ARTIFACTS / "today_A.parquet"

EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
TRADES_B_OUT = ARTIFACTS / "backtest_trades_B.parquet"
STATS_B_OUT = ARTIFACTS / "backtest_stats_B.json"
TRADE_STATS_B_OUT = ARTIFACTS / "trade_stats_B.json"
TODAY_B_OUT = ARTIFACTS / "today_B.parquet"

EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_C_OUT = ARTIFACTS / "backtest_trades_C.parquet"
STATS_C_OUT = ARTIFACTS / "backtest_stats_C.json"
TRADE_STATS_C_OUT = ARTIFACTS / "trade_stats_C.json"
TODAY_C_OUT = ARTIFACTS / "today_C.parquet"


# ============================================================
# IO HELPERS
# ============================================================

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_today(daily_df: pd.DataFrame, out_path: Path):
    if daily_df is None or daily_df.empty or "Date" not in daily_df.columns:
        return
    tmp = daily_df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    latest = tmp["Date"].max()
    today = tmp[tmp["Date"] == latest].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today.to_parquet(out_path, index=False)


# ============================================================
# 1) DATA LOADING (PIPELINE)
# ============================================================

def load_price_data_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    elif "Price" in df.columns:
        # allow already-prepared parquet
        df["Price"] = df["Price"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' or 'Price' found in parquet.")

    keep = ["Ticker", "Date", "Price", "Index"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in price parquet: {missing}")

    df = df[keep].drop_duplicates(subset=["Ticker", "Date"], keep="last")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_index_returns_parquet(path: str | Path) -> pd.DataFrame:
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
# 2) RETURN DEFINITIONS (PIPELINE)
# ============================================================

def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def add_relative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative daily growth factor = (1 + stock_ret) / (1 + index_ret)
    """
    df = df.copy()
    if "idx_ret_1d" not in df.columns:
        raise ValueError("idx_ret_1d missing. Merge index returns before calling add_relative_returns().")

    stock_ret = df.groupby("Ticker")["Price"].pct_change()
    idx_ret = df["idx_ret_1d"]
    denom = (1.0 + idx_ret).replace(0.0, np.nan)

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
# 3) MOMENTUM FEATURE ENGINE (PIPELINE)
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
    df["Momentum_Mid"] = (0.5 * df["30D zscore"] + 0.5 * df["45D zscore"])
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


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Early_Fast"] = (0.6 * df["Accel_Fast_z"] + 0.4 * df["Momentum_Fast"])
    df["Early_Mid"]  = (0.5 * df["Accel_Mid_z"]  + 0.5 * df["Momentum_Mid"])
    df["Early_Slow"] = (0.5 * df["Accel_Slow_z"] + 0.5 * df["Momentum_Slow"])

    df["Early Momentum Score"] = (
        0.5 * df["Early_Slow"] +
        0.3 * df["Early_Mid"] +
        0.2 * df["Early_Fast"]
    )

    return df.fillna(0.0)


def add_relative_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    PIPELINE function you pasted:
    - Build Rel_* returns by subtracting index momentum from stock returns
    - z-score cross-sectionally
    - Then use Rel_*_z as Momentum_* regimes
    """
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
        std = df.groupby("Date")[col].transform("std").replace(0, np.nan)
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
# 4) DAILY LISTS (PIPELINE)
# ============================================================

def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    records = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d]
        if snap.empty:
            continue
        picks = snap.sort_values("Momentum Score", ascending=False).head(top_n)
        for _, r in picks.iterrows():
            records.append({
                "Date": d,
                "Ticker": r["Ticker"],
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
# 5) BACKTEST MECHANICS (PIPELINE)
# ============================================================

def get_last_price(price_table: pd.DataFrame, ticker: str, date) -> float | None:
    try:
        s = price_table.loc[:date, ticker].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def simulate_single_bucket_as_unified(
    df_prices,
    price_table,
    daily_df,
    rebalance_interval,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital=100_000.0,
    dollars_per_name=5_000.0
):
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

    rebalance_dates = sorted(df_prices["Date"].unique())[::rebalance_interval]

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
        dailyA, lookback_days, w_momentum, w_early, w_consistency, as_of_date, top_n
    )
    selB = final_selection_from_daily(
        dailyB, lookback_days, w_momentum, w_early, w_consistency, as_of_date, top_n
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


def simulate_unified_portfolio(
    df_prices: pd.DataFrame,
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    rebalance_interval: int = 10,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    top_n: int = 10,
    total_capital: float = 100_000.0
):
    rebalance_dates = sorted(df_prices["Date"].unique())[::rebalance_interval]

    cash = total_capital
    portfolio = {}
    cost_basis = {}

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

        # SELL removed
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


# ============================================================
# 6) PERFORMANCE + TRADE STATS (PIPELINE)
# ============================================================

def compute_perf_stats_full(history_df: pd.DataFrame, value_col: str = "Portfolio Value") -> dict:
    base = {
        "Total Return (%)": 0.0,
        "CAGR (%)": 0.0,
        "Sharpe Ratio": 0.0,
        "Sortino Ratio": 0.0,
        "Max Drawdown (%)": 0.0,
    }
    if history_df is None or history_df.empty:
        return base
    if "Date" not in history_df.columns or value_col not in history_df.columns:
        return base

    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if len(df) < 2:
        return base

    start = float(df[value_col].iloc[0])
    end = float(df[value_col].iloc[-1])
    if start <= 0:
        return base

    ret = df[value_col].pct_change().dropna()
    mean = float(ret.mean()) if len(ret) else 0.0
    std = float(ret.std()) if len(ret) else 0.0

    sharpe = (math.sqrt(252) * mean / std) if std > 0 else 0.0

    downside = ret[ret < 0]
    dstd = float(downside.std()) if len(downside) else 0.0
    sortino = (math.sqrt(252) * mean / dstd) if dstd > 0 else 0.0

    roll_max = df[value_col].cummax()
    max_dd = float((df[value_col] / roll_max - 1.0).min() * 100.0)

    total_return = float((end / start - 1.0) * 100.0)

    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    cagr = float(((end / start) ** (365.0 / days) - 1.0) * 100.0) if days and days > 0 else 0.0

    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd,
    }


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    base = {
        "Number of Trades": 0,
        "Win Rate (%)": 0.0,
        "Average Win ($)": 0.0,
        "Average Loss ($)": 0.0,
        "Profit Factor": 0.0,
    }
    if trades_df is None or trades_df.empty or "PnL" not in trades_df.columns:
        return base

    df = trades_df.copy()
    if "Action" in df.columns:
        df = df[df["Action"].astype(str).str.lower() == "sell"]

    pnl = pd.to_numeric(df["PnL"], errors="coerce").dropna()
    if pnl.empty:
        return base

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    num = int(len(pnl))
    win_rate = float((pnl > 0).mean() * 100.0) if num else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float((-losses).sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0.0

    return {
        "Number of Trades": num,
        "Win Rate (%)": win_rate,
        "Average Win ($)": avg_win,
        "Average Loss ($)": avg_loss,
        "Profit Factor": profit_factor,
    }


# ============================================================
# MAIN (PIPELINE-EXACT)
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    print("📥 Loading index returns...")
    idx = load_index_returns_parquet(INDEX_PATH)

    # Price table used for all backtests (pipeline uses pivoted Price)
    price_table = base.pivot(index="Date", columns="Ticker", values="Price").sort_index()

    # ---------------- BUCKET A (ABSOLUTE) ----------------
    print("🧮 Building Bucket A signals...")
    dfA = calculate_momentum_features(add_absolute_returns(base), windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N_DAILY)

    print("📊 Running Bucket A backtest...")
    eqA, trA = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=price_table,
        daily_df=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N_FINAL,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=5_000,
    )

    eqA.to_parquet(EQUITY_A_OUT, index=False)
    trA.to_parquet(TRADES_A_OUT, index=False)
    write_json(STATS_A_OUT, compute_perf_stats_full(eqA))
    write_json(TRADE_STATS_A_OUT, compute_trade_stats(trA))
    write_today(dailyA, TODAY_A_OUT)

    # ---------------- BUCKET B (RELATIVE) ----------------
    print("🧮 Building Bucket B signals (pipeline-relative)...")

    dfB = base.copy()
    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
        validate="many_to_one"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

    # IMPORTANT: this matches your pasted pipeline main:
    # absolute 1D return -> momentum features -> index momentum -> relative regime scoring -> regime filters
    dfB = calculate_momentum_features(add_absolute_returns(dfB), windows=WINDOWS)

    idx_mom = compute_index_momentum(idx, windows=WINDOWS)

    dfB = dfB.merge(
        idx_mom[
            ["date", "index", "idx_5D", "idx_10D", "idx_30D", "idx_45D", "idx_60D", "idx_90D"]
        ],
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)

    # ✅ YOUR regime filters (exact)
    dfB = dfB[
        (dfB["Momentum_Slow"] > 1.0) &
        (dfB["Momentum_Mid"] > 0.5) &
        (dfB["Momentum_Fast"] > 1.0)
    ].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=TOP_N_DAILY)

    print("📊 Running Bucket B backtest...")
    eqB, trB = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=price_table,
        daily_df=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N_FINAL,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=5_000,
    )

    eqB.to_parquet(EQUITY_B_OUT, index=False)
    trB.to_parquet(TRADES_B_OUT, index=False)
    write_json(STATS_B_OUT, compute_perf_stats_full(eqB))
    write_json(TRADE_STATS_B_OUT, compute_trade_stats(trB))
    write_today(dailyB, TODAY_B_OUT)

    # ---------------- BUCKET C (UNIFIED) ----------------
    print("📊 Running Bucket C unified backtest...")
    eqC, trC = simulate_unified_portfolio(
        df_prices=base,
        price_table=price_table,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N_FINAL,
        total_capital=TOTAL_CAPITAL,
    )

    eqC.to_parquet(EQUITY_C_OUT, index=False)
    trC.to_parquet(TRADES_C_OUT, index=False)
    write_json(STATS_C_OUT, compute_perf_stats_full(eqC))
    write_json(TRADE_STATS_C_OUT, compute_trade_stats(trC))

    # Save "today_C" as the *latest target* snapshot (pipeline-equivalent)
    # We store the actual target weights for the last available date.
    last_date = max(sorted(base["Date"].unique()))
    tgt_last = build_unified_target(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=last_date,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N_FINAL,
        total_capital=TOTAL_CAPITAL,
        weight_A=0.20,
        weight_B=0.80
    )
    if isinstance(tgt_last, pd.DataFrame) and not tgt_last.empty:
        tgt_last.to_parquet(TODAY_C_OUT, index=False)

    print("✅ Backtest artifacts written (pipeline-exact).")


if __name__ == "__main__":
    main()
