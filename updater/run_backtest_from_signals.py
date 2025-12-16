# updater/run_backtest_from_signals.py

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from core.data_utils import load_price_data_parquet, filter_by_index
from core.selection import build_daily_lists, build_unified_target
from core.features import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_early_momentum,
    add_relative_regime_momentum_score,
    compute_index_momentum,
)

from core.backtest import (
    simulate_unified_portfolio,
    simulate_single_bucket_as_unified,
)

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")
PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

# --- Bucket A outputs ---
EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
TRADES_A_OUT = ARTIFACTS / "backtest_trades_A.parquet"
STATS_A_OUT = ARTIFACTS / "backtest_stats_A.json"
TRADE_STATS_A_OUT = ARTIFACTS / "trade_stats_A.json"
TODAY_A_OUT = ARTIFACTS / "today_A.parquet"

# --- Bucket B outputs ---
EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
TRADES_B_OUT = ARTIFACTS / "backtest_trades_B.parquet"
STATS_B_OUT = ARTIFACTS / "backtest_stats_B.json"
TRADE_STATS_B_OUT = ARTIFACTS / "trade_stats_B.json"
TODAY_B_OUT = ARTIFACTS / "today_B.parquet"

# --- Bucket C outputs ---
EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_C_OUT = ARTIFACTS / "backtest_trades_C.parquet"
STATS_C_OUT = ARTIFACTS / "backtest_stats_C.json"
TRADE_STATS_C_OUT = ARTIFACTS / "trade_stats_C.json"
TODAY_C_OUT = ARTIFACTS / "today_C.parquet"


# ============================================================
# HELPERS
# ============================================================

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _to_naive_dt(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def _normalize_price_table(price_table: pd.DataFrame) -> pd.DataFrame:
    pt = price_table.copy()
    idx = pd.to_datetime(pt.index, errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    pt.index = idx
    pt = pt.sort_index()
    return pt


def get_last_price(price_table: pd.DataFrame, ticker: str, asof: pd.Timestamp):
    """Last available price up to and including asof."""
    if ticker not in price_table.columns:
        return None
    s = price_table.loc[:asof, ticker].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def compute_perf_stats_full(history_df: pd.DataFrame, value_col: str) -> dict:
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
    df["Date"] = _to_naive_dt(df["Date"])
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
    """
    Robust trade stats:
    - If 'Action' exists, we compute on sells only (baseline convention)
    - If it doesn't exist, we compute on ALL rows (prevents KeyError)
    """
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


def write_today(daily_df: pd.DataFrame, out_path: Path):
    if daily_df is None or daily_df.empty or "Date" not in daily_df.columns:
        return
    tmp = daily_df.copy()
    tmp["Date"] = _to_naive_dt(tmp["Date"])
    tmp = tmp.dropna(subset=["Date"])
    if tmp.empty:
        return
    latest = tmp["Date"].max()
    today = tmp[tmp["Date"] == latest].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today.to_parquet(out_path, index=False)


# ============================================================
# C TRADES (GENERATE BLOTTTER WITHOUT CHANGING CORE)
# ============================================================

def build_unified_trades_over_time(
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    rebalance_interval: int,
    lookback_days: int,
    w_momentum: float,
    w_early: float,
    w_consistency: float,
    top_n: int,
    total_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dates = sorted(price_table.index.dropna().unique())
    rebalance_dates = dates[::rebalance_interval]
    if len(rebalance_dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    trades = []
    capital = float(total_capital)

    equity_rows = [{
        "Date": pd.Timestamp(rebalance_dates[0]),
        "Portfolio Value": capital,
        "PnL": 0.0,
    }]

    for i in range(1, len(rebalance_dates)):
        prev = pd.Timestamp(rebalance_dates[i - 1])
        curr = pd.Timestamp(rebalance_dates[i])

        target = build_unified_target(
            dailyA=dailyA,
            dailyB=dailyB,
            as_of_date=prev,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=capital,
        )

        period_pnl = 0.0

        for _, row in target.iterrows():
            t = str(row["Ticker"])
            alloc = float(row["Capital"])

            p0 = get_last_price(price_table, t, prev)
            p1 = get_last_price(price_table, t, curr)
            if p0 is None or p1 is None or p0 == 0:
                continue

            r = (p1 / p0) - 1.0
            pnl = alloc * r
            period_pnl += pnl

            trades.append(
                {
                    "Entry Date": prev,
                    "Exit Date": curr,
                    "Ticker": t,
                    "Capital": alloc,
                    "Entry Price": float(p0),
                    "Exit Price": float(p1),
                    "Return": float(r),
                    "PnL": float(pnl),
                    "Action": "Sell",
                }
            )

        capital += period_pnl
        equity_rows.append({"Date": curr, "Portfolio Value": float(capital), "PnL": float(period_pnl)})

    equity_df = pd.DataFrame(equity_rows).sort_values("Date").reset_index(drop=True)
    trades_df = pd.DataFrame(trades).sort_values(["Exit Date", "Ticker"]).reset_index(drop=True)
    return equity_df, trades_df


# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    price_table = base.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    price_table = _normalize_price_table(price_table)

    # --------------------------------------------------------
    # Build Bucket A signals (ABSOLUTE)
    # --------------------------------------------------------
    print("🧮 Building Bucket A signals...")
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    # --------------------------------------------------------
    # Build Bucket B signals (RELATIVE - cross-sectional rank)
    # --------------------------------------------------------
    print("🧮 Building Bucket B signals (relative ranking)...")

    # --------------------------------------------------------
    # Build Bucket B signals (TRUE RELATIVE MOMENTUM)
    # --------------------------------------------------------
    
    # 1) Start from base universe
    dfB = base.copy()
    
    # 2) Merge index returns (creates idx_ret_1d)
    idx = pd.read_parquet(ARTIFACTS / "index_returns_5y.parquet")
    idx.columns = idx.columns.str.lower().str.replace(" ", "_")
    idx["date"] = pd.to_datetime(idx["date"])
    
    # 🔧 normalize index column name ONCE
    if "index" not in idx.columns:
        for c in idx.columns:
            if c in ("index_name", "benchmark", "symbol", "ticker"):
                idx = idx.rename(columns={c: "index"})
                break
    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left"
    ).drop(columns=["date", "index"], errors="ignore")
    

    # 4) Absolute stock returns (needed for compounding)
    dfB = add_absolute_returns(dfB)
    
    # 5) Stock momentum features
    dfB = calculate_momentum_features(dfB, windows=WINDOWS)

    # 3) Drop rows without index data
    # --------------------------------------------------------
    # Normalize index return column
    # --------------------------------------------------------
    
    idx_ret_candidates = [
        "idx_ret_1d",
        "index_ret_1d",
        "ret_1d",
        "return_1d",
        "daily_return",
    ]
    
    found = None
    for c in idx_ret_candidates:
        if c in dfB.columns:
            found = c
            break
    
    if found is None:
        raise ValueError(
            f"No daily index return column found after merge. Columns: {dfB.columns.tolist()}"
        )
    
    # standardize name
    if found != "idx_ret_1d":
        dfB = dfB.rename(columns={found: "idx_ret_1d"})

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()
    
    # 6) Index momentum
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
    
    # 7) RELATIVE regime momentum (this is the key)
    # --------------------------------------------------------
    # NORMALIZE momentum column names for relative regime logic
    # --------------------------------------------------------
    
    # --------------------------------------------------------
    # Validate momentum columns produced by calculate_momentum_features
    # --------------------------------------------------------
    
    required = {"Momentum_Fast", "Momentum_Mid", "Momentum_Slow"}
    missing = required - set(dfB.columns)
    
    if missing:
        raise ValueError(
            f"Bucket B missing momentum columns from calculate_momentum_features: {missing}. "
            f"Columns present: {list(dfB.columns)}"
        )
    
    # --------------------------------------------------------
    # Bucket B compatibility: early momentum expects Momentum Score
    # --------------------------------------------------------
    
    # --- Ensure Bucket B always has the expected scoring columns ---
    # Some upstream builders produce either "Momentum Score" or "Relative_Momentum_Score" (or neither),
    # depending on which path ran. Normalize here so downstream never explodes.
    
    # 1) If Relative_Momentum_Score exists but Momentum Score doesn't, mirror it
    if "Relative_Momentum_Score" in dfB.columns and "Momentum Score" not in dfB.columns:
        dfB["Momentum Score"] = dfB["Relative_Momentum_Score"]
    
    # 2) If Momentum Score exists but Relative_Momentum_Score doesn't, mirror it
    if "Momentum Score" in dfB.columns and "Relative_Momentum_Score" not in dfB.columns:
        dfB["Relative_Momentum_Score"] = dfB["Momentum Score"]
    
    # 3) If neither exists, create a safe fallback from absolute returns (best effort)
    if "Momentum Score" not in dfB.columns and "Relative_Momentum_Score" not in dfB.columns:
        # Prefer an obvious absolute-return column name if present
        candidates = [
            "Absolute_Return",
            "Absolute_Returns",
            "abs_return",
            "abs_returns",
            "Return",
            "Returns",
            "Pct_Return",
            "Pct_Returns",
        ]
        base_col = next((c for c in candidates if c in dfB.columns), None)
    
        if base_col is None:
            raise ValueError(
                f"Bucket B missing Momentum Score and Relative_Momentum_Score, and no usable return column found. "
                f"Columns present: {list(dfB.columns)}"
            )
    
        # Use base returns as momentum proxy, and rank-normalize into Relative_Momentum_Score
        dfB["Momentum Score"] = dfB[base_col]
        dfB["Relative_Momentum_Score"] = (
            dfB.groupby("Date")["Momentum Score"]
               .rank(ascending=False, method="average")
        )


     add_relative_regime_momentum_score(dfB)
    
    # 8) Regime filters (same as local pipeline)
     dfB[
        (dfB["Momentum_Slow"] > 1.0) &
        (dfB["Momentum_Mid"] > 0.5) &
        (dfB["Momentum_Fast"] > 1.0)
    ].copy()
    
    # 9) Acceleration + early momentum
    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_early_momentum(dfB)
    
    # 10) Daily lists
    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    # --------------------------------------------------------
    # A backtest + trades (baseline-style)
    # --------------------------------------------------------
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
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name= 5_000
    )
    statsA = compute_perf_stats_full(eqA, value_col="Portfolio Value")
    tstatsA = compute_trade_stats(trA)

    eqA.to_parquet(EQUITY_A_OUT, index=False)
    trA.to_parquet(TRADES_A_OUT, index=False)
    write_json(STATS_A_OUT, statsA)
    write_json(TRADE_STATS_A_OUT, tstatsA)
    write_today(dailyA, TODAY_A_OUT)

    # --------------------------------------------------------
    # B backtest + trades (baseline-style)
    # --------------------------------------------------------
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
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name= 5_000
    )
    statsB = compute_perf_stats_full(eqB, value_col="Portfolio Value")
    tstatsB = compute_trade_stats(trB)
    
    eqB.to_parquet(EQUITY_B_OUT, index=False)
    trB.to_parquet(TRADES_B_OUT, index=False)
    write_json(STATS_B_OUT, statsB)
    write_json(TRADE_STATS_B_OUT, tstatsB)
    write_today(dailyB, TODAY_B_OUT)

    # --------------------------------------------------------
    # Bucket C unified backtest (UNCHANGED core call)
    # --------------------------------------------------------
    print("📊 Running unified backtest (Bucket C)...")
    equityC_core, target_last = simulate_unified_portfolio(
        df_prices=base,
        price_table=price_table,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
    )

    # --------------------------------------------------------
    # C equity + trades replay (so you get trade stats like baseline)
    # --------------------------------------------------------
    print("📊 Replaying Bucket C targets to generate trades...")
    eqC, trC = build_unified_trades_over_time(
        price_table=price_table,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
    )

    statsC = compute_perf_stats_full(eqC, value_col="Portfolio Value")
    tstatsC = compute_trade_stats(trC)

    eqC.to_parquet(EQUITY_C_OUT, index=False)
    trC.to_parquet(TRADES_C_OUT, index=False)
    write_json(STATS_C_OUT, statsC)
    write_json(TRADE_STATS_C_OUT, tstatsC)

    if isinstance(target_last, pd.DataFrame) and not target_last.empty:
        target_last.to_parquet(TODAY_C_OUT, index=False)

    print("✅ Backtest artifacts written.")


if __name__ == "__main__":
    main()
