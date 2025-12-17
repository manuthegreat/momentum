# updater/run_backtest_from_signals.py

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from core.data_utils import (
    load_price_data_parquet,
    load_index_returns_parquet,
    filter_by_index,
)

from core.selection import (
    build_daily_lists,
    build_unified_target,
)

from core.features import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_residual_momentum,
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
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

# Outputs
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
# HELPERS
# ============================================================

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_today(daily_df: pd.DataFrame, out_path: Path):
    if daily_df.empty:
        return
    latest = daily_df["Date"].max()
    daily_df[daily_df["Date"] == latest].to_parquet(out_path, index=False)


def compute_perf_stats_full(history_df: pd.DataFrame, value_col="Portfolio Value") -> dict:
    if history_df.empty or len(history_df) < 2:
        return {}

    df = history_df.sort_values("Date")
    start, end = df[value_col].iloc[0], df[value_col].iloc[-1]
    ret = df[value_col].pct_change().dropna()

    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    cagr = ((end / start) ** (1 / years) - 1) * 100 if years > 0 else 0

    sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else 0
    downside = ret[ret < 0]
    sortino = np.sqrt(252) * ret.mean() / downside.std() if downside.std() > 0 else 0

    drawdown = (df[value_col] / df[value_col].cummax() - 1).min() * 100

    return {
        "Total Return (%)": (end / start - 1) * 100,
        "CAGR (%)": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": drawdown,
    }


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    sells = trades_df[trades_df["Action"] == "Sell"]
    if sells.empty:
        return {}

    wins = sells[sells["PnL"] > 0]
    losses = sells[sells["PnL"] < 0]

    return {
        "Number of Trades": len(sells),
        "Win Rate (%)": len(wins) / len(sells) * 100,
        "Average Win ($)": wins["PnL"].mean() if not wins.empty else 0,
        "Average Loss ($)": losses["PnL"].mean() if not losses.empty else 0,
        "Profit Factor": wins["PnL"].sum() / abs(losses["PnL"].sum()) if not losses.empty else 0,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(exist_ok=True)

    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    idx = load_index_returns_parquet(INDEX_PATH)

    # ========================================================
    # BUCKET A — ABSOLUTE MOMENTUM (PIPELINE IDENTICAL)
    # ========================================================

    dfA = calculate_momentum_features(
        add_absolute_returns(base),
        windows=WINDOWS
    )
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)

    priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    eqA, trA = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=priceA,
        daily_df=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=5_000,
    )

    write_json(STATS_A_OUT, compute_perf_stats_full(eqA))
    write_json(TRADE_STATS_A_OUT, compute_trade_stats(trA))
    eqA.to_parquet(EQUITY_A_OUT, index=False)
    trA.to_parquet(TRADES_A_OUT, index=False)
    write_today(dailyA, TODAY_A_OUT)

    # ========================================================
    # BUCKET B — RELATIVE REGIME MOMENTUM (PIPELINE IDENTICAL)
    # ========================================================

    dfB = base.copy()

    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

    dfB = calculate_momentum_features(
        add_absolute_returns(dfB),
        windows=WINDOWS
    )

    idx_mom = compute_index_momentum(idx, windows=WINDOWS)

    dfB = dfB.merge(
        idx_mom[
            ["date", "index",
             "idx_5D", "idx_10D",
             "idx_30D", "idx_45D",
             "idx_60D", "idx_90D"]
        ],
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)

    dfB["Momentum Score"] = (
        0.5 * dfB["Rel_Slow_z"] +
        0.3 * dfB["Rel_Mid_z"] +
        0.2 * dfB["Rel_Fast_z"]
    )

    dfB = dfB[
        (dfB["Momentum_Slow"] > 1) &
        (dfB["Momentum_Mid"] > 0.5) &
        (dfB["Momentum_Fast"] > 1)
    ].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    priceB = dfB.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    eqB, trB = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=priceB,
        daily_df=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=5_000,
    )

    write_json(STATS_B_OUT, compute_perf_stats_full(eqB))
    write_json(TRADE_STATS_B_OUT, compute_trade_stats(trB))
    eqB.to_parquet(EQUITY_B_OUT, index=False)
    trB.to_parquet(TRADES_B_OUT, index=False)
    write_today(dailyB, TODAY_B_OUT)

    # ========================================================
    # BUCKET C — UNIFIED (PIPELINE IDENTICAL)
    # ========================================================

    eqC, trC = simulate_unified_portfolio(
        df_prices=base,
        price_table=priceA,
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

    write_json(STATS_C_OUT, compute_perf_stats_full(eqC))
    write_json(TRADE_STATS_C_OUT, compute_trade_stats(trC))
    eqC.to_parquet(EQUITY_C_OUT, index=False)
    trC.to_parquet(TRADES_C_OUT, index=False)

    print("✅ Pipeline-identical backtest complete.")


if __name__ == "__main__":
    main()
