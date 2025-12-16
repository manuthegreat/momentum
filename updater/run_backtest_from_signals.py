# updater/run_backtest_from_signals.py

import json
import pandas as pd
from pathlib import Path

from core.data_utils import load_price_data_parquet, filter_by_index
from core.backtest import (
    simulate_single_bucket,
    simulate_unified_portfolio,
    compute_performance_stats,
)
from core.selection import build_daily_lists
from core.features import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_early_momentum,
)

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")

PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"

EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"

STATS_A_OUT = ARTIFACTS / "backtest_stats_A.json"
STATS_B_OUT = ARTIFACTS / "backtest_stats_B.json"
STATS_C_OUT = ARTIFACTS / "backtest_stats_C.json"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000
CAPITAL_PER_TRADE = 5_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    price_table = (
        base.pivot(index="Date", columns="Ticker", values="Price")
        .sort_index()
    )

    # --------------------------------------------------------
    # Bucket A (ABSOLUTE MOMENTUM)
    # --------------------------------------------------------

    print("🧮 Building Bucket A signals...")
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    print("📊 Backtesting Bucket A...")
    equity_A = simulate_single_bucket(
        price_table=price_table,
        daily_df=dailyA,
        capital_per_trade=CAPITAL_PER_TRADE,
        rebalance_interval=REBALANCE_INTERVAL,
    )
    stats_A = compute_performance_stats(equity_A)

    equity_A.to_parquet(EQUITY_A_OUT, index=False)
    with open(STATS_A_OUT, "w") as f:
        json.dump(stats_A, f, indent=2)

    # --------------------------------------------------------
    # Bucket B (RELATIVE MOMENTUM)
    # --------------------------------------------------------

    print("🧮 Building Bucket B signals...")
    dfB = dfA.copy()  # relative filter already applied upstream
    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    print("📊 Backtesting Bucket B...")
    equity_B = simulate_single_bucket(
        price_table=price_table,
        daily_df=dailyB,
        capital_per_trade=CAPITAL_PER_TRADE,
        rebalance_interval=REBALANCE_INTERVAL,
    )
    stats_B = compute_performance_stats(equity_B)

    equity_B.to_parquet(EQUITY_B_OUT, index=False)
    with open(STATS_B_OUT, "w") as f:
        json.dump(stats_B, f, indent=2)

    # --------------------------------------------------------
    # Bucket C (UNIFIED PORTFOLIO)
    # --------------------------------------------------------

    print("📊 Running unified backtest (Bucket C)...")
    equity_C, trades_C = simulate_unified_portfolio(
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

    stats_C = compute_performance_stats(equity_C.rename(columns={"Portfolio Value": "Portfolio Value"}))

    equity_C.to_parquet(EQUITY_C_OUT, index=False)
    trades_C.to_parquet(ARTIFACTS / "backtest_trades_C.parquet", index=False)
    with open(STATS_C_OUT, "w") as f:
        json.dump(stats_C, f, indent=2)

    print("✅ All artifacts written successfully")

if __name__ == "__main__":
    main()
