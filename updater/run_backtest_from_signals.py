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

# --- OUTPUTS ---
EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
STATS_A_OUT  = ARTIFACTS / "backtest_stats_A.json"

EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
STATS_B_OUT  = ARTIFACTS / "backtest_stats_B.json"

EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_C_OUT = ARTIFACTS / "backtest_trades_C.parquet"
STATS_C_OUT  = ARTIFACTS / "backtest_stats_C.json"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

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

    # --------------------------------------------------------
    # FEATURE ENGINEERING (shared)
    # --------------------------------------------------------

    print("🧮 Computing features...")
    df = add_absolute_returns(base)
    df = calculate_momentum_features(df, windows=WINDOWS)
    df = add_regime_momentum_score(df)
    df = add_regime_acceleration(df)
    df = add_regime_early_momentum(df)

    # --------------------------------------------------------
    # BUCKET A — ABSOLUTE MOMENTUM
    # --------------------------------------------------------

    print("📊 Bucket A — building daily lists...")
    dailyA = build_daily_lists(df, top_n=TOP_N)

    print("📈 Bucket A — running backtest...")
    equity_A = simulate_single_bucket(
        price_table=base.pivot(index="Date", columns="Ticker", values="Price"),
        daily_df=dailyA,
        capital_per_trade=TOTAL_CAPITAL / TOP_N,
        rebalance_interval=REBALANCE_INTERVAL,
    )

    stats_A = compute_performance_stats(equity_A)

    equity_A.to_parquet(EQUITY_A_OUT, index=False)
    with open(STATS_A_OUT, "w") as f:
        json.dump(stats_A, f, indent=2)

    # --------------------------------------------------------
    # BUCKET B — RELATIVE MOMENTUM
    # --------------------------------------------------------

    print("📊 Bucket B — building daily lists...")
    dailyB = build_daily_lists(df, top_n=TOP_N)

    print("📈 Bucket B — running backtest...")
    equity_B = simulate_single_bucket(
        price_table=base.pivot(index="Date", columns="Ticker", values="Price"),
        daily_df=dailyB,
        capital_per_trade=TOTAL_CAPITAL / TOP_N,
        rebalance_interval=REBALANCE_INTERVAL,
    )

    stats_B = compute_performance_stats(equity_B)

    equity_B.to_parquet(EQUITY_B_OUT, index=False)
    with open(STATS_B_OUT, "w") as f:
        json.dump(stats_B, f, indent=2)

    # --------------------------------------------------------
    # BUCKET C — COMBINED (UNCHANGED LOGIC)
    # --------------------------------------------------------

    print("📊 Bucket C — running unified backtest...")
    price_table = base.pivot(index="Date", columns="Ticker", values="Price")

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

    stats_C = compute_performance_stats(equity_C)

    equity_C.to_parquet(EQUITY_C_OUT, index=False)
    trades_C.to_parquet(TRADES_C_OUT, index=False)

    with open(STATS_C_OUT, "w") as f:
        json.dump(stats_C, f, indent=2)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n✅ Backtest artifacts written:")
    print(f"  • {EQUITY_A_OUT}")
    print(f"  • {STATS_A_OUT}")
    print(f"  • {EQUITY_B_OUT}")
    print(f"  • {STATS_B_OUT}")
    print(f"  • {EQUITY_C_OUT}")
    print(f"  • {TRADES_C_OUT}")
    print(f"  • {STATS_C_OUT}")

if __name__ == "__main__":
    main()
