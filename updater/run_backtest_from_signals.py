# updater/run_backtest_from_signals.py

import json
import pandas as pd
from pathlib import Path

from core.data_utils import load_price_data_parquet, filter_by_index
from core.backtest import (
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

# Bucket A artifacts
A_TODAY_OUT = ARTIFACTS / "today_A.parquet"
A_STATS_OUT = ARTIFACTS / "backtest_stats_A.json"

# Bucket B artifacts
B_TODAY_OUT = ARTIFACTS / "today_B.parquet"
B_STATS_OUT = ARTIFACTS / "backtest_stats_B.json"

# Bucket C artifacts
EQUITY_OUT = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_OUT = ARTIFACTS / "backtest_trades_C.parquet"
STATS_OUT  = ARTIFACTS / "backtest_stats_C.json"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

WEIGHT_A = 0.20
WEIGHT_B = 0.80

# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    # --------------------------------------------------------
    # Build Bucket A — ABSOLUTE MOMENTUM
    # --------------------------------------------------------

    print("🧮 Building Bucket A signals...")
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    # 👉 NEW: Persist Bucket A signals
    dailyA.to_parquet(A_TODAY_OUT, index=False)

    # Optional but included: A-only backtest stats (parity with CLI)
    equity_A, _ = simulate_unified_portfolio(
        df_prices=base,
        price_table=base.pivot(index="Date", columns="Ticker", values="Price"),
        dailyA=dailyA,
        dailyB=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
    )
    stats_A = compute_performance_stats(equity_A)

    with open(A_STATS_OUT, "w") as f:
        json.dump(stats_A, f, indent=2)

    # --------------------------------------------------------
    # Build Bucket B — RELATIVE MOMENTUM
    # --------------------------------------------------------

    print("🧮 Building Bucket B signals...")
    dfB = dfA.copy()  # relative filters already applied upstream
    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    # 👉 NEW: Persist Bucket B signals
    dailyB.to_parquet(B_TODAY_OUT, index=False)

    equity_B, _ = simulate_unified_portfolio(
        df_prices=base,
        price_table=base.pivot(index="Date", columns="Ticker", values="Price"),
        dailyA=dailyB,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
    )
    stats_B = compute_performance_stats(equity_B)

    with open(B_STATS_OUT, "w") as f:
        json.dump(stats_B, f, indent=2)

    # --------------------------------------------------------
    # Price table for unified backtest
    # --------------------------------------------------------

    price_table = (
        base.pivot(index="Date", columns="Ticker", values="Price")
        .sort_index()
    )

    # --------------------------------------------------------
    # Run Bucket C — UNIFIED (A + B)
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

    stats_C = compute_performance_stats(equity_C)

    # --------------------------------------------------------
    # Persist Bucket C artifacts
    # --------------------------------------------------------

    equity_C.to_parquet(EQUITY_OUT, index=False)
    trades_C.to_parquet(TRADES_OUT, index=False)

    with open(STATS_OUT, "w") as f:
        json.dump(stats_C, f, indent=2)

    print("✅ Backtest artifacts written:")
    print(f"  • {A_TODAY_OUT}")
    print(f"  • {A_STATS_OUT}")
    print(f"  • {B_TODAY_OUT}")
    print(f"  • {B_STATS_OUT}")
    print(f"  • {EQUITY_OUT}")
    print(f"  • {TRADES_OUT}")
    print(f"  • {STATS_OUT}")

    print("\n📈 Bucket C Performance Summary:")
    for k, v in stats_C.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
