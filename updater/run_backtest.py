import pandas as pd
from pathlib import Path

from core.data import load_price_data_parquet, load_index_returns_parquet, filter_by_index
from core.features import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_early_momentum,
)
from core.selection import build_daily_lists
from core.backtest import (
    simulate_single_bucket,
    simulate_unified_portfolio,
    compute_performance_stats,
)

# =====================================================
# CONFIG
# =====================================================

ARTIFACTS = Path("artifacts")

PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"
INDEX_NAME = "SP500"

WINDOWS = (5, 10, 30, 45, 60, 90)
DAILY_TOP_N = 10
REBALANCE_INTERVAL = 10
CAPITAL_PER_TRADE = 5_000
TOTAL_CAPITAL = 100_000

# =====================================================
# LOAD DATA
# =====================================================

base = load_price_data_parquet(PRICE_PATH)
base = filter_by_index(base, INDEX_NAME)

idx = load_index_returns_parquet(INDEX_PATH)

# =====================================================
# BUCKET A — ABSOLUTE MOMENTUM
# =====================================================

dfA = add_absolute_returns(base)
dfA = calculate_momentum_features(dfA, windows=WINDOWS)
dfA = add_regime_momentum_score(dfA)
dfA = add_regime_acceleration(dfA)
dfA = add_regime_early_momentum(dfA)

dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()

histA = simulate_single_bucket(
    price_table=priceA,
    daily_df=dailyA,
    capital_per_trade=CAPITAL_PER_TRADE,
    rebalance_interval=REBALANCE_INTERVAL,
)

histA.to_parquet(ARTIFACTS / "history_A.parquet", index=False)

# =====================================================
# BUCKET B — RELATIVE MOMENTUM (FILTERED)
# =====================================================

dfB = base.merge(
    idx,
    left_on=["Date", "Index"],
    right_on=["date", "index"],
    how="left",
    validate="many_to_one"
).drop(columns=["date", "index"], errors="ignore")

dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

dfB = add_absolute_returns(dfB)
dfB = calculate_momentum_features(dfB, windows=WINDOWS)
dfB = add_regime_momentum_score(dfB)

# Relative trend filter
dfB = dfB[
    (dfB["Momentum_Slow"] > 0.5) &
    (dfB["Momentum_Mid"] > 0.25) &
    (dfB["Momentum_Fast"] > 0.5)
]

dfB = add_regime_acceleration(dfB)
dfB = add_regime_early_momentum(dfB)

dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

priceB = dfB.pivot(index="Date", columns="Ticker", values="Price").sort_index()

histB = simulate_single_bucket(
    price_table=priceB,
    daily_df=dailyB,
    capital_per_trade=CAPITAL_PER_TRADE,
    rebalance_interval=REBALANCE_INTERVAL,
)

histB.to_parquet(ARTIFACTS / "history_B.parquet", index=False)

# =====================================================
# BUCKET C — UNIFIED (80 / 20)
# =====================================================

histC, _ = simulate_unified_portfolio(
    df_prices=base,
    price_table=priceA,   # same prices
    dailyA=dailyA,
    dailyB=dailyB,
    rebalance_interval=REBALANCE_INTERVAL,
    lookback_days=10,
    w_momentum=0.5,
    w_early=0.3,
    w_consistency=0.2,
    top_n=10,
    total_capital=TOTAL_CAPITAL,
)

histC.to_parquet(ARTIFACTS / "history_C.parquet", index=False)

# =====================================================
# SUMMARY
# =====================================================

print("✅ Backtests complete")
print("Saved:")
print(" - history_A.parquet")
print(" - history_B.parquet")
print(" - history_C.parquet")

print("\nStats:")
print("A:", compute_performance_stats(histA))
print("B:", compute_performance_stats(histB))
print("C:", compute_performance_stats(histC))
