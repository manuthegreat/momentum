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
from core.backtest import simulate_single_bucket, compute_performance_stats

# =====================================================
# CONFIG
# =====================================================

ARTIFACTS = Path("artifacts")

PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_NAME = "SP500"

WINDOWS = (5, 10, 30, 45, 60, 90)
DAILY_TOP_N = 10
REBALANCE_INTERVAL = 10
CAPITAL_PER_TRADE = 5_000

# =====================================================
# LOAD DATA
# =====================================================

base = load_price_data_parquet(PRICE_PATH)
base = filter_by_index(base, INDEX_NAME)

# =====================================================
# FEATURE PIPELINE (ABSOLUTE MOMENTUM)
# =====================================================

df = add_absolute_returns(base)
df = calculate_momentum_features(df, windows=WINDOWS)
df = add_regime_momentum_score(df)
df = add_regime_acceleration(df)
df = add_regime_early_momentum(df)

daily = build_daily_lists(df, top_n=DAILY_TOP_N)

price_table = (
    df.pivot(index="Date", columns="Ticker", values="Price")
    .sort_index()
)

# =====================================================
# BACKTEST
# =====================================================

history = simulate_single_bucket(
    price_table=price_table,
    daily_df=daily,
    capital_per_trade=CAPITAL_PER_TRADE,
    rebalance_interval=REBALANCE_INTERVAL,
)

stats = compute_performance_stats(history)

history.to_parquet(ARTIFACTS / "history_A.parquet", index=False)

print("✅ Backtest complete")
print(stats)
