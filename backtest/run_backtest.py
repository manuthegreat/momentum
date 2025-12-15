from pathlib import Path
import pandas as pd

from pipeline import (
    load_price_data_parquet,
    load_index_returns_parquet,
    filter_by_index,
    simulate_single_bucket_as_unified,
    simulate_unified_portfolio,
    compute_performance_stats,
    compute_trade_stats,
)

# ================= CONFIG =================

ART = Path("artifacts/backtest")
ART.mkdir(parents=True, exist_ok=True)

TOTAL_CAPITAL = 100_000
REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

WEIGHT_A = 0.20
WEIGHT_B = 0.80

# ================= LOAD DATA =================

prices = load_price_data_parquet("artifacts/index_constituents_5yr.parquet")
prices = filter_by_index(prices, "SP500")

index = load_index_returns_parquet("artifacts/index_returns_5y.parquet")

# price table for execution
price_table = prices.pivot(index="Date", columns="Ticker", values="Price").sort_index()
