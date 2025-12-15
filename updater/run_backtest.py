# backtest/run_backtest.py

from pathlib import Path
import pandas as pd

# ============================================================
# IMPORT YOUR EXISTING PIPELINE FUNCTIONS
# ============================================================
# IMPORTANT:
# This assumes you have moved your big pipeline file into
# something like `pipeline.py` at repo root.
# If your file name is different, adjust the import accordingly.

from pipeline import (
    load_price_data_parquet,
    load_index_returns_parquet,
    filter_by_index,

    # Feature + daily list builders
    calculate_momentum_features,
    add_absolute_returns,
    add_relative_regime_momentum_score,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_residual_momentum,
    add_regime_early_momentum,
    compute_index_momentum,
    build_daily_lists,

    # Backtest engines
    simulate_single_bucket_as_unified,
    simulate_unified_portfolio,

    # Stats
    compute_performance_stats,
    compute_trade_stats,
)

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
ART = ROOT / "artifacts" / "backtest"
ART.mkdir(parents=True, exist_ok=True)

PRICE_PARQUET = ROOT / "artifacts" / "index_constituents_5yr.parquet"
INDEX_PARQUET = ROOT / "artifacts" / "index_returns_5y.parquet"

TOTAL_CAPITAL = 100_000

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

WEIGHT_A = 0.20
WEIGHT_B = 0.80

WINDOWS = (5, 10, 30, 45, 60, 90)

# ============================================================
# LOAD BASE DATA
# ============================================================

print("Loading price data...")
base = load_price_data_parquet(PRICE_PARQUET)
base = filter_by_index(base, "SP500")

print("Loading index data...")
idx = load_index_returns_parquet(INDEX_PARQUET)

# Execution price table
price_table = (
    base
    .pivot(index="Date", columns="Ticker", values="Price")
    .sort_index()
)

# ============================================================
# ---------------- BUCKET A (ABSOLUTE MOMENTUM) ----------------
# ============================================================

print("Running Bucket A backtest...")

dfA = calculate_momentum_features(
    add_absolute_returns(base),
    windows=WINDOWS
)

dfA = add_regime_momentum_score(dfA)
dfA = add_regime_acceleration(dfA)
dfA = add_regime_residual_momentum(dfA)
dfA = add_regime_early_momentum(dfA)

dailyA = build_daily_lists(dfA, top_n=TOP_N)

histA, tradesA = simulate_single_bucket_as_unified(
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
    dollars_per_name=5_000,
)

histA["Bucket"] = "A"
tradesA["Bucket"] = "A"

# ============================================================
# ---------------- BUCKET B (RELATIVE MOMENTUM) ----------------
# ============================================================

print("Running Bucket B backtest...")

dfB = base.copy()

# Merge index daily returns
dfB = dfB.merge(
    idx,
    left_on=["Date", "Index"],
    right_on=["date", "index"],
    how="left",
    validate="many_to_one"
).drop(columns=["date", "index"], errors="ignore")

dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

# Stock absolute returns
dfB = calculate_momentum_features(
    add_absolute_returns(dfB),
    windows=WINDOWS
)

# Index momentum
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

# Relative regime momentum
dfB = add_relative_regime_momentum_score(dfB)

# Triple gate (as per your philosophy)
dfB = dfB[
    (dfB["Momentum_Slow"] > 1.0) &
    (dfB["Momentum_Mid"] > 0.5) &
    (dfB["Momentum_Fast"] > 1.0)
].copy()

dfB = add_regime_acceleration(dfB)
dfB = add_regime_residual_momentum(dfB)
dfB = add_regime_early_momentum(dfB)

dailyB = build_daily_lists(dfB, top_n=TOP_N)

histB, tradesB = simulate_single_bucket_as_unified(
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
    dollars_per_name=5_000,
)

histB["Bucket"] = "B"
tradesB["Bucket"] = "B"

# ============================================================
# ---------------- BUCKET C (UNIFIED PORTFOLIO) ----------------
# ============================================================

print("Running Bucket C backtest...")

histC, tradesC = simulate_unified_portfolio(
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

histC["Bucket"] = "C"
tradesC["Bucket"] = "C"

# ============================================================
# SAVE HISTORY + TRADES
# ============================================================

history = pd.concat([histA, histB, histC], ignore_index=True)
trades = pd.concat([tradesA, tradesB, tradesC], ignore_index=True)

history.to_parquet(ART / "backtest_history.parquet", index=False)
trades.to_parquet(ART / "backtest_trades.parquet", index=False)

# ============================================================
# PRECOMPUTE SUMMARY STATS
# ============================================================

rows = []

for bucket, h in history.groupby("Bucket"):
    t = trades[trades["Bucket"] == bucket]

    perf = compute_performance_stats(h)
    trade_stats = compute_trade_stats(t)

    rows.append({
        "Bucket": bucket,
        **perf,
        **trade_stats,
    })

summary = pd.DataFrame(rows)
summary.to_parquet(ART / "backtest_summary.parquet", index=False)

print("Backtest complete.")
print(f"Artifacts written to: {ART.resolve()}")
