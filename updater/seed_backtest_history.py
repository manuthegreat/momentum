import pandas as pd
from pathlib import Path

from core.data_utils import (
    load_price_data_parquet,
    load_index_returns_parquet,
    filter_by_index,
)

from core.momentum_utils import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_residual_momentum,
    add_regime_early_momentum,
    add_relative_regime_momentum_score,
    compute_index_momentum,
    build_daily_lists,
    final_selection_from_daily,
    build_unified_target,
)

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")

PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"

OUT_PATH = ARTIFACTS / "backtest_signals.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)
DAILY_TOP_N = 10
FINAL_TOP_N = 10
LOOKBACK_DAYS = 10

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

WEIGHT_A = 0.20
WEIGHT_B = 0.80

# ============================================================
# LOAD DATA
# ============================================================

print("Loading historical price data...")
base = load_price_data_parquet(PRICE_PATH)
base = filter_by_index(base, "SP500")

print("Loading index returns...")
idx = load_index_returns_parquet(INDEX_PATH)

# ============================================================
# BUCKET A — ABSOLUTE MOMENTUM
# ============================================================

print("Computing Bucket A signals...")

dfA = calculate_momentum_features(
    add_absolute_returns(base),
    windows=WINDOWS
)

dfA = add_regime_momentum_score(dfA)
dfA = add_regime_acceleration(dfA)
dfA = add_regime_residual_momentum(dfA)
dfA = add_regime_early_momentum(dfA)

dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

# ============================================================
# BUCKET B — RELATIVE MOMENTUM
# ============================================================

print("Computing Bucket B signals...")

dfB = base.merge(
    idx,
    left_on=["Date", "Index"],
    right_on=["date", "index"],
    how="left",
    validate="many_to_one"
).drop(columns=["date", "index"], errors="ignore")

dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

dfB = calculate_momentum_features(
    add_absolute_returns(dfB),
    windows=WINDOWS
)

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

dfB = add_relative_regime_momentum_score(dfB)

# Optional regime filters (same as prod)
dfB = dfB[dfB["Momentum_Slow"] > 1]
dfB = dfB[dfB["Momentum_Mid"] > 0.5]
dfB = dfB[dfB["Momentum_Fast"] > 1]

dfB = add_regime_acceleration(dfB)
dfB = add_regime_residual_momentum(dfB)
dfB = add_regime_early_momentum(dfB)

dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

# ============================================================
# BUILD HISTORICAL BUCKET C (UNIFIED)
# ============================================================

print("Building historical Bucket C selections...")

all_dates = sorted(set(dailyA["Date"]).intersection(dailyB["Date"]))
records = []

for d in all_dates:
    tgt = build_unified_target(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=d,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000,
        weight_A=WEIGHT_A,
        weight_B=WEIGHT_B,
    )

    if tgt.empty:
        continue

    tgt = tgt.copy()
    tgt["Date"] = d
    tgt["Bucket"] = "C"

    records.append(tgt)

# ============================================================
# WRITE OUTPUT
# ============================================================

history = pd.concat(records, ignore_index=True)

history = history.sort_values(["Date", "Position_Size"], ascending=[True, False])

OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
history.to_parquet(OUT_PATH, index=False)

print(f"✅ Historical backtest written → {OUT_PATH}")
print(f"Rows: {len(history)}")
print(f"From {history['Date'].min().date()} to {history['Date'].max().date()}")
