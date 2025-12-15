import sys
from pathlib import Path

# -----------------------------------------------------
# Ensure project root is on PYTHONPATH
# -----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from core.data import (
    load_price_data_parquet,
    load_index_returns_parquet,
    filter_by_index,
)
from core.features import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_early_momentum,
)
from core.selection import (
    build_daily_lists,
    final_selection_from_daily,
    build_unified_target,
)

# =====================================================
# CONFIG
# =====================================================

ARTIFACTS = ROOT / "artifacts"

PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"

INDEX_NAME = "SP500"

WINDOWS = (5, 10, 30, 45, 60, 90)
DAILY_TOP_N = 10
FINAL_TOP_N = 10
LOOKBACK_DAYS = 10

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

TOTAL_CAPITAL = 100_000
WEIGHT_A = 0.20
WEIGHT_B = 0.80

ARTIFACTS.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

base = load_price_data_parquet(PRICE_PATH)
base = filter_by_index(base, INDEX_NAME)

idx = load_index_returns_parquet(INDEX_PATH)

# =====================================================
# BUCKET A
# =====================================================

dfA = add_absolute_returns(base)
dfA = calculate_momentum_features(dfA, windows=WINDOWS)
dfA = add_regime_momentum_score(dfA)
dfA = add_regime_acceleration(dfA)
dfA = add_regime_early_momentum(dfA)

dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

todayA = final_selection_from_daily(
    dailyA,
    lookback_days=LOOKBACK_DAYS,
    w_momentum=W_MOM,
    w_early=W_EARLY,
    w_consistency=W_CONS,
    top_n=FINAL_TOP_N,
)

todayA.to_parquet(ARTIFACTS / "today_A.parquet", index=False)

# =====================================================
# BUCKET B
# =====================================================

dfB = base.merge(
    idx,
    left_on=["Date", "Index"],
    right_on=["date", "index"],
    how="left",
    validate="many_to_one",
).drop(columns=["date", "index"], errors="ignore")

dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

dfB = add_absolute_returns(dfB)
dfB = calculate_momentum_features(dfB, windows=WINDOWS)
dfB = add_regime_momentum_score(dfB)

dfB = dfB[
    (dfB["Momentum_Slow"] > 0.5)
    & (dfB["Momentum_Mid"] > 0.25)
    & (dfB["Momentum_Fast"] > 0.5)
]

dfB = add_regime_acceleration(dfB)
dfB = add_regime_early_momentum(dfB)

dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

todayB = final_selection_from_daily(
    dailyB,
    lookback_days=LOOKBACK_DAYS,
    w_momentum=W_MOM,
    w_early=W_EARLY,
    w_consistency=W_CONS,
    top_n=FINAL_TOP_N,
)

todayB.to_parquet(ARTIFACTS / "today_B.parquet", index=False)

# =====================================================
# BUCKET C
# =====================================================

todayC = build_unified_target(
    dailyA=dailyA,
    dailyB=dailyB,
    as_of_date=None,
    lookback_days=LOOKBACK_DAYS,
    w_momentum=W_MOM,
    w_early=W_EARLY,
    w_consistency=W_CONS,
    top_n=FINAL_TOP_N,
    total_capital=TOTAL_CAPITAL,
    weight_A=WEIGHT_A,
    weight_B=WEIGHT_B,
)

todayC.to_parquet(ARTIFACTS / "today_C.parquet", index=False)

print("✅ Daily artifacts written")
