import pandas as pd
from pathlib import Path

from core.data_utils import (
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
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading historical price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    print("📥 Loading index returns...")
    idx = load_index_returns_parquet(INDEX_PATH)

    print("Index returns columns (before normalization):", list(idx.columns))

    # --------------------------------------------------------
    # Normalize historical index returns schema
    # (match live daily pipeline exactly)
    # --------------------------------------------------------
    
    if "index" not in idx.columns:
        if "Index" in idx.columns:
            idx = idx.rename(columns={"Index": "index"})
        elif "Ticker" in idx.columns:
            idx = idx.rename(columns={"Ticker": "index"})
        else:
            raise ValueError(
                f"Index returns parquet must contain an index identifier. "
                f"Found columns: {list(idx.columns)}"
            )
    
    if "date" not in idx.columns and "Date" in idx.columns:
        idx = idx.rename(columns={"Date": "date"})
    
    print("Index returns columns (after normalization):", list(idx.columns))


    # ========================================================
    # BUCKET A — ABSOLUTE MOMENTUM
    # ========================================================

    print("🧮 Computing Bucket A features...")
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    # ========================================================
    # BUCKET B — RELATIVE MOMENTUM
    # ========================================================

    print("🧮 Computing Bucket B features...")
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
    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

    # ========================================================
    # BUILD HISTORICAL BUCKET C
    # ========================================================

    print("🧱 Building historical Bucket C selections...")

    all_dates = sorted(
        set(dailyA["Date"]).intersection(dailyB["Date"])
    )

    records = []

    for d in all_dates:
        target = build_unified_target(
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

        if target.empty:
            continue

        target = target.copy()
        target["Date"] = d
        target["Bucket"] = "C"

        records.append(target)

    if not records:
        raise RuntimeError("No historical Bucket C records generated.")

    history = pd.concat(records, ignore_index=True)
    history = history.sort_values(
        ["Date", "Position_Size"],
        ascending=[True, False],
    )

    history.to_parquet(OUT_PATH, index=False)

    print(f"✅ Historical backtest signals written → {OUT_PATH}")
    print(f"📊 Rows: {len(history)}")
    print(
        f"📅 From {history['Date'].min().date()} "
        f"to {history['Date'].max().date()}"
    )


if __name__ == "__main__":
    main()
