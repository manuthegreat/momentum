# updater/run_backtest.py

from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# IMPORT YOUR REAL PIPELINE
# =========================

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
    simulate_single_bucket_as_unified,
    simulate_unified_portfolio,
    compute_performance_stats,
    compute_trade_stats,
)

# =========================
# PATHS & CONSTANTS
# =========================

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

PRICE_PATH = ART / "index_constituents_5yr.parquet"
INDEX_PATH = ART / "index_returns_5y.parquet"

REBALANCE_INTERVAL = 10
DAILY_TOP_N = 10
FINAL_TOP_N = 10
LOOKBACK_DAYS = 10
WINDOWS = (5, 10, 30, 45, 60, 90)

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

TOTAL_CAPITAL = 100_000
DOLLARS_PER_NAME = 5_000


# =========================
# MAIN
# =========================

def main():

    print("📥 Loading data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    idx = load_index_returns_parquet(INDEX_PATH)

    # =====================================================
    # BUCKET A — ABSOLUTE MOMENTUM (UNCHANGED LOGIC)
    # =====================================================

    print("🚀 Running Bucket A backtest...")

    dfA = calculate_momentum_features(
        add_absolute_returns(base),
        windows=WINDOWS
    )
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)

    priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    histA, tradesA = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=priceA,
        daily_df=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=DOLLARS_PER_NAME,
    )

    histA.to_parquet(ART / "history_A.parquet", index=False)

    # =====================================================
    # BUCKET B — RELATIVE MOMENTUM (UNCHANGED LOGIC)
    # =====================================================

    print("🚀 Running Bucket B backtest...")

    dfB = base.copy()

    dfB = dfB.merge(
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
            ["date", "index",
             "idx_5D", "idx_10D",
             "idx_30D", "idx_45D",
             "idx_60D", "idx_90D"]
        ],
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)

    # 🔒 Your triple-gating — untouched
    dfB = dfB[
        (dfB["Momentum_Slow"] > 1) &
        (dfB["Momentum_Mid"] > 0.5) &
        (dfB["Momentum_Fast"] > 1)
    ].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    priceB = dfB.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

    histB, tradesB = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=priceB,
        daily_df=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=DOLLARS_PER_NAME,
    )

    histB.to_parquet(ART / "history_B.parquet", index=False)

    # =====================================================
    # BUCKET C — UNIFIED PORTFOLIO (80/20)
    # =====================================================

    print("🚀 Running Bucket C unified backtest...")

    histC, tradesC = simulate_unified_portfolio(
        df_prices=base,
        price_table=priceA,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=TOTAL_CAPITAL,
    )

    histC.to_parquet(ART / "history_C.parquet", index=False)

    print("✅ Backtest complete. Artifacts written:")
    print("   - history_A.parquet")
    print("   - history_B.parquet")
    print("   - history_C.parquet")


if __name__ == "__main__":
    main()
