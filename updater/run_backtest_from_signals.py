# updater/run_backtest_from_signals.py
# Thin wrapper around PIPELINE CODE — NO REIMPLEMENTATION

import json
from pathlib import Path
import pandas as pd

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
    add_relative_regime_momentum_score,
    compute_index_momentum,
)

from core.selection import build_daily_lists
from core.backtest import (
    simulate_single_bucket_as_unified,
    simulate_unified_portfolio,
)

# ============================================================
# CONFIG (unchanged)
# ============================================================

ARTIFACTS = Path("artifacts")
PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

# Outputs
EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
TRADES_A_OUT = ARTIFACTS / "backtest_trades_A.parquet"
STATS_A_OUT = ARTIFACTS / "backtest_stats_A.json"
TODAY_A_OUT = ARTIFACTS / "today_A.parquet"

EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
TRADES_B_OUT = ARTIFACTS / "backtest_trades_B.parquet"
STATS_B_OUT = ARTIFACTS / "backtest_stats_B.json"
TODAY_B_OUT = ARTIFACTS / "today_B.parquet"

EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"
STATS_C_OUT = ARTIFACTS / "backtest_stats_C.json"
TODAY_C_OUT = ARTIFACTS / "today_C.parquet"


# ============================================================
# HELPERS
# ============================================================

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_today(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return
    last_date = df["Date"].max()
    df[df["Date"] == last_date].to_parquet(path, index=False)


# ============================================================
# MAIN — PIPELINE PARITY
# ============================================================

def main():
    ARTIFACTS.mkdir(exist_ok=True)

    # ---------- Load data ----------
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    idx = load_index_returns_parquet(INDEX_PATH)

    price_table = base.pivot(index="Date", columns="Ticker", values="Price").sort_index()

    # ---------- BUCKET A ----------
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    eqA, trA = simulate_single_bucket_as_unified(
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

    eqA.to_parquet(EQUITY_A_OUT, index=False)
    trA.to_parquet(TRADES_A_OUT, index=False)
    write_json(STATS_A_OUT, eqA.attrs.get("stats", {}))
    write_today(dailyA, TODAY_A_OUT)

    # ---------- BUCKET B ----------
    dfB = base.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()
    dfB = add_absolute_returns(dfB)
    dfB = calculate_momentum_features(dfB, WINDOWS)

    idx_mom = compute_index_momentum(idx, WINDOWS)
    dfB = dfB.merge(
        idx_mom,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)

    # YOUR regime filters — preserved
    dfB = dfB[
        (dfB["Momentum_Slow"] > 1.0) &
        (dfB["Momentum_Mid"] > 0.5) &
        (dfB["Momentum_Fast"] > 1.0)
    ]

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    eqB, trB = simulate_single_bucket_as_unified(
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

    eqB.to_parquet(EQUITY_B_OUT, index=False)
    trB.to_parquet(TRADES_B_OUT, index=False)
    write_json(STATS_B_OUT, eqB.attrs.get("stats", {}))
    write_today(dailyB, TODAY_B_OUT)

    # ---------- BUCKET C ----------
    eqC, target_last = simulate_unified_portfolio(
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

    eqC.to_parquet(EQUITY_C_OUT, index=False)
    write_json(STATS_C_OUT, eqC.attrs.get("stats", {}))

    if isinstance(target_last, pd.DataFrame):
        target_last.to_parquet(TODAY_C_OUT, index=False)

    print("✅ GitHub backtest is now pipeline-identical.")


if __name__ == "__main__":
    main()
