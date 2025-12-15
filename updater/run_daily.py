import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from momentum.core.data_utils import (
    load_price_data_parquet,
    load_index_returns_parquet
)

# ---- IMPORT YOUR EXISTING FUNCTIONS ----
# If your functions live in one file, move them into a module (recommended).
# Example:
# from engine import (
#   load_price_data_parquet, load_index_returns_parquet, filter_by_index,
#   add_absolute_returns, calculate_momentum_features,
#   add_regime_momentum_score, add_regime_acceleration, add_regime_residual_momentum, add_regime_early_momentum,
#   compute_index_momentum, add_relative_regime_momentum_score,
#   build_daily_lists, final_selection_from_daily,
# )
#
# For now, if everything is in the same file, you can inline this runner at the bottom of that file.

# ============================================================
# Utility: get "today" list from final_selection (BUY/HOLD/SELL)
# ============================================================

def today_trades_from_two_buckets(
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    lookback_days: int,
    w_momentum: float,
    w_early: float,
    w_consistency: float,
    top_n: int,
    weight_B: float = 0.80,
    weight_A: float = 0.20,
    dollars_total: float = 100_000.0,
):
    """
    Produces Bucket C "today's trades" with fixed A/B capital split.
    - First compute final lists for A and B for today and yesterday
    - Then size positions according to 80/20 capital split
    - If ticker appears in both, it gets BOTH allocations (adds up)
    """

    # --- sanity ---
    if dailyA.empty and dailyB.empty:
        return pd.DataFrame(columns=["Ticker", "Position_Size", "Action"])

    # Determine today's date from available daily frames
    dates = []
    if not dailyA.empty:
        dates += list(dailyA["Date"].unique())
    if not dailyB.empty:
        dates += list(dailyB["Date"].unique())
    dates = sorted(pd.to_datetime(pd.Series(dates)).unique())
    if len(dates) == 0:
        return pd.DataFrame(columns=["Ticker", "Position_Size", "Action"])

    today = pd.Timestamp(dates[-1])
    prev = pd.Timestamp(dates[-2]) if len(dates) >= 2 else None

    # Helper: get final selection tickers as-of date
    def final_set(daily_df, as_of_date):
        sel = final_selection_from_daily(
            daily_df,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            as_of_date=as_of_date,
            top_n=top_n
        )
        return sel

    selA_today = final_set(dailyA, today) if not dailyA.empty else pd.DataFrame()
    selB_today = final_set(dailyB, today) if not dailyB.empty else pd.DataFrame()

    # If either is empty, still allow the other to dominate
    A_names = list(selA_today["Ticker"]) if not selA_today.empty else []
    B_names = list(selB_today["Ticker"]) if not selB_today.empty else []

    A_cap = dollars_total * weight_A
    B_cap = dollars_total * weight_B

    # Allocate equally within bucket (simple, no overfit)
    A_per = (A_cap / len(A_names)) if len(A_names) > 0 else 0.0
    B_per = (B_cap / len(B_names)) if len(B_names) > 0 else 0.0

    # Build target position sizes (sum if overlap)
    target = {}
    for t in A_names:
        target[t] = target.get(t, 0.0) + A_per
    for t in B_names:
        target[t] = target.get(t, 0.0) + B_per

    target_df = pd.DataFrame(
        [{"Ticker": t, "Position_Size": float(sz)} for t, sz in target.items()]
    ).sort_values("Position_Size", ascending=False).reset_index(drop=True)

    # --- Action labeling (BUY/HOLD/SELL) vs previous target ---
    if prev is not None:
        selA_prev = final_set(dailyA, prev) if not dailyA.empty else pd.DataFrame()
        selB_prev = final_set(dailyB, prev) if not dailyB.empty else pd.DataFrame()

        A_prev = list(selA_prev["Ticker"]) if not selA_prev.empty else []
        B_prev = list(selB_prev["Ticker"]) if not selB_prev.empty else []

        prev_target = {}
        A_prev_per = (A_cap / len(A_prev)) if len(A_prev) > 0 else 0.0
        B_prev_per = (B_cap / len(B_prev)) if len(B_prev) > 0 else 0.0

        for t in A_prev:
            prev_target[t] = prev_target.get(t, 0.0) + A_prev_per
        for t in B_prev:
            prev_target[t] = prev_target.get(t, 0.0) + B_prev_per

        prev_set = set(prev_target.keys())
        today_set = set(target.keys())

        # Actions for today's names
        target_df["Action"] = np.where(
            target_df["Ticker"].isin(prev_set),
            "HOLD",
            "BUY"
        )

        # Add sells (names that were in prev but not now)
        sells = sorted(list(prev_set - today_set))
        if sells:
            sells_df = pd.DataFrame({
                "Ticker": sells,
                "Position_Size": [0.0] * len(sells),
                "Action": ["SELL"] * len(sells)
            })
            target_df = pd.concat([target_df, sells_df], ignore_index=True)

    else:
        target_df["Action"] = "BUY"

    return target_df, today


def main():
    # Paths inside repo (GitHub Actions + Streamlit use these)
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    artifacts_dir = os.path.abspath(artifacts_dir)

    price_path = os.path.join(artifacts_dir, "index_constituents_5yr.parquet")
    index_path = os.path.join(artifacts_dir, "index_returns_5y.parquet")

    out_today_path = os.path.join(artifacts_dir, "today_C.parquet")
    out_meta_path = os.path.join(artifacts_dir, "metadata.json")

    # ---- YOUR KNOBS ----
    REBALANCE_INTERVAL = 10
    DAILY_TOP_N = 10
    FINAL_TOP_N = 10
    LOOKBACK_DAYS = 10
    WINDOWS = (5, 10, 30, 45, 60, 90)

    W_MOM = 0.50
    W_EARLY = 0.30
    W_CONS = 0.20

    TOTAL_CAPITAL = 100_000.0
    WEIGHT_B = 0.80
    WEIGHT_A = 0.20

    # ---- Load base data ----
    base = load_price_data_parquet(price_path)
    base = filter_by_index(base, "SP500")
    idx = load_index_returns_parquet(index_path)

    # ---- Build Bucket A daily list ----
    dfA = calculate_momentum_features(add_absolute_returns(base), windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)
    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    # ---- Build Bucket B daily list ----
    dfB = base.copy()
    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
        validate="many_to_one"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

    dfB = calculate_momentum_features(add_absolute_returns(dfB), windows=WINDOWS)

    idx_mom = compute_index_momentum(idx, windows=WINDOWS)
    dfB = dfB.merge(
        idx_mom[["date","index","idx_5D","idx_10D","idx_30D","idx_45D","idx_60D","idx_90D"]],
        left_on=["Date","Index"],
        right_on=["date","index"],
        how="left"
    ).drop(columns=["date","index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)
    dfB = dfB[dfB["Momentum_Slow"] > 1].copy()
    dfB = dfB[dfB["Momentum_Mid"] > 0.5].copy()
    dfB = dfB[dfB["Momentum_Fast"] > 1].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

    # ---- Compute Bucket C today trades (80/20) ----
    todayC, as_of_date = today_trades_from_two_buckets(
        dailyA=dailyA,
        dailyB=dailyB,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        weight_B=WEIGHT_B,
        weight_A=WEIGHT_A,
        dollars_total=TOTAL_CAPITAL
    )

    # Persist minimal artifact for Streamlit
    todayC.to_parquet(out_today_path, index=False)

    meta = {
        "as_of_date": str(as_of_date.date()),
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_capital": TOTAL_CAPITAL,
        "weight_B": WEIGHT_B,
        "weight_A": WEIGHT_A,
        "rows": int(len(todayC)),
    }
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", out_today_path)
    print("Wrote:", out_meta_path)
    print(todayC)


if __name__ == "__main__":
    main()
