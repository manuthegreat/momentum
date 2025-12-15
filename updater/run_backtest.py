from pathlib import Path
import pandas as pd
import numpy as np

from core.data_utils import load_price_data_parquet
from core.momentum_utils import (
    build_daily_lists,
    final_selection_from_daily,
)

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

INITIAL_CAPITAL = 1_000_000
WEIGHT_A = 0.5
WEIGHT_B = 0.5


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Return"] = df.groupby("Ticker")["Price"].pct_change()
    return df.dropna(subset=["Return"])


def generate_signals_from_existing_logic(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Uses EXISTING momentum_utils logic.
    No new functions. No refactors.
    """

    all_days = []

    for dt, day_df in prices.groupby("Date"):
        daily_lists = build_daily_lists(day_df)
        final = final_selection_from_daily(daily_lists)

        if final is None or final.empty:
            continue

        final = final.copy()
        final["Date"] = dt
        all_days.append(final)

    if not all_days:
        raise RuntimeError("No signals generated in backtest")

    return pd.concat(all_days, ignore_index=True)


def backtest_bucket(signals: pd.DataFrame, returns: pd.DataFrame, capital: float):
    df = (
        signals.merge(
            returns[["Date", "Ticker", "Return"]],
            on=["Date", "Ticker"],
            how="inner",
        )
        .sort_values("Date")
    )

    daily = (
        df.groupby("Date")["Return"]
        .mean()
        .rename("Portfolio_Return")
        .to_frame()
    )

    daily["Portfolio_Value"] = capital * (1 + daily["Portfolio_Return"]).cumprod()
    return daily.reset_index()


def main():

    print("Loading price data...")
    prices = load_price_data_parquet(ART / "index_constituents_5yr.parquet")

    if "Close" in prices.columns and "Price" not in prices.columns:
        prices["Price"] = prices["Close"]

    assert {"Date", "Ticker", "Price"}.issubset(prices.columns)

    print("Computing returns...")
    returns = compute_daily_returns(prices)

    print("Generating signals using EXISTING logic...")
    signals = generate_signals_from_existing_logic(prices)

    bucketA = signals[signals["Bucket"] == "A"]
    bucketB = signals[signals["Bucket"] == "B"]

    print("Backtesting Bucket A...")
    histA = backtest_bucket(bucketA, returns, INITIAL_CAPITAL * WEIGHT_A)

    print("Backtesting Bucket B...")
    histB = backtest_bucket(bucketB, returns, INITIAL_CAPITAL * WEIGHT_B)

    print("Combining Bucket C...")
    histC = (
        histA.merge(histB, on="Date", suffixes=("_A", "_B"))
    )
    histC["Portfolio_Value"] = (
        histC["Portfolio_Value_A"] + histC["Portfolio_Value_B"]
    )
    histC = histC[["Date", "Portfolio_Value"]]

    print("Saving artifacts...")
    histA.to_parquet(ART / "history_A.parquet", index=False)
    histB.to_parquet(ART / "history_B.parquet", index=False)
    histC.to_parquet(ART / "history_C.parquet", index=False)

    print("✅ Backtest complete.")


if __name__ == "__main__":
    main()
