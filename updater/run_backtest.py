# updater/run_backtest.py

from pathlib import Path
import pandas as pd
import numpy as np

from core.data_utils import (
    load_price_data_parquet,
)
from core.momentum_utils import (
    calculate_momentum_features,
    add_absolute_returns,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_residual_momentum,
    add_regime_early_momentum,
    build_daily_lists,
    final_selection_from_daily,
)


ART = Path("artifacts")
ART.mkdir(exist_ok=True)

INITIAL_CAPITAL = 1_000_000
WEIGHT_A = 0.5
WEIGHT_B = 0.5


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily returns per ticker
    """
    df = price_df.sort_values(["Ticker", "Date"]).copy()
    df["Return"] = df.groupby("Ticker")["Price"].pct_change()
    return df.dropna(subset=["Return"])


def backtest_bucket(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Equal-weight daily rebalance backtest for one bucket
    """

    df = (
        signals.merge(
            returns[["Date", "Ticker", "Return"]],
            on=["Date", "Ticker"],
            how="inner",
        )
        .sort_values("Date")
    )

    daily_returns = (
        df.groupby("Date")["Return"]
        .mean()
        .rename("Portfolio_Return")
        .to_frame()
    )

    daily_returns["Portfolio_Value"] = capital * (
        1 + daily_returns["Portfolio_Return"]
    ).cumprod()

    daily_returns = daily_returns.reset_index()
    return daily_returns


def main():

    print("Loading price data...")
    prices = load_price_data_parquet(ART / "index_constituents_5yr.parquet")

    prices = prices.rename(columns={"Close": "Price"}) if "Close" in prices else prices
    assert "Price" in prices.columns, "Price column missing"

    returns = compute_daily_returns(prices)

    print("Generating daily signals...")
    signals = generate_daily_signals(prices)

    bucketA = signals[signals["Bucket"] == "A"]
    bucketB = signals[signals["Bucket"] == "B"]

    print("Running backtest for Bucket A...")
    histA = backtest_bucket(
        bucketA,
        returns,
        capital=INITIAL_CAPITAL * WEIGHT_A,
    )

    print("Running backtest for Bucket B...")
    histB = backtest_bucket(
        bucketB,
        returns,
        capital=INITIAL_CAPITAL * WEIGHT_B,
    )

    print("Combining Bucket C (A + B)...")
    histC = (
        histA[["Date", "Portfolio_Value"]]
        .merge(
            histB[["Date", "Portfolio_Value"]],
            on="Date",
            suffixes=("_A", "_B"),
        )
    )

    histC["Portfolio_Value"] = (
        histC["Portfolio_Value_A"] + histC["Portfolio_Value_B"]
    )

    histC = histC[["Date", "Portfolio_Value"]]

    print("Saving artifacts...")
    histA.to_parquet(ART / "history_A.parquet", index=False)
    histB.to_parquet(ART / "history_B.parquet", index=False)
    histC.to_parquet(ART / "history_C.parquet", index=False)

    print("Backtest complete.")


if __name__ == "__main__":
    main()
