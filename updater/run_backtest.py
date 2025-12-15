from pathlib import Path
import pandas as pd

from core.data_utils import (
    load_price_data_parquet,
    filter_by_index,
)

from core.momentum_utils import (
    compute_absolute_momentum,
    compute_relative_momentum,
)

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

PRICE_PATH = ART / "index_constituents_5yr.parquet"
BENCHMARK_TICKER = "^GSPC"

INITIAL_CAPITAL = 1_000_000
WEIGHT_A = 0.5
WEIGHT_B = 0.5


def compute_next_day_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Next-day return to avoid lookahead:
    signal on Date t -> apply return from t to t+1
    """
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["NextReturn"] = df.groupby("Ticker")["Price"].pct_change().shift(-1)
    return df.dropna(subset=["NextReturn"])[["Date", "Ticker", "NextReturn"]]


def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors run_daily.py logic across ALL dates:
      - Bucket A: top 10 absolute momentum each date
      - Bucket B: top 10 relative momentum each date
    """
    # Ensure Price exists
    prices = prices.copy()
    if "Price" not in prices.columns:
        if "Adj Close" in prices.columns:
            prices["Price"] = prices["Adj Close"]
        elif "Close" in prices.columns:
            prices["Price"] = prices["Close"]
        else:
            raise ValueError("No Price/Adj Close/Close column found")

    prices = prices.sort_values(["Ticker", "Date"])

    # Use same index filtering pattern as run_daily.py
    prices = filter_by_index(prices, "SP500")

    bench = prices[prices["Ticker"] == BENCHMARK_TICKER].copy()
    stocks = prices[prices["Ticker"] != BENCHMARK_TICKER].copy()

    if bench.empty:
        raise RuntimeError(
            f"Benchmark ticker {BENCHMARK_TICKER} not found in {PRICE_PATH}. "
            "Either include it in the parquet or change BENCHMARK_TICKER."
        )

    abs_mom = compute_absolute_momentum(stocks)
    rel_mom = compute_relative_momentum(stocks, bench)

    top_abs = (
        abs_mom.sort_values("ret", ascending=False)
        .groupby("Date", as_index=False)
        .head(10)
        .loc[:, ["Date", "Ticker"]]
        .assign(Bucket="A")
    )

    top_rel = (
        rel_mom.sort_values("rel_ret", ascending=False)
        .groupby("Date", as_index=False)
        .head(10)
        .loc[:, ["Date", "Ticker"]]
        .assign(Bucket="B")
    )

    signals = pd.concat([top_abs, top_rel], ignore_index=True)

    # If a ticker appears in both buckets same day, keep both rows (they're separate sleeves)
    return signals


def backtest_bucket(signals: pd.DataFrame, next_returns: pd.DataFrame, capital: float) -> pd.DataFrame:
    df = (
        signals.merge(next_returns, on=["Date", "Ticker"], how="inner")
        .sort_values("Date")
        .copy()
    )

    # Equal-weight within bucket each day
    daily = (
        df.groupby("Date")["NextReturn"]
        .mean()
        .rename("Portfolio_Return")
        .to_frame()
    )

    daily["Portfolio_Value"] = capital * (1 + daily["Portfolio_Return"]).cumprod()
    return daily.reset_index()


def main():
    print("Loading price data...")
    prices = load_price_data_parquet(PRICE_PATH)

    print("Computing next-day returns...")
    next_returns = compute_next_day_returns(prices)

    print("Generating A/B signals (run_daily logic across full history)...")
    signals = generate_signals(prices)

    bucketA = signals[signals["Bucket"] == "A"]
    bucketB = signals[signals["Bucket"] == "B"]

    print("Backtesting Bucket A...")
    histA = backtest_bucket(bucketA, next_returns, INITIAL_CAPITAL * WEIGHT_A)

    print("Backtesting Bucket B...")
    histB = backtest_bucket(bucketB, next_returns, INITIAL_CAPITAL * WEIGHT_B)

    print("Combining Bucket C...")
    histC = histA.merge(histB, on="Date", suffixes=("_A", "_B"))
    histC["Portfolio_Value"] = histC["Portfolio_Value_A"] + histC["Portfolio_Value_B"]
    histC = histC[["Date", "Portfolio_Value"]]

    print("Saving artifacts...")
    histA.to_parquet(ART / "history_A.parquet", index=False)
    histB.to_parquet(ART / "history_B.parquet", index=False)
    histC.to_parquet(ART / "history_C.parquet", index=False)

    print("✅ Backtest complete.")


if __name__ == "__main__":
    main()
