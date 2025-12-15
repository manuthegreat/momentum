import pandas as pd
import numpy as np

def compute_absolute_momentum(df: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = (
        df.groupby("Ticker")["Price"]
        .pct_change(lookback)
    )
    return df.dropna(subset=["ret"])


def compute_relative_momentum(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback: int = 126
) -> pd.DataFrame:

    s = compute_absolute_momentum(stock_df, lookback)
    b = compute_absolute_momentum(benchmark_df, lookback)

    b = b.rename(columns={"ret": "bench_ret"})[["Date", "bench_ret"]]

    out = s.merge(b, on="Date", how="inner")
    out["rel_ret"] = out["ret"] - out["bench_ret"]

    return out.dropna(subset=["rel_ret"])
