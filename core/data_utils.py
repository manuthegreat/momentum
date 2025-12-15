import pandas as pd
import numpy as np

def load_price_data_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' found in parquet.")

    keep = ["Ticker", "Date", "Price", "Index"]
    df = df[keep].drop_duplicates(subset=["Ticker", "Date"], keep="last")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_index_returns_parquet(path: str) -> pd.DataFrame:
    idx = pd.read_parquet(path)
    idx.columns = idx.columns.str.lower().str.replace(" ", "_")
    idx["date"] = pd.to_datetime(idx["date"])

    if "index_name" in idx.columns:
        idx = idx.rename(columns={"index_name": "index"})

    frames = []
    for _, g in idx.groupby("index"):
        g = g.sort_values("date").copy()
        close = g["close"]
        g["idx_ret_1d"] = close.pct_change()
        frames.append(g[["date", "index", "idx_ret_1d"]])

    return pd.concat(frames, ignore_index=True)
