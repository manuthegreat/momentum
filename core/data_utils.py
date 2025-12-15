import pandas as pd
from pathlib import Path


def load_price_data_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load stock-level price data.

    Required columns:
      - Ticker
      - Date
      - Index
      - Price OR Adj Close OR Close
    """
    df = pd.read_parquet(path)

    required = {"Ticker", "Date", "Index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])

    if "Price" not in df.columns:
        if "Adj Close" in df.columns:
            df["Price"] = df["Adj Close"]
        elif "Close" in df.columns:
            df["Price"] = df["Close"]
        else:
            raise ValueError("No Price / Adj Close / Close column found")

    df = (
        df[["Ticker", "Date", "Price", "Index"]]
        .drop_duplicates(["Ticker", "Date"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    return df


def load_index_returns_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load index-level daily returns.

    Required columns (case-insensitive):
      - date
      - index
      - close
    """
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    required = {"date", "index", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required index columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["index", "date"]).reset_index(drop=True)

    # Daily index returns
    df["idx_ret_1d"] = df.groupby("index")["close"].pct_change()

    return df


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Filter universe by index membership.
    """
    return df[df["Index"] == index_name].copy()
