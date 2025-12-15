import pandas as pd
from pathlib import Path


def load_price_data_parquet(path: str | Path) -> pd.DataFrame:
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


def load_index_returns_parquet(path):
    """
    Load index returns parquet.
    Accepts either:
      - Date as index
      - Date as column
    Always returns Date-indexed DataFrame
    """
    df = pd.read_parquet(path)

    # Case 1: already indexed
    if df.index.name is not None:
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # Case 2: Date column exists
    date_cols = [c for c in df.columns if c.lower() in ("date", "datetime")]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        return df.sort_index()

    # Otherwise → genuinely invalid
    raise ValueError(
        "Index parquet must contain a Date column or indexed dates"
    )


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    return df[df["Index"] == index_name].copy()
