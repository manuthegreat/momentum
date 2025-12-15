import pandas as pd
from pathlib import Path

def load_price_data_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalize date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # 🔑 Normalize price column ONCE
    if "Price" not in df.columns:
        if "Adj Close" in df.columns:
            df["Price"] = df["Adj Close"]
        elif "Close" in df.columns:
            df["Price"] = df["Close"]
        else:
            raise ValueError(
                "No Price / Adj Close / Close column found in parquet"
            )

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df

def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Filter universe by index membership.
    Assumes `Index` column exists.
    """
    if "Index" not in df.columns:
        return df  # safe no-op
    return df[df["Index"] == index_name].copy()
