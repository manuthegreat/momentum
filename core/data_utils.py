import pandas as pd
from pathlib import Path

def load_price_data_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
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
