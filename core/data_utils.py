import pandas as pd
from pathlib import Path

def load_price_data_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = {"Ticker", "Date"}
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

    if "Index" not in df.columns:
        raise ValueError("Index column required for momentum pipeline")

    return (
        df[["Ticker", "Date", "Price", "Index"]]
        .drop_duplicates(["Ticker", "Date"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

def load_index_returns_parquet(path: str) -> pd.DataFrame:
    """
    Load index returns from a parquet file.

    Expected columns (flexible):
      - Date (or date)
      - Ticker / Index / Symbol (one of these)
      - Return / Returns (or a numeric column you use downstream)

    Returns a cleaned DataFrame sorted by [Ticker, Date] when possible.
    """
    df = pd.read_parquet(path)

    # Normalize column names gently
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Date handling
    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        raise ValueError(f"Expected a Date column in {path}. Found: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Try to sort sensibly if there is a ticker-like column
    ticker_col = None
    for c in ["Ticker", "Index", "Symbol", "ticker", "index", "symbol"]:
        if c in df.columns:
            ticker_col = c
            break

    if ticker_col:
        df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)
    else:
        df = df.sort_values([date_col]).reset_index(drop=True)

    # Standardize to Date if downstream expects it
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    return df

def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Filter universe by index membership.
    Assumes `Index` column exists.
    """
    if "Index" not in df.columns:
        return df  # safe no-op
    return df[df["Index"] == index_name].copy()
