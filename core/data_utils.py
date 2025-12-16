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


def load_index_returns_parquet(path: Path) -> pd.DataFrame:
    """
    Load index daily returns and normalize schema ONCE.
    Required output columns:
      - date
      - index
      - idx_ret_1d
    """
    df = pd.read_parquet(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- normalize date ---
    if "date" not in df.columns:
        raise ValueError(f"[IDX] No 'date' column found. Columns: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # --- normalize index name ---
    if "index" not in df.columns:
        # common real-world variants
        candidates = [
            "benchmark",
            "index_name",
            "symbol",
            "ticker",
            "sp500",
            "sp_500",
        ]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"[IDX] No index identifier column found. "
                f"Expected one of {candidates}. Columns: {list(df.columns)}"
            )
        df = df.rename(columns={found: "index"})

    # --- normalize daily return ---
    if "idx_ret_1d" not in df.columns:
        candidates = [
            "return_1d",
            "ret_1d",
            "daily_return",
            "pct_return",
        ]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"[IDX] No daily return column found. "
                f"Expected one of {candidates}. Columns: {list(df.columns)}"
            )
        df = df.rename(columns={found: "idx_ret_1d"})

    # final contract enforcement
    required = {"date", "index", "idx_ret_1d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[IDX] Missing required index columns after normalization: {missing}")

    return df.sort_values(["index", "date"]).reset_index(drop=True)


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Filter universe by index membership.
    """
    return df[df["Index"] == index_name].copy()
