import pandas as pd
from pathlib import Path

# ================================
# Paths
# ================================
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

HIST_BASE_PATH = ARTIFACTS / "today_C.parquet"
IDX_RETURNS_PATH = ARTIFACTS / "index_returns_5y.parquet"

OUT_A = ARTIFACTS / "history_A.parquet"
OUT_B = ARTIFACTS / "history_B.parquet"
OUT_C = ARTIFACTS / "history_C.parquet"


# ================================
# Helpers
# ================================
def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe has a lowercase 'date' column.
    Accepts date as index or as 'Date' / 'date' column.
    """
    df = df.copy()

    # If index is datetime → reset
    if df.index.name is not None:
        df = df.reset_index()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(f"No date column found. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])
    return df


# ================================
# Main
# ================================
def main():
    print("Loading historical base signals...")
    base = pd.read_parquet(HIST_BASE_PATH)
    base = normalize_date_column(base)

    print("Loading index returns...")
    idx_returns = pd.read_parquet(IDX_RETURNS_PATH)
    idx_returns = normalize_date_column(idx_returns)

    # -------------------------------
    # Bucket A (raw historical signals)
    # -------------------------------
    dfA = base.copy()
    dfA.to_parquet(OUT_A, index=False)
    print(f"Saved {OUT_A}")

    # -------------------------------
    # Bucket B (signals + index returns)
    # -------------------------------
    dfB = base.merge(
        idx_returns,
        on="date",
        how="left",
        suffixes=("", "_index"),
    )
    dfB.to_parquet(OUT_B, index=False)
    print(f"Saved {OUT_B}")

    # -------------------------------
    # Bucket C (final backtest view)
    # -------------------------------
    dfC = dfB.copy()
    dfC.to_parquet(OUT_C, index=False)
    print(f"Saved {OUT_C}")

    print("✅ Backtest run complete")


if __name__ == "__main__":
    main()
