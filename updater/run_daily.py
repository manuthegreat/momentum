import pandas as pd
from pathlib import Path

# ================================
# Paths
# ================================
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

TODAY_PATH = ARTIFACTS / "today_C.parquet"
IDX_RETURNS_PATH = ARTIFACTS / "index_returns_5y.parquet"

OUT_A = ARTIFACTS / "today_A.parquet"
OUT_B = ARTIFACTS / "today_B.parquet"
OUT_C = ARTIFACTS / "today_C.parquet"


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
    print("Loading daily base signals (Bucket C)...")
    base = pd.read_parquet(TODAY_PATH)
    base = normalize_date_column(base)

    print("Loading index returns...")
    idx_returns = pd.read_parquet(IDX_RETURNS_PATH)
    idx_returns = normalize_date_column(idx_returns)

    # -------------------------------
    # Bucket A (raw signals)
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
    # Bucket C (final = same as base for now)
    # -------------------------------
    dfC = base.copy()
    dfC.to_parquet(OUT_C, index=False)
    print(f"Saved {OUT_C}")

    print("✅ Daily run complete")


if __name__ == "__main__":
    main()
