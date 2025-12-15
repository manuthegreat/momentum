# updater/run_daily.py

import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_FILE = ARTIFACTS_DIR / "daily_signals.parquet"
BASE_SIGNALS_FILE = ARTIFACTS_DIR / "today_C.parquet"


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize date column.
    Accepts: date, Date, asof
    Outputs: date (datetime64)
    """
    df = df.copy()

    if "date" in df.columns:
        col = "date"
    elif "Date" in df.columns:
        col = "Date"
    elif "asof" in df.columns:
        col = "asof"
    else:
        raise ValueError(f"No date column found. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df[col])
    return df.drop(columns=[c for c in ["Date", "asof"] if c in df.columns])


def main():
    print("Loading daily base signals (Bucket C)...")

    base = pd.read_parquet(BASE_SIGNALS_FILE)
    base = normalize_date_column(base)

    # Enforce schema consistency
    base = base.rename(columns={
        "ticker": "ticker",
        "count": "count",
        "position_size": "position_size",
        "action": "action",
    })

    base = base.sort_values(["date", "ticker"])

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    base.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved daily signals → {OUTPUT_FILE}")
    print(f"Rows: {len(base)} | Date: {base['date'].max().date()}")


if __name__ == "__main__":
    main()
