# updater/run_backtest.py

import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

BASE_SIGNALS_FILE = ARTIFACTS_DIR / "today_C.parquet"
OUTPUT_FILE = ARTIFACTS_DIR / "backtest_signals.parquet"


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
    print("Loading historical base signals...")

    base = pd.read_parquet(BASE_SIGNALS_FILE)
    base = normalize_date_column(base)

    # Sort for backtest usage
    base = base.sort_values(["date", "ticker"])

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    base.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved backtest signals → {OUTPUT_FILE}")
    print(
        f"Rows: {len(base)} | "
        f"From {base['date'].min().date()} "
        f"To {base['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
