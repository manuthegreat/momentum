import pandas as pd
from pathlib import Path
from datetime import date
import json

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

PRICE_PATH = ART / "index_constituents_5yr.parquet"

if not PRICE_PATH.exists():
    raise FileNotFoundError("index_constituents_5yr.parquet not found in artifacts/")

df = pd.read_parquet(PRICE_PATH)

if df.empty:
    raise ValueError("Seed parquet is empty")

# ---- Ensure Date exists ----
if "Date" not in df.columns:
    raise ValueError("Date column missing in seed_today.py input")

df["Date"] = pd.to_datetime(df["Date"])

# ---- Ensure Price exists ----
if "Price" not in df.columns:
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No Price / Adj Close / Close column found")

# ---- Last price per ticker (SAFE) ----
df = df.sort_values(["Ticker", "Date"])
out = (
    df.groupby("Ticker", as_index=False)
      .tail(1)[["Ticker", "Price"]]
      .head(10)
)

out["Position_Size"] = 10_000
out["Action"] = "HOLD"
out["AsOf"] = date.today().isoformat()

out.to_parquet(ART / "today_C.parquet", index=False)

meta = {
    "as_of": date.today().isoformat(),
    "status": "seed",
    "rows": int(len(out))
}

(ART / "metadata.json").write_text(json.dumps(meta, indent=2))
