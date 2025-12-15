import pandas as pd
from pathlib import Path
from datetime import date
import json

# Path to raw universe parquet
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

df = pd.read_parquet(ART / "index_constituents_5yr.parquet")

# Ensure a Price column exists
if "Price" not in df.columns:
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No Price / Adj Close / Close column found in seed_today.py")

# Pick last price per ticker as dummy fallback
df = df.sort_values("Date")
out = df.groupby("Ticker").tail(1)[["Ticker", "Price"]].head(10)

out["Position_Size"] = 10000
out["Action"] = "HOLD"
out["AsOf"] = date.today().isoformat()

out.to_parquet(ART / "today_C.parquet", index=False)

meta = {
    "as_of": date.today().isoformat(),
    "status": "seed",
    "rows": len(out)
}

(ART / "metadata.json").write_text(json.dumps(meta, indent=2))
