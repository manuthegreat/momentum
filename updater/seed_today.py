import pandas as pd
from pathlib import Path
from datetime import date
import json

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

SRC = ART / "index_constituents_5yr.parquet"

if not SRC.exists():
    raise FileNotFoundError("Missing index_constituents_5yr.parquet")

df = pd.read_parquet(SRC)

if "Price" not in df.columns:
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No Price column found")

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
