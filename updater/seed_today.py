import pandas as pd
from pathlib import Path
import json
from datetime import datetime

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

df = pd.read_parquet(ART / "index_constituents_5yr.parquet")

# Dummy "today" output
out = (
    df.sort_values("Date")
      .groupby("Ticker")
      .tail(1)[["Ticker", "Price"]]
      .head(10)
)

out["Position_Size"] = 10_000
out["Action"] = "HOLD"

out.to_parquet(ART / "today_C.parquet", index=False)

meta = {
    "as_of": datetime.utcnow().isoformat(),
    "weights": {"A": 0.2, "B": 0.8},
    "status": "seed"
}

(ART / "metadata.json").write_text(json.dumps(meta, indent=2))
