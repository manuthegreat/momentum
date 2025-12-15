import pandas as pd
from pathlib import Path
from datetime import date

from momentum.core.data_utils import (
    load_price_data_parquet,
    filter_by_index,
)

from momentum.core.momentum_utils import (
    compute_absolute_momentum,
    compute_relative_momentum,
)

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

PRICE_PATH = ART / "index_constituents_5yr.parquet"
BENCHMARK_TICKER = "^GSPC"

def main():
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    # --- Benchmark ---
    bench = base[base["Ticker"] == BENCHMARK_TICKER].copy()
    stocks = base[base["Ticker"] != BENCHMARK_TICKER].copy()

    abs_mom = compute_absolute_momentum(stocks)
    rel_mom = compute_relative_momentum(stocks, bench)

    # --- Top selections ---
    top_abs = (
        abs_mom.sort_values("ret", ascending=False)
        .groupby("Date")
        .head(10)
    )

    top_rel = (
        rel_mom.sort_values("rel_ret", ascending=False)
        .groupby("Date")
        .head(10)
    )

    # --- Today snapshot (Bucket C logic: 80/20 fixed) ---
    today = abs_mom["Date"].max()

    a_today = top_abs[top_abs["Date"] == today][["Ticker"]]
    b_today = top_rel[top_rel["Date"] == today][["Ticker"]]

    out = (
        pd.concat([a_today.assign(Bucket="A"),
                   b_today.assign(Bucket="B")])
        .value_counts(["Ticker"])
        .reset_index(name="Count")
    )

    out["Position_Size"] = out["Count"].map({1: 2000, 2: 10000})
    out["Action"] = "HOLD"
    out["AsOf"] = date.today().isoformat()

    out.to_parquet(ART / "today_C.parquet", index=False)

    print("✅ today_C.parquet written")

if __name__ == "__main__":
    main()
