# updater/run_backtest.py

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from core.data_utils import (
    load_price_data_parquet,
    load_index_returns_parquet,
)

from core.momentum_utils import (
    calculate_momentum_features,
    build_bucket_A,
    build_bucket_B,
    simulate_bucket_A,
    simulate_bucket_B,
    simulate_bucket_C,
    compute_performance_stats,
    compute_trade_stats,
)

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def main():

    print("🔹 Loading data...")
    prices = load_price_data_parquet(ART / "index_constituents_5yr.parquet")
    index_returns = load_index_returns_parquet(ART / "index_returns_5y.parquet")

    print("🔹 Calculating momentum features...")
    features = calculate_momentum_features(prices)

    print("🔹 Building buckets...")
    bucketA = build_bucket_A(features)
    bucketB = build_bucket_B(features)

    print("🔹 Running backtests...")
    histA, tradesA = simulate_bucket_A(bucketA, prices)
    histB, tradesB = simulate_bucket_B(bucketB, prices)
    histC, tradesC, todayC = simulate_bucket_C(bucketA, bucketB, prices)

    print("🔹 Computing stats...")
    statsA = compute_performance_stats(histA)
    statsB = compute_performance_stats(histB)
    statsC = compute_performance_stats(histC)

    trade_statsA = compute_trade_stats(tradesA)
    trade_statsB = compute_trade_stats(tradesB)
    trade_statsC = compute_trade_stats(tradesC)

    print("🔹 Saving artifacts...")
    histA.to_parquet(ART / "history_A.parquet", index=False)
    histB.to_parquet(ART / "history_B.parquet", index=False)
    histC.to_parquet(ART / "history_C.parquet", index=False)

    todayC.to_parquet(ART / "today_C.parquet", index=False)

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "A": statsA,
        "B": statsB,
        "C": statsC,
    }

    pd.Series(summary).to_json(ART / "backtest_summary.json", indent=2)

    print("✅ Backtest complete")

if __name__ == "__main__":
    main()
