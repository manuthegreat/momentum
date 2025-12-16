# updater/run_backtest_from_signals.py

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from core.data_utils import load_price_data_parquet, filter_by_index
from core.backtest import (
    simulate_unified_portfolio,          # Bucket C (unchanged)
    simulate_single_bucket_as_unified,   # Bucket A & B (FIX)
)
from core.selection import build_daily_lists, build_unified_target
from core.features import (
    add_absolute_returns,
    calculate_momentum_features,
    add_regime_momentum_score,
    add_regime_acceleration,
    add_regime_early_momentum,
)

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")
PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)

REBALANCE_INTERVAL = 10
LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

# --- Bucket A ---
EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
TRADES_A_OUT = ARTIFACTS / "backtest_trades_A.parquet"
STATS_A_OUT = ARTIFACTS / "backtest_stats_A.json"
TRADE_STATS_A_OUT = ARTIFACTS / "trade_stats_A.json"
TODAY_A_OUT = ARTIFACTS / "today_A.parquet"

# --- Bucket B ---
EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
TRADES_B_OUT = ARTIFACTS / "backtest_trades_B.parquet"
STATS_B_OUT = ARTIFACTS / "backtest_stats_B.json"
TRADE_STATS_B_OUT = ARTIFACTS / "trade_stats_B.json"
TODAY_B_OUT = ARTIFACTS / "today_B.parquet"

# --- Bucket C ---
EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_C_OUT = ARTIFACTS / "backtest_trades_C.parquet"
STATS_C_OUT = ARTIFACTS / "backtest_stats_C.json"
TRADE_STATS_C_OUT = ARTIFACTS / "trade_stats_C.json"
TODAY_C_OUT = ARTIFACTS / "today_C.parquet"

# ============================================================
# HELPERS
# ============================================================

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_perf_stats(history_df: pd.DataFrame) -> dict:
    if history_df.empty or len(history_df) < 2:
        return {
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
        }

    df = history_df.sort_values("Date").copy()
    start = df["Portfolio Value"].iloc[0]
    end = df["Portfolio Value"].iloc[-1]

    ret = df["Portfolio Value"].pct_change().dropna()

    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25

    sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else 0.0
    downside = ret[ret < 0]
    sortino = np.sqrt(252) * ret.mean() / downside.std() if downside.std() > 0 else 0.0

    drawdown = (df["Portfolio Value"] / df["Portfolio Value"].cummax() - 1).min() * 100

    return {
        "Total Return (%)": float((end / start - 1) * 100),
        "CAGR (%)": float(((end / start) ** (1 / years) - 1) * 100) if years > 0 else 0.0,
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Max Drawdown (%)": float(drawdown),
    }


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    sells = trades_df[trades_df["Action"] == "Sell"]
    if sells.empty:
        return {
            "Number of Trades": 0,
            "Win Rate (%)": 0.0,
            "Average Win ($)": 0.0,
            "Average Loss ($)": 0.0,
            "Profit Factor": 0.0,
        }

    wins = sells[sells["PnL"] > 0]
    losses = sells[sells["PnL"] < 0]

    gross_win = wins["PnL"].sum()
    gross_loss = abs(losses["PnL"].sum())

    return {
        "Number of Trades": int(len(sells)),
        "Win Rate (%)": float(len(wins) / len(sells) * 100),
        "Average Win ($)": float(wins["PnL"].mean()) if not wins.empty else 0.0,
        "Average Loss ($)": float(losses["PnL"].mean()) if not losses.empty else 0.0,
        "Profit Factor": float(gross_win / gross_loss) if gross_loss > 0 else 0.0,
    }


def write_today(daily_df: pd.DataFrame, out: Path):
    if daily_df.empty:
        return
    d = daily_df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    latest = d["Date"].max()
    d[d["Date"] == latest].to_parquet(out, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading prices...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    price_table = base.pivot(index="Date", columns="Ticker", values="Price").sort_index()

    # ========================================================
    # BUCKET A — ABSOLUTE MOMENTUM
    # ========================================================

    print("🧮 Bucket A signals...")
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    histA, tradesA = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=price_table,
        daily_df=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=5_000,
    )

    statsA = compute_perf_stats(histA)
    tstatsA = compute_trade_stats(tradesA)

    histA.to_parquet(EQUITY_A_OUT, index=False)
    tradesA.to_parquet(TRADES_A_OUT, index=False)
    write_json(STATS_A_OUT, statsA)
    write_json(TRADE_STATS_A_OUT, tstatsA)
    write_today(dailyA, TODAY_A_OUT)

    # ========================================================
    # BUCKET B — RELATIVE MOMENTUM
    # ========================================================

    print("🧮 Bucket B signals...")
    dfB = dfA.copy()  # RELATIVE logic already upstream in your pipeline
    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    histB, tradesB = simulate_single_bucket_as_unified(
        df_prices=base,
        price_table=price_table,
        daily_df=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
        dollars_per_name=5_000,
    )

    statsB = compute_perf_stats(histB)
    tstatsB = compute_trade_stats(tradesB)

    histB.to_parquet(EQUITY_B_OUT, index=False)
    tradesB.to_parquet(TRADES_B_OUT, index=False)
    write_json(STATS_B_OUT, statsB)
    write_json(TRADE_STATS_B_OUT, tstatsB)
    write_today(dailyB, TODAY_B_OUT)

    # ========================================================
    # BUCKET C — UNIFIED PORTFOLIO (UNCHANGED)
    # ========================================================

    print("📊 Bucket C unified backtest...")
    histC, tradesC = simulate_unified_portfolio(
        df_prices=base,
        price_table=price_table,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=TOP_N,
        total_capital=TOTAL_CAPITAL,
    )

    statsC = compute_perf_stats(histC)
    tstatsC = compute_trade_stats(tradesC)

    histC.to_parquet(EQUITY_C_OUT, index=False)
    tradesC.to_parquet(TRADES_C_OUT, index=False)
    write_json(STATS_C_OUT, statsC)
    write_json(TRADE_STATS_C_OUT, tstatsC)

    if isinstance(tradesC, pd.DataFrame) and not tradesC.empty:
        last_date = histC["Date"].max()
        tgt = build_unified_target(
            dailyA, dailyB,
            as_of_date=last_date,
            lookback_days=LOOKBACK_DAYS,
            w_momentum=W_MOM,
            w_early=W_EARLY,
            w_consistency=W_CONS,
            top_n=TOP_N,
            total_capital=TOTAL_CAPITAL,
        )
        tgt.to_parquet(TODAY_C_OUT, index=False)

    print("✅ All artifacts written successfully.")


if __name__ == "__main__":
    main()
