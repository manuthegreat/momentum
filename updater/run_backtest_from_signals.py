# updater/run_backtest_from_signals.py

import json
import math
import pandas as pd
from pathlib import Path

from core.data_utils import load_price_data_parquet, filter_by_index
from core.backtest import (
    simulate_unified_portfolio,  # Bucket C stays exactly as-is
)
from core.selection import build_daily_lists
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

WEIGHT_A = 0.20
WEIGHT_B = 0.80

# --- Bucket A outputs ---
EQUITY_A_OUT = ARTIFACTS / "backtest_equity_A.parquet"
TRADES_A_OUT = ARTIFACTS / "backtest_trades_A.parquet"
STATS_A_OUT = ARTIFACTS / "backtest_stats_A.json"
TRADE_STATS_A_OUT = ARTIFACTS / "trade_stats_A.json"
TODAY_A_OUT = ARTIFACTS / "today_A.parquet"

# --- Bucket B outputs ---
EQUITY_B_OUT = ARTIFACTS / "backtest_equity_B.parquet"
TRADES_B_OUT = ARTIFACTS / "backtest_trades_B.parquet"
STATS_B_OUT = ARTIFACTS / "backtest_stats_B.json"
TRADE_STATS_B_OUT = ARTIFACTS / "trade_stats_B.json"
TODAY_B_OUT = ARTIFACTS / "today_B.parquet"

# --- Bucket C outputs ---
EQUITY_C_OUT = ARTIFACTS / "backtest_equity_C.parquet"
TRADES_C_OUT = ARTIFACTS / "backtest_trades_C.parquet"
STATS_C_OUT = ARTIFACTS / "backtest_stats_C.json"
TRADE_STATS_C_OUT = ARTIFACTS / "trade_stats_C.json"
TODAY_C_OUT = ARTIFACTS / "today_C.parquet"

# ============================================================
# HELPERS
# ============================================================

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and math.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_perf_stats_full(history_df: pd.DataFrame, value_col: str) -> dict:
    """
    Produces keys aligned with your PC baseline:
    Total Return (%), CAGR (%), Sharpe Ratio, Sortino Ratio, Max Drawdown (%)
    """
    base = {
        "Total Return (%)": 0.0,
        "CAGR (%)": 0.0,
        "Sharpe Ratio": 0.0,
        "Sortino Ratio": 0.0,
        "Max Drawdown (%)": 0.0,
    }

    if history_df is None or history_df.empty:
        return base
    if "Date" not in history_df.columns or value_col not in history_df.columns:
        return base

    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if len(df) < 2:
        return base

    start = _safe_float(df[value_col].iloc[0], 0.0)
    end = _safe_float(df[value_col].iloc[-1], 0.0)
    if start <= 0:
        return base

    ret = df[value_col].pct_change().dropna()
    mean = float(ret.mean()) if len(ret) else 0.0
    std = float(ret.std()) if len(ret) else 0.0

    sharpe = (math.sqrt(252) * mean / std) if std > 0 else 0.0

    downside = ret[ret < 0]
    dstd = float(downside.std()) if len(downside) else 0.0
    sortino = (math.sqrt(252) * mean / dstd) if dstd > 0 else 0.0

    roll_max = df[value_col].cummax()
    max_dd = float((df[value_col] / roll_max - 1.0).min() * 100.0)

    total_return = float((end / start - 1.0) * 100.0)

    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    cagr = float(((end / start) ** (365.0 / days) - 1.0) * 100.0) if days and days > 0 else 0.0

    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd,
    }


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    """
    Produces trade stats aligned with your baseline:
    Number of Trades, Win Rate (%), Average Win ($), Average Loss ($), Profit Factor
    """
    base = {
        "Number of Trades": 0,
        "Win Rate (%)": 0.0,
        "Average Win ($)": 0.0,
        "Average Loss ($)": 0.0,
        "Profit Factor": 0.0,
    }

    if trades_df is None or trades_df.empty or "PnL" not in trades_df.columns:
        return base

    pnl = pd.to_numeric(trades_df["PnL"], errors="coerce").dropna()
    if pnl.empty:
        return base

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    num = int(len(pnl))
    win_rate = float((pnl > 0).mean() * 100.0) if num else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float((-losses).sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0.0

    return {
        "Number of Trades": num,
        "Win Rate (%)": win_rate,
        "Average Win ($)": avg_win,
        "Average Loss ($)": avg_loss,
        "Profit Factor": profit_factor,
    }


def simulate_bucket_rebalance_trades(
    price_table: pd.DataFrame,
    daily_list_df: pd.DataFrame,
    capital_per_trade: float,
    rebalance_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A/B backtest:
    - Rebalance every N days
    - Allocate fixed capital_per_trade per selected name
    Outputs:
      equity_df: Date, Portfolio Value
      trades_df: one row per (rebalance, ticker) with entry/exit and PnL
    """
    if daily_list_df is None or daily_list_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ddf = daily_list_df.copy()
    if "Date" not in ddf.columns or "Ticker" not in ddf.columns:
        return pd.DataFrame(), pd.DataFrame()

    ddf["Date"] = pd.to_datetime(ddf["Date"], errors="coerce")
    ddf = ddf.dropna(subset=["Date"])
    if ddf.empty:
        return pd.DataFrame(), pd.DataFrame()

    rebalance_dates = sorted(ddf["Date"].unique())
    rebalance_dates = rebalance_dates[::rebalance_interval]
    if len(rebalance_dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    trades = []
    equity_rows = []
    portfolio_value = 0.0

    for i in range(1, len(rebalance_dates)):
        entry_dt = pd.Timestamp(rebalance_dates[i - 1])
        exit_dt = pd.Timestamp(rebalance_dates[i])

        picks = (
            ddf.loc[ddf["Date"] == entry_dt, "Ticker"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not picks:
            equity_rows.append({"Date": exit_dt, "Portfolio Value": portfolio_value})
            continue

        period_value = 0.0

        for t in picks:
            if t not in price_table.columns:
                continue

            p0 = price_table.loc[:entry_dt, t].dropna()
            p1 = price_table.loc[:exit_dt, t].dropna()
            if p0.empty or p1.empty:
                continue

            entry_px = float(p0.iloc[-1])
            exit_px = float(p1.iloc[-1])
            alloc = float(capital_per_trade)

            r = (exit_px / entry_px - 1.0)
            pnl = alloc * r
            period_value += alloc + pnl

            trades.append(
                {
                    "Entry Date": entry_dt,
                    "Exit Date": exit_dt,
                    "Ticker": t,
                    "Capital": alloc,
                    "Entry Price": entry_px,
                    "Exit Price": exit_px,
                    "Return": r,
                    "PnL": pnl,
                }
            )

        portfolio_value = period_value
        equity_rows.append({"Date": exit_dt, "Portfolio Value": portfolio_value})

    equity_df = pd.DataFrame(equity_rows).sort_values("Date").reset_index(drop=True)
    trades_df = pd.DataFrame(trades).sort_values(["Exit Date", "Ticker"]).reset_index(drop=True)
    return equity_df, trades_df


def write_today(df: pd.DataFrame, out_path: Path):
    """
    Writes the latest Date slice (keeps ALL columns, no schema assumptions).
    """
    if df is None or df.empty or "Date" not in df.columns:
        return

    tmp = df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
    tmp = tmp.dropna(subset=["Date"])
    if tmp.empty:
        return

    latest = tmp["Date"].max()
    today = tmp[tmp["Date"] == latest].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today.to_parquet(out_path, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    # price table for backtests
    price_table = (
        base.pivot(index="Date", columns="Ticker", values="Price")
        .sort_index()
    )

    # --------------------------------------------------------
    # Build Bucket A signals (ABSOLUTE)
    # --------------------------------------------------------
    print("🧮 Building Bucket A signals...")
    dfA = add_absolute_returns(base)
    dfA = calculate_momentum_features(dfA, windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_early_momentum(dfA)

    dailyA = build_daily_lists(dfA, top_n=TOP_N)

    # --------------------------------------------------------
    # Build Bucket B signals (RELATIVE already baked upstream)
    # --------------------------------------------------------
    print("🧮 Building Bucket B signals...")
    dfB = dfA.copy()  # unchanged
    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    # --------------------------------------------------------
    # Backtest A
    # --------------------------------------------------------
    print("📊 Running Bucket A backtest...")
    eqA, trA = simulate_bucket_rebalance_trades(
        price_table=price_table,
        daily_list_df=dailyA[["Date", "Ticker"]].copy() if not dailyA.empty else pd.DataFrame(),
        capital_per_trade=5_000,
        rebalance_interval=REBALANCE_INTERVAL,
    )
    statsA = compute_perf_stats_full(eqA, value_col="Portfolio Value")
    tstatsA = compute_trade_stats(trA)

    eqA.to_parquet(EQUITY_A_OUT, index=False)
    trA.to_parquet(TRADES_A_OUT, index=False)
    write_json(STATS_A_OUT, statsA)
    write_json(TRADE_STATS_A_OUT, tstatsA)
    write_today(dailyA, TODAY_A_OUT)

    # --------------------------------------------------------
    # Backtest B
    # --------------------------------------------------------
    print("📊 Running Bucket B backtest...")
    eqB, trB = simulate_bucket_rebalance_trades(
        price_table=price_table,
        daily_list_df=dailyB[["Date", "Ticker"]].copy() if not dailyB.empty else pd.DataFrame(),
        capital_per_trade=5_000,
        rebalance_interval=REBALANCE_INTERVAL,
    )
    statsB = compute_perf_stats_full(eqB, value_col="Portfolio Value")
    tstatsB = compute_trade_stats(trB)

    eqB.to_parquet(EQUITY_B_OUT, index=False)
    trB.to_parquet(TRADES_B_OUT, index=False)
    write_json(STATS_B_OUT, statsB)
    write_json(TRADE_STATS_B_OUT, tstatsB)
    write_today(dailyB, TODAY_B_OUT)

    # --------------------------------------------------------
    # Bucket C unified backtest (UNCHANGED)
    # --------------------------------------------------------
    print("📊 Running unified backtest (Bucket C)...")
    equityC, tradesC = simulate_unified_portfolio(
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

    # C perf stats using your baseline keys
    if equityC is not None and not equityC.empty:
        if "Portfolio Value" in equityC.columns:
            statsC = compute_perf_stats_full(equityC, value_col="Portfolio Value")
        elif "Equity" in equityC.columns:
            tmp = equityC.rename(columns={"Equity": "Portfolio Value"})
            statsC = compute_perf_stats_full(tmp, value_col="Portfolio Value")
        else:
            statsC = {
                "Total Return (%)": 0.0,
                "CAGR (%)": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "Max Drawdown (%)": 0.0,
            }
    else:
        statsC = {
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
        }

    # C trade stats: only if tradesC has PnL
    tstatsC = compute_trade_stats(tradesC) if isinstance(tradesC, pd.DataFrame) else {
        "Number of Trades": 0,
        "Win Rate (%)": 0.0,
        "Average Win ($)": 0.0,
        "Average Loss ($)": 0.0,
        "Profit Factor": 0.0,
    }

    equityC.to_parquet(EQUITY_C_OUT, index=False)
    if isinstance(tradesC, pd.DataFrame):
        tradesC.to_parquet(TRADES_C_OUT, index=False)
    write_json(STATS_C_OUT, statsC)
    write_json(TRADE_STATS_C_OUT, tstatsC)

    # Optional: if you already generate today_C elsewhere, keep it.
    # If not, we can create it later from your unified target / output artifacts.

    print("✅ Backtest artifacts written:")
    print(f"  • {EQUITY_A_OUT}")
    print(f"  • {TRADES_A_OUT}")
    print(f"  • {STATS_A_OUT}")
    print(f"  • {TRADE_STATS_A_OUT}")
    print(f"  • {TODAY_A_OUT}")
    print(f"  • {EQUITY_B_OUT}")
    print(f"  • {TRADES_B_OUT}")
    print(f"  • {STATS_B_OUT}")
    print(f"  • {TRADE_STATS_B_OUT}")
    print(f"  • {TODAY_B_OUT}")
    print(f"  • {EQUITY_C_OUT}")
    print(f"  • {TRADES_C_OUT}")
    print(f"  • {STATS_C_OUT}")
    print(f"  • {TRADE_STATS_C_OUT}")


if __name__ == "__main__":
    main()
