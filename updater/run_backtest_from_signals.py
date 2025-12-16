# updater/run_backtest_from_signals.py

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from core.data_utils import load_price_data_parquet, filter_by_index
from core.backtest import simulate_unified_portfolio  # C remains unchanged
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

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _to_naive_dt(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def _normalize_price_table(price_table: pd.DataFrame) -> pd.DataFrame:
    pt = price_table.copy()
    idx = pd.to_datetime(pt.index, errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    pt.index = idx
    pt = pt.sort_index()
    return pt


def get_last_price(price_table: pd.DataFrame, ticker: str, asof: pd.Timestamp):
    """Last available price up to and including asof."""
    if ticker not in price_table.columns:
        return None
    s = price_table.loc[:asof, ticker].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def compute_perf_stats_full(history_df: pd.DataFrame, value_col: str) -> dict:
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
    df["Date"] = _to_naive_dt(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if len(df) < 2:
        return base

    start = float(df[value_col].iloc[0])
    end = float(df[value_col].iloc[-1])
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
    Robust trade stats:
    - If 'Action' exists, we compute on sells only (baseline convention)
    - If it doesn't exist, we compute on ALL rows (prevents KeyError)
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

    df = trades_df.copy()

    if "Action" in df.columns:
        df = df[df["Action"].astype(str).str.lower() == "sell"]

    pnl = pd.to_numeric(df["PnL"], errors="coerce").dropna()
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


def write_today(daily_df: pd.DataFrame, out_path: Path):
    if daily_df is None or daily_df.empty or "Date" not in daily_df.columns:
        return
    tmp = daily_df.copy()
    tmp["Date"] = _to_naive_dt(tmp["Date"])
    tmp = tmp.dropna(subset=["Date"])
    if tmp.empty:
        return
    latest = tmp["Date"].max()
    today = tmp[tmp["Date"] == latest].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today.to_parquet(out_path, index=False)


# ============================================================
# BUCKET B: RELATIVE MOMENTUM (CROSS-SECTIONAL RANK)
# ============================================================

def make_relative_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # find the momentum column dynamically
    candidates = [
        "RegimeMomentumScore",
        "Regime_Momentum_Score",
        "MomentumScore",
    ]

    score_col = next((c for c in candidates if c in out.columns), None)

    if score_col is None:
        raise ValueError(
            f"No momentum score column found. "
            f"Available columns: {list(out.columns)}"
        )

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", score_col])

    # Cross-sectional ranking (relative momentum)
    out[score_col] = (
        out.groupby("Date")[score_col]
           .rank(pct=True, ascending=False)   # ← IMPORTANT
    )

    return out


# ============================================================
# A/B BACKTEST (MATCHES YOUR BASELINE STYLE)
# ============================================================

def backtest_single_bucket_like_baseline(
    price_table: pd.DataFrame,
    daily_df: pd.DataFrame,
    rebalance_interval: int,
    capital_per_trade: float = 5_000,
    start_capital: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if daily_df is None or daily_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = daily_df.copy()
    if "Date" not in d.columns or "Ticker" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()

    d["Date"] = _to_naive_dt(d["Date"])
    d = d.dropna(subset=["Date"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    reb_dates = sorted(d["Date"].unique())
    reb_dates = reb_dates[::rebalance_interval]
    if len(reb_dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    first_dt = pd.Timestamp(reb_dates[0])
    first_picks = (
        d.loc[d["Date"] == first_dt, "Ticker"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    capital = float(start_capital) if start_capital is not None else float(capital_per_trade) * float(len(first_picks))
    if capital <= 0:
        capital = float(capital_per_trade) * 10.0

    trades = []
    equity_rows = [{"Date": first_dt, "Portfolio Value": capital, "PnL": 0.0}]

    for i in range(1, len(reb_dates)):
        entry_dt = pd.Timestamp(reb_dates[i - 1])
        exit_dt = pd.Timestamp(reb_dates[i])

        picks = (
            d.loc[d["Date"] == entry_dt, "Ticker"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        period_pnl = 0.0

        for t in picks:
            p0 = get_last_price(price_table, t, entry_dt)
            p1 = get_last_price(price_table, t, exit_dt)
            if p0 is None or p1 is None or p0 == 0:
                continue

            r = (p1 / p0) - 1.0
            pnl = float(capital_per_trade) * float(r)
            period_pnl += pnl

            trades.append(
                {
                    "Entry Date": entry_dt,
                    "Exit Date": exit_dt,
                    "Ticker": t,
                    "Capital": float(capital_per_trade),
                    "Entry Price": float(p0),
                    "Exit Price": float(p1),
                    "Return": float(r),
                    "PnL": float(pnl),
                }
            )

        capital += period_pnl
        equity_rows.append({"Date": exit_dt, "Portfolio Value": float(capital), "PnL": float(period_pnl)})

    equity_df = pd.DataFrame(equity_rows).sort_values("Date").reset_index(drop=True)
    trades_df = pd.DataFrame(trades).sort_values(["Exit Date", "Ticker"]).reset_index(drop=True) if trades else pd.DataFrame()
    return equity_df, trades_df


# ============================================================
# C TRADES (GENERATE BLOTTTER WITHOUT CHANGING CORE)
# ============================================================

def build_unified_trades_over_time(
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    rebalance_interval: int,
    lookback_days: int,
    w_momentum: float,
    w_early: float,
    w_consistency: float,
    top_n: int,
    total_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dates = sorted(price_table.index.dropna().unique())
    rebalance_dates = dates[::rebalance_interval]
    if len(rebalance_dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    trades = []
    capital = float(total_capital)

    equity_rows = [{
        "Date": pd.Timestamp(rebalance_dates[0]),
        "Portfolio Value": capital,
        "PnL": 0.0,
    }]

    for i in range(1, len(rebalance_dates)):
        prev = pd.Timestamp(rebalance_dates[i - 1])
        curr = pd.Timestamp(rebalance_dates[i])

        target = build_unified_target(
            dailyA=dailyA,
            dailyB=dailyB,
            as_of_date=prev,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=capital,
        )

        period_pnl = 0.0

        for _, row in target.iterrows():
            t = str(row["Ticker"])
            alloc = float(row["Capital"])

            p0 = get_last_price(price_table, t, prev)
            p1 = get_last_price(price_table, t, curr)
            if p0 is None or p1 is None or p0 == 0:
                continue

            r = (p1 / p0) - 1.0
            pnl = alloc * r
            period_pnl += pnl

            trades.append(
                {
                    "Entry Date": prev,
                    "Exit Date": curr,
                    "Ticker": t,
                    "Capital": alloc,
                    "Entry Price": float(p0),
                    "Exit Price": float(p1),
                    "Return": float(r),
                    "PnL": float(pnl),
                    "Action": "Sell",
                }
            )

        capital += period_pnl
        equity_rows.append({"Date": curr, "Portfolio Value": float(capital), "PnL": float(period_pnl)})

    equity_df = pd.DataFrame(equity_rows).sort_values("Date").reset_index(drop=True)
    trades_df = pd.DataFrame(trades).sort_values(["Exit Date", "Ticker"]).reset_index(drop=True)
    return equity_df, trades_df


# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("📥 Loading price data...")
    base = load_price_data_parquet(PRICE_PATH)
    base = filter_by_index(base, "SP500")

    price_table = base.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    price_table = _normalize_price_table(price_table)

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
    # Build Bucket B signals (RELATIVE - cross-sectional rank)
    # --------------------------------------------------------
    print("🧮 Building Bucket B signals (relative ranking)...")
    dfB = add_absolute_returns(base)
    dfB = calculate_momentum_features(dfB, windows=WINDOWS)
    dfB = add_regime_momentum_score(dfB)   # SAME as A
    dfB = make_relative_bucket(dfB)        # 🔥 KEY DIFFERENCE
    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=TOP_N)

    # --------------------------------------------------------
    # A backtest + trades (baseline-style)
    # --------------------------------------------------------
    print("📊 Running Bucket A backtest...")
    eqA, trA = backtest_single_bucket_like_baseline(
        price_table=price_table,
        daily_df=dailyA,
        rebalance_interval=REBALANCE_INTERVAL,
        capital_per_trade=5_000,
    )
    statsA = compute_perf_stats_full(eqA, value_col="Portfolio Value")
    tstatsA = compute_trade_stats(trA)

    eqA.to_parquet(EQUITY_A_OUT, index=False)
    trA.to_parquet(TRADES_A_OUT, index=False)
    write_json(STATS_A_OUT, statsA)
    write_json(TRADE_STATS_A_OUT, tstatsA)
    write_today(dailyA, TODAY_A_OUT)

    # --------------------------------------------------------
    # B backtest + trades (baseline-style)
    # --------------------------------------------------------
    print("📊 Running Bucket B backtest...")
    eqB, trB = backtest_single_bucket_like_baseline(
        price_table=price_table,
        daily_df=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        capital_per_trade=5_000,
    )
    statsB = compute_perf_stats_full(eqB, value_col="Portfolio Value")
    tstatsB = compute_trade_stats(trB)

    eqB.to_parquet(EQUITY_B_OUT, index=False)
    trB.to_parquet(TRADES_B_OUT, index=False)
    write_json(STATS_B_OUT, statsB)
    write_json(TRADE_STATS_B_OUT, tstatsB)
    write_today(dailyB, TODAY_B_OUT)

    # --------------------------------------------------------
    # Bucket C unified backtest (UNCHANGED core call)
    # --------------------------------------------------------
    print("📊 Running unified backtest (Bucket C)...")
    equityC_core, target_last = simulate_unified_portfolio(
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

    # --------------------------------------------------------
    # C equity + trades replay (so you get trade stats like baseline)
    # --------------------------------------------------------
    print("📊 Replaying Bucket C targets to generate trades...")
    eqC, trC = build_unified_trades_over_time(
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

    statsC = compute_perf_stats_full(eqC, value_col="Portfolio Value")
    tstatsC = compute_trade_stats(trC)

    eqC.to_parquet(EQUITY_C_OUT, index=False)
    trC.to_parquet(TRADES_C_OUT, index=False)
    write_json(STATS_C_OUT, statsC)
    write_json(TRADE_STATS_C_OUT, tstatsC)

    if isinstance(target_last, pd.DataFrame) and not target_last.empty:
        target_last.to_parquet(TODAY_C_OUT, index=False)

    print("✅ Backtest artifacts written.")


if __name__ == "__main__":
    main()
