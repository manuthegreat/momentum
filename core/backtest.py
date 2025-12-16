import numpy as np
import pandas as pd
from core.selection import build_unified_target

def get_last_price(price_table, ticker, date):
    try:
        s = price_table.loc[:date, ticker].dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None


def simulate_single_bucket(
    price_table,
    daily_df,
    capital_per_trade=5_000,
    rebalance_interval=10,
):
    dates = sorted(daily_df["Date"].unique())[::rebalance_interval]
    portfolio = {}
    history = []

    for d in dates:
        picks = daily_df[daily_df["Date"] == d]["Ticker"]

        portfolio = {}
        value = 0.0

        for t in picks:
            px = get_last_price(price_table, t, d)
            if px:
                shares = capital_per_trade / px
                portfolio[t] = shares
                value += shares * px

        history.append({"Date": d, "Portfolio Value": value})

    return pd.DataFrame(history)

def simulate_single_bucket_as_unified(
    df_prices,
    price_table,
    daily_df,
    rebalance_interval,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital,
    dollars_per_name,
):
    """
    Single-bucket backtest that mirrors the unified capital mechanics.
    This is what your PC baseline effectively does for A and B.
    """

    dates = sorted(price_table.index.unique())
    rebalance_dates = dates[::rebalance_interval]

    equity = []
    trades = []
    capital = total_capital

    for i in range(1, len(rebalance_dates)):
        prev = rebalance_dates[i - 1]
        curr = rebalance_dates[i]

        picks = daily_df[daily_df["Date"] == prev]["Ticker"].dropna().unique().tolist()
        if not picks:
            equity.append({"Date": curr, "Portfolio Value": capital})
            continue

        period_pnl = 0.0

        for t in picks:
            if t not in price_table.columns:
                continue

            p0 = price_table.loc[prev, t]
            p1 = price_table.loc[curr, t]

            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                continue

            alloc = dollars_per_name
            r = (p1 / p0 - 1.0)
            pnl = alloc * r
            period_pnl += pnl

            trades.append({
                "Entry Date": prev,
                "Exit Date": curr,
                "Ticker": t,
                "Capital": alloc,
                "Entry Price": float(p0),
                "Exit Price": float(p1),
                "Return": float(r),
                "PnL": float(pnl),
                "Action": "Sell",
            })

        capital += period_pnl
        equity.append({"Date": curr, "Portfolio Value": capital})

    return pd.DataFrame(equity), pd.DataFrame(trades)


def compute_performance_stats(history_df):
    if len(history_df) < 2:
        return {}

    start = history_df["Portfolio Value"].iloc[0]
    end = history_df["Portfolio Value"].iloc[-1]

    ret = history_df["Portfolio Value"].pct_change().dropna()

    return {
        "Total Return (%)": (end / start - 1) * 100,
        "Sharpe": np.sqrt(252) * ret.mean() / ret.std() if ret.std() else 0.0,
        "Max Drawdown (%)": (
            (history_df["Portfolio Value"] /
             history_df["Portfolio Value"].cummax() - 1).min() * 100
        )
    }

def simulate_unified_portfolio(
    df_prices,
    price_table,
    dailyA,
    dailyB,
    rebalance_interval=10,
    lookback_days=10,
    w_momentum=0.5,
    w_early=0.3,
    w_consistency=0.2,
    top_n=10,
    total_capital=100_000,
):
    """
    Unified backtest: recompute combined portfolio at each rebalance date
    """

    dates = sorted(price_table.index.unique())
    rebalance_dates = dates[::rebalance_interval]

    equity = []
    capital = total_capital

    target = pd.DataFrame()

    for i in range(1, len(rebalance_dates)):
        prev = rebalance_dates[i - 1]
        curr = rebalance_dates[i]

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

        pnl = 0.0

        for _, row in target.iterrows():
            t = row["Ticker"]
            alloc = row["Capital"]

            if t not in price_table.columns:
                continue

            p0 = price_table.loc[prev, t]
            p1 = price_table.loc[curr, t]

            if pd.isna(p0) or pd.isna(p1):
                continue

            pnl += alloc * (p1 / p0 - 1)

        capital += pnl

        equity.append(
            {
                "Date": curr,
                "Portfolio Value": capital,
                "PnL": pnl,
            }
        )
        
    equity_df = pd.DataFrame(equity).sort_values("Date").reset_index(drop=True)
    return pd.DataFrame(equity), target
