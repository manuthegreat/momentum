import numpy as np
import pandas as pd


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
