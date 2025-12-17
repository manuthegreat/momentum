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
    total_capital=100_000.0,
    dollars_per_name=5_000.0
):
    cash = total_capital
    portfolio = {}        # ticker -> shares
    cost_basis = {}       # ticker -> invested dollars

    history = []
    trades = []

    rebalance_dates = sorted(df_prices["Date"].unique())[::rebalance_interval]

    for d in rebalance_dates:

        # ---- build target ----
        sel = final_selection_from_daily(
            daily_df,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            as_of_date=d,
            top_n=top_n,
        )

        target = set(sel["Ticker"]) if not sel.empty else set()
        target_sizes = {t: dollars_per_name for t in target}
        held = set(portfolio)

        # ---- SELL removed ----
        for t in held - target:
            px = get_last_price(price_table, t, d)
            if px is not None:
                shares = portfolio[t]
                proceeds = shares * px
                pnl = proceeds - cost_basis[t]
                cash += proceeds
                trades.append(
                    {"Date": d, "Ticker": t, "Action": "Sell", "Price": px, "PnL": pnl}
                )
            portfolio.pop(t, None)
            cost_basis.pop(t, None)

        # ---- BUY / RESIZE ----
        for t in target:
            px = get_last_price(price_table, t, d)
            if px is None:
                continue

            target_value = target_sizes[t]
            current_shares = portfolio.get(t, 0.0)
            current_value = current_shares * px
            delta = target_value - current_value

            if delta > 0 and delta <= cash:
                shares = delta / px
                portfolio[t] = current_shares + shares
                cost_basis[t] = cost_basis.get(t, 0.0) + delta
                cash -= delta
                trades.append(
                    {"Date": d, "Ticker": t, "Action": "Buy", "Price": px, "PnL": 0.0}
                )

            elif delta < 0:
                trim_value = -delta
                trim_shares = trim_value / px
                portfolio[t] = current_shares - trim_shares
                cost_basis[t] -= trim_value
                cash += trim_value
                trades.append(
                    {"Date": d, "Ticker": t, "Action": "Resize", "Price": px, "PnL": 0.0}
                )

        # ---- ALWAYS MARK TO MARKET (THIS IS THE FIX) ----
        nav = cash + sum(
            get_last_price(price_table, t, d) * s
            for t, s in portfolio.items()
            if get_last_price(price_table, t, d) is not None
        )

        history.append({"Date": d, "Portfolio Value": nav})

    return pd.DataFrame(history), pd.DataFrame(trades)


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
    rebalance_interval,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital=100_000.0,
):
    cash = total_capital
    portfolio = {}
    cost_basis = {}

    history = []
    trades = []

    rebalance_dates = sorted(df_prices["Date"].unique())[::rebalance_interval]

    for d in rebalance_dates:

        target_df = build_unified_target(
            dailyA,
            dailyB,
            as_of_date=d,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=total_capital,
            weight_A=0.20,
            weight_B=0.80,
        )

        target_sizes = (
            dict(zip(target_df["Ticker"], target_df["Position_Size"]))
            if not target_df.empty
            else {}
        )

        target = set(target_sizes)
        held = set(portfolio)

        # ---- SELL removed ----
        for t in held - target:
            px = get_last_price(price_table, t, d)
            if px is not None:
                shares = portfolio[t]
                proceeds = shares * px
                pnl = proceeds - cost_basis.get(t, 0.0)
                cash += proceeds
                trades.append(
                    {"Date": d, "Ticker": t, "Action": "Sell", "Price": px, "PnL": pnl}
                )
            portfolio.pop(t, None)
            cost_basis.pop(t, None)

        # ---- BUY / RESIZE ----
        for t, target_value in target_sizes.items():
            px = get_last_price(price_table, t, d)
            if px is None:
                continue

            current_shares = portfolio.get(t, 0.0)
            current_value = current_shares * px
            delta = target_value - current_value

            if delta > 0 and delta <= cash:
                shares = delta / px
                portfolio[t] = current_shares + shares
                cost_basis[t] = cost_basis.get(t, 0.0) + delta
                cash -= delta
                trades.append(
                    {"Date": d, "Ticker": t, "Action": "Buy", "Price": px, "PnL": 0.0}
                )

            elif delta < 0:
                trim_value = -delta
                trim_shares = trim_value / px
                portfolio[t] = current_shares - trim_shares
                cost_basis[t] -= trim_value
                cash += trim_value
                trades.append(
                    {"Date": d, "Ticker": t, "Action": "Resize", "Price": px, "PnL": 0.0}
                )

        # ---- ALWAYS MARK TO MARKET (THIS IS THE FIX) ----
        nav = cash + sum(
            get_last_price(price_table, t, d) * s
            for t, s in portfolio.items()
            if get_last_price(price_table, t, d) is not None
        )

        history.append({"Date": d, "Portfolio Value": nav})

    return pd.DataFrame(history), pd.DataFrame(trades)
