# updater/run_backtest_from_signals.py
# SINGLE-ENGINE, SIGNAL-DRIVEN, NAV-COMPOUNDING IMPLEMENTATION

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS = Path("artifacts")
PRICE_PATH = ARTIFACTS / "index_constituents_5yr.parquet"
INDEX_PATH = ARTIFACTS / "index_returns_5y.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)

LOOKBACK_DAYS = 10
TOP_N = 10
TOTAL_CAPITAL = 100_000

W_MOM = 0.50
W_EARLY = 0.30
W_CONS = 0.20

WEIGHT_A = 0.20
WEIGHT_B = 0.80

# ============================================================
# IO
# ============================================================

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# DATA LOADING
# ============================================================

def load_price_data(path):
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = df.get("Adj Close", df.get("Close", df["Price"]))
    return df[["Ticker", "Date", "Price", "Index"]].sort_values(["Ticker", "Date"])


def load_index_returns(path):
    idx = pd.read_parquet(path)
    idx.columns = idx.columns.str.lower().str.replace(" ", "_")
    idx["date"] = pd.to_datetime(idx["date"])

    out = []
    for _, g in idx.groupby("index"):
        g = g.sort_values("date")
        close = g["close"]
        g["idx_ret_1d"] = close.pct_change()
        g["idx_1d"] = 1 + g["idx_ret_1d"]
        for w in WINDOWS:
            g[f"idx_{w}D"] = (
                g["idx_1d"].rolling(w, min_periods=w).apply(np.prod, raw=True) - 1
            )
        out.append(g)

    return pd.concat(out, ignore_index=True)


# ============================================================
# FEATURES
# ============================================================

def add_absolute_returns(df):
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def calculate_momentum(df):
    df = df.copy()
    for w in WINDOWS:
        df[f"{w}D Return"] = (
            df.groupby("Ticker")["1D Return"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )

        mean = df.groupby("Date")[f"{w}D Return"].transform("mean")
        std = df.groupby("Date")[f"{w}D Return"].transform("std").replace(0, np.nan)
        df[f"{w}D z"] = ((df[f"{w}D Return"] - mean) / std).fillna(0.0)

    return df.fillna(0.0)


def add_absolute_regimes(df):
    df["Momentum_Fast"] = 0.6 * df["5D z"] + 0.4 * df["10D z"]
    df["Momentum_Mid"]  = 0.5 * df["30D z"] + 0.5 * df["45D z"]
    df["Momentum_Slow"] = 0.5 * df["60D z"] + 0.5 * df["90D z"]
    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )
    return df


def add_relative_regimes(df):
    df["Rel_Fast"] = (
        0.6 * df["5D Return"] + 0.4 * df["10D Return"]
        - (0.6 * df["idx_5D"] + 0.4 * df["idx_10D"])
    )
    df["Rel_Mid"] = (
        0.5 * df["30D Return"] + 0.5 * df["45D Return"]
        - (0.5 * df["idx_30D"] + 0.5 * df["idx_45D"])
    )
    df["Rel_Slow"] = (
        0.5 * df["60D Return"] + 0.5 * df["90D Return"]
        - (0.5 * df["idx_60D"] + 0.5 * df["idx_90D"])
    )

    for c in ["Rel_Fast", "Rel_Mid", "Rel_Slow"]:
        mean = df.groupby("Date")[c].transform("mean")
        std = df.groupby("Date")[c].transform("std").replace(0, np.nan)
        df[c] = ((df[c] - mean) / std).fillna(0.0)

    df["Momentum_Fast"] = df["Rel_Fast"]
    df["Momentum_Mid"]  = df["Rel_Mid"]
    df["Momentum_Slow"] = df["Rel_Slow"]

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )
    return df


# ============================================================
# DAILY → TARGET
# ============================================================

def build_daily(df):
    out = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d]
        picks = snap.sort_values("Momentum Score", ascending=False).head(TOP_N)
        for _, r in picks.iterrows():
            out.append({"Date": d, "Ticker": r["Ticker"], "Score": r["Momentum Score"]})
    return pd.DataFrame(out)


def build_target(daily, as_of):
    window = daily[daily["Date"] <= as_of].tail(LOOKBACK_DAYS)
    if window.empty:
        return []
    agg = window.groupby("Ticker").size().sort_values(ascending=False)
    return list(agg.head(TOP_N).index)


# ============================================================
# SINGLE EXECUTION ENGINE (FINAL)
# ============================================================

def run_engine(price_table, daily, capital):
    cash = capital
    portfolio = {}
    history = []
    last_target = None

    for d in sorted(daily["Date"].unique()):
        target = tuple(sorted(build_target(daily, d)))
        if not target or target == last_target:
            nav = cash + sum(
                price_table.loc[d, t] * s
                for t, s in portfolio.items()
                if t in price_table.columns and d in price_table.index
            )
            history.append({"Date": d, "Portfolio Value": nav})
            continue

        last_target = target
        nav = cash + sum(
            price_table.loc[d, t] * s
            for t, s in portfolio.items()
            if t in price_table.columns and d in price_table.index
        )

        dollars = nav / len(target)
        new_port = {}

        for t in target:
            px = price_table.loc[d, t]
            shares = dollars / px
            new_port[t] = shares

        portfolio = new_port
        cash = 0.0

        nav = sum(
            price_table.loc[d, t] * s
            for t, s in portfolio.items()
        )
        history.append({"Date": d, "Portfolio Value": nav})

    return pd.DataFrame(history)


# ============================================================
# MAIN
# ============================================================

def main():
    base = load_price_data(PRICE_PATH)
    idx = load_index_returns(INDEX_PATH)

    price_table = base.pivot(index="Date", columns="Ticker", values="Price")

    # -------- A --------
    dfA = add_absolute_regimes(calculate_momentum(add_absolute_returns(base)))
    dailyA = build_daily(dfA)

    # -------- B --------
    dfB = base.merge(idx, left_on=["Date", "Index"], right_on=["date", "index"])
    dfB = add_relative_regimes(calculate_momentum(add_absolute_returns(dfB)))

    dfB = dfB[
        (dfB["Momentum_Slow"] > 1.0) &
        (dfB["Momentum_Mid"] > 0.5) &
        (dfB["Momentum_Fast"] > 1.0)
    ]

    dailyB = build_daily(dfB)

    # -------- C --------
    dailyC = pd.concat([
        dailyA.assign(weight=WEIGHT_A),
        dailyB.assign(weight=WEIGHT_B)
    ])

    eqA = run_engine(price_table, dailyA, TOTAL_CAPITAL)
    eqB = run_engine(price_table, dailyB, TOTAL_CAPITAL)
    eqC = run_engine(price_table, dailyC, TOTAL_CAPITAL)

    eqA.to_parquet(ARTIFACTS / "backtest_equity_A.parquet")
    eqB.to_parquet(ARTIFACTS / "backtest_equity_B.parquet")
    eqC.to_parquet(ARTIFACTS / "backtest_equity_C.parquet")

    print("✅ FINAL engine complete. No parallel logic. No drift.")


if __name__ == "__main__":
    main()
