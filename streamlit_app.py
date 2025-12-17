import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# ============================================================
# 1) DATA LOADING
# ============================================================

def load_price_data_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' found in parquet.")

    keep = ["Ticker", "Date", "Price", "Index"]
    df = df[keep].drop_duplicates(subset=["Ticker", "Date"], keep="last")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_index_returns_parquet(path: str) -> pd.DataFrame:
    idx = pd.read_parquet(path)
    idx.columns = idx.columns.str.lower().str.replace(" ", "_")
    idx["date"] = pd.to_datetime(idx["date"])

    if "index_name" in idx.columns:
        idx = idx.rename(columns={"index_name": "index"})

    frames = []
    for _, g in idx.groupby("index"):
        g = g.sort_values("date").copy()
        close = g["close"]

        g["idx_ret_1d"] = close.pct_change()
        g["idx_ret_20d"] = close.pct_change(20)
        g["idx_ret_60d"] = close.pct_change(60)
        g["idx_uptrend"] = (g["idx_ret_60d"] > 0).astype(int)

        frames.append(
            g[["date", "index", "idx_ret_1d", "idx_ret_20d", "idx_ret_60d", "idx_uptrend"]]
        )

    return pd.concat(frames, ignore_index=True)


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    return df[df["Index"] == index_name].reset_index(drop=True)

# ============================================================
# 2) RETURNS
# ============================================================

def add_absolute_returns(df):
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def compute_index_momentum(idx, windows):
    idx = idx.sort_values(["index", "date"]).copy()
    idx["idx_1d"] = 1.0 + idx["idx_ret_1d"]

    for w in windows:
        idx[f"idx_{w}D"] = (
            idx.groupby("index")["idx_1d"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )

    return idx

# ============================================================
# 3) MOMENTUM ENGINE (UNCHANGED)
# ============================================================

def calculate_momentum_features(df, windows):
    df = df.copy()
    for w in windows:
        r = f"{w}D Return"
        z = f"{w}D zscore"
        dz = f"{w}D zscore change"

        df[r] = (
            df.groupby("Ticker")["1D Return"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )

        mean = df.groupby("Date")[r].transform("mean")
        std = df.groupby("Date")[r].transform("std").replace(0, np.nan)
        df[z] = ((df[r] - mean) / std)

        df[dz] = (
            df.groupby("Ticker")[z]
            .diff()
            .ewm(span=w, adjust=False)
            .mean()
        )

    df[df.select_dtypes("number").columns] = df.select_dtypes("number").fillna(0)
    return df


def add_regime_momentum_score(df):
    df = df.copy()
    df["Momentum_Fast"] = 0.6 * df["5D zscore"] + 0.4 * df["10D zscore"]
    df["Momentum_Mid"]  = 0.5 * df["30D zscore"] + 0.5 * df["45D zscore"]
    df["Momentum_Slow"] = 0.5 * df["60D zscore"] + 0.5 * df["90D zscore"]

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )
    return df.fillna(0)


def add_regime_acceleration(df):
    df = df.copy()
    df["Accel_Fast"] = df.groupby("Ticker")["Momentum_Fast"].diff()
    df["Accel_Mid"]  = df.groupby("Ticker")["Momentum_Mid"].diff()
    df["Accel_Slow"] = df.groupby("Ticker")["Momentum_Slow"].diff()

    def z(x):
        s = x.std()
        return ((x - x.mean()) / s).fillna(0) if s and s > 0 else (x - x.mean()).fillna(0)

    df["Accel_Fast_z"] = df.groupby("Date")["Accel_Fast"].transform(z)
    df["Accel_Mid_z"]  = df.groupby("Date")["Accel_Mid"].transform(z)
    df["Accel_Slow_z"] = df.groupby("Date")["Accel_Slow"].transform(z)

    return df.fillna(0)


def add_regime_early_momentum(df):
    df = df.copy()
    df["Early Momentum Score"] = (
        0.5 * (0.5 * df["Accel_Slow_z"] + 0.5 * df["Momentum_Slow"]) +
        0.3 * (0.5 * df["Accel_Mid_z"]  + 0.5 * df["Momentum_Mid"]) +
        0.2 * (0.6 * df["Accel_Fast_z"] + 0.4 * df["Momentum_Fast"])
    )
    return df.fillna(0)

# ============================================================
# 4) DAILY LISTS
# ============================================================

def build_daily_lists(df, top_n):
    out = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d].sort_values("Momentum Score", ascending=False).head(top_n)
        for _, r in snap.iterrows():
            out.append({
                "Date": d,
                "Ticker": r["Ticker"],
                "Momentum Score": r["Momentum Score"],
                "Early Momentum Score": r["Early Momentum Score"]
            })
    return pd.DataFrame(out)

# ============================================================
# MAIN (PIPELINE EXECUTION)
# ============================================================

st.title("📈 Momentum Strategy — Pipeline Replica")

ARTIFACTS = "artifacts"
PRICE_PATH = f"{ARTIFACTS}/index_constituents_5yr.parquet"
INDEX_PATH = f"{ARTIFACTS}/index_returns_5y.parquet"

WINDOWS = (5, 10, 30, 45, 60, 90)
TOP_N = 10

base = load_price_data_parquet(PRICE_PATH)
base = filter_by_index(base, "SP500")
idx = load_index_returns_parquet(INDEX_PATH)

dfA = calculate_momentum_features(add_absolute_returns(base), WINDOWS)
dfA = add_regime_momentum_score(dfA)
dfA = add_regime_acceleration(dfA)
dfA = add_regime_early_momentum(dfA)

dailyA = build_daily_lists(dfA, TOP_N)

st.subheader("Bucket A — Daily Signals")
st.dataframe(dailyA.tail(20), use_container_width=True)
