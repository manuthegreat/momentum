# update_parquet.py
# Nightly batch job:
# 1) Refresh raw artifacts (constituents OHLC + index returns)
# 2) Run the EXACT SAME pipeline you previously ran inside Streamlit
# 3) Persist "reference" parquets so Streamlit becomes UI-only
#
# IMPORTANT: Core calculation logic has been copied verbatim from streamlit_app.py
#           (only packaging / persistence / caching has been added).

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from io import StringIO

ARTIFACTS_DIR = "artifacts"

RAW_CONSTITUENTS_PATH = os.path.join(ARTIFACTS_DIR, "index_constituents_5yr.parquet")
RAW_INDEX_RETURNS_PATH = os.path.join(ARTIFACTS_DIR, "index_returns_5y.parquet")

# Reference outputs (consumed by Streamlit UI)
BASE_SP500_PATH = os.path.join(ARTIFACTS_DIR, "base_SP500_5y.parquet")
DAILYA_PATH = os.path.join(ARTIFACTS_DIR, "dailyA.parquet")
DAILYB_PATH = os.path.join(ARTIFACTS_DIR, "dailyB.parquet")
SCOREA_PATH = os.path.join(ARTIFACTS_DIR, "scoreA.parquet")
SCOREB_PATH = os.path.join(ARTIFACTS_DIR, "scoreB.parquet")
PRICEA_PATH = os.path.join(ARTIFACTS_DIR, "priceA.parquet")

BUCKETC_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_history.parquet")
BUCKETC_TRADES_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_trades.parquet")
BUCKETC_STATS_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_stats.parquet")
BUCKETC_TRADE_STATS_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_trade_stats.parquet")

PREVIEW_LATEST_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_preview_latest.parquet")

YAHOO_META_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "yahoo_metadata.parquet")


# ============================================================
# 1. UNIVERSE BUILDERS (RAW)
# ============================================================

def get_sp500_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break

    df["Ticker"] = df["Symbol"].str.replace(".", "-", regex=False)
    df["Name"] = df["Security"]
    df["Sector"] = df["GICS Sector"]

    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe():
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index", "code"]):
            df = t.copy()
            break

    if df is None:
        raise ValueError("No HSI table found")

    df.columns = [str(c).lower() for c in df.columns]

    ticker_col = next(
        (c for c in df.columns if "ticker" in c or "code" in c or "sehk" in c),
        None
    )
    if ticker_col is None:
        raise ValueError("No HSI ticker column")

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)")
        .iloc[:, 0]
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

    name_col = "name" if "name" in df.columns else df.columns[0]
    df["Name"] = df[name_col]

    if "sub-index" in df.columns:
        df["Sector"] = df["sub-index"]
    elif "industry" in df.columns:
        df["Sector"] = df["industry"]
    else:
        df["Sector"] = None

    return df[["Ticker", "Name", "Sector"]]


def get_sti_universe():
    data = [
        ("D05.SI", "DBS Group Holdings", "Financials"),
        ("U11.SI", "United Overseas Bank", "Financials"),
        ("O39.SI", "OCBC", "Financials"),
        ("C07.SI", "Jardine Matheson", "Conglomerate"),
        ("C09.SI", "City Developments", "Real Estate"),
        ("C38U.SI", "CICT", "Real Estate"),
        ("Z74.SI", "Singtel", "Telecom"),
    ]
    return pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])


# ============================================================
# 2. DOWNLOAD CONSTITUENT OHLC (5Y) (RAW)
# ============================================================

def download_5yr_ohlc(tickers, label):
    print(f"\nDownloading {label} ({len(tickers)} tickers)")
    frames = []
    batch_size = 40

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        data = yf.download(
            batch,
            period="5y",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )

        for t in batch:
            try:
                df = data[t].dropna()
                if df.empty:
                    continue
                df = df.reset_index()
                df["Ticker"] = t
                df["Index"] = label
                frames.append(df)
            except Exception:
                continue

    return frames


# ============================================================
# 3. DOWNLOAD INDEX RETURNS (5Y) (RAW)
# ============================================================

def download_index_5y(ticker, label):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.dropna().reset_index()

    df["index_name"] = label
    df["ticker"] = ticker

    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)
    df["ret_60d"] = df["close"].pct_change(60)

    return df


# ============================================================
# 4. YAHOO METADATA CACHE (FULL UNIVERSE, INCREMENTAL)
# ============================================================

def _fetch_one_yahoo_metadata(ticker: str) -> dict:
    name = sector = industry = None
    try:
        tk = yf.Ticker(ticker)

        # 1) Try fast_info first (same as Streamlit)
        fi = getattr(tk, "fast_info", {}) or {}
        name = fi.get("shortName") or fi.get("longName")

        # 2) Try full info only if needed
        info = tk.get_info() or {}
        name = name or info.get("shortName") or info.get("longName")
        sector = info.get("sector")
        industry = info.get("industry")

    except Exception:
        pass

    return {"Ticker": ticker, "Name": name, "Sector": sector, "Industry": industry}


def update_metadata_cache(
    all_tickers: list[str],
    cache_path: str = YAHOO_META_CACHE_PATH,
    ttl_days: int = 90,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """
    Full universe metadata, but incremental:
    - Loads cache if exists
    - Fetches missing tickers + tickers older than TTL
    - Saves back to cache
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    now = pd.Timestamp.utcnow().normalize()

    if os.path.exists(cache_path):
        cache = pd.read_parquet(cache_path)
        if "LastUpdated" in cache.columns:
            cache["LastUpdated"] = pd.to_datetime(cache["LastUpdated"], errors="coerce")
        else:
            cache["LastUpdated"] = pd.NaT
    else:
        cache = pd.DataFrame(columns=["Ticker", "Name", "Sector", "Industry", "LastUpdated"])

    cache = cache.drop_duplicates(subset=["Ticker"], keep="last")
    cache_tickers = set(cache["Ticker"].astype(str).tolist()) if not cache.empty else set()

    all_set = set(map(str, all_tickers))
    missing = sorted(list(all_set - cache_tickers))

    stale = []
    if not cache.empty:
        cutoff = now - pd.Timedelta(days=ttl_days)
        stale = cache.loc[cache["LastUpdated"].isna() | (cache["LastUpdated"] < cutoff), "Ticker"].astype(str).tolist()
        stale = sorted(list(set(stale) & all_set))

    to_fetch = sorted(list(set(missing + stale)))

    if to_fetch:
        print(f"\nYahoo metadata: fetching {len(to_fetch)} tickers (missing={len(missing)}, stale={len(stale)})")
        rows = []
        for i, t in enumerate(to_fetch, 1):
            row = _fetch_one_yahoo_metadata(t)
            row["LastUpdated"] = now
            rows.append(row)

            if sleep_s and i < len(to_fetch):
                time.sleep(sleep_s)

        upd = pd.DataFrame(rows)
        cache = pd.concat([cache, upd], ignore_index=True)
        cache = cache.drop_duplicates(subset=["Ticker"], keep="last")
        cache.to_parquet(cache_path, index=False)
        print(f"Saved metadata cache: {cache_path}")
    else:
        print("\nYahoo metadata: cache already up-to-date for this universe.")

    return cache


# ============================================================
# 5. PIPELINE FUNCTIONS (COPIED FROM STREAMLIT, LOGIC UNCHANGED)
# ============================================================

def load_price_data_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' found in parquet.")

    keep = ["Ticker", "Date", "Price", "Index"] + ohlc_cols
    missing = [c for c in ["Ticker", "Date", "Price", "Index"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in price parquet: {missing}")

    df = df[keep].drop_duplicates(subset=["Ticker", "Date"], keep="last")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_index_returns_parquet(path: str) -> pd.DataFrame:
    idx = pd.read_parquet(path)
    idx.columns = idx.columns.str.lower().str.replace(" ", "_")
    idx["date"] = pd.to_datetime(idx["date"])

    if "index_name" in idx.columns:
        idx = idx.rename(columns={"index_name": "index"})

    required = {"date", "close", "index"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(f"Missing required columns in index parquet: {sorted(missing)}")

    frames = []
    for _, g in idx.groupby("index"):
        g = g.sort_values("date").copy()
        close = g["close"]

        g["idx_ret_1d"] = close.pct_change()
        g["idx_ret_20d"] = close.pct_change(20)
        g["idx_ret_60d"] = close.pct_change(60)
        g["idx_uptrend"] = (g["idx_ret_60d"] > 0).astype(int)

        frames.append(g[["date", "index", "idx_ret_1d", "idx_ret_20d", "idx_ret_60d", "idx_uptrend"]])

    return pd.concat(frames, ignore_index=True)


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    return df[df["Index"] == index_name].reset_index(drop=True)


def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def compute_index_momentum(idx: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    idx = idx.copy().sort_values(["index", "date"])
    idx["idx_1d"] = 1.0 + idx["idx_ret_1d"]

    for w in windows:
        idx[f"idx_{w}D"] = (
            idx.groupby("index")["idx_1d"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )
    return idx


def calculate_momentum_features(df: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
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

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)
    return df


def add_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Momentum_Fast"] = (0.6 * df["5D zscore"] + 0.4 * df["10D zscore"])
    df["Momentum_Mid"] = (0.5 * df["30D zscore"] + 0.5 * df["45D zscore"])
    df["Momentum_Slow"] = (0.5 * df["60D zscore"] + 0.5 * df["90D zscore"])

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )
    return df.fillna(0.0)


def add_regime_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Accel_Fast"] = df.groupby("Ticker")["Momentum_Fast"].diff()
    df["Accel_Mid"] = df.groupby("Ticker")["Momentum_Mid"].diff()
    df["Accel_Slow"] = df.groupby("Ticker")["Momentum_Slow"].diff()

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    df["Accel_Fast_z"] = df.groupby("Date")["Accel_Fast"].transform(zscore_safe)
    df["Accel_Mid_z"] = df.groupby("Date")["Accel_Mid"].transform(zscore_safe)
    df["Accel_Slow_z"] = df.groupby("Date")["Accel_Slow"].transform(zscore_safe)

    df["Acceleration Score"] = (
        0.5 * df["Accel_Fast_z"] +
        0.3 * df["Accel_Mid_z"] +
        0.2 * df["Accel_Slow_z"]
    )
    return df.fillna(0.0)


def add_regime_residual_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Residual_Momentum"] = (
        df["Momentum_Fast"] -
        df.groupby("Ticker")["Momentum_Slow"].transform("mean")
    )

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    df["Residual_Momentum_z"] = df.groupby("Date")["Residual_Momentum"].transform(zscore_safe)
    return df.fillna(0.0)


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Early_Fast"] = (0.6 * df["Accel_Fast_z"] + 0.4 * df["Momentum_Fast"])
    df["Early_Mid"] = (0.5 * df["Accel_Mid_z"] + 0.5 * df["Momentum_Mid"])
    df["Early_Slow"] = (0.5 * df["Accel_Slow_z"] + 0.5 * df["Momentum_Slow"])

    df["Early Momentum Score"] = (
        0.5 * df["Early_Slow"] +
        0.3 * df["Early_Mid"] +
        0.2 * df["Early_Fast"]
    )
    return df.fillna(0.0)


def add_relative_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses df's existing stock returns features (5D/10D/30D/45D/60D/90D Return)
    and index momentum columns (idx_5D..idx_90D) to build relative regimes,
    then sets Momentum_* and Momentum Score from relative z-scores.
    """
    df = df.copy()

    # Relative regime components
    df["Rel_Slow"] = (
        0.5 * df["60D Return"] + 0.5 * df["90D Return"]
    ) - (
        0.5 * df["idx_60D"] + 0.5 * df["idx_90D"]
    )

    df["Rel_Mid"] = (
        0.5 * df["30D Return"] + 0.5 * df["45D Return"]
    ) - (
        0.5 * df["idx_30D"] + 0.5 * df["idx_45D"]
    )

    df["Rel_Fast"] = (
        0.6 * df["5D Return"] + 0.4 * df["10D Return"]
    ) - (
        0.6 * df["idx_5D"] + 0.4 * df["idx_10D"]
    )

    # Cross-sectional z-scores AFTER benchmark subtraction
    for col in ["Rel_Slow", "Rel_Mid", "Rel_Fast"]:
        mean = df.groupby("Date")[col].transform("mean")
        std = df.groupby("Date")[col].transform("std").replace(0, np.nan)
        df[col + "_z"] = ((df[col] - mean) / std).fillna(0.0)

    # Define regime scaffolding using RELATIVE z-scores
    df["Momentum_Slow"] = df["Rel_Slow_z"]
    df["Momentum_Mid"] = df["Rel_Mid_z"]
    df["Momentum_Fast"] = df["Rel_Fast_z"]

    # Momentum Score built from relative regimes
    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )

    return df.fillna(0.0)


def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    records = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d].sort_values("Momentum Score", ascending=False).head(top_n).copy()
        if snap.empty:
            continue

        # rank 1..top_n (1 = best)
        snap["Rank"] = np.arange(1, len(snap) + 1)

        for _, r in snap.iterrows():
            records.append({
                "Date": d,
                "Ticker": r["Ticker"],
                "Rank": int(r["Rank"]),
                "Momentum Score": float(r["Momentum Score"]),
                "Early Momentum Score": float(r["Early Momentum Score"]),
            })
    return pd.DataFrame(records)


def final_selection_from_daily(
    daily_df: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    as_of_date=None,
    top_n: int = 10
) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    if as_of_date is None:
        as_of_date = daily_df["Date"].max()

    dates = sorted(
        daily_df.loc[daily_df["Date"] <= as_of_date, "Date"].unique(),
        reverse=True
    )[:lookback_days]

    if not dates:
        return pd.DataFrame()

    window = daily_df[daily_df["Date"].isin(dates)]
    if window.empty:
        return pd.DataFrame()

    agg = (
        window.groupby("Ticker")
        .agg(
            Momentum_Score=("Momentum Score", "mean"),
            Early_Momentum_Score=("Early Momentum Score", "mean"),
            Appearances=("Date", "count"),
            Rank_Mean=("Rank", "mean"),
            Rank_Std=("Rank", "std"),
        )
        .reset_index()
    )

    # std can be NaN when only 1 appearance; treat that as perfectly stable (std=0)
    agg["Rank_Std"] = agg["Rank_Std"].fillna(0.0)

    agg["Consistency"] = agg["Appearances"] / len(dates)
    agg["Weighted_Score"] = (
        w_momentum * agg["Momentum_Score"] +
        w_early * agg["Early_Momentum_Score"] +
        w_consistency * agg["Consistency"]
    )

    # Consistency already 0..1
    agg["ConsistencyScore"] = agg["Consistency"] * 100.0

    # Rank stability from Rank_Std (lower is better)
    # Normalize: std >= (top_n/2) is considered "very unstable" => score goes to 0
    max_std = max(1.0, top_n / 2.0)
    agg["RankStabilityScore"] = (1.0 - (agg["Rank_Std"] / max_std)).clip(0.0, 1.0) * 100.0

    # 50/50 confidence
    agg["Signal_Confidence"] = 0.5 * agg["ConsistencyScore"] + 0.5 * agg["RankStabilityScore"]

    return (
        agg.sort_values("Weighted_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def monthly_rebalance_dates(df_prices: pd.DataFrame) -> list:
    dates = pd.to_datetime(df_prices["Date"]).dropna().sort_values().unique()
    if len(dates) == 0:
        return []
    tmp = pd.DataFrame({"Date": dates})
    tmp["Month"] = tmp["Date"].dt.to_period("M")
    return tmp.groupby("Month")["Date"].max().tolist()


def get_last_price(price_table: pd.DataFrame, ticker: str, date) -> float | None:
    try:
        s = price_table.loc[:date, ticker].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def build_bucket_c_signals(
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    as_of_date,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital,
    weight_A=0.20,
    weight_B=0.80,
) -> pd.DataFrame:
    """
    Bucket C = combine final selection from A and B, allocate dollars by weights,
    and aggregate diagnostics per ticker (showing Consistency etc.).
    """
    selA = final_selection_from_daily(
        dailyA,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    ).copy()

    selB = final_selection_from_daily(
        dailyB,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    ).copy()

    frames = []

    if not selA.empty:
        selA["Bucket"] = "A"
        selA["Position_Size"] = (total_capital * weight_A) / len(selA)
        frames.append(selA)

    if not selB.empty:
        selB["Bucket"] = "B"
        selB["Position_Size"] = (total_capital * weight_B) / len(selB)
        frames.append(selB)

    if not frames:
        return pd.DataFrame()

    combo = pd.concat(frames, ignore_index=True)

    # Aggregate per ticker
    out = (
        combo.groupby("Ticker", as_index=False)
        .agg(
            Position_Size=("Position_Size", "sum"),
            Weighted_Score=("Weighted_Score", "max"),
            Momentum_Score=("Momentum_Score", "max"),
            Early_Momentum_Score=("Early_Momentum_Score", "max"),
            Consistency=("Consistency", "max"),

            # ✅ add these
            Rank_Std=("Rank_Std", "min"),  # smaller = more stable
            RankStabilityScore=("RankStabilityScore", "max"),
            Signal_Confidence=("Signal_Confidence", "max"),

            Bucket_Source=("Bucket", lambda x: "+".join(sorted(set(x))))
        )
        .sort_values(["Position_Size", "Signal_Confidence"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Normalize source formatting
    out["Bucket_Source"] = out["Bucket_Source"].replace({"A+B": "A+B", "A": "A", "B": "B"})
    return out


def simulate_unified_portfolio(
    df_prices: pd.DataFrame,
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    top_n: int = 10,
    total_capital: float = 100_000.0,
    weight_A: float = 0.20,
    weight_B: float = 0.80,
):
    rebalance_dates = monthly_rebalance_dates(df_prices)

    cash = total_capital
    portfolio = {}  # ticker -> shares
    cost_basis = {}  # ticker -> dollars invested

    history = []
    trades = []

    for d in rebalance_dates:
        target_df = build_bucket_c_signals(
            dailyA=dailyA,
            dailyB=dailyB,
            as_of_date=d,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=total_capital,
            weight_A=weight_A,
            weight_B=weight_B
        )
        if target_df.empty:
            continue

        target_sizes = dict(zip(target_df["Ticker"], target_df["Position_Size"]))
        target = set(target_sizes.keys())
        held = set(portfolio.keys())

        # SELL
        for t in list(held - target):
            px = get_last_price(price_table, t, d)
            if px is None or px <= 0:
                continue
            shares = portfolio[t]
            proceeds = shares * px
            cash += proceeds
            pnl = proceeds - cost_basis.get(t, 0.0)
            trades.append({"Date": d, "Ticker": t, "Action": "Sell", "Price": px, "PnL": pnl})
            portfolio.pop(t, None)
            cost_basis.pop(t, None)

        # BUY / RESIZE
        for t, target_dollars in target_sizes.items():
            px = get_last_price(price_table, t, d)
            if px is None or px <= 0:
                continue

            current_shares = portfolio.get(t, 0.0)
            current_value = current_shares * px
            delta_value = target_dollars - current_value

            if delta_value > 0:
                if delta_value > cash:
                    continue
                delta_shares = delta_value / px
                portfolio[t] = current_shares + delta_shares
                cost_basis[t] = cost_basis.get(t, 0.0) + delta_value
                cash -= delta_value
                trades.append({"Date": d, "Ticker": t, "Action": "Buy", "Price": px, "PnL": 0.0})

            elif delta_value < 0:
                trim_value = -delta_value
                trim_shares = trim_value / px
                portfolio[t] = current_shares - trim_shares
                cost_basis[t] = cost_basis.get(t, 0.0) - trim_value
                cash += trim_value
                trades.append({"Date": d, "Ticker": t, "Action": "Resize", "Price": px, "PnL": 0.0})

        holdings_value = sum(
            get_last_price(price_table, t, d) * s
            for t, s in portfolio.items()
            if get_last_price(price_table, t, d) is not None
        )
        nav = cash + holdings_value
        history.append({"Date": d, "Portfolio Value": nav})

    return pd.DataFrame(history), pd.DataFrame(trades)


def compute_performance_stats(history_df: pd.DataFrame) -> dict:
    df = history_df.sort_values("Date").copy()
    if df.empty or len(df) < 2:
        return {"Message": "Not enough history"}

    start = float(df["Portfolio Value"].iloc[0])
    end = float(df["Portfolio Value"].iloc[-1])
    if start <= 0:
        return {"Message": "Invalid start value"}

    total_return = (end / start - 1) * 100
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25

    cagr = ((end / start) ** (1 / years) - 1) * 100 if (end > 0 and years > 0) else np.nan

    ret = df["Portfolio Value"].pct_change().dropna()

    std = ret.std()
    sharpe = np.sqrt(252) * ret.mean() / std if (std is not None and std > 0) else np.nan

    downside = ret[ret < 0]
    dstd = downside.std()
    sortino = np.sqrt(252) * ret.mean() / dstd if (dstd is not None and dstd > 0) else np.nan

    rolling_max = df["Portfolio Value"].cummax()
    drawdown = (df["Portfolio Value"] - rolling_max) / rolling_max

    return {
        "Total Return (%)": float(total_return),
        "CAGR (%)": float(cagr) if pd.notna(cagr) else np.nan,
        "Sharpe Ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Sortino Ratio": float(sortino) if pd.notna(sortino) else np.nan,
        "Max Drawdown (%)": float(drawdown.min() * 100),
    }


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {"Message": "No trades"}
    sells = trades_df[trades_df["Action"] == "Sell"].copy()
    if sells.empty:
        return {"Message": "No closed trades"}

    wins = sells[sells["PnL"] > 0]
    losses = sells[sells["PnL"] < 0]

    total_win = wins["PnL"].sum()
    total_loss = abs(losses["PnL"].sum())

    return {
        "Number of Trades": int(len(sells)),
        "Win Rate (%)": float(len(wins) / len(sells) * 100),
        "Average Win ($)": float(wins["PnL"].mean()) if not wins.empty else 0.0,
        "Average Loss ($)": float(losses["PnL"].mean()) if not losses.empty else 0.0,
        "Profit Factor": float(total_win / total_loss) if total_loss > 0 else np.nan,
    }


def build_bucket_c_signal_preview(
    dailyA,
    dailyB,
    as_of_date,
    lookback_days,
    w_momentum,
    w_early,
    w_consistency,
    top_n,
    total_capital,
    weight_A=0.20,
    weight_B=0.80,
):
    tgtC = build_bucket_c_signals(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        top_n=top_n,
        total_capital=total_capital,
        weight_A=weight_A,
        weight_B=weight_B,
    )

    if tgtC is None or tgtC.empty:
        return pd.DataFrame()

    out = tgtC.copy()
    out = out.rename(columns={"Position_Size": "Target_$"})
    total = float(out["Target_$"].sum()) if "Target_$" in out.columns else 0.0
    out["Weight_%"] = (out["Target_$"] / total * 100.0) if total > 0 else np.nan

    # Order / keep columns for the Signals tab (includes Consistency)
    cols = [
        "Ticker", "Name",
        "Target_$", "Weight_%",
        "Signal_Confidence",
        "Weighted_Score", "Momentum_Score", "Early_Momentum_Score",
        "Sector", "Industry",
    ]

    out["Consistency"] = out["Consistency"] * 100.0  # convert to %

    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(["Signal_Confidence", "Weighted_Score"], ascending=[False, False]).reset_index(drop=True)

    return out


# ============================================================
# 6. RAW REFRESH + REFERENCE BUILD
# ============================================================

def refresh_raw_parquets():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ---------- Build universes ----------
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    sti = get_sti_universe()

    # ---------- Download constituents ----------
    frames = []
    frames += download_5yr_ohlc(sp500["Ticker"].tolist(), "SP500")
    frames += download_5yr_ohlc(hsi["Ticker"].tolist(), "HSI")
    frames += download_5yr_ohlc(sti["Ticker"].tolist(), "STI")

    full_constituents = pd.concat(frames, ignore_index=True)
    full_constituents.to_parquet(RAW_CONSTITUENTS_PATH, index=False)
    print(f"Saved {RAW_CONSTITUENTS_PATH}")

    # ---------- Download index returns ----------
    index_map = {
        "^GSPC": "SP500",
        "^HSI": "HSI",
        "^STI": "STI",
        "^VIX": "VIX",
    }

    idx_frames = []
    for t, lbl in index_map.items():
        df = download_index_5y(t, lbl)
        if df is not None:
            idx_frames.append(df)

    full_index = pd.concat(idx_frames, ignore_index=True)
    full_index.to_parquet(RAW_INDEX_RETURNS_PATH, index=False)
    print(f"Saved {RAW_INDEX_RETURNS_PATH}")

    return full_constituents


def build_reference_parquets():
    # (Same params as Streamlit)
    DAILY_TOP_N = 10
    FINAL_TOP_N = 10
    LOOKBACK_DAYS = 10
    WINDOWS = (5, 10, 30, 45, 60, 90)

    W_MOM = 0.50
    W_EARLY = 0.30
    W_CONS = 0.20

    # Load raw artifacts (just refreshed)
    base = load_price_data_parquet(RAW_CONSTITUENTS_PATH)
    base = filter_by_index(base, "SP500")
    idx = load_index_returns_parquet(RAW_INDEX_RETURNS_PATH)

    # Save base SP500 for Streamlit charts
    base.to_parquet(BASE_SP500_PATH, index=False)
    print(f"Saved {BASE_SP500_PATH}")

    # ---------------- Bucket A (absolute) ----------------
    dfA = calculate_momentum_features(add_absolute_returns(base), windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)

    priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    scoreA = dfA[["Date", "Ticker", "Momentum Score", "Early Momentum Score"]].copy()

    dailyA.to_parquet(DAILYA_PATH, index=False)
    scoreA.to_parquet(SCOREA_PATH, index=False)
    priceA.reset_index().to_parquet(PRICEA_PATH, index=False)

    print(f"Saved {DAILYA_PATH}")
    print(f"Saved {SCOREA_PATH}")
    print(f"Saved {PRICEA_PATH}")

    # ---------------- Bucket B (relative) ----------------
    dfB = base.copy()
    dfB = dfB.merge(
        idx,
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left",
        validate="many_to_one"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = dfB[dfB["idx_ret_1d"].notna()].copy()

    dfB = calculate_momentum_features(add_absolute_returns(dfB), windows=WINDOWS)

    idx_mom = compute_index_momentum(idx, windows=WINDOWS)
    dfB = dfB.merge(
        idx_mom[["date", "index", "idx_5D", "idx_10D", "idx_30D", "idx_45D", "idx_60D", "idx_90D"]],
        left_on=["Date", "Index"],
        right_on=["date", "index"],
        how="left"
    ).drop(columns=["date", "index"], errors="ignore")

    dfB = add_relative_regime_momentum_score(dfB)

    # Filters (as in your original long version)
    dfB = dfB[dfB["Momentum_Slow"] > 1].copy()
    dfB = dfB[dfB["Momentum_Mid"] > 0.5].copy()
    dfB = dfB[dfB["Momentum_Fast"] > 1].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)
    scoreB = dfB[["Date", "Ticker", "Momentum Score", "Early Momentum Score"]].copy()

    dailyB.to_parquet(DAILYB_PATH, index=False)
    scoreB.to_parquet(SCOREB_PATH, index=False)

    print(f"Saved {DAILYB_PATH}")
    print(f"Saved {SCOREB_PATH}")

    # ---------------- Bucket C (unified) backtest ----------------
    histU, tradesU = simulate_unified_portfolio(
        df_prices=base,
        price_table=priceA,
        dailyA=dailyA,
        dailyB=dailyB,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000.0,
        weight_A=0.20,
        weight_B=0.80
    )

    statsU = compute_performance_stats(histU)
    trade_statsU = compute_trade_stats(tradesU)

    histU.to_parquet(BUCKETC_HISTORY_PATH, index=False)
    tradesU.to_parquet(BUCKETC_TRADES_PATH, index=False)

    pd.DataFrame([{"Metric": k, "Value": v} for k, v in statsU.items()]).to_parquet(BUCKETC_STATS_PATH, index=False)
    pd.DataFrame([{"Metric": k, "Value": v} for k, v in trade_statsU.items()]).to_parquet(BUCKETC_TRADE_STATS_PATH, index=False)

    print(f"Saved {BUCKETC_HISTORY_PATH}")
    print(f"Saved {BUCKETC_TRADES_PATH}")
    print(f"Saved {BUCKETC_STATS_PATH}")
    print(f"Saved {BUCKETC_TRADE_STATS_PATH}")

    # ---------------- Latest preview (ready-to-display) ----------------
    common_dates = sorted(set(dailyA["Date"]).intersection(set(dailyB["Date"])))
    signal_date = common_dates[-1] if common_dates else None
    if signal_date is None:
        print("WARNING: No overlapping signal dates between dailyA and dailyB. Preview not saved.")
        return

    preview = build_bucket_c_signal_preview(
        dailyA=dailyA,
        dailyB=dailyB,
        as_of_date=signal_date,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000.0,
        weight_A=0.20,
        weight_B=0.80
    )

    # Full-universe metadata cache (incremental)
    # Use ALL tickers present in raw constituents parquet (SP500+HSI+STI)
    raw_const = pd.read_parquet(RAW_CONSTITUENTS_PATH)
    universe_tickers = sorted(raw_const["Ticker"].astype(str).unique().tolist())
    meta = update_metadata_cache(universe_tickers, cache_path=YAHOO_META_CACHE_PATH, ttl_days=90, sleep_s=0.0)

    preview = preview.drop(columns=["Name", "Sector", "Industry"], errors="ignore").merge(
        meta[["Ticker", "Name", "Sector", "Industry"]],
        on="Ticker",
        how="left"
    )

    # Stamp the signal date (handy for UI)
    preview["Signal_Date"] = pd.to_datetime(signal_date)

    preview.to_parquet(PREVIEW_LATEST_PATH, index=False)
    print(f"Saved {PREVIEW_LATEST_PATH}")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1) Refresh raw parquets
    refresh_raw_parquets()

    # 2) Build reference parquets (heavy lifting)
    build_reference_parquets()

    print("\n✅ Overnight batch complete.")


if __name__ == "__main__":
    main()
