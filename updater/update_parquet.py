# update_parquet.py
# Nightly batch job:
# 1) Refresh raw artifacts (constituents OHLC + index returns) from Yahoo
# 2) Run consolidated signals pipeline (Weekly Swing, Fibonacci, Momentum Bucket C)
# 3) Persist signal parquets so Streamlit becomes UI-only

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from updater.golden_source_validation import GoldenSourceCheck, validate_golden_source

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "artifacts"

RAW_CONSTITUENTS_PATH = os.path.join(ARTIFACTS_DIR, "index_constituents_5yr.parquet")
RAW_INDEX_RETURNS_PATH = os.path.join(ARTIFACTS_DIR, "index_returns_5y.parquet")

WEEKLY_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "weekly_swing_signals.parquet")
FIB_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "fib_signals.parquet")
MOMENTUM_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "momentum_bucketc_signals.parquet")
ACTION_LIST_PATH = os.path.join(ARTIFACTS_DIR, "action_list.parquet")
GOLDEN_WEEKLY_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "weekly_swing_signals_golden.parquet")
GOLDEN_FIB_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "fib_signals_golden.parquet")


# ============================================================
# 1) UNIVERSE BUILDERS (RAW)
# ============================================================

def get_sp500_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break

    df["Ticker"] = df["Symbol"].str.replace(".", "-", regex=False)
    df["Name"] = df["Security"]
    df["Sector"] = df["GICS Sector"]

    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
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
        None,
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


def get_sti_universe() -> pd.DataFrame:
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
# 2) DOWNLOAD CONSTITUENT OHLC (5Y) (RAW)
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
            progress=False,
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
# 3) DOWNLOAD INDEX RETURNS (5Y) (RAW)
# ============================================================

def download_index_5y(ticker, label):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
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
# 4) LOAD + STANDARDIZE PARQUETS (PRICES + INDEX RETURNS)
# ============================================================

def load_prices_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    colmap = {}
    for c in df.columns:
        cl = str(c).strip()
        if cl.lower() == "date":
            colmap[c] = "date"
        elif cl.lower() == "ticker":
            colmap[c] = "ticker"
        elif cl.lower() == "index":
            colmap[c] = "index"
        elif cl.lower() == "open":
            colmap[c] = "open"
        elif cl.lower() == "high":
            colmap[c] = "high"
        elif cl.lower() == "low":
            colmap[c] = "low"
        elif cl.lower() == "close":
            colmap[c] = "close"
        elif cl.lower() in ("adj close", "adj_close", "adjclose"):
            colmap[c] = "adj_close"
        elif cl.lower() == "volume":
            colmap[c] = "volume"

    df = df.rename(columns=colmap)

    required = {"date", "ticker", "open", "high", "low", "close"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}. Found: {sorted(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str)

    if "index" not in df.columns:
        df["index"] = "UNKNOWN"

    if "volume" not in df.columns:
        df["volume"] = 0.0

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close"]).copy()
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)

    df["turnover"] = df["close"].astype(float) * df["volume"].astype(float)

    return df


def load_index_returns_from_parquet(path: str) -> pd.DataFrame:
    idx = pd.read_parquet(path).copy()
    idx.columns = [str(c).strip().lower() for c in idx.columns]

    if "date" not in idx.columns:
        raise ValueError(f"Index parquet missing 'date'. Found: {sorted(idx.columns)}")
    if "index_name" not in idx.columns:
        raise ValueError(f"Index parquet missing 'index_name'. Found: {sorted(idx.columns)}")

    idx["date"] = pd.to_datetime(idx["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    idx["index_name"] = idx["index_name"].astype(str).str.upper().str.strip()

    for rc in ["ret_1d", "ret_5d", "ret_20d", "ret_60d"]:
        if rc in idx.columns:
            idx[rc] = pd.to_numeric(idx[rc], errors="coerce")

    idx = idx.dropna(subset=["date", "index_name"]).copy()
    idx = idx.sort_values(["index_name", "date"]).drop_duplicates(subset=["index_name", "date"], keep="last").reset_index(drop=True)

    return idx


def infer_index_name_for_row(ticker: str, index_col_val: str) -> str:
    t = str(ticker).upper().strip()
    idxv = str(index_col_val).upper().strip() if index_col_val is not None else ""

    if "SP500" in idxv or "S&P" in idxv:
        return "SP500"
    if idxv == "HSI" or "HANG" in idxv:
        return "HSI"
    if idxv == "STI" or "STRAITS" in idxv:
        return "STI"

    if t.endswith(".HK"):
        return "HSI"
    if t.endswith(".SI"):
        return "STI"
    return "SP500"


# ============================================================
# 5) SYSTEM 1: WEEKLY SWING
# ============================================================

@dataclass
class WeeklySwingConfig:
    max_open_positions: int = 20
    max_entries_per_day: int = 10

    profit_target_mult: float = 1.10
    holding_days_max: int = 30

    tight_range_5d_max: float = 0.12
    close_pos20_min: float = 0.60
    sma20_slope_floor_mult: float = -0.002

    pause_days_min: int = 4
    pause_days_max: int = 9
    pause_near_high_frac: float = 0.97
    pause_range20_max: float = 0.22

    fib_lookback: int = 10
    fib_frac: float = 0.382

    pullback_min_pct_below_close: float = 0.02
    pullback_max_pct_below_close: float = 0.25
    max_hh_dist_from_close: float = 0.20

    turnover_baseline_days: int = 10
    turnover_expansion_min_A: float = 1.0
    turnover_expansion_min_C: float = 1.0

    breakout_buffer_pct: float = 0.0

    adv_turnover_20_min: float = 0.0


def _hh_and_low_since_hh_py(high: np.ndarray, low: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    hh = np.full(n, np.nan, dtype=float)
    low_since = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i < window - 1:
            continue
        s = i - window + 1
        win_high = high[s: i + 1]
        win_low = low[s: i + 1]

        k = int(np.argmax(win_high))
        hh_i = float(win_high[k])
        low_i = float(np.min(win_low[k:]))

        hh[i] = hh_i
        low_since[i] = low_i

    return hh, low_since


def weekly_add_indicators(df: pd.DataFrame, cfg: WeeklySwingConfig) -> pd.DataFrame:
    d = df.sort_values(["ticker", "date"]).copy()

    g = d.groupby("ticker", sort=False)

    close = d["close"].astype(float)
    d["sma20"] = g["close"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    d["sma50"] = g["close"].rolling(50, min_periods=50).mean().reset_index(level=0, drop=True)
    d["sma20_slope5"] = d["sma20"] - g["sma20"].shift(5)

    d["prev_close"] = g["close"].shift(1)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    prev_close = d["prev_close"].astype(float)

    tr1 = (high - low).to_numpy()
    tr2 = (high - prev_close).abs().to_numpy()
    tr3 = (low - prev_close).abs().to_numpy()
    d["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))

    d["atr5"] = g["tr"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    d["atr20"] = g["tr"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    d["hh5"] = g["high"].rolling(5, min_periods=5).max().reset_index(level=0, drop=True)
    d["ll5"] = g["low"].rolling(5, min_periods=5).min().reset_index(level=0, drop=True)
    d["range5_pct"] = (d["hh5"] - d["ll5"]) / close

    d["hh20"] = g["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    d["ll20"] = g["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    d["range20_pct"] = (d["hh20"] - d["ll20"]) / close

    denom20 = (d["hh20"] - d["ll20"]).replace(0, np.nan)
    d["close_pos_20"] = (close - d["ll20"]) / denom20

    d["turnover_5d"] = g["turnover"].rolling(5, min_periods=5).sum().reset_index(level=0, drop=True)
    d["adv_turnover_20"] = g["turnover"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    d["avg_turnover_10d"] = (
        g["turnover"]
        .rolling(cfg.turnover_baseline_days, min_periods=cfg.turnover_baseline_days)
        .mean()
        .reset_index(level=0, drop=True)
    )
    d["baseline_5d_turnover"] = d["avg_turnover_10d"] * 5.0

    d["hh10_close"] = g["close"].rolling(10, min_periods=10).max().reset_index(level=0, drop=True)

    d["hh_fib"] = np.nan
    d["low_since_hh_fib"] = np.nan

    for _, sub in d.groupby("ticker", sort=False):
        if len(sub) < cfg.fib_lookback:
            continue
        hi = sub["high"].to_numpy(dtype=np.float64)
        lo = sub["low"].to_numpy(dtype=np.float64)
        hh, low_since = _hh_and_low_since_hh_py(hi, lo, cfg.fib_lookback)
        d.loc[sub.index, "hh_fib"] = hh
        d.loc[sub.index, "low_since_hh_fib"] = low_since

    return d


def weekly_universe_filter_parquet_only(df: pd.DataFrame, cfg: WeeklySwingConfig) -> pd.DataFrame:
    needed = [
        "sma20",
        "sma50",
        "sma20_slope5",
        "atr5",
        "atr20",
        "hh5",
        "ll5",
        "range5_pct",
        "range20_pct",
        "close_pos_20",
        "turnover_5d",
        "baseline_5d_turnover",
        "adv_turnover_20",
        "hh10_close",
        "hh_fib",
        "low_since_hh_fib",
    ]
    d = df.dropna(subset=needed).copy()
    if d.empty:
        return d

    if cfg.adv_turnover_20_min and cfg.adv_turnover_20_min > 0:
        d = d[d["adv_turnover_20"].astype(float) >= float(cfg.adv_turnover_20_min)].copy()

    return d


def weekly_score(out: pd.DataFrame) -> pd.DataFrame:
    o = out.copy()
    denom = o["baseline_5d_turnover"].replace(0, np.nan)
    o["turnover_expansion"] = (o["turnover_5d"] / denom).replace([np.inf, -np.inf], np.nan)

    o["range_compression_score"] = 1.0 / (o["range5_pct"].clip(lower=1e-6))
    o["dist_20dma"] = (o["close"] - o["sma20"]) / o["sma20"]

    for col in ["turnover_expansion", "range_compression_score", "dist_20dma"]:
        o[col + "_r"] = o.groupby("signal_date")[col].rank(pct=True)

    o["score"] = (
        0.5 * o["turnover_expansion_r"] +
        0.3 * o["range_compression_score_r"] +
        0.2 * o["dist_20dma_r"]
    )
    return o


def weekly_detect_setups(df_raw: pd.DataFrame, cfg: WeeklySwingConfig) -> pd.DataFrame:
    df = df_raw.sort_values(["ticker", "date"]).copy()
    df = weekly_universe_filter_parquet_only(df, cfg)
    if df.empty:
        return pd.DataFrame()

    denom = df["baseline_5d_turnover"].replace(0, np.nan)
    df["turnover_expansion"] = (df["turnover_5d"] / denom).replace([np.inf, -np.inf], np.nan)

    near_high = df["close"] >= (cfg.pause_near_high_frac * df["hh10_close"])
    contraction = (df["range5_pct"] < df["range20_pct"]) | (df["atr5"] < df["atr20"])
    df["pause_day"] = (near_high & contraction).astype(int)

    df["pause_count"] = (
        df.groupby("ticker")["pause_day"]
        .transform(lambda x: x.rolling(cfg.pause_days_max, min_periods=cfg.pause_days_max).sum())
    )

    trend_up = (
        (df["close"] > df["sma20"]) &
        (df["sma20"] > df["sma50"]) &
        (df["sma20_slope5"] >= cfg.sma20_slope_floor_mult * df["sma20"])
    )

    vol_compress = (df["atr5"] < df["atr20"]) & (df["range5_pct"] < df["range20_pct"])
    tight = df["range5_pct"] <= cfg.tight_range_5d_max
    location = df["close_pos_20"] >= cfg.close_pos20_min
    vol_confirm_a = df["turnover_expansion"] >= cfg.turnover_expansion_min_A
    mask_a = trend_up & vol_compress & tight & location & vol_confirm_a
    a_frame = df.loc[mask_a, :].copy()
    if not a_frame.empty:
        a_frame["signal_date"] = a_frame["date"].dt.normalize()
        a_frame["setup_tag"] = "VOL_COMPRESSION_BREAKOUT"
        a_frame["entry_type"] = "BREAKOUT"
        a_frame["breakout_level"] = a_frame["hh5"]
        a_frame["pullback_level"] = np.nan
        a_frame["stop_level"] = a_frame["ll5"]

    mask_pause_base = (
        trend_up &
        (df["pause_count"] >= cfg.pause_days_min) &
        (df["pause_count"] <= cfg.pause_days_max) &
        (df["range20_pct"] <= cfg.pause_range20_max) &
        (df["hh_fib"] <= df["close"] * (1.0 + cfg.max_hh_dist_from_close))
    )
    pb = df.loc[mask_pause_base, :].copy()
    if not pb.empty:
        pb["signal_date"] = pb["date"].dt.normalize()

    c1 = pb.copy()
    if not c1.empty:
        c1 = c1[c1["turnover_expansion"] >= cfg.turnover_expansion_min_C].copy()
        c1["setup_tag"] = "TREND_PAUSE_BREAKOUT"
        c1["entry_type"] = "BREAKOUT"
        c1["breakout_level"] = c1["hh5"]
        c1["pullback_level"] = np.nan
        c1["stop_level"] = c1["ll5"]

    c2 = pb.copy()
    if not c2.empty:
        hh = c2["hh_fib"].astype(float)
        lsh = c2["low_since_hh_fib"].astype(float)
        rng = (hh - lsh).clip(lower=1e-9)
        fib38 = hh - cfg.fib_frac * rng

        c2["setup_tag"] = "TREND_PAUSE_38PULLBACK"
        c2["entry_type"] = "PULLBACK"
        c2["breakout_level"] = c2["hh5"]
        c2["pullback_level"] = fib38
        c2["stop_level"] = c2["ll5"]

        close = c2["close"].astype(float)
        min_level = close * (1.0 - cfg.pullback_max_pct_below_close)
        max_level = close * (1.0 - cfg.pullback_min_pct_below_close)
        c2 = c2[(c2["pullback_level"] >= min_level) & (c2["pullback_level"] <= max_level)].copy()

    frames = [x for x in [a_frame, c1, c2] if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["signal_date", "ticker", "setup_tag"], keep="first")
    out = weekly_score(out)

    keep = [
        "signal_date",
        "ticker",
        "setup_tag",
        "entry_type",
        "score",
        "breakout_level",
        "pullback_level",
        "stop_level",
        "close",
        "sma20",
        "sma50",
        "turnover_expansion",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values(["signal_date", "score"], ascending=[False, False]).reset_index(drop=True)


def weekly_current_signals(df_prices: pd.DataFrame, cfg: WeeklySwingConfig, n_days: int = 3) -> pd.DataFrame:
    d = weekly_add_indicators(df_prices, cfg)
    sig = weekly_detect_setups(d, cfg)
    if sig.empty:
        return sig

    cal = pd.to_datetime(pd.Index(sorted(d["date"].dt.normalize().unique()))).normalize()
    last_dates = cal[-n_days:] if len(cal) >= n_days else cal

    sig = sig[pd.to_datetime(sig["signal_date"]).isin(last_dates)].copy()
    if sig.empty:
        return sig

    sig = sig.sort_values(["ticker", "signal_date", "score"], ascending=[True, False, False])
    sig = sig.groupby("ticker", as_index=False).head(1)

    sig["System"] = sig["setup_tag"].apply(lambda x: f"weekly_swing::{x}")
    sig["Signal"] = sig["entry_type"]
    return sig.sort_values(["signal_date", "score"], ascending=[False, False]).reset_index(drop=True)


# ============================================================
# 6) SYSTEM 2: FIBONACCI
# ============================================================

LOOKBACK_DAYS = 300


def find_swing_as_of_quick(group: pd.DataFrame, current_date: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS):
    window = group[
        (group["date"] <= current_date)
        & (group["date"] >= (current_date - pd.Timedelta(days=lookback_days)))
    ].copy()
    if len(window) < 10:
        return None

    highs = window["high"].values
    lows = window["low"].values
    dates = window["date"].values

    look = 5
    pivots = []
    for i in range(look, len(highs) - look):
        if highs[i] == max(highs[i - look: i + look + 1]):
            pivots.append(i)
    if not pivots:
        return None

    best_rel_idx = max(pivots, key=lambda idx: highs[idx])
    swing_high_price = float(highs[best_rel_idx])
    swing_high_date = pd.to_datetime(dates[best_rel_idx])

    prior_segment = window.iloc[: best_rel_idx + 1]
    low_pos = prior_segment["low"].idxmin()

    swing_low_price = float(group.loc[low_pos, "low"])
    swing_low_date = pd.to_datetime(group.loc[low_pos, "date"])

    if swing_low_price >= swing_high_price:
        return None

    swing_range = swing_high_price - swing_low_price
    return {
        "Swing Low Date": swing_low_date,
        "Swing Low Price": swing_low_price,
        "Swing High Date": swing_high_date,
        "Swing High Price": swing_high_price,
        "Retrace 50": swing_high_price - 0.50 * swing_range,
        "Retrace 61": swing_high_price - 0.618 * swing_range,
        "Stop Consider (78.6%)": swing_high_price - 0.786 * swing_range,
    }


def fib_build_watchlist(df: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    rows = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        latest_price = float(g["close"].iloc[-1])
        latest_date = pd.to_datetime(g["date"].iloc[-1])

        swing = find_swing_as_of_quick(g, latest_date, lookback_days)
        if swing is None:
            continue

        post_high = g[(g["date"] > swing["Swing High Date"]) & (g["date"] <= latest_date)]
        if (not post_high.empty) and (post_high["low"] < swing["Stop Consider (78.6%)"]).any():
            continue

        retracement = (swing["Swing High Price"] - latest_price) / (swing["Swing High Price"] - swing["Swing Low Price"])
        if retracement >= 0.38:
            rows.append({
                "Ticker": ticker,
                "Latest Date": latest_date,
                "Latest Price": latest_price,
                "Swing Low Date": swing["Swing Low Date"],
                "Swing Low Price": float(swing["Swing Low Price"]),
                "Swing High Date": swing["Swing High Date"],
                "Swing High Price": float(swing["Swing High Price"]),
                "Swing Range": float(swing["Swing High Price"] - swing["Swing Low Price"]),
                "Retracement": float(retracement),
            })
    return pd.DataFrame(rows)


def shape_priority(shape: str) -> int:
    order = {
        "consolidation under BOS": 1,
        "rounded recovery": 2,
        "strong recovery": 3,
        "normal recovery": 4,
        "V-reversal": 5,
        "volatile pullback": 6,
        "insufficient data": 7,
    }
    return order.get(shape, 7)


def setup_shape(g: pd.DataFrame, retr_low_date, last_local_high):
    post = g[g["date"] > retr_low_date].copy()
    if post.empty or len(post) < 6:
        return "insufficient data"

    closes = post["close"].values
    highs = post["high"].values
    lows = post["low"].values
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, 1)
    slope = coeffs[0]

    fitted = np.polyval(coeffs, x)
    noise = np.std(closes - fitted)
    noise_ratio = noise / np.mean(closes)

    total_up = closes[-1] - closes[0]
    range_up = max(closes) - min(closes)
    recovery_pct = 0 if range_up == 0 else total_up / range_up

    if last_local_high is not None and np.isfinite(last_local_high) and last_local_high != 0:
        dist_to_bos = (last_local_high - closes[-1]) / last_local_high
    else:
        dist_to_bos = None

    if dist_to_bos is not None and dist_to_bos < 0.02 and noise_ratio < 0.008:
        return "consolidation under BOS"
    if slope > 0 and noise_ratio < 0.015 and recovery_pct > 0.60:
        return "rounded recovery"
    if slope > 0 and recovery_pct > 0.75:
        return "strong recovery"
    if slope > np.mean(closes) * 0.0008 and recovery_pct > 0.85 and noise_ratio < 0.02:
        return "V-reversal"
    if noise_ratio > 0.03:
        return "volatile pullback"
    return "normal recovery"


def fib_confirmation_engine(df_prices: pd.DataFrame, watch: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in watch.iterrows():
        ticker = row["Ticker"]
        swing_low_date = row["Swing Low Date"]
        swing_high_date = row["Swing High Date"]

        g = df_prices[df_prices["ticker"] == ticker].sort_values("date").copy()
        if g.empty or len(g) < 40:
            continue

        fib50 = row["Swing High Price"] - 0.50 * (row["Swing High Price"] - row["Swing Low Price"])
        fib786 = row["Swing High Price"] - 0.786 * (row["Swing High Price"] - row["Swing Low Price"])

        correction = g[(g["date"] > swing_high_date) & (g["date"] <= row["Latest Date"])].copy()
        if correction.empty:
            continue

        retr_idx = correction["low"].idxmin()
        retr_low_price = float(correction.loc[retr_idx, "low"])
        retr_low_date = pd.to_datetime(correction.loc[retr_idx, "date"])

        post = g[g["date"] > retr_low_date].copy()

        retr_in_zone = (retr_low_price <= fib50) and (retr_low_price >= fib786)
        no_lower_after = True if post.empty else (post["low"].min() >= retr_low_price)
        retracement_floor_respected = retr_in_zone and no_lower_after

        higher_low_found = False
        hl_price = np.nan
        if len(post) >= 3:
            lows = post["low"].values
            pivot_lows = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    pivot_lows.append(i)
            for idx in pivot_lows:
                pivot_low = lows[idx]
                if pivot_low <= retr_low_price:
                    continue
                if post["low"].iloc[idx + 1:].min() < pivot_low:
                    continue
                has_green_follow_through = False
                if idx + 1 < len(post):
                    if post["close"].iloc[idx + 1] > post["open"].iloc[idx + 1]:
                        has_green_follow_through = True

                broke_minor_high = False
                if idx + 2 < len(post):
                    minor_high = max(post["high"].iloc[idx: idx + 2])
                    if post["high"].iloc[idx + 2:].max() > minor_high:
                        broke_minor_high = True

                if has_green_follow_through or broke_minor_high:
                    higher_low_found = True
                    hl_price = float(pivot_low)
                    break

        bullish_candle = False
        corr = correction.reset_index(drop=True)
        for i in range(2, len(corr)):
            o = corr["open"].iloc[i]
            c = corr["close"].iloc[i]
            h = corr["high"].iloc[i]
            l = corr["low"].iloc[i]
            body = abs(c - o)
            range_ = max(h - l, 1e-9)
            lower_wick = (o - l) if c >= o else (c - l)

            o1 = corr["open"].iloc[i - 1]
            c1 = corr["close"].iloc[i - 1]
            o2 = corr["open"].iloc[i - 2]
            c2 = corr["close"].iloc[i - 2]

            in_fib_zone = (l <= fib50) and (l >= fib786)

            hammer = (
                in_fib_zone
                and lower_wick > 0.6 * range_
                and c >= o
            )

            engulf = (
                in_fib_zone
                and (c > o1)
                and (o < c1)
                and (c1 < o1)
            )

            morning_star = (
                in_fib_zone
                and (c1 < o1)
                and (abs(c2 - o2) <= 0.3 * (corr["high"].iloc[i - 2] - corr["low"].iloc[i - 2]))
                and (c > (o1 + c1) / 2)
            )

            piercing = (
                in_fib_zone
                and (c1 < o1)
                and (o < c1)
                and (c > (o1 + c1) / 2)
            )

            tweezer = (
                abs(l - corr["low"].iloc[i - 1]) <= 0.2 * range_
                and in_fib_zone
                and (c >= o)
            )

            strong_reversal = (
                in_fib_zone
                and c >= l + 0.6 * range_
            )

            if hammer or engulf or morning_star or piercing or tweezer or strong_reversal:
                bullish_candle = True
                break

        corr2 = g[(g["date"] > retr_low_date) & (g["date"] < row["Latest Date"])].copy()
        if corr2.empty or len(corr2) < 3:
            last_local_high = np.nan
            bos = False
        else:
            highs = corr2["high"].values
            pivot_highs = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 2]:
                    pivot_highs.append(highs[i])
            bos_level = max(pivot_highs) if pivot_highs else float(corr2["high"].max())
            last_local_high = float(bos_level)
            post2 = g[g["date"] > retr_low_date]
            bos = (post2["close"] > bos_level).any()

        gp = g.copy()
        gp["SMA10"] = gp["close"].rolling(10).mean()

        delta = gp["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()
        rs = roll_up / roll_down
        gp["RSI"] = 100 - (100 / (1 + rs))

        ema12 = gp["close"].ewm(span=12, adjust=False).mean()
        ema26 = gp["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        gp["MACDH"] = macd - signal

        last_row = gp.iloc[-1]
        close_now = float(last_row["close"])
        sma10 = float(last_row["SMA10"]) if pd.notna(last_row["SMA10"]) else np.nan
        rsi_now = float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else np.nan
        macdh_now = float(last_row["MACDH"]) if pd.notna(last_row["MACDH"]) else np.nan

        cond1 = close_now > sma10
        cond2 = macdh_now > 0
        cond3 = rsi_now > 50
        two_of_three = (int(cond1) + int(cond2) + int(cond3)) >= 2

        macd_line = macd.iloc[-1]
        macd_line_prev = macd.iloc[-2]
        macd_cross_up = macd_line > macd_line_prev

        rsi_strong = rsi_now > 55
        last3_high = gp["high"].iloc[-3:].max()
        price_breakout = close_now > last3_high

        breakout_momentum = macd_cross_up or (rsi_strong and price_breakout)

        momentum_ok = two_of_three or breakout_momentum

        shape = setup_shape(g, retr_low_date, last_local_high)
        shape_pr = shape_priority(shape)

        retracement_held = (retracement_floor_respected and higher_low_found and bullish_candle)
        uptrend_resumed = (bos and momentum_ok)

        final_signal = "BUY" if (retracement_held and uptrend_resumed) else "WATCH" if retracement_held else "INVALID"

        bos_prox = 0.0
        if np.isfinite(last_local_high) and last_local_high != 0:
            bos_prox = float(np.clip(1 - ((last_local_high - close_now) / max(close_now, 1e-9)), 0, 1))

        readiness = 100 * (
            0.25 * int(retracement_held) +
            0.20 * int(higher_low_found) +
            0.15 * int(bullish_candle) +
            0.20 * int(momentum_ok) +
            0.20 * bos_prox
        )

        results.append({
            "ticker": ticker,
            "System": "fibonacci",
            "Signal": final_signal,
            "Signal_Date": pd.to_datetime(row["Latest Date"]).normalize(),
            "READINESS_SCORE": round(float(np.clip(readiness, 0, 100)), 2),
            "LastLocalHigh": float(last_local_high) if np.isfinite(last_local_high) else np.nan,
            "HL_Price": float(hl_price) if np.isfinite(hl_price) else np.nan,
            "LatestPrice": close_now,
            "Shape": shape,
            "ShapePriority": shape_pr,
            "SwingLow": float(row["Swing Low Price"]),
            "SwingHigh": float(row["Swing High Price"]),
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    sig_rank = out["Signal"].map({"BUY": 0, "WATCH": 1, "INVALID": 2}).fillna(9)
    out = out.assign(_r=sig_rank).sort_values(["_r", "READINESS_SCORE"], ascending=[True, False]).drop(columns=["_r"])
    return out.reset_index(drop=True)


# ============================================================
# 7) SYSTEM 3: MOMENTUM BUCKET C
# ============================================================

def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values(["ticker", "date"]).copy()
    d["1D Return"] = d.groupby("ticker")["close"].pct_change()
    return d


def attach_benchmark_and_alpha(df_prices: pd.DataFrame, idx_returns: pd.DataFrame) -> pd.DataFrame:
    d = df_prices.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values(["ticker", "date"]).copy()

    d["stock_ret_1d"] = d.groupby("ticker")["close"].pct_change()

    d["index_name"] = [
        infer_index_name_for_row(t, idxv) for t, idxv in zip(d["ticker"].astype(str), d.get("index", "UNKNOWN"))
    ]

    idx = idx_returns.copy()
    idx = idx[idx["index_name"].isin(["SP500", "HSI", "STI"])].copy()

    if "ret_1d" not in idx.columns:
        raise ValueError(f"Index returns parquet must contain 'ret_1d'. Found: {sorted(idx.columns)}")

    idx_small = idx[["date", "index_name", "ret_1d"]].rename(columns={"ret_1d": "bm_ret_1d"}).copy()

    d = d.merge(idx_small, on=["date", "index_name"], how="left")

    d["alpha_1d"] = d["stock_ret_1d"] - d["bm_ret_1d"]

    d["bm_ret_1d"] = d["bm_ret_1d"].fillna(0.0)
    d["alpha_1d"] = d["alpha_1d"].fillna(0.0)

    return d


def calculate_momentum_features(
    df: pd.DataFrame,
    windows=(5, 10, 30, 45, 60, 90),
    base_col: str = "1D Return",
) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["ticker", "date"]).copy()

    if base_col not in d.columns:
        raise ValueError(f"calculate_momentum_features: missing base_col='{base_col}'")

    gross_col = "__gross__"
    d[gross_col] = 1.0 + pd.to_numeric(d[base_col], errors="coerce").fillna(0.0)

    for w in windows:
        r = f"{w}D Return"
        z = f"{w}D zscore"
        dz = f"{w}D zscore change"

        d[r] = (
            d.groupby("ticker")[gross_col]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1.0
        )

        mean = d.groupby("date")[r].transform("mean")
        std = d.groupby("date")[r].transform("std").replace(0, np.nan)
        d[z] = ((d[r] - mean) / std)

        d[dz] = (
            d.groupby("ticker")[z]
            .diff()
            .ewm(span=w, adjust=False)
            .mean()
        )

    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].fillna(0.0)

    if gross_col in d.columns:
        d = d.drop(columns=[gross_col])

    return d


def add_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Momentum_Fast"] = (0.6 * d["5D zscore"] + 0.4 * d["10D zscore"])
    d["Momentum_Mid"] = (0.5 * d["30D zscore"] + 0.5 * d["45D zscore"])
    d["Momentum_Slow"] = (0.5 * d["60D zscore"] + 0.5 * d["90D zscore"])
    d["Momentum Score"] = (0.5 * d["Momentum_Slow"] + 0.3 * d["Momentum_Mid"] + 0.2 * d["Momentum_Fast"])
    return d.fillna(0.0)


def add_regime_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Accel_Fast"] = d.groupby("ticker")["Momentum_Fast"].diff()
    d["Accel_Mid"] = d.groupby("ticker")["Momentum_Mid"].diff()
    d["Accel_Slow"] = d.groupby("ticker")["Momentum_Slow"].diff()

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    d["Accel_Fast_z"] = d.groupby("date")["Accel_Fast"].transform(zscore_safe)
    d["Accel_Mid_z"] = d.groupby("date")["Accel_Mid"].transform(zscore_safe)
    d["Accel_Slow_z"] = d.groupby("date")["Accel_Slow"].transform(zscore_safe)

    d["Acceleration Score"] = (0.5 * d["Accel_Fast_z"] + 0.3 * d["Accel_Mid_z"] + 0.2 * d["Accel_Slow_z"])
    return d.fillna(0.0)


def add_regime_residual_momentum(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Residual_Momentum"] = d["Momentum_Fast"] - d.groupby("ticker")["Momentum_Slow"].transform("mean")

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    d["Residual_Momentum_z"] = d.groupby("date")["Residual_Momentum"].transform(zscore_safe)
    return d.fillna(0.0)


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Early_Fast"] = (0.6 * d["Accel_Fast_z"] + 0.4 * d["Momentum_Fast"])
    d["Early_Mid"] = (0.5 * d["Accel_Mid_z"] + 0.5 * d["Momentum_Mid"])
    d["Early_Slow"] = (0.5 * d["Accel_Slow_z"] + 0.5 * d["Momentum_Slow"])
    d["Early Momentum Score"] = (0.5 * d["Early_Slow"] + 0.3 * d["Early_Mid"] + 0.2 * d["Early_Fast"])
    return d.fillna(0.0)


def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    records = []
    for d0 in sorted(df["date"].unique()):
        snap = df[df["date"] == d0].sort_values("Momentum Score", ascending=False).head(top_n).copy()
        if snap.empty:
            continue
        snap["Rank"] = np.arange(1, len(snap) + 1)
        for _, r in snap.iterrows():
            records.append({
                "date": d0,
                "ticker": r["ticker"],
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
    top_n: int = 10,
) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    if as_of_date is None:
        as_of_date = daily_df["date"].max()

    dates = sorted(daily_df.loc[daily_df["date"] <= as_of_date, "date"].unique(), reverse=True)[:lookback_days]
    if not dates:
        return pd.DataFrame()

    window = daily_df[daily_df["date"].isin(dates)]
    if window.empty:
        return pd.DataFrame()

    agg = (
        window.groupby("ticker")
        .agg(
            Momentum_Score=("Momentum Score", "mean"),
            Early_Momentum_Score=("Early Momentum Score", "mean"),
            Appearances=("date", "count"),
            Rank_Mean=("Rank", "mean"),
            Rank_Std=("Rank", "std"),
        )
        .reset_index()
    )
    agg["Rank_Std"] = agg["Rank_Std"].fillna(0.0)
    agg["Consistency"] = agg["Appearances"] / len(dates)

    agg["Weighted_Score"] = (
        w_momentum * agg["Momentum_Score"] +
        w_early * agg["Early_Momentum_Score"] +
        w_consistency * agg["Consistency"]
    )

    agg["ConsistencyScore"] = agg["Consistency"] * 100.0
    max_std = max(1.0, top_n / 2.0)
    agg["RankStabilityScore"] = (1.0 - (agg["Rank_Std"] / max_std)).clip(0.0, 1.0) * 100.0
    agg["Signal_Confidence"] = 0.5 * agg["ConsistencyScore"] + 0.5 * agg["RankStabilityScore"]

    return agg.sort_values("Weighted_Score", ascending=False).head(top_n).reset_index(drop=True)


def momentum_bucketC_latest(
    df_prices: pd.DataFrame,
    index_returns: pd.DataFrame,
    top_n_daily: int = 10,
    final_top_n: int = 10,
    lookback_days: int = 10,
) -> pd.DataFrame:
    d0 = df_prices.copy()

    if "index" in d0.columns and (d0["index"] != "UNKNOWN").any():
        if (d0["index"].astype(str) == "SP500").any():
            d0 = d0[d0["index"].astype(str) == "SP500"].copy()

    if d0["date"].nunique() < 120:
        return pd.DataFrame()

    df_a = add_absolute_returns(d0)
    df_a = calculate_momentum_features(df_a, base_col="1D Return")
    df_a = add_regime_momentum_score(df_a)
    df_a = add_regime_acceleration(df_a)
    df_a = add_regime_residual_momentum(df_a)
    df_a = add_regime_early_momentum(df_a)
    daily_a = build_daily_lists(df_a, top_n=top_n_daily)

    d_alpha = attach_benchmark_and_alpha(d0, index_returns)

    df_b = d_alpha.copy()
    df_b = calculate_momentum_features(df_b, base_col="alpha_1d")
    df_b = add_regime_momentum_score(df_b)
    df_b = add_regime_acceleration(df_b)
    df_b = add_regime_residual_momentum(df_b)
    df_b = add_regime_early_momentum(df_b)

    df_b = df_b[(df_b["Momentum_Slow"] > 0.25) & (df_b["Momentum_Mid"] > 0.10)].copy()
    daily_b = build_daily_lists(df_b, top_n=top_n_daily)

    common_dates = sorted(set(daily_a["date"]).intersection(set(daily_b["date"])))
    if not common_dates:
        return pd.DataFrame()
    as_of = common_dates[-1]

    sel_a = final_selection_from_daily(daily_a, lookback_days=lookback_days, as_of_date=as_of, top_n=final_top_n).copy()
    sel_b = final_selection_from_daily(daily_b, lookback_days=lookback_days, as_of_date=as_of, top_n=final_top_n).copy()

    frames = []
    total_capital = 100_000.0
    weight_a, weight_b = 0.20, 0.80

    if not sel_a.empty:
        sel_a["Bucket"] = "A"
        sel_a["Target_dollars"] = (total_capital * weight_a) / len(sel_a)
        frames.append(sel_a)
    if not sel_b.empty:
        sel_b["Bucket"] = "B"
        sel_b["Target_dollars"] = (total_capital * weight_b) / len(sel_b)
        frames.append(sel_b)

    if not frames:
        return pd.DataFrame()

    combo = pd.concat(frames, ignore_index=True)

    out = (
        combo.groupby("ticker", as_index=False)
        .agg(
            Target_dollars=("Target_dollars", "sum"),
            Signal_Confidence=("Signal_Confidence", "max"),
            Weighted_Score=("Weighted_Score", "max"),
            Momentum_Score=("Momentum_Score", "max"),
            Early_Momentum_Score=("Early_Momentum_Score", "max"),
            Consistency=("Consistency", "max"),
            Bucket_Source=("Bucket", lambda x: "+".join(sorted(set(x)))),
        )
        .sort_values(["Signal_Confidence", "Weighted_Score"], ascending=[False, False])
        .reset_index(drop=True)
    )

    out["System"] = "momentum_bucketC"
    out["Signal"] = "HOLDINGS_CANDIDATE"
    out["Signal_Date"] = pd.to_datetime(as_of)

    total = float(out["Target_dollars"].sum()) if len(out) else 0.0
    out["Weight_%"] = (out["Target_dollars"] / total * 100.0) if total > 0 else np.nan
    out["Consistency_%"] = out["Consistency"] * 100.0

    keep = [
        "ticker",
        "System",
        "Signal",
        "Signal_Date",
        "Weight_%",
        "Target_dollars",
        "Signal_Confidence",
        "Weighted_Score",
        "Bucket_Source",
        "Consistency_%",
    ]
    return out[keep].head(25).reset_index(drop=True)


# ============================================================
# 8) RAW REFRESH + SIGNAL BUILD
# ============================================================

def refresh_raw_parquets():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    sti = get_sti_universe()

    frames = []
    frames += download_5yr_ohlc(sp500["Ticker"].tolist(), "SP500")
    frames += download_5yr_ohlc(hsi["Ticker"].tolist(), "HSI")
    frames += download_5yr_ohlc(sti["Ticker"].tolist(), "STI")

    full_constituents = pd.concat(frames, ignore_index=True)
    full_constituents.to_parquet(RAW_CONSTITUENTS_PATH, index=False)
    print(f"Saved {RAW_CONSTITUENTS_PATH}")

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


def build_signal_parquets():
    df_prices = load_prices_from_parquet(RAW_CONSTITUENTS_PATH)
    idx_returns = load_index_returns_from_parquet(RAW_INDEX_RETURNS_PATH)

    weekly_cfg = WeeklySwingConfig(
        adv_turnover_20_min=0.0,
        breakout_buffer_pct=0.001,
    )

    weekly_sig = weekly_current_signals(df_prices, weekly_cfg, n_days=3)
    if not weekly_sig.empty:
        keep = [
            "ticker",
            "System",
            "Signal",
            "signal_date",
            "score",
            "breakout_level",
            "pullback_level",
            "stop_level",
            "close",
            "turnover_expansion",
        ]
        keep = [c for c in keep if c in weekly_sig.columns]
        weekly_sig = weekly_sig[keep].sort_values(["signal_date", "score"], ascending=[False, False]).reset_index(drop=True)

    watch = fib_build_watchlist(df_prices, lookback_days=LOOKBACK_DAYS)
    fib_sig = pd.DataFrame()
    if watch is not None and not watch.empty:
        fib_sig = fib_confirmation_engine(df_prices, watch)
        if not fib_sig.empty:
            fib_sig = fib_sig.sort_values(["Signal", "READINESS_SCORE"], ascending=[True, False]).reset_index(drop=True)

    mom_sig = momentum_bucketC_latest(
        df_prices=df_prices,
        index_returns=idx_returns,
        top_n_daily=10,
        final_top_n=10,
        lookback_days=10,
    )

    def _best_weekly(x: pd.DataFrame) -> pd.Series:
        x = x.sort_values("score", ascending=False)
        r = x.iloc[0]
        return pd.Series({
            "weekly_tag": r.get("System", np.nan),
            "weekly_signal": r.get("Signal", np.nan),
            "weekly_score": r.get("score", np.nan),
            "weekly_date": r.get("signal_date", np.nan),
            "weekly_breakout": r.get("breakout_level", np.nan),
            "weekly_stop": r.get("stop_level", np.nan),
        })

    def _best_fib(x: pd.DataFrame) -> pd.Series:
        order = {"BUY": 0, "WATCH": 1, "INVALID": 2}
        x = x.copy()
        x["_o"] = x["Signal"].map(order).fillna(9)
        x = x.sort_values(["_o", "READINESS_SCORE"], ascending=[True, False])
        r = x.iloc[0]
        return pd.Series({
            "fib_signal": r.get("Signal", np.nan),
            "fib_readiness": r.get("READINESS_SCORE", np.nan),
            "fib_shape": r.get("Shape", np.nan),
            "fib_last_local_high": r.get("LastLocalHigh", np.nan),
        })

    def _best_mom(x: pd.DataFrame) -> pd.Series:
        x = x.sort_values(["Signal_Confidence", "Weight_%"], ascending=[False, False])
        r = x.iloc[0]
        return pd.Series({
            "mom_signal": r.get("Signal", np.nan),
            "mom_conf": r.get("Signal_Confidence", np.nan),
            "mom_weight": r.get("Weight_%", np.nan),
            "mom_bucket": r.get("Bucket_Source", np.nan),
            "mom_date": r.get("Signal_Date", np.nan),
        })

    if weekly_sig is not None and not weekly_sig.empty:
        w = weekly_sig.groupby("ticker").apply(_best_weekly).reset_index()
    else:
        w = pd.DataFrame(columns=["ticker"])

    if fib_sig is not None and not fib_sig.empty:
        f = fib_sig.groupby("ticker").apply(_best_fib).reset_index()
    else:
        f = pd.DataFrame(columns=["ticker"])

    if mom_sig is not None and not mom_sig.empty:
        m = mom_sig.groupby("ticker").apply(_best_mom).reset_index()
    else:
        m = pd.DataFrame(columns=["ticker"])

    combined = w.merge(f, on="ticker", how="outer").merge(m, on="ticker", how="outer")

    combined["weekly_score_100"] = pd.to_numeric(combined.get("weekly_score"), errors="coerce") * 100.0
    combined["fib_readiness"] = pd.to_numeric(combined.get("fib_readiness"), errors="coerce")
    combined["mom_conf"] = pd.to_numeric(combined.get("mom_conf"), errors="coerce")

    fib_bump = combined["fib_signal"].map({"BUY": 15, "WATCH": 5, "INVALID": 0}).fillna(0)

    combined["ACTION_SCORE"] = (
        0.45 * combined["weekly_score_100"].fillna(0) +
        0.35 * combined["fib_readiness"].fillna(0) +
        0.20 * combined["mom_conf"].fillna(0) +
        fib_bump
    )

    view_cols = [
        "ticker",
        "ACTION_SCORE",
        "weekly_signal",
        "weekly_tag",
        "weekly_date",
        "weekly_breakout",
        "weekly_stop",
        "fib_signal",
        "fib_readiness",
        "fib_shape",
        "mom_conf",
        "mom_weight",
        "mom_bucket",
        "mom_date",
    ]
    view_cols = [c for c in view_cols if c in combined.columns]

    combined = combined.sort_values("ACTION_SCORE", ascending=False).reset_index(drop=True)

    checks = [
        GoldenSourceCheck(
            label="weekly",
            path=GOLDEN_WEEKLY_SIGNALS_PATH,
            key_columns=["signal_date", "ticker"],
            compare_columns=[
                "System",
                "Signal",
                "score",
                "breakout_level",
                "pullback_level",
                "stop_level",
                "close",
                "turnover_expansion",
            ],
        ),
        GoldenSourceCheck(
            label="fib",
            path=GOLDEN_FIB_SIGNALS_PATH,
            key_columns=["Signal_Date", "ticker"],
            compare_columns=[
                "System",
                "Signal",
                "READINESS_SCORE",
                "LastLocalHigh",
                "HL_Price",
                "LatestPrice",
                "Shape",
                "ShapePriority",
            ],
        ),
    ]
    validation_messages = validate_golden_source(
        checks,
        {
            "weekly": weekly_sig,
            "fib": fib_sig,
        },
    )
    for message in validation_messages:
        print(message)

    weekly_sig.to_parquet(WEEKLY_SIGNALS_PATH, index=False)
    fib_sig.to_parquet(FIB_SIGNALS_PATH, index=False)
    mom_sig.to_parquet(MOMENTUM_SIGNALS_PATH, index=False)
    combined[view_cols].to_parquet(ACTION_LIST_PATH, index=False)

    print(f"Saved {WEEKLY_SIGNALS_PATH}")
    print(f"Saved {FIB_SIGNALS_PATH}")
    print(f"Saved {MOMENTUM_SIGNALS_PATH}")
    print(f"Saved {ACTION_LIST_PATH}")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    refresh_raw_parquets()
    build_signal_parquets()

    print("\n✅ Overnight batch complete.")


if __name__ == "__main__":
    main()
