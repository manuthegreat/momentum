# update_parquet.py
# Nightly batch job:
# 1) Refresh raw artifacts (constituents OHLC + index returns) from Yahoo
# 2) Run consolidated signals pipeline (Weekly Swing, Fibonacci, Momentum Bucket C)
# 3) Persist signal parquets so Streamlit becomes UI-only

from __future__ import annotations

import os
import sys
import warnings
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

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

UNIVERSE_RULES = {
    "HK": dict(
        mcap_min=5e9,
        mcap_max=250e9,
        adv_turnover_20_min=100e6,
    ),
    "US": dict(
        mcap_min=2e9,
        mcap_max=150e9,
        adv_turnover_20_min=75e6,
    ),
    "SG": dict(
        mcap_min=1e9,
        mcap_max=80e9,
        adv_turnover_20_min=10e6,
    ),
}


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

    df["index_name"] = df.apply(
        lambda row: infer_index_name_for_row(row["ticker"], row["index"]),
        axis=1,
    )
    df["country"] = df["index_name"].map({"SP500": "US", "HSI": "HK", "STI": "SG"}).fillna("US")

    if "market_cap" not in df.columns or df["market_cap"].isna().all():
        tickers = df["ticker"].dropna().astype(str).unique().tolist()

        def _fundamentals_one(tkr: str) -> Tuple[str, Optional[float], Optional[float]]:
            last_err = None
            for attempt in range(3):
                try:
                    yt = yf.Ticker(tkr)

                    shares = None
                    mcap_now = None

                    fi = getattr(yt, "fast_info", None)
                    if fi is not None and hasattr(fi, "get"):
                        try:
                            shares = fi.get("shares", None) or fi.get("shares_outstanding", None)
                            mcap_now = fi.get("market_cap", None) or fi.get("marketCap", None)
                        except Exception:
                            pass

                    if shares is None or mcap_now is None:
                        try:
                            info = yt.get_info()
                            if shares is None:
                                shares = info.get("sharesOutstanding", None)
                            if mcap_now is None:
                                mcap_now = info.get("marketCap", None)
                        except Exception:
                            pass

                    def _clean(x):
                        try:
                            x = float(x)
                            return x if np.isfinite(x) and x > 0 else None
                        except Exception:
                            return None

                    return tkr, _clean(shares), _clean(mcap_now)
                except Exception as e:
                    last_err = e
                    time.sleep(min(2.0, 0.2 * (attempt + 1)))

            print(f"[WARN] fundamentals failed for {tkr}: {last_err}")
            return tkr, None, None

        shares_map: Dict[str, Optional[float]] = {}
        mcap_now_map: Dict[str, Optional[float]] = {}

        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_fundamentals_one, t): t for t in tickers}
            for fut in as_completed(futs):
                tkr, so, mc = fut.result()
                shares_map[tkr] = so
                mcap_now_map[tkr] = mc

        df["shares_outstanding"] = pd.to_numeric(df["ticker"].map(shares_map), errors="coerce")
        market_cap_hist = df["shares_outstanding"] * df["close"].astype(float)
        market_cap_now = pd.to_numeric(df["ticker"].map(mcap_now_map), errors="coerce")
        df["market_cap"] = market_cap_hist.fillna(market_cap_now)
    else:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

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
# 5) SYSTEM 1: WEEKLY SWING (ALPHA5)
# ============================================================

def _norm_date(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce").tz_localize(None).normalize()


def _norm_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.tz_localize(None).dt.normalize()
    return out


def _to_num(s, fill: Optional[float] = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(fill) if fill is not None else x


def _rolling_prod_1p(ret: pd.Series, window: int) -> pd.Series:
    gross = 1.0 + ret.fillna(0.0)
    return (
        gross.rolling(window, min_periods=window)
        .apply(np.prod, raw=True) - 1.0
    )


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    mean = df.groupby("date")[col].transform("mean")
    std = df.groupby("date")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).fillna(0.0)


def normalize_01_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    x = df[col].astype(float)
    lo = df.groupby("date")[col].transform("min")
    hi = df.groupby("date")[col].transform("max")
    denom = (hi - lo).replace(0, np.nan)
    return ((x - lo) / denom).fillna(0.0).clip(0.0, 1.0)


def attach_benchmark_and_alpha(
    df_prices: pd.DataFrame,
    idx_returns: pd.DataFrame,
    bm_lookback: int = 20,
) -> pd.DataFrame:
    d = _norm_date_col(df_prices, "date").sort_values(["ticker", "date"]).copy()
    d["stock_ret_1d"] = d.groupby("ticker")["close"].pct_change()

    idx_col = d["index"] if "index" in d.columns else "UNKNOWN"
    d["index_name"] = [
        infer_index_name_for_row(t, idxv)
        for t, idxv in zip(d["ticker"].astype(str), idx_col)
    ]

    idx = idx_returns.copy()
    if "ret_1d" not in idx.columns:
        raise ValueError("Index returns parquet must contain 'ret_1d'.")

    idx = _norm_date_col(idx, "date")
    idx["index_name"] = idx["index_name"].astype(str).str.upper().str.strip()
    idx = idx[idx["index_name"].isin(["SP500", "HSI", "STI"])].copy()

    idx_small = idx[["date", "index_name", "ret_1d"]].rename(columns={"ret_1d": "bm_ret_1d"}).copy()
    idx_small["bm_ret_1d"] = _to_num(idx_small["bm_ret_1d"], fill=0.0)

    idx_small = idx_small.sort_values(["index_name", "date"]).copy()
    idx_small["bm_cumret_lb"] = (
        idx_small.groupby("index_name", sort=False)["bm_ret_1d"]
        .apply(lambda s: _rolling_prod_1p(s, int(bm_lookback)))
        .reset_index(level=0, drop=True)
    )
    idx_small["bm_cumret_lb"] = _to_num(idx_small["bm_cumret_lb"], fill=0.0)

    d = d.merge(idx_small, on=["date", "index_name"], how="left")
    d["bm_ret_1d"] = _to_num(d["bm_ret_1d"], fill=0.0)
    d["bm_cumret_lb"] = _to_num(d["bm_cumret_lb"], fill=0.0)

    d["alpha_1d"] = (_to_num(d["stock_ret_1d"], fill=0.0) - d["bm_ret_1d"]).fillna(0.0)
    return d


def calculate_momentum_features(
    df: pd.DataFrame,
    windows=(5, 10, 30, 45, 60, 90),
    base_col: str = "alpha_1d",
) -> pd.DataFrame:
    d = df.sort_values(["ticker", "date"]).copy()
    if base_col not in d.columns:
        raise ValueError(f"calculate_momentum_features: missing base_col='{base_col}'")

    base = _to_num(d[base_col], fill=0.0)

    for w in windows:
        r = f"{w}D Return"
        z = f"{w}D zscore"
        dz = f"{w}D zscore change"

        d[r] = (
            d.groupby("ticker", sort=False)[base_col]
            .apply(lambda s: _rolling_prod_1p(_to_num(s, fill=0.0), int(w)))
            .reset_index(level=0, drop=True)
        )

        d[z] = _zscore_by_date(d, r)

        d[dz] = (
            d.groupby("ticker", sort=False)[z]
            .diff()
            .ewm(span=int(w), adjust=False)
            .mean()
        )

    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].fillna(0.0)
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
    d["Accel_Fast"] = d.groupby("ticker", sort=False)["Momentum_Fast"].diff()
    d["Accel_Mid"] = d.groupby("ticker", sort=False)["Momentum_Mid"].diff()
    d["Accel_Slow"] = d.groupby("ticker", sort=False)["Momentum_Slow"].diff()

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


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Early_Fast"] = (0.6 * d["Accel_Fast_z"] + 0.4 * d["Momentum_Fast"])
    d["Early_Mid"] = (0.5 * d["Accel_Mid_z"] + 0.5 * d["Momentum_Mid"])
    d["Early_Slow"] = (0.5 * d["Accel_Slow_z"] + 0.5 * d["Momentum_Slow"])
    d["Early Momentum Score"] = (0.5 * d["Early_Slow"] + 0.3 * d["Early_Mid"] + 0.2 * d["Early_Fast"])
    return d.fillna(0.0)


@dataclass
class Alpha5Config:
    min_sma_trend: bool = True
    min_alpha_slow: float = 0.0

    bm_gate_on: bool = True
    bm_lookback: int = 20
    bm_min_cum_ret: float = 0.0

    max_signals: int = 25

    mce_fast_abs_max: float = 0.40
    mce_mid_abs_max: float = 0.35
    mce_atr_ratio_pct_max: float = 0.30
    mce_near_high_max_dist: float = 0.10

    fbd_lookback: int = 8
    fbd_reclaim_days: int = 4
    fbd_expansion_mult: float = 1.15
    fbd_close_pos_min: float = 0.65

    rot_slow_pct_min: float = 0.60
    rot_slow_pct_max: float = 0.85
    rot_accel_mid_z_min: float = 0.75
    rot_near_high_max_dist: float = 0.14
    rot_atr_ratio_pct_max: float = 0.45

    shock_lookback: int = 12
    shock_tr_mult: float = 1.80
    shock_close_pos_min: float = 0.60
    shock_consol_days: int = 5
    shock_consol_range5_pct_max: float = 0.12

    min_R: float = 3.0
    fresh_days: int = 7

    w_struct: float = 0.50
    w_engine: float = 0.30
    w_geom: float = 0.20


def add_daily_indicators(df_prices: pd.DataFrame) -> pd.DataFrame:
    d = _norm_date_col(df_prices, "date").sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker", sort=False)

    d["sma20"] = g["close"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    d["sma50"] = g["close"].rolling(50, min_periods=50).mean().reset_index(level=0, drop=True)

    d["prev_close"] = g["close"].shift(1)
    high = _to_num(d["high"])
    low = _to_num(d["low"])
    prev_close = _to_num(d["prev_close"])

    tr1 = (high - low).to_numpy()
    tr2 = (high - prev_close).abs().to_numpy()
    tr3 = (low - prev_close).abs().to_numpy()
    d["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))

    d["atr5"] = g["tr"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    d["atr20"] = g["tr"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    d["hh5"] = g["high"].rolling(5, min_periods=5).max().reset_index(level=0, drop=True)
    d["ll5"] = g["low"].rolling(5, min_periods=5).min().reset_index(level=0, drop=True)

    d["hh20"] = g["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    d["ll20"] = g["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)

    close = _to_num(d["close"])
    denom20 = (d["hh20"] - d["ll20"]).replace(0, np.nan)
    d["close_pos_20"] = ((close - d["ll20"]) / denom20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["near_high_20_dist"] = ((d["hh20"] - close) / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["range5_pct"] = ((d["hh5"] - d["ll5"]) / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["atr_ratio"] = (d["atr5"] / d["atr20"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return d


def build_alpha_momentum_panel(df_prices: pd.DataFrame, idx_returns: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    d = attach_benchmark_and_alpha(df_prices, idx_returns, bm_lookback=int(cfg.bm_lookback))
    d = d.dropna(subset=["date", "ticker", "close"]).copy()
    d = calculate_momentum_features(d, base_col="alpha_1d")
    d = add_regime_momentum_score(d)
    d = add_regime_acceleration(d)
    d = add_regime_early_momentum(d)

    keep = [
        "date", "ticker", "close",
        "Momentum_Fast", "Momentum_Mid", "Momentum_Slow", "Momentum Score",
        "Accel_Fast_z", "Accel_Mid_z", "Accel_Slow_z", "Acceleration Score",
        "Early Momentum Score",
        "stock_ret_1d", "alpha_1d",
        "index_name", "bm_ret_1d", "bm_cumret_lb",
    ]
    keep = [c for c in keep if c in d.columns]
    return d[keep].copy()


def structural_gate(latest: pd.DataFrame, cfg: Alpha5Config) -> pd.Series:
    ok = pd.Series(True, index=latest.index)

    if cfg.min_sma_trend:
        ok &= (latest["close"] > latest["sma50"]) & (latest["sma20"] > latest["sma50"])

    ok &= (latest["Momentum_Slow"] > cfg.min_alpha_slow)

    if cfg.bm_gate_on and "bm_cumret_lb" in latest.columns:
        ok &= (latest["bm_cumret_lb"] >= cfg.bm_min_cum_ret)

    return ok.fillna(False)


def engine_mce(latest: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    d = latest.copy()
    d["atr_ratio_pct"] = d.groupby("date")["atr_ratio"].rank(pct=True)

    cond = (
        (d["Momentum_Slow"] > 0) &
        (d["Momentum_Fast"].abs() <= cfg.mce_fast_abs_max) &
        (d["Momentum_Mid"].abs() <= cfg.mce_mid_abs_max) &
        (d["Accel_Fast_z"] > 0) &
        (d["atr_ratio_pct"] <= cfg.mce_atr_ratio_pct_max) &
        (d["near_high_20_dist"] <= cfg.mce_near_high_max_dist)
    )

    out = d.loc[cond, ["ticker", "date", "close", "hh5", "ll5", "sma20", "sma50", "atr20", "atr_ratio_pct", "near_high_20_dist"]].copy()
    if out.empty:
        return out

    out["engine"] = "MCE"
    out["entry_type"] = "BREAKOUT"
    out["entry_level"] = out["hh5"]
    out["stop_level"] = out["ll5"]
    out["timing_bonus"] = (1.0 - out["atr_ratio_pct"]).clip(0, 1)
    return out


def engine_fbd(df_all: pd.DataFrame, latest: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    d = df_all.sort_values(["ticker", "date"]).copy()
    latest_tickers = set(latest["ticker"].astype(str))
    out_rows = []

    for t, g in d.groupby("ticker", sort=False):
        if t not in latest_tickers:
            continue

        g = g.dropna(subset=["sma20", "atr20", "ll20", "tr", "ll5", "high", "low", "close"]).copy()
        if len(g) < 60:
            continue

        g_tail = g.tail(cfg.fbd_lookback + cfg.fbd_reclaim_days + 2).copy()
        if g_tail.empty:
            continue

        breakdown_mask = (g_tail["close"] < g_tail["ll20"]) | (g_tail["close"] < g_tail["sma20"])
        if not breakdown_mask.any():
            continue

        bd_idx = g_tail.index[np.where(breakdown_mask.values)[0][-1]]
        bd_date = _norm_date(g_tail.loc[bd_idx, "date"])
        bd_level = float(min(g_tail.loc[bd_idx, "ll20"], g_tail.loc[bd_idx, "sma20"]))

        after = g_tail[g_tail["date"] > bd_date].head(cfg.fbd_reclaim_days)
        if after.empty:
            continue

        reclaim = after[after["close"] > after["sma20"]]
        if reclaim.empty:
            continue

        r0 = reclaim.iloc[-1]
        rng = float(r0["high"] - r0["low"])
        close_pos = float(r0["close"] - r0["low"]) / max(rng, 1e-9)
        exp_ok = (float(r0["tr"]) >= cfg.fbd_expansion_mult * float(r0["atr20"])) and (close_pos >= cfg.fbd_close_pos_min)
        if not exp_ok:
            continue

        lt = latest[latest["ticker"] == t].iloc[-1]
        if float(lt["Momentum_Slow"]) <= 0:
            continue

        out_rows.append({
            "ticker": t,
            "date": _norm_date(r0["date"]),
            "close": float(r0["close"]),
            "engine": "FBD",
            "entry_type": "RECLAIM",
            "entry_level": float(r0["sma20"]),
            "stop_level": float(min(r0["low"], r0["ll5"])),
            "timing_bonus": 1.0,
            "bd_level": bd_level,
        })

    return pd.DataFrame(out_rows)


def engine_rotation(latest: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    d = latest.copy()
    d["slow_pct"] = d.groupby("date")["Momentum_Slow"].rank(pct=True)
    d["atr_ratio_pct"] = d.groupby("date")["atr_ratio"].rank(pct=True)

    cond = (
        (d["slow_pct"] >= cfg.rot_slow_pct_min) &
        (d["slow_pct"] <= cfg.rot_slow_pct_max) &
        (d["Accel_Mid_z"] >= cfg.rot_accel_mid_z_min) &
        (d["near_high_20_dist"] <= cfg.rot_near_high_max_dist) &
        (d["atr_ratio_pct"] <= cfg.rot_atr_ratio_pct_max)
    )

    out = d.loc[cond, ["ticker", "date", "close", "hh5", "ll5", "sma20", "sma50", "atr20", "slow_pct", "atr_ratio_pct", "near_high_20_dist"]].copy()
    if out.empty:
        return out

    out["engine"] = "ROT"
    out["entry_type"] = "BREAKOUT"
    out["entry_level"] = out["hh5"]
    out["stop_level"] = out["ll5"]
    out["timing_bonus"] = (0.6 * (1.0 - out["near_high_20_dist"].clip(0, 1)) + 0.4 * out["slow_pct"]).clip(0, 1)
    return out


def engine_shock_absorption(df_all: pd.DataFrame, latest: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    d = df_all.sort_values(["ticker", "date"]).copy()
    latest_tickers = set(latest["ticker"].astype(str))
    out_rows = []

    for t, g in d.groupby("ticker", sort=False):
        if t not in latest_tickers:
            continue

        g = g.dropna(subset=["atr20", "tr", "hh5", "ll5", "range5_pct", "high", "low", "close"]).copy()
        if len(g) < 80:
            continue

        tail = g.tail(cfg.shock_lookback + cfg.shock_consol_days + 10).copy()
        if tail.empty:
            continue

        shock_mask = (tail["tr"] >= cfg.shock_tr_mult * tail["atr20"])
        if not shock_mask.any():
            continue

        sidx = tail.index[np.where(shock_mask.values)[0][-1]]
        srow = tail.loc[sidx]
        rng = float(srow["high"] - srow["low"])
        close_pos = float(srow["close"] - srow["low"]) / max(rng, 1e-9)
        if close_pos < cfg.shock_close_pos_min:
            continue

        shock_low = float(srow["low"])
        shock_date = _norm_date(srow["date"])

        post = tail[tail["date"] > shock_date].head(cfg.shock_consol_days)
        if len(post) < cfg.shock_consol_days:
            continue

        holds = float(post["low"].min()) >= shock_low
        tight = float(post["range5_pct"].iloc[-1]) <= cfg.shock_consol_range5_pct_max
        if not (holds and tight):
            continue

        lt = latest[latest["ticker"] == t].iloc[-1]
        if float(lt["Momentum_Slow"]) <= 0:
            continue

        entry = float(post["hh5"].iloc[-1])
        stop = float(min(shock_low, post["ll5"].iloc[-1]))

        out_rows.append({
            "ticker": t,
            "date": _norm_date(post["date"].iloc[-1]),
            "close": float(post["close"].iloc[-1]),
            "engine": "SHOCK",
            "entry_type": "BREAKOUT",
            "entry_level": entry,
            "stop_level": stop,
            "timing_bonus": 1.0,
            "shock_date": shock_date,
        })

    return pd.DataFrame(out_rows)


def compute_geometry(entry: pd.Series, stop: pd.Series, cfg: Alpha5Config) -> pd.DataFrame:
    e = _to_num(entry)
    s = _to_num(stop)
    risk = (e - s).clip(lower=1e-9)

    target = e + cfg.min_R * risk
    geom_score = (1.0 / risk).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.DataFrame({
        "risk_per_share": risk,
        "target_level": target,
        "R_multiple": float(cfg.min_R),
        "geom_raw": geom_score,
    })


_SIGNALS_MERGE_COLS = [
    "Momentum_Slow", "Momentum Score", "Early Momentum Score",
    "Accel_Fast_z", "Accel_Mid_z", "near_high_20_dist", "atr_ratio",
    "sma20", "sma50", "hh20", "ll20", "range5_pct", "close",
    "bm_ret_1d", "bm_cumret_lb",
]
_SIGNALS_MUST_HAVE = ["close", "Momentum_Slow", "Accel_Fast_z"]

_SIGNALS_KEEP_BASE = [
    "ticker", "System", "Signal_Date",
    "FINAL_ALPHA_SCORE",
    "engines_fired",
    "entry_type", "entry_level", "stop_level", "target_level", "R_multiple",
    "risk_per_share",
    "close",
    "event_date",
]
_SIGNALS_KEEP_EXTRA = [
    "struct_score", "geom_score", "engine_score",
    "Momentum_Slow", "Early Momentum Score", "Accel_Fast_z", "Accel_Mid_z",
]


def _run_engines(d_all: pd.DataFrame, latest_g: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    if latest_g is None or latest_g.empty:
        return pd.DataFrame()

    engines = [
        engine_mce(latest_g, cfg),
        engine_fbd(d_all, latest_g, cfg),
        engine_rotation(latest_g, cfg),
        engine_shock_absorption(d_all, latest_g, cfg),
    ]
    engines = [x for x in engines if x is not None and not x.empty]
    return pd.concat(engines, ignore_index=True) if engines else pd.DataFrame()


def _merge_latest_scoring(engines: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    add_cols = [c for c in _SIGNALS_MERGE_COLS if c in latest.columns]
    engines = engines.drop(columns=[c for c in add_cols if c in engines.columns], errors="ignore")
    return engines.merge(latest[["ticker", "date"] + add_cols], on=["ticker", "date"], how="left")


def _postprocess_engines(engines: pd.DataFrame, latest: pd.DataFrame, as_of: pd.Timestamp, cfg: Alpha5Config) -> pd.DataFrame:
    if engines is None or engines.empty:
        return pd.DataFrame()

    engines = engines.copy()
    engines["event_date"] = pd.to_datetime(engines["date"]).dt.normalize()
    engines["date"] = as_of

    engines = engines[engines["event_date"] >= (as_of - pd.Timedelta(days=int(cfg.fresh_days)))].copy()
    if engines.empty:
        return pd.DataFrame()

    engines = _merge_latest_scoring(engines, latest)

    must_have = [c for c in _SIGNALS_MUST_HAVE if c in engines.columns]
    engines = engines.dropna(subset=must_have).copy()
    if engines.empty:
        return pd.DataFrame()

    geom = compute_geometry(engines["entry_level"], engines["stop_level"], cfg)
    engines = pd.concat([engines.reset_index(drop=True), geom.reset_index(drop=True)], axis=1)

    engines["engine"] = engines["engine"].astype(str)
    agg = (
        engines.groupby(["ticker", "date"], as_index=False)
        .agg(
            engines_fired=("engine", lambda x: "+".join(sorted(set(x)))),
            engine_count=("engine", lambda x: len(set(x))),
        )
    )
    return engines.merge(agg, on=["ticker", "date"], how="left")


def _score_and_select(engines: pd.DataFrame, as_of: pd.Timestamp, cfg: Alpha5Config, include_scores: bool) -> pd.DataFrame:
    engines = engines.copy()

    engines["struct_raw"] = (
        0.65 * engines["Momentum_Slow"].fillna(0.0)
        + 0.35 * engines["Early Momentum Score"].fillna(0.0)
    )
    engines["struct_score"] = normalize_01_by_date(engines, "struct_raw")
    engines["geom_score"] = normalize_01_by_date(engines, "geom_raw")
    engines["engine_score"] = (engines["engine_count"].astype(float) / 4.0).clip(0.0, 1.0)

    engines["FINAL_ALPHA_SCORE"] = (
        float(cfg.w_struct) * engines["struct_score"] +
        float(cfg.w_engine) * engines["engine_score"] +
        float(cfg.w_geom) * engines["geom_score"]
    ).clip(0.0, 1.0)

    engines = engines.sort_values(["ticker", "FINAL_ALPHA_SCORE"], ascending=[True, False]).copy()
    best = engines.groupby("ticker", as_index=False).head(1)

    best = (
        best.sort_values(["FINAL_ALPHA_SCORE", "engine_count", "struct_score"], ascending=[False, False, False])
        .head(int(cfg.max_signals))
        .reset_index(drop=True)
    )

    best["System"] = "weekly_swing_alpha5"
    best["Signal"] = "TRADE_CANDIDATE"
    best["Signal_Date"] = pd.to_datetime(as_of)

    keep = _SIGNALS_KEEP_BASE + (_SIGNALS_KEEP_EXTRA if include_scores else [])
    keep = [c for c in keep if c in best.columns]
    return best[keep].copy()


def _build_signals_from_d_all(d_all: pd.DataFrame, cfg: Alpha5Config, include_scores: bool) -> pd.DataFrame:
    as_of = _norm_date(d_all["date"].max())
    latest = d_all[d_all["date"] == as_of].copy()
    if latest.empty:
        return pd.DataFrame()

    gate = structural_gate(latest, cfg)
    latest_g = latest.loc[gate].copy()
    if latest_g.empty:
        return pd.DataFrame()

    engines = _run_engines(d_all, latest_g, cfg)
    if engines.empty:
        return pd.DataFrame()

    engines = _postprocess_engines(engines, latest, as_of, cfg)
    if engines.empty:
        return pd.DataFrame()

    return _score_and_select(engines, as_of, cfg, include_scores=include_scores)


def _precompute_alpha5_d_all(df_prices: pd.DataFrame, idx_returns: pd.DataFrame, cfg: Alpha5Config) -> pd.DataFrame:
    w = add_daily_indicators(df_prices)
    w = w.dropna(subset=["sma20", "sma50", "atr20", "hh5", "ll5", "hh20", "ll20"]).copy()
    if w.empty:
        return pd.DataFrame()

    mom_panel = build_alpha_momentum_panel(df_prices, idx_returns, cfg=cfg)
    mom = mom_panel.dropna(subset=["Momentum_Slow", "Accel_Fast_z", "Accel_Mid_z"]).copy()
    if mom.empty:
        return pd.DataFrame()

    d_all = w.merge(
        mom[
            [
                "date", "ticker",
                "Momentum_Fast", "Momentum_Mid", "Momentum_Slow", "Momentum Score",
                "Accel_Fast_z", "Accel_Mid_z", "Accel_Slow_z", "Early Momentum Score",
                "bm_ret_1d", "bm_cumret_lb",
            ]
        ],
        on=["date", "ticker"],
        how="left",
    )
    d_all = d_all.dropna(subset=["Momentum_Slow"]).copy()
    d_all["date"] = pd.to_datetime(d_all["date"]).dt.normalize()
    d_all["ticker"] = d_all["ticker"].astype(str)
    return d_all.sort_values(["ticker", "date"]).reset_index(drop=True)


def weekly_swing_alpha5_signals(
    df_prices: pd.DataFrame,
    idx_returns: pd.DataFrame,
    cfg: Optional[Alpha5Config] = None,
) -> pd.DataFrame:
    cfg = cfg or Alpha5Config()
    d_all = _precompute_alpha5_d_all(df_prices, idx_returns, cfg=cfg)
    if d_all.empty:
        return pd.DataFrame()
    return _build_signals_from_d_all(d_all=d_all, cfg=cfg, include_scores=True)


def weekly_swing_alpha5_signals_from_panel(
    d_all: pd.DataFrame,
    mom_panel: Optional[pd.DataFrame] = None,
    cfg: Optional[Alpha5Config] = None,
) -> pd.DataFrame:
    cfg = cfg or Alpha5Config()
    _ = mom_panel
    if d_all is None or d_all.empty:
        return pd.DataFrame()
    return _build_signals_from_d_all(d_all=d_all, cfg=cfg, include_scores=False)


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
    watch = pd.DataFrame(rows)
    if watch.empty:
        return watch

    watch["Retracement %"] = watch["Retracement"] * 100.0
    watch["Prime Setup"] = watch["Retracement %"].between(50, 78.6)
    return watch[watch["Prime Setup"]].reset_index(drop=True)


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
        if g.empty:
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

        if np.isfinite(last_local_high) and last_local_high != 0:
            bos_dist = last_local_high - close_now
            raw = 1 - bos_dist / max(close_now, 1e-9)
            bos_prox = float(np.clip(raw, 0, 1))
        else:
            bos_prox = 0.0

        if final_signal == "BUY":
            readiness = 100.0
        else:
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

    weekly_cfg = Alpha5Config(
        max_signals=25,
        min_R=3.0,
        fresh_days=7,
        bm_gate_on=True,
        bm_lookback=20,
        bm_min_cum_ret=0.0,
    )

    weekly_sig = weekly_swing_alpha5_signals(df_prices, idx_returns, weekly_cfg)
    if not weekly_sig.empty:
        keep = [
            "ticker",
            "System",
            "Signal",
            "Signal_Date",
            "FINAL_ALPHA_SCORE",
            "entry_level",
            "stop_level",
            "close",
            "risk_per_share",
            "target_level",
            "event_date",
        ]
        keep = [c for c in keep if c in weekly_sig.columns]
        weekly_sig = weekly_sig[keep].sort_values(["Signal_Date", "FINAL_ALPHA_SCORE"], ascending=[False, False]).reset_index(drop=True)

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
        x = x.sort_values("FINAL_ALPHA_SCORE", ascending=False)
        r = x.iloc[0]
        return pd.Series({
            "weekly_tag": r.get("System", np.nan),
            "weekly_signal": r.get("Signal", np.nan),
            "weekly_score": r.get("FINAL_ALPHA_SCORE", np.nan),
            "weekly_date": r.get("Signal_Date", np.nan),
            "weekly_breakout": r.get("entry_level", np.nan),
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
