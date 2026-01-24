from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from updater.update_parquet import (
    WeeklySwingConfig,
    load_prices_from_parquet,
    load_index_returns_from_parquet,
    weekly_add_indicators,
    weekly_detect_setups,
    momentum_bucketC_latest,
)


# =========================
# 0) CONFIG
# =========================
ARTIFACTS_DIR = "artifacts"

PRICES_PATH = os.path.join(ARTIFACTS_DIR, "index_constituents_5yr.parquet")
INDEX_RETURNS_PATH = os.path.join(ARTIFACTS_DIR, "index_returns_5y.parquet")

BACKTEST_EQUITY_S1_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_system1.parquet")
BACKTEST_EQUITY_S2_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_system2.parquet")
BACKTEST_EQUITY_S3_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_system3.parquet")
BACKTEST_EQUITY_COMBINED_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_combined.parquet")
BACKTEST_TRADES_PATH = os.path.join(ARTIFACTS_DIR, "backtest_trades.parquet")
BACKTEST_POSITIONS_PATH = os.path.join(ARTIFACTS_DIR, "backtest_positions.parquet")
BACKTEST_STATS_PATH = os.path.join(ARTIFACTS_DIR, "backtest_stats.json")

BASE_CCY = "USD"
FX_HKD_PER_USD = 7.75
FX_SGD_PER_USD = 1.30

# -------------------------
# System 3 (Momentum)
# -------------------------
# ✅ Only add cash at t0 (no monthly contrib)
S3_INITIAL_CASH = 100_000.0
S3_MONTHLY_CONTRIB = 0.0  # force off

# -------------------------
# System 2 (Fib)
# -------------------------
S2_TRADE_DOLLARS = 10_000.0
S2_INITIAL_CASH = 500_000.0
S2_PROFIT_TGT = 0.08

# -------------------------
# System 1 (Weekly swing)
# -------------------------
S1_TRADE_DOLLARS = 5_000.0
S1_INITIAL_CASH = 250_000.0
S1_SCORE_MIN = 0.7
S1_PROFIT_TGT = 0.08
S1_MAX_HOLD_DAYS = 15
S1_USE_STOPLEVEL = True
S1_MAX_POSITIONS = 20

# Fib speed controls
FIB_LOOKBACK_DAYS = 300
FIB_USE_NUMBA = True
FIB_PARALLEL = True
FIB_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Weekly signal config (same defaults as signals batch)
WEEKLY_CFG = WeeklySwingConfig(
    adv_turnover_20_min=0.0,
    breakout_buffer_pct=0.0,
)


# =========================
# 1) HELPERS
# =========================
def fx_to_usd(ticker: str) -> float:
    t = ticker.upper().strip()
    if t.endswith(".HK"):
        return 1.0 / FX_HKD_PER_USD
    if t.endswith(".SI"):
        return 1.0 / FX_SGD_PER_USD
    return 1.0


def build_weekly_signals(df_prices: pd.DataFrame, cfg: WeeklySwingConfig) -> pd.DataFrame:
    d = df_prices.copy()
    d = weekly_add_indicators(d, cfg)
    sig = weekly_detect_setups(d, cfg)
    if sig is None or sig.empty:
        return pd.DataFrame()
    sig = sig.copy()
    sig["signal_date"] = pd.to_datetime(sig["signal_date"]).dt.normalize()
    return sig


def momentum_bucketC_latest_asof(
    df_prices: pd.DataFrame,
    index_returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    top_n_daily: int = 10,
    final_top_n: int = 10,
    lookback_days: int = 10,
) -> pd.DataFrame:
    d0 = df_prices[df_prices["date"] <= as_of_date]
    idx = index_returns[index_returns["date"] <= as_of_date]
    if d0.empty or idx.empty:
        return pd.DataFrame()
    return momentum_bucketC_latest(
        df_prices=d0,
        index_returns=idx,
        top_n_daily=top_n_daily,
        final_top_n=final_top_n,
        lookback_days=lookback_days,
    )


def fib_levels_daily(g: pd.DataFrame, lookback_days=300) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    dates = pd.to_datetime(g["date"]).to_numpy()
    highs = g["high"].to_numpy(float)
    lows = g["low"].to_numpy(float)

    n = len(g)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    target_382 = np.full(n, np.nan)

    use_numba = False
    if FIB_USE_NUMBA:
        try:
            from numba import njit  # type: ignore

            @njit(cache=True)
            def _fib_numba(dates_i64, highs_arr, lows_arr, lookback_days_i64):
                n0 = len(dates_i64)
                sh = np.full(n0, np.nan)
                sl = np.full(n0, np.nan)
                tgt = np.full(n0, np.nan)
                day_ns = 86_400_000_000_000
                for i in range(n0):
                    start_ts = dates_i64[i] - lookback_days_i64 * day_ns
                    j = np.searchsorted(dates_i64, start_ts, side="left")
                    if i - j + 1 < 30:
                        continue
                    max_h = highs_arr[j]
                    max_k = j
                    for k in range(j + 1, i + 1):
                        v = highs_arr[k]
                        if v > max_h:
                            max_h = v
                            max_k = k
                    min_l = lows_arr[j]
                    for k in range(j, max_k + 1):
                        v = lows_arr[k]
                        if v < min_l:
                            min_l = v
                    if min_l >= max_h:
                        continue
                    sh[i] = max_h
                    sl[i] = min_l
                    tgt[i] = max_h - 0.382 * (max_h - min_l)
                return sh, sl, tgt

            dates_i64 = dates.astype("datetime64[ns]").astype(np.int64)
            swing_high, swing_low, target_382 = _fib_numba(dates_i64, highs, lows, int(lookback_days))
            use_numba = True
        except Exception:
            use_numba = False

    if not use_numba:
        dates_i64 = dates.astype("datetime64[ns]").astype(np.int64)
        day_ns = 86_400_000_000_000
        for i in range(n):
            start_ts = dates_i64[i] - lookback_days * day_ns
            j = int(np.searchsorted(dates_i64, start_ts, side="left"))
            if i - j + 1 < 30:
                continue
            win_highs = highs[j : i + 1]
            if win_highs.size == 0:
                continue
            k = int(np.argmax(win_highs))
            hi_idx = j + k
            hi = float(highs[hi_idx])
            lo = float(np.min(lows[j : hi_idx + 1]))
            if lo >= hi:
                continue
            swing_high[i] = hi
            swing_low[i] = lo
            target_382[i] = hi - 0.382 * (hi - lo)

    out = g[["date", "ticker", "open", "high", "low", "close"]].copy()
    out["fib_swing_high"] = swing_high
    out["fib_swing_low"] = swing_low
    out["fib_target_382"] = target_382
    return out


def _fib_worker(args):
    _, g, lookback_days = args
    return fib_levels_daily(g, lookback_days=lookback_days)


def build_fib_signals(df_prices: pd.DataFrame, lookback_days=300) -> pd.DataFrame:
    groups = [(t, g.copy(), lookback_days) for t, g in df_prices.groupby("ticker", sort=False)]
    frames: List[pd.DataFrame] = []

    if FIB_PARALLEL and len(groups) > 1:
        try:
            with ProcessPoolExecutor(max_workers=FIB_MAX_WORKERS) as ex:
                futs = [ex.submit(_fib_worker, args) for args in groups]
                for fut in as_completed(futs):
                    frames.append(fut.result())
        except Exception:
            frames = [fib_levels_daily(g, lookback_days=lookback_days) for _, g, _ in groups]
    else:
        frames = [fib_levels_daily(g, lookback_days=lookback_days) for _, g, _ in groups]

    fib_daily = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

    fib_daily["prev_close"] = fib_daily.groupby("ticker")["close"].shift(1)
    fib_daily["prev_target"] = fib_daily.groupby("ticker")["fib_target_382"].shift(1)

    fib_daily["buy_signal"] = (
        np.isfinite(fib_daily["fib_target_382"])
        & np.isfinite(fib_daily["prev_target"])
        & (fib_daily["prev_close"] < fib_daily["prev_target"])
        & (fib_daily["close"] >= fib_daily["fib_target_382"])
    )
    return fib_daily


# =========================
# 2) BACKTEST CORE
# =========================
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px_local: float
    shares: float
    target_px_local: Optional[float] = None
    stop_px_local: Optional[float] = None
    max_exit_date: Optional[pd.Timestamp] = None


class Portfolio:
    def __init__(self, name: str):
        self.name = name
        self.cash_usd = 0.0
        self.positions: Dict[str, Position] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        self.external_flows: Dict[pd.Timestamp, float] = {}

    def value_usd(self, close_ffill: Dict[str, np.ndarray], date_to_i: Dict[pd.Timestamp, int], date: pd.Timestamp) -> float:
        v = float(self.cash_usd)
        i = date_to_i.get(date, None)
        if i is None:
            return v
        for p in self.positions.values():
            arr = close_ffill.get(p.ticker)
            if arr is None:
                continue
            px = float(arr[i])
            if not np.isfinite(px):
                continue
            v += float(p.shares) * px * fx_to_usd(p.ticker)
        return float(v)

    def mark_to_market(self, close_ffill, date_to_i, date):
        d = pd.Timestamp(date).normalize()
        self.equity_curve.append({"date": d, "equity_usd": self.value_usd(close_ffill, date_to_i, d)})

    def add_flow(self, date: pd.Timestamp, amt_usd: float):
        if amt_usd == 0:
            return
        d = pd.Timestamp(date).normalize()
        self.external_flows[d] = float(self.external_flows.get(d, 0.0) + float(amt_usd))

    def buy_usd(
        self,
        ticker: str,
        date: pd.Timestamp,
        px_local: float,
        dollars_usd: float,
        target_px_local=None,
        stop_px_local=None,
        max_exit_date=None,
    ):
        if dollars_usd <= 0:
            return

        px_local = float(px_local)
        if not np.isfinite(px_local) or px_local <= 0:
            return

        cost_usd = float(dollars_usd)
        if self.cash_usd + 1e-12 < cost_usd:
            return

        fx = fx_to_usd(ticker)
        add_shares = (cost_usd / fx) / px_local
        if add_shares <= 0 or not np.isfinite(add_shares):
            return

        d = pd.Timestamp(date).normalize()

        self.cash_usd -= cost_usd

        if ticker in self.positions:
            p = self.positions[ticker]
            old_shares = float(p.shares)
            new_shares = old_shares + float(add_shares)

            if new_shares > 0:
                p.entry_px_local = (p.entry_px_local * old_shares + px_local * float(add_shares)) / new_shares
            p.shares = new_shares

            if target_px_local is not None:
                p.target_px_local = float(target_px_local)
            if stop_px_local is not None:
                p.stop_px_local = float(stop_px_local)
            if max_exit_date is not None:
                p.max_exit_date = pd.Timestamp(max_exit_date).normalize()

        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                entry_date=d,
                entry_px_local=px_local,
                shares=float(add_shares),
                target_px_local=None if target_px_local is None else float(target_px_local),
                stop_px_local=None if stop_px_local is None else float(stop_px_local),
                max_exit_date=None if max_exit_date is None else pd.Timestamp(max_exit_date).normalize(),
            )

        self.trades.append(
            {
                "portfolio": self.name,
                "ticker": ticker,
                "side": "BUY",
                "date": d,
                "px_local": float(px_local),
                "usd": float(cost_usd),
                "shares": float(add_shares),
            }
        )

    def sell_all(self, ticker: str, date: pd.Timestamp, px_local: float):
        p = self.positions.get(ticker)
        if p is None:
            return

        px_local = float(px_local)
        if not np.isfinite(px_local) or px_local <= 0:
            return

        fx = fx_to_usd(ticker)
        proceeds_usd = float(p.shares) * px_local * fx
        if not np.isfinite(proceeds_usd):
            return

        d = pd.Timestamp(date).normalize()
        self.cash_usd += float(proceeds_usd)
        self.trades.append(
            {
                "portfolio": self.name,
                "ticker": ticker,
                "side": "SELL",
                "date": d,
                "px_local": float(px_local),
                "usd": float(proceeds_usd),
                "shares": float(p.shares),
            }
        )
        del self.positions[ticker]


# =========================
# 3) STATS
# =========================
def compute_stats_twr(curve: pd.DataFrame, flows_by_date: Optional[Dict[pd.Timestamp, float]] = None) -> dict:
    curve = curve.sort_values("date").copy()
    curve["date"] = pd.to_datetime(curve["date"]).dt.normalize()

    eq = curve["equity_usd"].astype(float).to_numpy()
    dates = curve["date"].to_list()

    if flows_by_date:
        flow = np.array([float(flows_by_date.get(d, 0.0)) for d in dates], dtype=float)
    else:
        flow = np.zeros(len(eq), dtype=float)

    rets = np.zeros(len(eq), dtype=float)
    for i in range(1, len(eq)):
        denom = eq[i - 1] + flow[i]
        if denom > 0 and np.isfinite(denom) and np.isfinite(eq[i]):
            rets[i] = eq[i] / denom - 1.0
        else:
            rets[i] = 0.0

    total_return = float(np.prod(1.0 + rets[1:]) - 1.0) if len(eq) > 1 else 0.0
    days = max(1, int((dates[-1] - dates[0]).days)) if len(dates) else 1
    cagr = float((1.0 + total_return) ** (365.0 / days) - 1.0) if days > 0 else 0.0

    eq_s = pd.Series(eq)
    roll_max = eq_s.cummax()
    dd = (eq_s / roll_max - 1.0).to_numpy()
    max_dd = float(np.nanmin(dd)) if len(dd) else 0.0

    vol = float(np.std(rets[1:], ddof=1)) if len(rets) > 2 else 0.0
    mean = float(np.mean(rets[1:])) if len(rets) > 1 else 0.0
    sharpe = float((mean / vol) * np.sqrt(252)) if vol and vol > 0 else 0.0

    return {
        "Start Equity": float(eq[0]) if len(eq) else 0.0,
        "End Equity": float(eq[-1]) if len(eq) else 0.0,
        "Total Return (TWR)": float(total_return),
        "CAGR (TWR)": float(cagr),
        "Max Drawdown (raw EQ)": float(max_dd),
        "Sharpe (daily, TWR)": float(sharpe),
    }


# =========================
# 4) CALENDAR HELPERS
# =========================
def last_trading_day_each_month(calendar: List[pd.Timestamp]) -> List[pd.Timestamp]:
    cal = pd.to_datetime(pd.Index(calendar)).sort_values()
    months = cal.to_period("M")
    out: List[pd.Timestamp] = []
    for m in months.unique():
        out.append(cal[months == m].max())
    return out


def next_trading_date_for_ticker(dates_by_ticker: Dict[str, np.ndarray], ticker: str, d: pd.Timestamp) -> Optional[pd.Timestamp]:
    arr = dates_by_ticker.get(ticker)
    if arr is None or len(arr) == 0:
        return None
    j = int(np.searchsorted(arr, np.datetime64(pd.Timestamp(d).to_datetime64()), side="right"))
    if j >= len(arr):
        return None
    return pd.Timestamp(arr[j]).to_pydatetime().replace(tzinfo=None)


def nth_trading_date_for_ticker(dates_by_ticker: Dict[str, np.ndarray], ticker: str, d: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
    arr = dates_by_ticker.get(ticker)
    if arr is None or len(arr) == 0:
        return None
    i = int(np.searchsorted(arr, np.datetime64(pd.Timestamp(d).to_datetime64()), side="left"))
    if i >= len(arr):
        return None
    j = min(i + (n - 1), len(arr) - 1)
    return pd.Timestamp(arr[j]).to_pydatetime().replace(tzinfo=None)


# =========================
# 5) MAIN
# =========================
def main():
    df = load_prices_from_parquet(PRICES_PATH)
    idx = load_index_returns_from_parquet(INDEX_RETURNS_PATH)

    dates_by_ticker: Dict[str, np.ndarray] = {}
    for t, g in df.groupby("ticker", sort=False):
        dates_by_ticker[str(t)] = g["date"].sort_values().to_numpy(dtype="datetime64[ns]")

    calendar = sorted(pd.to_datetime(df["date"]).dt.normalize().unique())
    total_days = len(calendar)
    date_to_i: Dict[pd.Timestamp, int] = {pd.Timestamp(d).normalize(): i for i, d in enumerate(calendar)}

    px_open = {(r.ticker, r.date): float(r.open) for r in df.itertuples(index=False)}
    px_high = {(r.ticker, r.date): float(r.high) for r in df.itertuples(index=False)}
    px_low = {(r.ticker, r.date): float(r.low) for r in df.itertuples(index=False)}

    close_ffill: Dict[str, np.ndarray] = {}
    cal_index = pd.Index(calendar)
    for t, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")["close"].astype(float)
        s = s.reindex(cal_index).ffill()
        close_ffill[str(t)] = s.to_numpy(dtype=float)

    month_ends = set(last_trading_day_each_month(calendar))

    p1 = Portfolio("System1_Weekly")
    p2 = Portfolio("System2_Fib")
    p3 = Portfolio("System3_Momentum")

    p1.cash_usd = S1_INITIAL_CASH
    p2.cash_usd = S2_INITIAL_CASH
    p3.cash_usd = S3_INITIAL_CASH
    if total_days > 0:
        p3.add_flow(pd.Timestamp(calendar[0]).normalize(), S3_INITIAL_CASH)

    weekly_all = build_weekly_signals(df, WEEKLY_CFG)
    if not weekly_all.empty:
        weekly_all = weekly_all[weekly_all["score"] >= S1_SCORE_MIN].copy()
        weekly_all["signal_date"] = pd.to_datetime(weekly_all["signal_date"]).dt.normalize()

    s1_entries_by_date: Dict[pd.Timestamp, List[dict]] = {}
    if weekly_all is not None and not weekly_all.empty:
        for s in weekly_all.itertuples(index=False):
            tk = str(s.ticker)
            sig_d = pd.Timestamp(s.signal_date).normalize()
            entry_d = next_trading_date_for_ticker(dates_by_ticker, tk, sig_d)
            if entry_d is None:
                continue
            entry_d = pd.Timestamp(entry_d).normalize()

            max_exit = nth_trading_date_for_ticker(dates_by_ticker, tk, entry_d, S1_MAX_HOLD_DAYS)
            max_exit = pd.Timestamp(max_exit).normalize() if max_exit is not None else None

            stop = None
            if S1_USE_STOPLEVEL and hasattr(s, "stop_level"):
                try:
                    stop = float(s.stop_level) if np.isfinite(float(s.stop_level)) else None
                except Exception:
                    stop = None

            s1_entries_by_date.setdefault(entry_d, []).append({"ticker": tk, "stop": stop, "max_exit": max_exit})

    fib_daily = build_fib_signals(df, lookback_days=FIB_LOOKBACK_DAYS)
    fib_signal_dates = fib_daily[fib_daily["buy_signal"]].copy()

    s2_entries_by_date: Dict[pd.Timestamp, List[dict]] = {}
    if not fib_signal_dates.empty:
        for r in fib_signal_dates.itertuples(index=False):
            tk = str(r.ticker)
            sig_d = pd.Timestamp(r.date).normalize()
            entry_d = next_trading_date_for_ticker(dates_by_ticker, tk, sig_d)
            if entry_d is None:
                continue
            entry_d = pd.Timestamp(entry_d).normalize()

            stop_px = float(r.fib_swing_low) if np.isfinite(float(r.fib_swing_low)) else None
            s2_entries_by_date.setdefault(entry_d, []).append({"ticker": tk, "stop": stop_px})

    s3_rebalance_by_date: Dict[pd.Timestamp, pd.Timestamp] = {}
    for d0 in sorted(month_ends):
        d0 = pd.Timestamp(d0).normalize()
        i = date_to_i.get(d0, None)
        if i is None:
            continue
        if i + 1 < len(calendar):
            trade_d = pd.Timestamp(calendar[i + 1]).normalize()
            s3_rebalance_by_date[trade_d] = d0

    start_ts = time.time()
    last_status_ts = start_ts
    status_interval_sec = 0.5

    for i, d0 in enumerate(calendar):
        d0 = pd.Timestamp(d0).normalize()

        now = time.time()
        if now - last_status_ts >= status_interval_sec or i == 0 or i == total_days - 1:
            pct = (i + 1) / total_days if total_days else 1.0
            elapsed = now - start_ts
            eta = (elapsed / pct - elapsed) if pct > 0 else 0.0
            print(f"\rBacktest progress: {pct:6.2%} | day {i+1}/{total_days} | ETA {eta:6.1f}s", end="", flush=True)
            last_status_ts = now

        if d0 in s3_rebalance_by_date:
            asof = s3_rebalance_by_date[d0]

            mom = momentum_bucketC_latest_asof(
                df_prices=df,
                index_returns=idx,
                as_of_date=pd.Timestamp(asof),
                top_n_daily=10,
                final_top_n=10,
                lookback_days=10,
            )

            if mom is not None and not mom.empty:
                targets = mom["ticker"].astype(str).tolist()

                for tk in list(p3.positions.keys()):
                    if tk not in targets:
                        px = px_open.get((tk, d0), np.nan)
                        if np.isfinite(px):
                            p3.sell_all(tk, d0, px)

                eq = p3.value_usd(close_ffill, date_to_i, d0)
                if eq > 0 and len(targets) > 0:
                    per = eq / len(targets)
                    for tk in targets:
                        px = px_open.get((tk, d0), np.nan)
                        if not np.isfinite(px):
                            continue
                        if tk in p3.positions:
                            cur_usd = p3.positions[tk].shares * px * fx_to_usd(tk)
                            diff = per - cur_usd
                            if diff > 1:
                                p3.buy_usd(tk, d0, px, min(diff, p3.cash_usd))
                        else:
                            p3.buy_usd(tk, d0, px, min(per, p3.cash_usd))

        for tk in list(p2.positions.keys()):
            hi = px_high.get((tk, d0), np.nan)
            lo = px_low.get((tk, d0), np.nan)
            if not np.isfinite(hi) or not np.isfinite(lo):
                continue

            pos = p2.positions[tk]

            if pos.stop_px_local is not None and lo <= pos.stop_px_local:
                p2.sell_all(tk, d0, pos.stop_px_local)
                continue

            if pos.target_px_local is not None and hi >= pos.target_px_local:
                p2.sell_all(tk, d0, pos.target_px_local)
                continue

        if d0 in s2_entries_by_date:
            for o in s2_entries_by_date[d0]:
                tk = o["ticker"]
                if tk in p2.positions:
                    continue
                px = px_open.get((tk, d0), np.nan)
                if not np.isfinite(px):
                    continue

                target_px = float(px) * (1.0 + S2_PROFIT_TGT)
                stop_px = o["stop"]

                p2.buy_usd(
                    tk,
                    d0,
                    float(px),
                    min(S2_TRADE_DOLLARS, p2.cash_usd),
                    target_px_local=target_px,
                    stop_px_local=stop_px,
                )

        for tk in list(p1.positions.keys()):
            hi = px_high.get((tk, d0), np.nan)
            lo = px_low.get((tk, d0), np.nan)
            if not np.isfinite(hi) or not np.isfinite(lo):
                continue

            pos = p1.positions[tk]

            tgt = pos.entry_px_local * (1.0 + S1_PROFIT_TGT)
            if hi >= tgt:
                p1.sell_all(tk, d0, tgt)
                continue

            if S1_USE_STOPLEVEL and pos.stop_px_local is not None and lo <= pos.stop_px_local:
                p1.sell_all(tk, d0, pos.stop_px_local)
                continue

            if pos.max_exit_date is not None and d0 >= pos.max_exit_date:
                px = px_open.get((tk, d0), np.nan)
                if not np.isfinite(px):
                    arr = close_ffill.get(tk)
                    if arr is None:
                        continue
                    ii = date_to_i.get(d0, None)
                    if ii is None or not np.isfinite(arr[ii]):
                        continue
                    px = float(arr[ii])
                p1.sell_all(tk, d0, float(px))
                continue

        if d0 in s1_entries_by_date:
            for o in s1_entries_by_date[d0]:
                tk = o["ticker"]
                if tk in p1.positions:
                    continue
                if len(p1.positions) >= S1_MAX_POSITIONS:
                    continue
                px = px_open.get((tk, d0), np.nan)
                if not np.isfinite(px):
                    continue
                p1.buy_usd(
                    tk,
                    d0,
                    float(px),
                    min(S1_TRADE_DOLLARS, p1.cash_usd),
                    stop_px_local=o["stop"],
                    max_exit_date=o["max_exit"],
                )

        p1.mark_to_market(close_ffill, date_to_i, d0)
        p2.mark_to_market(close_ffill, date_to_i, d0)
        p3.mark_to_market(close_ffill, date_to_i, d0)

    print()

    c1 = pd.DataFrame(p1.equity_curve)
    c2 = pd.DataFrame(p2.equity_curve)
    c3 = pd.DataFrame(p3.equity_curve)

    combined = c1.merge(c2, on="date", suffixes=("_s1", "_s2")).merge(c3, on="date", suffixes=("", "_s3"))
    combined["equity_usd"] = combined["equity_usd_s1"] + combined["equity_usd_s2"] + combined["equity_usd"]
    combined = combined[["date", "equity_usd"]].copy()

    combined_flows: Dict[pd.Timestamp, float] = {}
    for d, amt in p1.external_flows.items():
        combined_flows[d] = combined_flows.get(d, 0.0) + amt
    for d, amt in p2.external_flows.items():
        combined_flows[d] = combined_flows.get(d, 0.0) + amt
    for d, amt in p3.external_flows.items():
        combined_flows[d] = combined_flows.get(d, 0.0) + amt

    stats = {
        "System1": compute_stats_twr(c1, p1.external_flows),
        "System2": compute_stats_twr(c2, p2.external_flows),
        "System3": compute_stats_twr(c3, p3.external_flows),
        "Combined": compute_stats_twr(combined, combined_flows),
    }

    trades = pd.DataFrame(p1.trades + p2.trades + p3.trades)
    asof_date = pd.Timestamp(calendar[-1]).normalize() if len(calendar) else pd.Timestamp.today().normalize()

    positions = []
    for port in (p1, p2, p3):
        for tk, pos in port.positions.items():
            positions.append(
                {
                    "portfolio": port.name,
                    "ticker": tk,
                    "entry_date": pos.entry_date,
                    "avg_entry_px_local": pos.entry_px_local,
                    "shares": pos.shares,
                    "target_px_local": pos.target_px_local,
                    "stop_px_local": pos.stop_px_local,
                    "max_exit_date": pos.max_exit_date,
                    "asof_date": asof_date,
                }
            )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    c1.to_parquet(BACKTEST_EQUITY_S1_PATH, index=False)
    c2.to_parquet(BACKTEST_EQUITY_S2_PATH, index=False)
    c3.to_parquet(BACKTEST_EQUITY_S3_PATH, index=False)
    combined.to_parquet(BACKTEST_EQUITY_COMBINED_PATH, index=False)
    trades.to_parquet(BACKTEST_TRADES_PATH, index=False)
    pd.DataFrame(positions).to_parquet(BACKTEST_POSITIONS_PATH, index=False)

    with open(BACKTEST_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {BACKTEST_EQUITY_S1_PATH}")
    print(f"Saved {BACKTEST_EQUITY_S2_PATH}")
    print(f"Saved {BACKTEST_EQUITY_S3_PATH}")
    print(f"Saved {BACKTEST_EQUITY_COMBINED_PATH}")
    print(f"Saved {BACKTEST_TRADES_PATH}")
    print(f"Saved {BACKTEST_POSITIONS_PATH}")
    print(f"Saved {BACKTEST_STATS_PATH}")


if __name__ == "__main__":
    main()
