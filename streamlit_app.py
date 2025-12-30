# streamlit_app.py
# CLEAN 2-TAB VERSION
# Tab 1: Signals (click a ticker -> chart updates below)
# Tab 2: Backtest Results (Overview from your current code)
#
# ✅ Strategy math is untouched.
# ✅ Monthly rebalance schedule remains month-end (as in your current version).
# ✅ Removed tabs: Current Portfolio / Monthly Rebalances / Diagnostics / Analytics

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")

# ============================================================
# 1) DATA LOADING
# ============================================================

def load_price_data_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep OHLC if present (for candlesticks). This does NOT change your strategy math.
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


# ============================================================
# 2) RETURN DEFINITIONS (ONLY DIFFERENCE BETWEEN A & B)
# ============================================================

def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1
    return df


def add_relative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative daily growth factor = (1 + stock_ret) / (1 + index_ret)
    """
    df = df.copy()
    if "idx_ret_1d" not in df.columns:
        raise ValueError("idx_ret_1d missing. Merge index returns before calling add_relative_returns().")

    stock_ret = df.groupby("Ticker")["Price"].pct_change()
    idx_ret = df["idx_ret_1d"]

    denom = (1.0 + idx_ret).replace(0.0, np.nan)
    df["1D Return"] = (1.0 + stock_ret) / denom
    return df


def compute_index_momentum(idx: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    idx = idx.copy()
    idx = idx.sort_values(["index", "date"])

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
# 3) MOMENTUM FEATURE ENGINE (SHARED)
# ============================================================

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
    df["Momentum_Mid"]  = (0.5 * df["30D zscore"] + 0.5 * df["45D zscore"])
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
    df["Accel_Mid"]  = df.groupby("Ticker")["Momentum_Mid"].diff()
    df["Accel_Slow"] = df.groupby("Ticker")["Momentum_Slow"].diff()

    def zscore_safe(x: pd.Series) -> pd.Series:
        s = x.std()
        if s == 0 or pd.isna(s):
            return (x - x.mean()).fillna(0.0)
        return ((x - x.mean()) / s).fillna(0.0)

    df["Accel_Fast_z"] = df.groupby("Date")["Accel_Fast"].transform(zscore_safe)
    df["Accel_Mid_z"]  = df.groupby("Date")["Accel_Mid"].transform(zscore_safe)
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
    df["Early_Mid"]  = (0.5 * df["Accel_Mid_z"] + 0.5 * df["Momentum_Mid"])
    df["Early_Slow"] = (0.5 * df["Accel_Slow_z"] + 0.5 * df["Momentum_Slow"])

    df["Early Momentum Score"] = (
        0.5 * df["Early_Slow"] +
        0.3 * df["Early_Mid"] +
        0.2 * df["Early_Fast"]
    )

    return df.fillna(0.0)


def add_relative_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

    for col in ["Rel_Slow", "Rel_Mid", "Rel_Fast"]:
        mean = df.groupby("Date")[col].transform("mean")
        std  = df.groupby("Date")[col].transform("std").replace(0, np.nan)
        df[col + "_z"] = ((df[col] - mean) / std).fillna(0.0)

    df["Momentum_Slow"] = df["Rel_Slow_z"]
    df["Momentum_Mid"]  = df["Rel_Mid_z"]
    df["Momentum_Fast"] = df["Rel_Fast_z"]

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"] +
        0.3 * df["Momentum_Mid"] +
        0.2 * df["Momentum_Fast"]
    )

    return df.fillna(0.0)


# ============================================================
# 4) DAILY LISTS (BASE BUY-LIST PER DAY)
# ============================================================

def get_daily_momentum_topn(df_date: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df_date.empty:
        return pd.DataFrame()
    return df_date.sort_values("Momentum Score", ascending=False).head(top_n)


def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    records = []
    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d]
        picks = get_daily_momentum_topn(snap, top_n=top_n)
        if picks.empty:
            continue
        for _, r in picks.iterrows():
            records.append({
                "Date": d,
                "Ticker": r["Ticker"],
                "Momentum Score": float(r["Momentum Score"]),
                "Early Momentum Score": float(r["Early Momentum Score"]),
            })
    return pd.DataFrame(records)


# ============================================================
# 4B) PERSISTENCE / CONSISTENCY SCORING (RESTORED)
# ============================================================

def final_selection_from_daily(
    daily_df: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    as_of_date=None,
    top_n: int = 10
) -> pd.DataFrame:
    if daily_df.empty:
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
            Appearances=("Date", "count")
        )
        .reset_index()
    )

    agg["Consistency"] = agg["Appearances"] / len(dates)
    agg["Weighted_Score"] = (
        w_momentum * agg["Momentum_Score"] +
        w_early * agg["Early_Momentum_Score"] +
        w_consistency * agg["Consistency"]
    )

    return (
        agg.sort_values("Weighted_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 5) BACKTEST ENGINE (MONTH-END REBALANCE)
# ============================================================

def get_last_price(price_table: pd.DataFrame, ticker: str, date) -> float | None:
    try:
        s = price_table.loc[:date, ticker].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def monthly_rebalance_dates(df_prices: pd.DataFrame) -> list:
    """Month-end rebalance dates = last available trading day in each month."""
    dates = pd.to_datetime(df_prices["Date"]).dropna().sort_values().unique()
    if len(dates) == 0:
        return []
    tmp = pd.DataFrame({"Date": dates})
    tmp["Month"] = tmp["Date"].dt.to_period("M")
    return tmp.groupby("Month")["Date"].max().tolist()


def build_unified_target(
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
    weight_B=0.80
) -> pd.DataFrame:

    selA = final_selection_from_daily(
        dailyA,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    )

    selB = final_selection_from_daily(
        dailyB,
        lookback_days=lookback_days,
        w_momentum=w_momentum,
        w_early=w_early,
        w_consistency=w_consistency,
        as_of_date=as_of_date,
        top_n=top_n
    )

    frames = []

    if not selA.empty:
        dollars_per_name_A = (total_capital * weight_A) / len(selA)
        tmpA = selA[["Ticker"]].copy()
        tmpA["Position_Size"] = dollars_per_name_A
        frames.append(tmpA)

    if not selB.empty:
        dollars_per_name_B = (total_capital * weight_B) / len(selB)
        tmpB = selB[["Ticker"]].copy()
        tmpB["Position_Size"] = dollars_per_name_B
        frames.append(tmpB)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    return (
        combined
        .groupby("Ticker", as_index=False)["Position_Size"]
        .sum()
    )


def simulate_unified_portfolio(
    df_prices: pd.DataFrame,
    price_table: pd.DataFrame,
    dailyA: pd.DataFrame,
    dailyB: pd.DataFrame,
    rebalance_interval: int = 10,  # kept for signature compatibility; unused in monthly mode
    lookback_days: int = 10,
    w_momentum: float = 0.50,
    w_early: float = 0.30,
    w_consistency: float = 0.20,
    top_n: int = 10,
    total_capital: float = 100_000.0
):
    rebalance_dates = monthly_rebalance_dates(df_prices)

    cash = total_capital
    portfolio = {}   # ticker -> shares
    cost_basis = {}  # ticker -> dollars invested

    history = []
    trades = []

    for d in rebalance_dates:
        target_df = build_unified_target(
            dailyA, dailyB,
            as_of_date=d,
            lookback_days=lookback_days,
            w_momentum=w_momentum,
            w_early=w_early,
            w_consistency=w_consistency,
            top_n=top_n,
            total_capital=total_capital,
            weight_A=0.20,
            weight_B=0.80
        )
        if target_df.empty:
            continue

        target_sizes = dict(zip(target_df["Ticker"], target_df["Position_Size"]))
        target = set(target_sizes.keys())
        held = set(portfolio.keys())

        # SELL
        for t in held - target:
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


# ============================================================
# 6) PERFORMANCE + TRADE STATS
# ============================================================

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

    if start <= 0 or end <= 0 or years <= 0:
        cagr = np.nan
    else:
        cagr = ((end / start) ** (1 / years) - 1) * 100

    ret = df["Portfolio Value"].pct_change().dropna()
    sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() and ret.std() > 0 else np.nan
    sortino = np.nan
    downside = ret[ret < 0]
    if downside.std() and downside.std() > 0:
        sortino = np.sqrt(252) * ret.mean() / downside.std()

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


def plot_equity_and_drawdown(history_df: pd.DataFrame, title: str):
    if history_df is None or history_df.empty:
        st.info("No equity history to display.")
        return

    df = history_df.sort_values("Date").copy()
    df["Rolling_Max"] = df["Portfolio Value"].cummax()
    df["Drawdown"] = (df["Portfolio Value"] - df["Rolling_Max"]) / df["Rolling_Max"]

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=df["Date"], y=df["Portfolio Value"], mode="lines", name="Portfolio Value"))
    fig_eq.update_layout(
        title=f"{title} — Equity Curve",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Portfolio Value", tickformat=",.0f"),
        template="plotly_white",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df["Date"], y=df["Drawdown"], mode="lines", name="Drawdown"))
    fig_dd.update_layout(
        title=f"{title} — Drawdown",
        height=220,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Drawdown", tickformat=".0%"),
        template="plotly_white",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ============================================================
# SIGNALS UI HELPERS (CLICK TABLE -> CHART)
# ============================================================

def pick_single_row(df: pd.DataFrame, key: str, label_col: str = "Ticker"):
    """
    Single-row selection using st.dataframe selection_mode="single-row".
    Returns selected value or None.
    """
    if df is None or df.empty or label_col not in df.columns:
        st.dataframe(df, use_container_width=True)
        return None

    view = df.reset_index(drop=True)

    event = st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode="single-row",
    )

    rows = getattr(event, "selection", {}).get("rows", []) if hasattr(event, "selection") else []
    if not rows:
        return None

    val = str(view.iloc[rows[0]][label_col])
    if val == "TOTAL":
        return None
    return val


def _trade_markers_for_ticker(trades_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["Date", "Action", "Price"])
    x = trades_df[trades_df["Ticker"] == ticker].copy()
    if x.empty:
        return x
    x["Date"] = pd.to_datetime(x["Date"])
    return x.sort_values("Date")


def plot_ticker_price_with_trades_and_momentum(
    base_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    score_df: pd.DataFrame,
    ticker: str,
    lookback_days: int = 500
):
    if not ticker:
        return

    px = base_df[base_df["Ticker"] == ticker].copy()
    if px.empty:
        st.warning(f"No price history for {ticker}")
        return

    px = px.sort_values("Date")
    if lookback_days and len(px) > lookback_days:
        px = px.tail(lookback_days)

    sc = score_df[score_df["Ticker"] == ticker].copy() if score_df is not None else pd.DataFrame()
    if not sc.empty:
        sc["Date"] = pd.to_datetime(sc["Date"])
        sc = sc.sort_values("Date")
        if lookback_days and len(sc) > lookback_days:
            sc = sc.tail(lookback_days)

    tdf = _trade_markers_for_ticker(trades_df, ticker)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
    )

    has_ohlc = all(c in px.columns for c in ["Open", "High", "Low", "Close"])

    # --- Price panel ---
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=px["Date"], open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"],
                name="Price",
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(go.Scatter(x=px["Date"], y=px["Price"], mode="lines", name="Price"), row=1, col=1)

    # Trade markers
    if not tdf.empty:
        if "Price" not in tdf.columns or tdf["Price"].isna().all():
            s = px.set_index("Date")["Price"]
            tdf["Price"] = [float(s.loc[:d].iloc[-1]) if len(s.loc[:d]) else np.nan for d in tdf["Date"]]

        def add_marker(action, symbol):
            sub = tdf[tdf["Action"].str.upper() == action].copy()
            if sub.empty:
                return
            fig.add_trace(
                go.Scatter(
                    x=sub["Date"], y=sub["Price"],
                    mode="markers+text",
                    text=[action] * len(sub),
                    textposition="top center",
                    name=action,
                    marker=dict(size=11, symbol=symbol),
                ),
                row=1, col=1
            )

        add_marker("BUY", "triangle-up")
        add_marker("SELL", "triangle-down")
        add_marker("RESIZE", "diamond")

    # --- Momentum panel ---
    if not sc.empty:
        fig.add_trace(go.Scatter(x=sc["Date"], y=sc["Momentum Score"], mode="lines", name="Momentum Score"), row=2, col=1)
        fig.add_trace(go.Scatter(x=sc["Date"], y=sc["Early Momentum Score"], mode="lines", name="Early Momentum"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Momentum Score"), row=2, col=1)

    fig.update_layout(
        height=720,
        margin=dict(l=30, r=20, t=40, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        title=f"{ticker} — Price + Trades + Momentum"
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PIPELINE (UNCHANGED MATH)
# ============================================================

@st.cache_data(show_spinner=False)
def run_full_pipeline():
    parquet_path = "artifacts/index_constituents_5yr.parquet"
    index_path = "artifacts/index_returns_5y.parquet"

    REBALANCE_INTERVAL = 10
    DAILY_TOP_N = 10
    FINAL_TOP_N = 10
    LOOKBACK_DAYS = 10
    CAPITAL_PER_TRADE = 5000
    WINDOWS = (5, 10, 30, 45, 60, 90)

    W_MOM = 0.50
    W_EARLY = 0.30
    W_CONS = 0.20

    base = load_price_data_parquet(parquet_path)
    base = filter_by_index(base, "SP500")
    idx = load_index_returns_parquet(index_path)

    # ---------------- BUCKET A ----------------
    dfA = calculate_momentum_features(add_absolute_returns(base), windows=WINDOWS)
    dfA = add_regime_momentum_score(dfA)
    dfA = add_regime_acceleration(dfA)
    dfA = add_regime_residual_momentum(dfA)
    dfA = add_regime_early_momentum(dfA)
    priceA = dfA.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    dailyA = build_daily_lists(dfA, top_n=DAILY_TOP_N)

    # ---------------- BUCKET B ----------------
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

    dfB["Momentum Score"] = (
        0.5 * dfB["Rel_Slow_z"] +
        0.3 * dfB["Rel_Mid_z"] +
        0.2 * dfB["Rel_Fast_z"]
    )

    dfB = dfB[dfB["Momentum_Slow"] > 1].copy()
    dfB = dfB[dfB["Momentum_Mid"] > 0.5].copy()
    dfB = dfB[dfB["Momentum_Fast"] > 1].copy()

    dfB = add_regime_acceleration(dfB)
    dfB = add_regime_residual_momentum(dfB)
    dfB = add_regime_early_momentum(dfB)

    dailyB = build_daily_lists(dfB, top_n=DAILY_TOP_N)

    # ---------------- BUCKET C (UNIFIED BACKTEST) ----------------
    histU, tradesU = simulate_unified_portfolio(
        df_prices=base,
        price_table=priceA,
        dailyA=dailyA,
        dailyB=dailyB,
        rebalance_interval=REBALANCE_INTERVAL,
        lookback_days=LOOKBACK_DAYS,
        w_momentum=W_MOM,
        w_early=W_EARLY,
        w_consistency=W_CONS,
        top_n=FINAL_TOP_N,
        total_capital=100_000.0
    )

    statsU = compute_performance_stats(histU)
    trade_statsU = compute_trade_stats(tradesU)

    scoreA = dfA[["Date", "Ticker", "Momentum Score", "Early Momentum Score"]].copy()
    scoreB = dfB[["Date", "Ticker", "Momentum Score", "Early Momentum Score"]].copy()

    return {
        "params": {
            "REBALANCE_INTERVAL": REBALANCE_INTERVAL,
            "DAILY_TOP_N": DAILY_TOP_N,
            "FINAL_TOP_N": FINAL_TOP_N,
            "LOOKBACK_DAYS": LOOKBACK_DAYS,
            "CAPITAL_PER_TRADE": CAPITAL_PER_TRADE,
            "WINDOWS": WINDOWS,
            "W_MOM": W_MOM,
            "W_EARLY": W_EARLY,
            "W_CONS": W_CONS,
        },
        "BucketC": {"history": histU, "trades": tradesU, "stats": statsU, "trade_stats": trade_statsU},
        "internals": {
            "base": base,
            "dailyA": dailyA,
            "dailyB": dailyB,
            "scoreA": scoreA,
            "scoreB": scoreB,
        }
    }


def _stats_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"Metric": list(d.keys()), "Value": list(d.values())})


# ============================================================
# UI (2 TABS)
# ============================================================

st.title("📈 Momentum Portfolio")

with st.spinner("Running full pipeline from artifacts…"):
    out = run_full_pipeline()

params = out["params"]
bucketC = out["BucketC"]
internals = out["internals"]

base = internals["base"]
dailyA = internals["dailyA"]
dailyB = internals["dailyB"]
scoreA = internals.get("scoreA", pd.DataFrame())
scoreB = internals.get("scoreB", pd.DataFrame())

hist = bucketC["history"]
trades = bucketC["trades"]
stats = bucketC["stats"]
trade_stats = bucketC["trade_stats"]

# Common dates (for "as-of")
all_dates = sorted(set(dailyA["Date"]).intersection(set(dailyB["Date"])))
as_of = all_dates[-1] if all_dates else None

# Session state for selection in Signals tab
if "signals_selected_ticker" not in st.session_state:
    st.session_state.signals_selected_ticker = None

tab_signals, tab_backtest = st.tabs(["Signals", "Backtest Results"])

# =========================
# TAB 1: SIGNALS
# =========================
with tab_signals:
    st.markdown("### Signals")
    if as_of is None:
        st.info("No common dates found across signal sources.")
    else:
        st.caption(f"As-of date: {pd.to_datetime(as_of).date()}")

        # Build "today" signal tables (top N) from score snapshots
        top_n = int(params.get("FINAL_TOP_N", 10))

        snapA = scoreA[scoreA["Date"] == as_of].copy() if not scoreA.empty else pd.DataFrame()
        snapB = scoreB[scoreB["Date"] == as_of].copy() if not scoreB.empty else pd.DataFrame()

        def _prep(df, label):
            if df is None or df.empty:
                return pd.DataFrame()
            outdf = df[["Ticker", "Momentum Score", "Early Momentum Score"]].copy()
            outdf = outdf.sort_values("Momentum Score", ascending=False).head(top_n).reset_index(drop=True)
            outdf.insert(0, "Source", label)
            return outdf

        sigA = _prep(snapA, "A (Absolute)")
        sigB = _prep(snapB, "B (Relative)")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Top Signals — A (Absolute)")
            selA = pick_single_row(sigA, key="sigA_table", label_col="Ticker") if not sigA.empty else None
            if sigA.empty:
                st.info("No signals available for A on this date.")

        with c2:
            st.markdown("#### Top Signals — B (Relative)")
            selB = pick_single_row(sigB, key="sigB_table", label_col="Ticker") if not sigB.empty else None
            if sigB.empty:
                st.info("No signals available for B on this date.")

        # Update selection if user clicked
        if selA:
            st.session_state.signals_selected_ticker = selA
        if selB:
            st.session_state.signals_selected_ticker = selB

        selected = st.session_state.signals_selected_ticker

        st.markdown("---")
        st.markdown("### Chart")

        if not selected:
            st.info("Click a ticker in either table above to show its chart.")
        else:
            st.markdown(f"Selected: `{selected}`")

            score_source = st.radio(
                "Score source for the chart",
                options=["scoreA", "scoreB"],
                horizontal=True,
                index=0,
                key="signals_score_source",
            )
            score_df = internals.get(score_source, pd.DataFrame())

            plot_ticker_price_with_trades_and_momentum(
                base_df=base,
                trades_df=trades,
                score_df=score_df,
                ticker=selected,
                lookback_days=500
            )

# =========================
# TAB 2: BACKTEST (OVERVIEW)
# =========================
with tab_backtest:
    st.markdown("### Backtest Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", f"{stats.get('Total Return (%)', np.nan):.2f}")
    c2.metric("CAGR (%)", f"{stats.get('CAGR (%)', np.nan):.2f}")
    c3.metric("Sharpe", f"{stats.get('Sharpe Ratio', np.nan):.2f}")
    c4.metric("Max Drawdown (%)", f"{stats.get('Max Drawdown (%)', np.nan):.2f}")

    st.markdown("### Equity & Drawdown")
    plot_equity_and_drawdown(hist, title="Momentum Portfolio")

    st.markdown("### Trade Statistics")
    st.dataframe(_stats_to_df(trade_stats), use_container_width=True)

    with st.expander("Show full backtest trades"):
        if trades is None or trades.empty:
            st.write("No trades.")
        else:
            st.dataframe(trades.sort_values("Date"), use_container_width=True)
