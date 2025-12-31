# streamlit_app.py
# UI-ONLY VERSION:
# - Loads precomputed reference parquets produced by update_parquet.py
# - Displays Signals + Sector exposure + Chart
# - Displays Backtest Results
#
# IMPORTANT: No strategy calculations happen here.

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

ARTIFACTS_DIR = "artifacts"

BASE_SP500_PATH = os.path.join(ARTIFACTS_DIR, "base_SP500_5y.parquet")
SCOREA_PATH = os.path.join(ARTIFACTS_DIR, "scoreA.parquet")
SCOREB_PATH = os.path.join(ARTIFACTS_DIR, "scoreB.parquet")

BUCKETC_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_history.parquet")
BUCKETC_TRADES_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_trades.parquet")
BUCKETC_STATS_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_stats.parquet")
BUCKETC_TRADE_STATS_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_trade_stats.parquet")

PREVIEW_LATEST_PATH = os.path.join(ARTIFACTS_DIR, "bucketC_preview_latest.parquet")


st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")



# ============================================================
# LOAD REFERENCE ARTIFACTS (FAST)
# ============================================================

@st.cache_data(show_spinner=False)
def load_reference_artifacts():
    required = [
        BASE_SP500_PATH,
        BUCKETC_HISTORY_PATH,
        BUCKETC_TRADES_PATH,
        BUCKETC_STATS_PATH,
        BUCKETC_TRADE_STATS_PATH,
        PREVIEW_LATEST_PATH,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required reference artifacts. Run update_parquet.py overnight to generate them:\n"
            + "\n".join(missing)
        )

    base = pd.read_parquet(BASE_SP500_PATH)
    base["Date"] = pd.to_datetime(base["Date"])

    hist = pd.read_parquet(BUCKETC_HISTORY_PATH)
    if "Date" in hist.columns:
        hist["Date"] = pd.to_datetime(hist["Date"])

    trades = pd.read_parquet(BUCKETC_TRADES_PATH)
    if "Date" in trades.columns:
        trades["Date"] = pd.to_datetime(trades["Date"])

    stats_df = pd.read_parquet(BUCKETC_STATS_PATH)
    trade_stats_df = pd.read_parquet(BUCKETC_TRADE_STATS_PATH)

    preview = pd.read_parquet(PREVIEW_LATEST_PATH)
    if "Signal_Date" in preview.columns:
        preview["Signal_Date"] = pd.to_datetime(preview["Signal_Date"])

    return base, hist, trades, stats_df, trade_stats_df, preview


@st.cache_data(show_spinner=False)
def load_score_df(which: str) -> pd.DataFrame:
    path = SCOREA_PATH if which == "A" else SCOREB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing score parquet: {path}")
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ============================================================
# PLOTTING + UI HELPERS (UNCHANGED UI LOGIC)
# ============================================================

def compute_stats_dict(stats_df: pd.DataFrame) -> dict:
    if stats_df is None or stats_df.empty:
        return {}
    if set(["Metric", "Value"]).issubset(stats_df.columns):
        return dict(zip(stats_df["Metric"].astype(str), stats_df["Value"]))
    return {}


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


def pick_single_row(df: pd.DataFrame, key: str, label_col: str = "Ticker", column_config=None):
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
        column_config=column_config,
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
# UI
# ============================================================

st.title("📈 Momentum Portfolio (UI-only)")

try:
    base, hist, trades, stats_df, trade_stats_df, preview = load_reference_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

stats = compute_stats_dict(stats_df)
trade_stats = compute_stats_dict(trade_stats_df)

signal_date = None
if preview is not None and not preview.empty and "Signal_Date" in preview.columns:
    signal_date = pd.to_datetime(preview["Signal_Date"].max())
else:
    signal_date = None

tab_signals, tab_backtest = st.tabs(
    ["Signals", "Overview"]
)


with tab_signals:
    st.markdown("### Next Rebalance Preview (Option A)")

    if signal_date is not None:
        st.caption(
            f"Signals as of **{signal_date.date()}** · "
            f"Rebalance: Month-end"
        )
    else:
        st.caption("Signals date not found in preview parquet.")

    chart_source = st.radio(
        "Chart momentum source (visual only)",
        options=["Absolute (A)", "Relative (B)"],
        horizontal=True,
        index=0
    )
    score_for_chart = load_score_df("A" if chart_source.startswith("Absolute") else "B")

    if preview is None or preview.empty:
        st.info("No signals available.")
    else:
        st.markdown("### Sector exposure")

        sector_df = preview.copy()
        sector_df["Sector"] = sector_df["Sector"].fillna("Unknown")

        if "Weight_%" in sector_df.columns and sector_df["Weight_%"].notna().any():
            expo = sector_df.groupby("Sector", as_index=False)["Weight_%"].sum()
            values_col = "Weight_%"
            title = "Sector Exposure (by weight %)"
        else:
            expo = sector_df.groupby("Sector", as_index=False)["Target_$"].sum()
            values_col = "Target_$"
            title = "Sector Exposure (by $)"

        fig_sector = go.Figure(
            data=[go.Pie(labels=expo["Sector"], values=expo[values_col], hole=0.45)]
        )
        fig_sector.update_layout(
            title=title,
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
            template="plotly_white"
        )
        st.plotly_chart(fig_sector, use_container_width=True)

        st.markdown("#### Target portfolio (click a row to update chart)")
        battery_cfg = {
            "Signal_Confidence": st.column_config.ProgressColumn(
                "Signal Strength",
                min_value=0,
                max_value=100,
                format="%d%%",
            )
        }

        selected = pick_single_row(
            preview,
            key="signal_preview",
            label_col="Ticker",
            column_config=battery_cfg
        )

        if selected:
            st.markdown("#### Selected ticker chart")
            plot_ticker_price_with_trades_and_momentum(
                base_df=base,
                trades_df=trades,
                score_df=score_for_chart,
                ticker=selected,
                lookback_days=500
            )

with tab_backtest:
    st.markdown("### Backtest Results (Bucket C)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", f"{stats.get('Total Return (%)', np.nan):.2f}" if "Total Return (%)" in stats else "—")
    c2.metric("CAGR (%)", f"{stats.get('CAGR (%)', np.nan):.2f}" if "CAGR (%)" in stats else "—")
    c3.metric("Sharpe", f"{stats.get('Sharpe Ratio', np.nan):.2f}" if "Sharpe Ratio" in stats else "—")
    c4.metric("Max Drawdown (%)", f"{stats.get('Max Drawdown (%)', np.nan):.2f}" if "Max Drawdown (%)" in stats else "—")

    st.markdown("### Equity & Drawdown")
    plot_equity_and_drawdown(hist, title="Momentum Portfolio")

    st.markdown("### Trade Statistics")
    if trade_stats_df is None or trade_stats_df.empty:
        st.write("No trade stats.")
    else:
        st.dataframe(trade_stats_df, use_container_width=True)

    with st.expander("Show full backtest trades"):
        if trades is None or trades.empty:
            st.write("No trades.")
        else:
            x = trades.copy()
            x["Date"] = pd.to_datetime(x["Date"])
            st.dataframe(x.sort_values("Date"), use_container_width=True)

