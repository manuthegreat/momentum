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
# Helper functions
# ============================================================

def compute_monthly_returns_from_history(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty or "Date" not in hist.columns or "Portfolio Value" not in hist.columns:
        return pd.DataFrame()
    df = hist.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    # history is already monthly (month-end), but grouping is safe
    m = df.groupby("Month", as_index=False).agg(
        Date=("Date", "max"),
        Portfolio_Value=("Portfolio Value", "last")
    )
    m["Return_%"] = m["Portfolio_Value"].pct_change() * 100.0
    m["PnL_$"] = m["Portfolio_Value"].diff()
    return m


def realized_pnl_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["Action"] = t["Action"].astype(str).str.upper()
    sells = t[t["Action"] == "SELL"].copy()
    if sells.empty or "PnL" not in sells.columns:
        return pd.DataFrame()
    out = (sells.groupby("Ticker", as_index=False)["PnL"].sum()
           .rename(columns={"PnL": "Realized_PnL_$"})
           .sort_values("Realized_PnL_$", ascending=False)
           .reset_index(drop=True))
    return out


def trade_activity_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty or "Date" not in trades.columns:
        return pd.DataFrame()
    t = trades.copy()
    t["Date"] = pd.to_datetime(t["Date"])
    t["Month"] = t["Date"].dt.to_period("M").astype(str)
    t["Action"] = t["Action"].astype(str).str.upper()
    pivot = (t.groupby(["Month", "Action"]).size()
             .unstack(fill_value=0)
             .reset_index())
    return pivot


def mtm_from_preview(base: pd.DataFrame, preview: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Treat preview Target_$ as the model portfolio initiated on Signal_Date.
    Compute shares at Signal_Date and mark-to-market to latest available price date.
    Uses ONLY base price history (no strategy calcs).
    """
    if preview is None or preview.empty:
        return pd.DataFrame(), None, None

    if "Signal_Date" not in preview.columns:
        return pd.DataFrame(), None, None

    signal_date = pd.to_datetime(preview["Signal_Date"].max())
    if pd.isna(signal_date):
        return pd.DataFrame(), None, None

    # latest available price date in base
    base_dates = pd.to_datetime(base["Date"]).dropna()
    latest_date = base_dates.max() if len(base_dates) else signal_date

    # quick price lookup: last price <= date per ticker
    b = base.copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Ticker", "Date"])

    # we prefer Close if present, else Price
    price_col = "Close" if "Close" in b.columns else "Price"

    def last_px(ticker: str, d: pd.Timestamp) -> float | None:
        s = b.loc[(b["Ticker"] == ticker) & (b["Date"] <= d), price_col]
        if s.empty:
            return None
        v = s.iloc[-1]
        try:
            return float(v)
        except Exception:
            return None

    if "Target_$" not in preview.columns:
        return pd.DataFrame(), signal_date, latest_date

    rows = []
    total = float(preview["Target_$"].sum()) if preview["Target_$"].notna().any() else 0.0

    for _, r in preview.iterrows():
        tkr = str(r["Ticker"])
        dollars = float(r["Target_$"]) if pd.notna(r["Target_$"]) else 0.0

        entry_px = last_px(tkr, signal_date)
        mark_px = last_px(tkr, latest_date)

        if entry_px is None or entry_px <= 0:
            shares = np.nan
            pnl = np.nan
            pnl_pct = np.nan
            mkt = np.nan
        else:
            shares = dollars / entry_px
            if mark_px is None or mark_px <= 0:
                pnl = np.nan
                pnl_pct = np.nan
                mkt = np.nan
            else:
                mkt = shares * mark_px
                pnl = shares * (mark_px - entry_px)
                pnl_pct = (mark_px / entry_px - 1.0) * 100.0

        rows.append({
            "Ticker": tkr,
            "Name": r.get("Name", None),
            "Sector": r.get("Sector", None),
            "Target_$": dollars,
            "Weight_%": (dollars / total * 100.0) if total > 0 else np.nan,
            "Signal_Date": signal_date.date(),
            "Entry_Price": entry_px,
            "Latest_Date": latest_date.date(),
            "Latest_Price": mark_px,
            "Shares": shares,
            "Market_Value_$": mkt,
            "PnL_$": pnl,
            "PnL_%": pnl_pct,
        })

    out = pd.DataFrame(rows).sort_values("Target_$", ascending=False).reset_index(drop=True)

    if not out.empty:
        total_row = {
            "Ticker": "TOTAL",
            "Name": None,
            "Sector": None,
            "Target_$": out["Target_$"].sum(),
            "Weight_%": out["Weight_%"].sum(skipna=True),
            "Signal_Date": "",
            "Entry_Price": np.nan,
            "Latest_Date": "",
            "Latest_Price": np.nan,
            "Shares": np.nan,
            "Market_Value_$": out["Market_Value_$"].sum(skipna=True),
            "PnL_$": out["PnL_$"].sum(skipna=True),
            "PnL_%": np.nan,
        }
        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    return out, signal_date, latest_date




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

tab_signals, tab_overview, tab_current, tab_rebalances, tab_analytics, tab_backtest = st.tabs(
    ["Signals", "Overview", "Current Portfolio (MTM)", "Rebalance History", "Analytics", "Backtest Results"]
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


with tab_overview:
    st.markdown("### Performance summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (%)", f"{stats.get('Total Return (%)', np.nan):.2f}" if "Total Return (%)" in stats else "—")
    c2.metric("CAGR (%)", f"{stats.get('CAGR (%)', np.nan):.2f}" if "CAGR (%)" in stats else "—")
    c3.metric("Sharpe", f"{stats.get('Sharpe Ratio', np.nan):.2f}" if "Sharpe Ratio" in stats else "—")
    c4.metric("Max Drawdown (%)", f"{stats.get('Max Drawdown (%)', np.nan):.2f}" if "Max Drawdown (%)" in stats else "—")

    st.markdown("### Monthly performance (from bucketC_history)")
    monthly = compute_monthly_returns_from_history(hist)
    if monthly.empty:
        st.info("Not enough history to compute monthly returns.")
    else:
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly["Month"], y=monthly["Return_%"], name="Monthly Return (%)"))
        fig_m.update_layout(height=320, margin=dict(l=40, r=40, t=40, b=40), template="plotly_white")
        st.plotly_chart(fig_m, use_container_width=True)

        with st.expander("Show monthly table"):
            st.dataframe(monthly, use_container_width=True)

    st.markdown("### Equity & Drawdown")
    plot_equity_and_drawdown(hist, title="Momentum Portfolio")


with tab_current:
    st.markdown("### Model portfolio mark-to-market (from latest Signals)")
    mtm, mtm_entry, mtm_latest = mtm_from_preview(base, preview)

    if mtm.empty:
        st.info("MTM table not available (preview missing Target_$ or Signal_Date).")
    else:
        st.caption(
            f"Computed from **bucketC_preview_latest.parquet**: "
            f"entry={mtm_entry.date() if mtm_entry is not None else '—'} → "
            f"latest={mtm_latest.date() if mtm_latest is not None else '—'}"
        )

        selected = pick_single_row(mtm, key="mtm_table", label_col="Ticker")
        st.dataframe(mtm, use_container_width=True)

        if selected:
            st.markdown("#### Selected ticker chart")
            # reuse your existing chart source toggle
            chart_source = st.radio(
                "Chart momentum source (visual only)",
                options=["Absolute (A)", "Relative (B)"],
                horizontal=True,
                index=0,
                key="mtm_chart_source"
            )
            score_for_chart = load_score_df("A" if chart_source.startswith("Absolute") else "B")
            plot_ticker_price_with_trades_and_momentum(
                base_df=base,
                trades_df=trades,
                score_df=score_for_chart,
                ticker=selected,
                lookback_days=500
            )


with tab_rebalances:
    st.markdown("### Rebalance history (from bucketC_trades)")
    if trades is None or trades.empty:
        st.info("No trades to display.")
    else:
        t = trades.copy()
        t["Date"] = pd.to_datetime(t["Date"])
        t["Action"] = t["Action"].astype(str).str.upper()

        st.markdown("#### Trade activity by month")
        activity = trade_activity_by_month(t)
        st.dataframe(activity, use_container_width=True)

        st.markdown("#### Trades blotter (filter by rebalance date)")
        dates = sorted(t["Date"].dropna().unique())
        d = st.selectbox("Select rebalance date", dates, index=len(dates)-1)
        sub = t[t["Date"] == pd.to_datetime(d)].sort_values(["Action", "Ticker"])
        st.dataframe(sub, use_container_width=True)


with tab_analytics:
    st.markdown("### Analytics (realized PnL + best/worst months)")

    # realized PnL
    realized = realized_pnl_by_ticker(trades)
    if realized.empty:
        st.info("No realized PnL yet (no SELL trades).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top realized PnL (click)")
            sel1 = pick_single_row(realized.head(15), key="real_top", label_col="Ticker")
        with c2:
            st.markdown("#### Bottom realized PnL (click)")
            bottom = realized.tail(15).sort_values("Realized_PnL_$", ascending=True).reset_index(drop=True)
            sel2 = pick_single_row(bottom, key="real_bot", label_col="Ticker")

        chosen = sel1 or sel2
        if chosen:
            st.markdown("#### Selected ticker chart")
            chart_source = st.radio(
                "Chart momentum source (visual only)",
                options=["Absolute (A)", "Relative (B)"],
                horizontal=True,
                index=0,
                key="ana_chart_source"
            )
            score_for_chart = load_score_df("A" if chart_source.startswith("Absolute") else "B")
            plot_ticker_price_with_trades_and_momentum(
                base_df=base,
                trades_df=trades,
                score_df=score_for_chart,
                ticker=chosen,
                lookback_days=500
            )

    st.markdown("---")
    monthly = compute_monthly_returns_from_history(hist)
    st.markdown("#### Best / worst months (from bucketC_history)")
    if monthly.empty or "Return_%" not in monthly.columns:
        st.info("Monthly table not available.")
    else:
        best = monthly.sort_values("Return_%", ascending=False).head(6)
        worst = monthly.sort_values("Return_%", ascending=True).head(6)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Best months**")
            st.dataframe(best[["Month", "Return_%", "PnL_$"]], use_container_width=True)
        with cc2:
            st.markdown("**Worst months**")
            st.dataframe(worst[["Month", "Return_%", "PnL_$"]], use_container_width=True)

