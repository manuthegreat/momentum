# streamlit_app.py
# UI-ONLY VERSION:
# - Loads precomputed signal parquets produced by update_parquet.py
# - Displays signals from all three systems + combined action list
#
# IMPORTANT: No strategy calculations happen here.

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = "artifacts"

WEEKLY_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "weekly_swing_signals.parquet")
FIB_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "fib_signals.parquet")
MOMENTUM_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "momentum_bucketc_signals.parquet")
ACTION_LIST_PATH = os.path.join(ARTIFACTS_DIR, "action_list.parquet")
BACKTEST_EQUITY_S1_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_system1.parquet")
BACKTEST_EQUITY_S2_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_system2.parquet")
BACKTEST_EQUITY_S3_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_system3.parquet")
BACKTEST_EQUITY_COMBINED_PATH = os.path.join(ARTIFACTS_DIR, "backtest_equity_combined.parquet")
BACKTEST_TRADES_PATH = os.path.join(ARTIFACTS_DIR, "backtest_trades.parquet")
BACKTEST_POSITIONS_PATH = os.path.join(ARTIFACTS_DIR, "backtest_positions.parquet")
BACKTEST_STATS_PATH = os.path.join(ARTIFACTS_DIR, "backtest_stats.json")

st.set_page_config(page_title="Momentum Signals Dashboard", layout="wide")


def _supports_column_config() -> bool:
    return hasattr(st, "column_config")


def _format_percentage(series: pd.Series, decimals: int = 0) -> pd.Series:
    fmt = f"{{:.{decimals}f}}%"
    return series.map(lambda value: fmt.format(value) if pd.notna(value) else "")


def _normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        normalized = frame.convert_dtypes(dtype_backend="numpy_nullable")
    except TypeError:
        normalized = frame.convert_dtypes()

    for column in normalized.columns:
        dtype = normalized[column].dtype
        dtype_name = str(dtype).lower()
        if "string" in dtype_name or "unicode" in dtype_name:
            normalized[column] = normalized[column].astype("string[python]")
            continue
        if "arrow" in dtype_name or "pyarrow" in dtype_name:
            normalized[column] = normalized[column].astype(object)
            continue
        if hasattr(pd, "ArrowDtype") and isinstance(dtype, pd.ArrowDtype):
            normalized[column] = normalized[column].astype(object)

    return normalized


def _streamlit_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    object_cols = safe.select_dtypes(include=["object", "string"]).columns
    if not object_cols.empty:
        safe[object_cols] = safe[object_cols].astype("string[python]").astype(object)
    for column in safe.columns:
        dtype_name = str(safe[column].dtype).lower()
        if "pyarrow" in dtype_name or "arrow" in dtype_name:
            safe[column] = safe[column].astype("string[python]").astype(object)
        elif hasattr(pd, "ArrowDtype") and isinstance(safe[column].dtype, pd.ArrowDtype):
            safe[column] = safe[column].astype(object)
    return safe


def _latest_date(frame: pd.DataFrame, column: str) -> pd.Timestamp | None:
    if column not in frame.columns or frame.empty:
        return None
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max()


def _monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame[["date", "equity_usd"]].copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce")
    view["equity_usd"] = pd.to_numeric(view["equity_usd"], errors="coerce")
    view = view.dropna(subset=["date", "equity_usd"]).sort_values("date")
    if view.empty:
        return pd.DataFrame()

    view["month"] = view["date"].dt.to_period("M").dt.to_timestamp()
    grouped = view.groupby("month", as_index=False)["equity_usd"].agg(["first", "last"]).reset_index()
    grouped = grouped.rename(columns={"first": "Start", "last": "End", "month": "Month"})
    grouped["Month %"] = (grouped["End"] / grouped["Start"] - 1.0) * 100.0
    first_equity = grouped["Start"].iloc[0]
    grouped["Cum %"] = (grouped["End"] / first_equity - 1.0) * 100.0
    return grouped[["Month", "Start", "End", "Month %", "Cum %"]]


def _render_monthly_summary(frame: pd.DataFrame, label: str) -> None:
    summary = _monthly_summary(frame)
    if summary.empty:
        return

    st.markdown(f"#### {label}: Monthly Summary")
    summary_view = summary.copy()
    summary_view["Start"] = summary_view["Start"].map(lambda v: f"${v:,.0f}")
    summary_view["End"] = summary_view["End"].map(lambda v: f"${v:,.0f}")
    summary_view["Month %"] = summary_view["Month %"].map(lambda v: f"{v:.2f}%")
    summary_view["Cum %"] = summary_view["Cum %"].map(lambda v: f"{v:.2f}%")
    st.dataframe(_streamlit_safe_frame(summary_view), use_container_width=True)


def _filter_dataframe(frame: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    filtered = frame.copy()
    with st.expander("Filters", expanded=False):
        filter_cols = st.multiselect(
            "Filter columns",
            options=list(frame.columns),
            key=f"{key_prefix}-filter-cols",
        )
        for col in filter_cols:
            series = frame[col]
            col_key = f"{key_prefix}-{col}"
            if pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors="coerce")
                min_date = series.min()
                max_date = series.max()
                if pd.isna(min_date) or pd.isna(max_date):
                    continue
                date_range = st.date_input(
                    f"{col} range",
                    value=(min_date.date(), max_date.date()),
                    key=col_key,
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = series.dt.date.between(start_date, end_date)
                    filtered = filtered[mask]
            elif pd.api.types.is_numeric_dtype(series):
                min_val = float(series.min())
                max_val = float(series.max())
                if min_val == max_val:
                    st.caption(f"{col}: {min_val}")
                    continue
                step = (max_val - min_val) / 100 if max_val > min_val else 1.0
                selected = st.slider(
                    f"{col} range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                    key=col_key,
                )
                filtered = filtered[series.between(selected[0], selected[1])]
            else:
                unique = sorted(series.dropna().astype(str).unique())
                if len(unique) <= 50:
                    selected = st.multiselect(
                        f"{col} values",
                        options=unique,
                        default=unique,
                        key=col_key,
                    )
                    if selected:
                        filtered = filtered[series.astype(str).isin(selected)]
                else:
                    query = st.text_input(f"{col} contains", key=col_key)
                    if query:
                        filtered = filtered[series.astype(str).str.contains(query, case=False, na=False)]
    return filtered


@st.cache_data(show_spinner=False)
def load_signal_artifacts():
    required = [
        WEEKLY_SIGNALS_PATH,
        FIB_SIGNALS_PATH,
        MOMENTUM_SIGNALS_PATH,
        ACTION_LIST_PATH,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required signal artifacts. Run update_parquet.py to generate them:\n"
            + "\n".join(missing)
        )

    weekly = _normalize_dataframe(pd.read_parquet(WEEKLY_SIGNALS_PATH))
    fib = _normalize_dataframe(pd.read_parquet(FIB_SIGNALS_PATH))
    mom = _normalize_dataframe(pd.read_parquet(MOMENTUM_SIGNALS_PATH))
    action_list = _normalize_dataframe(pd.read_parquet(ACTION_LIST_PATH))

    if "signal_date" in weekly.columns:
        weekly["signal_date"] = pd.to_datetime(weekly["signal_date"])
    if "Signal_Date" in fib.columns:
        fib["Signal_Date"] = pd.to_datetime(fib["Signal_Date"])
    if "Signal_Date" in mom.columns:
        mom["Signal_Date"] = pd.to_datetime(mom["Signal_Date"])
    if "mom_date" in action_list.columns:
        action_list["mom_date"] = pd.to_datetime(action_list["mom_date"])
    if "weekly_date" in action_list.columns:
        action_list["weekly_date"] = pd.to_datetime(action_list["weekly_date"])

    return weekly, fib, mom, action_list


@st.cache_data(show_spinner=False)
def load_backtest_artifacts():
    required = [
        BACKTEST_EQUITY_S1_PATH,
        BACKTEST_EQUITY_S2_PATH,
        BACKTEST_EQUITY_S3_PATH,
        BACKTEST_EQUITY_COMBINED_PATH,
        BACKTEST_TRADES_PATH,
        BACKTEST_POSITIONS_PATH,
    ]
    if not all(os.path.exists(path) for path in required):
        return None

    equity_s1 = _normalize_dataframe(pd.read_parquet(BACKTEST_EQUITY_S1_PATH))
    equity_s2 = _normalize_dataframe(pd.read_parquet(BACKTEST_EQUITY_S2_PATH))
    equity_s3 = _normalize_dataframe(pd.read_parquet(BACKTEST_EQUITY_S3_PATH))
    equity_combined = _normalize_dataframe(pd.read_parquet(BACKTEST_EQUITY_COMBINED_PATH))
    trades = _normalize_dataframe(pd.read_parquet(BACKTEST_TRADES_PATH))
    positions = _normalize_dataframe(pd.read_parquet(BACKTEST_POSITIONS_PATH))

    stats = None
    if os.path.exists(BACKTEST_STATS_PATH):
        with open(BACKTEST_STATS_PATH, "r") as f:
            stats = json.load(f)

    for frame in (equity_s1, equity_s2, equity_s3, equity_combined, trades, positions):
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        if "entry_date" in frame.columns:
            frame["entry_date"] = pd.to_datetime(frame["entry_date"])

    if stats is not None:
        stats = pd.DataFrame(stats).T.reset_index().rename(columns={"index": "Portfolio"})

    return {
        "equity_s1": equity_s1,
        "equity_s2": equity_s2,
        "equity_s3": equity_s3,
        "equity_combined": equity_combined,
        "trades": trades,
        "positions": positions,
        "stats": stats,
    }


st.title("📊 Consolidated Momentum Signals")

try:
    weekly_sig, fib_sig, mom_sig, action_list = load_signal_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Weekly Swing", f"{len(weekly_sig):,}")
c2.metric("Fibonacci", f"{len(fib_sig):,}")
c3.metric("Momentum Bucket C", f"{len(mom_sig):,}")
c4.metric("Action List", f"{len(action_list):,}")


tab_weekly, tab_fib, tab_mom, tab_action, tab_backtest = st.tabs(
    ["Weekly Swing", "Fibonacci", "Momentum Bucket C", "Action List", "Backtests"]
)

with tab_weekly:
    st.markdown("### Weekly Swing Signals")

    if weekly_sig.empty:
        st.info("No weekly swing signals found in the latest batch.")
    else:
        latest_weekly = _latest_date(weekly_sig, "signal_date")
        if latest_weekly is not None:
            st.caption(f"Latest weekly signal date: {latest_weekly.date()}")
        dates = sorted(weekly_sig["signal_date"].dt.date.unique())
        selected = st.selectbox("Signal date", options=dates, index=len(dates) - 1)
        view = weekly_sig[weekly_sig["signal_date"].dt.date == selected].copy()

        st.caption(f"Signals for {selected}")
        filtered = _filter_dataframe(view, "weekly")
        st.dataframe(_streamlit_safe_frame(filtered), use_container_width=True)

with tab_fib:
    st.markdown("### Fibonacci Confirmation Signals")

    if fib_sig.empty:
        st.info("No Fibonacci signals found in the latest batch.")
    else:
        latest_fib = _latest_date(fib_sig, "Signal_Date")
        if latest_fib is not None:
            st.caption(f"Latest Fibonacci signal date: {latest_fib.date()}")
        signal_options = ["BUY", "WATCH", "INVALID"]
        selected_signals = st.multiselect(
            "Signal filter",
            options=signal_options,
            default=["BUY", "WATCH"],
        )
        view = fib_sig[fib_sig["Signal"].isin(selected_signals)].copy()
        filtered = _filter_dataframe(view, "fib")
        st.dataframe(_streamlit_safe_frame(filtered), use_container_width=True)

with tab_mom:
    st.markdown("### Momentum Bucket C (Latest Candidates)")

    if mom_sig.empty:
        st.info("No momentum bucket C candidates found in the latest batch.")
    else:
        signal_date = pd.to_datetime(mom_sig["Signal_Date"].max()).date()
        st.caption(f"Signals as of {signal_date}")
        filtered = _filter_dataframe(mom_sig, "mom")

        if _supports_column_config():
            pct_cfg = {
                "Weight_%": st.column_config.NumberColumn("Weight %", format="%.2f"),
                "Signal_Confidence": st.column_config.ProgressColumn(
                    "Signal Strength",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
            }
            st.dataframe(
                _streamlit_safe_frame(filtered),
                use_container_width=True,
                column_config=pct_cfg,
            )
        else:
            view = filtered.copy()
            if "Weight_%" in view.columns:
                view["Weight_%"] = view["Weight_%"].map(
                    lambda value: f"{value:.2f}" if pd.notna(value) else ""
                )
            if "Signal_Confidence" in view.columns:
                view["Signal_Confidence"] = _format_percentage(
                    view["Signal_Confidence"], decimals=0
                )
            st.dataframe(_streamlit_safe_frame(view), use_container_width=True)

with tab_action:
    st.markdown("### Action List (Best Signal Per Ticker)")

    if action_list.empty:
        st.info("No action list generated in the latest batch.")
    else:
        filtered = _filter_dataframe(action_list, "action")
        if _supports_column_config():
            pct_cfg = {
                "mom_conf": st.column_config.ProgressColumn(
                    "Momentum Confidence",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
            }
            st.dataframe(
                _streamlit_safe_frame(filtered),
                use_container_width=True,
                column_config=pct_cfg,
            )
        else:
            view = filtered.copy()
            if "mom_conf" in view.columns:
                view["mom_conf"] = _format_percentage(view["mom_conf"], decimals=0)
            st.dataframe(_streamlit_safe_frame(view), use_container_width=True)

with tab_backtest:
    st.markdown("### Backtested Results")
    backtests = load_backtest_artifacts()

    if backtests is None:
        st.info(
            "Backtest artifacts not found. Run `python updater/run_backtest_systems.py` "
            "after the daily update to generate them."
        )
    else:
        stats = backtests["stats"]
        if stats is not None and not stats.empty:
            st.subheader("Summary Stats")
            st.dataframe(_streamlit_safe_frame(stats), use_container_width=True)

        st.subheader("Monthly Summaries")
        _render_monthly_summary(backtests["equity_s1"], "System 1 (Weekly)")
        _render_monthly_summary(backtests["equity_s2"], "System 2 (Fib)")
        _render_monthly_summary(backtests["equity_s3"], "System 3 (Momentum)")
        _render_monthly_summary(backtests["equity_combined"], "Combined")

        st.subheader("Recent Trades")
        trades = backtests["trades"]
        if trades.empty:
            st.info("No trades found in the backtest output.")
        else:
            portfolio_options = sorted(trades["portfolio"].dropna().unique())
            selected_portfolios = st.multiselect(
                "Portfolios",
                options=portfolio_options,
                default=portfolio_options,
                key="backtest-trades-portfolios",
            )
            view = trades.copy()
            if selected_portfolios:
                view = view[view["portfolio"].isin(selected_portfolios)]
            view = view.sort_values("date", ascending=False).head(200)
            st.dataframe(_streamlit_safe_frame(view), use_container_width=True)

        st.subheader("Open Positions (End of Backtest)")
        positions = backtests["positions"]
        if positions.empty:
            st.info("No open positions at the end of the backtest.")
        else:
            st.dataframe(_streamlit_safe_frame(positions), use_container_width=True)
