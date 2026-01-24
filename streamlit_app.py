# streamlit_app.py
# UI-ONLY VERSION:
# - Loads precomputed signal parquets produced by update_parquet.py
# - Displays signals from all three systems + combined action list
#
# IMPORTANT: No strategy calculations happen here.

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = "artifacts"

WEEKLY_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "weekly_swing_signals.parquet")
FIB_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "fib_signals.parquet")
MOMENTUM_SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "momentum_bucketc_signals.parquet")
ACTION_LIST_PATH = os.path.join(ARTIFACTS_DIR, "action_list.parquet")

st.set_page_config(page_title="Momentum Signals Dashboard", layout="wide")


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

    weekly = pd.read_parquet(WEEKLY_SIGNALS_PATH)
    fib = pd.read_parquet(FIB_SIGNALS_PATH)
    mom = pd.read_parquet(MOMENTUM_SIGNALS_PATH)
    action_list = pd.read_parquet(ACTION_LIST_PATH)

    if "signal_date" in weekly.columns:
        weekly["signal_date"] = pd.to_datetime(weekly["signal_date"])
    if "Signal_Date" in mom.columns:
        mom["Signal_Date"] = pd.to_datetime(mom["Signal_Date"])
    if "mom_date" in action_list.columns:
        action_list["mom_date"] = pd.to_datetime(action_list["mom_date"])
    if "weekly_date" in action_list.columns:
        action_list["weekly_date"] = pd.to_datetime(action_list["weekly_date"])

    return weekly, fib, mom, action_list


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


tab_weekly, tab_fib, tab_mom, tab_action = st.tabs(
    ["Weekly Swing", "Fibonacci", "Momentum Bucket C", "Action List"]
)

with tab_weekly:
    st.markdown("### Weekly Swing Signals")

    if weekly_sig.empty:
        st.info("No weekly swing signals found in the latest batch.")
    else:
        dates = sorted(weekly_sig["signal_date"].dt.date.unique())
        selected = st.selectbox("Signal date", options=dates, index=len(dates) - 1)
        view = weekly_sig[weekly_sig["signal_date"].dt.date == selected].copy()

        st.caption(f"Signals for {selected}")
        st.dataframe(view, use_container_width=True)

with tab_fib:
    st.markdown("### Fibonacci Confirmation Signals")

    if fib_sig.empty:
        st.info("No Fibonacci signals found in the latest batch.")
    else:
        signal_options = ["BUY", "WATCH", "INVALID"]
        selected_signals = st.multiselect(
            "Signal filter",
            options=signal_options,
            default=["BUY", "WATCH"],
        )
        view = fib_sig[fib_sig["Signal"].isin(selected_signals)].copy()
        st.dataframe(view, use_container_width=True)

with tab_mom:
    st.markdown("### Momentum Bucket C (Latest Candidates)")

    if mom_sig.empty:
        st.info("No momentum bucket C candidates found in the latest batch.")
    else:
        signal_date = pd.to_datetime(mom_sig["Signal_Date"].max()).date()
        st.caption(f"Signals as of {signal_date}")

        pct_cfg = {
            "Weight_%": st.column_config.NumberColumn("Weight %", format="%.2f"),
            "Signal_Confidence": st.column_config.ProgressColumn("Signal Strength", min_value=0, max_value=100, format="%d%%"),
        }
        st.dataframe(mom_sig, use_container_width=True, column_config=pct_cfg)

with tab_action:
    st.markdown("### Action List (Best Signal Per Ticker)")

    if action_list.empty:
        st.info("No action list generated in the latest batch.")
    else:
        pct_cfg = {
            "mom_conf": st.column_config.ProgressColumn("Momentum Confidence", min_value=0, max_value=100, format="%d%%"),
        }
        st.dataframe(action_list, use_container_width=True, column_config=pct_cfg)
