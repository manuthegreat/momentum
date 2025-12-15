import streamlit as st
import pandas as pd
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Momentum Strategy Dashboard",
    layout="wide",
)

ARTIFACTS = Path("artifacts")

TODAY_A = ARTIFACTS / "today_A.parquet"
TODAY_B = ARTIFACTS / "today_B.parquet"
TODAY_C = ARTIFACTS / "today_C.parquet"

HISTORY_A = ARTIFACTS / "history_A.parquet"
HISTORY_B = ARTIFACTS / "history_B.parquet"
HISTORY_C = ARTIFACTS / "history_C.parquet"

# =====================================================
# HELPERS
# =====================================================

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def perf_stats(hist: pd.DataFrame) -> dict:
    if hist.empty or len(hist) < 2:
        return {}

    start = hist["Portfolio Value"].iloc[0]
    end = hist["Portfolio Value"].iloc[-1]

    ret = hist["Portfolio Value"].pct_change().dropna()

    return {
        "Start ($)": f"{start:,.0f}",
        "End ($)": f"{end:,.0f}",
        "Total Return (%)": f"{(end / start - 1) * 100:.2f}",
        "Sharpe": f"{(ret.mean() / ret.std() * (252 ** 0.5)):.2f}" if ret.std() else "—",
        "Max DD (%)": f"{((hist['Portfolio Value'] / hist['Portfolio Value'].cummax() - 1).min() * 100):.2f}",
    }

# =====================================================
# LOAD DATA
# =====================================================

todayA = load_parquet(TODAY_A)
todayB = load_parquet(TODAY_B)
todayC = load_parquet(TODAY_C)

historyA = load_parquet(HISTORY_A)
historyB = load_parquet(HISTORY_B)
historyC = load_parquet(HISTORY_C)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Momentum Dashboard")

section = st.sidebar.radio(
    "View",
    ["Overview", "Signals", "Backtests"],
)

# =====================================================
# OVERVIEW
# =====================================================

if section == "Overview":

    st.title("Momentum Strategy — Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Bucket A", len(todayA))
    c2.metric("Bucket B", len(todayB))
    c3.metric("Bucket C", len(todayC))

    st.subheader("Unified Portfolio (Bucket C)")
    st.dataframe(todayC, use_container_width=True)

# =====================================================
# SIGNALS
# =====================================================

elif section == "Signals":

    bucket = st.selectbox("Select Bucket", ["A", "B", "C"])

    st.subheader(f"Today's Signals — Bucket {bucket}")

    if bucket == "A":
        st.dataframe(todayA, use_container_width=True)
    elif bucket == "B":
        st.dataframe(todayB, use_container_width=True)
    else:
        st.dataframe(todayC, use_container_width=True)

# =====================================================
# BACKTESTS
# =====================================================

elif section == "Backtests":

    bucket = st.selectbox("Select Backtest", ["A", "B", "C"])

    if bucket == "A":
        hist = historyA
    elif bucket == "B":
        hist = historyB
    else:
        hist = historyC

    if hist.empty:
        st.warning("Backtest not available.")
    else:
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.sort_values("Date")

        st.subheader(f"Bucket {bucket} — Performance")

        stats = perf_stats(hist)
        cols = st.columns(len(stats))
        for c, (k, v) in zip(cols, stats.items()):
            c.metric(k, v)

        st.line_chart(hist.set_index("Date")["Portfolio Value"], use_container_width=True)
        st.dataframe(hist, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================

st.caption("Momentum system • Deterministic • Artifact-driven • Production-grade")
