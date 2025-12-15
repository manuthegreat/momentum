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

# =====================================================
# HELPERS
# =====================================================

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def format_percent(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)
    return df


# =====================================================
# LOAD DATA
# =====================================================

todayA = load_parquet(TODAY_A)
todayB = load_parquet(TODAY_B)
todayC = load_parquet(TODAY_C)

historyA = load_parquet(HISTORY_A)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Momentum Dashboard")

view = st.sidebar.radio(
    "Select View",
    [
        "📈 Overview",
        "🅰️ Bucket A — Absolute Momentum",
        "🅱️ Bucket B — Relative Momentum",
        "🅲 Bucket C — Unified Portfolio",
        "📊 Backtest Results",
    ],
)

# =====================================================
# OVERVIEW
# =====================================================

if view == "📈 Overview":

    st.title("📈 Momentum Strategy — Daily Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Bucket A Names", len(todayA))

    with col2:
        st.metric("Bucket B Names", len(todayB))

    with col3:
        st.metric("Unified Portfolio Names", len(todayC))

    st.divider()

    st.subheader("🅲 Today's Unified Portfolio (80 / 20 Allocation)")
    if todayC.empty:
        st.info("No signals generated today.")
    else:
        st.dataframe(
            todayC.sort_values("Position_Size", ascending=False),
            use_container_width=True,
        )

# =====================================================
# BUCKET A
# =====================================================

elif view == "🅰️ Bucket A — Absolute Momentum":

    st.title("🅰️ Bucket A — Absolute Momentum")

    if todayA.empty:
        st.warning("No signals available.")
    else:
        df = todayA.copy()
        df = format_percent(
            df,
            ["Momentum_Score", "Early_Momentum_Score", "Consistency", "Weighted_Score"],
        )

        st.dataframe(
            df.sort_values("Weighted_Score", ascending=False),
            use_container_width=True,
        )

# =====================================================
# BUCKET B
# =====================================================

elif view == "🅱️ Bucket B — Relative Momentum":

    st.title("🅱️ Bucket B — Relative Momentum")

    if todayB.empty:
        st.warning("No signals available.")
    else:
        df = todayB.copy()
        df = format_percent(
            df,
            ["Momentum_Score", "Early_Momentum_Score", "Consistency", "Weighted_Score"],
        )

        st.dataframe(
            df.sort_values("Weighted_Score", ascending=False),
            use_container_width=True,
        )

# =====================================================
# BUCKET C
# =====================================================

elif view == "🅲 Bucket C — Unified Portfolio":

    st.title("🅲 Bucket C — Unified Portfolio")

    if todayC.empty:
        st.warning("No unified portfolio available.")
    else:
        st.subheader("Target Allocations")

        st.dataframe(
            todayC.sort_values("Position_Size", ascending=False),
            use_container_width=True,
        )

        total = todayC["Position_Size"].sum()
        st.metric("Total Capital Allocated", f"${total:,.0f}")

# =====================================================
# BACKTEST
# =====================================================

elif view == "📊 Backtest Results":

    st.title("📊 Backtest — Bucket A")

    if historyA.empty or len(historyA) < 2:
        st.warning("Backtest history not available.")
    else:
        hist = historyA.copy()
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.sort_values("Date")

        col1, col2, col3 = st.columns(3)

        start = hist["Portfolio Value"].iloc[0]
        end = hist["Portfolio Value"].iloc[-1]
        total_return = (end / start - 1) * 100

        col1.metric("Start Value", f"${start:,.0f}")
        col2.metric("End Value", f"${end:,.0f}")
        col3.metric("Total Return", f"{total_return:.2f}%")

        st.divider()

        st.subheader("Equity Curve")
        st.line_chart(
            hist.set_index("Date")["Portfolio Value"],
            use_container_width=True,
        )

        st.subheader("Backtest Data")
        st.dataframe(hist, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================

st.divider()
st.caption(
    "Momentum Strategy Dashboard • Deterministic pipeline • GitHub Actions powered"
)
