import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("📈 Momentum Strategy Dashboard")

ARTIFACTS = Path("artifacts")

# ---------------------------------------------------------
# Load helpers
# ---------------------------------------------------------
def load_csv(name):
    path = ARTIFACTS / name
    if not path.exists():
        st.warning(f"Missing artifact: {name}")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["Date"])

signals = load_csv("signals.csv")
equity  = load_csv("backtest_equity.csv")
stats   = load_csv("backtest_stats.csv")

# ---------------------------------------------------------
# Normalize signals (Position_Size)
# ---------------------------------------------------------
TOTAL_CAPITAL = 100_000
BUCKET_WEIGHTS = {"A": 0.2, "B": 0.8, "C": 1.0}

if not signals.empty:
    if "Position_Size" not in signals.columns:
        def infer_position_size(df):
            out = df.copy()
            for b, w in BUCKET_WEIGHTS.items():
                mask = out["Bucket"] == b
                n = mask.sum()
                if n > 0:
                    out.loc[mask, "Position_Size"] = TOTAL_CAPITAL * w / n
            return out

        signals = infer_position_size(signals)

# ---------------------------------------------------------
# TODAY VIEW
# ---------------------------------------------------------
st.header("🟢 Today’s Signals")

if signals.empty:
    st.info("No signals available")
else:
    latest_date = signals["Date"].max()
    today = signals[signals["Date"] == latest_date]

    for bucket in ["A", "B", "C"]:
        df = today[today["Bucket"] == bucket]
        if df.empty:
            continue

        st.subheader(f"Bucket {bucket}")
        st.dataframe(
            df.sort_values("Position_Size", ascending=False)
            .reset_index(drop=True)
        )

# ---------------------------------------------------------
# BACKTEST RESULTS
# ---------------------------------------------------------
st.header("📊 Backtest Results")

if equity.empty:
    st.info("No backtest equity available")
else:
    if "Equity" in equity.columns:
        y_col = "Equity"
    elif "Portfolio Value" in equity.columns:
        y_col = "Portfolio Value"
    else:
        st.error("Unknown equity schema")
        st.stop()

    st.subheader("Equity Curve")
    st.line_chart(equity.set_index("Date")[y_col])

# ---------------------------------------------------------
# PERFORMANCE STATS
# ---------------------------------------------------------
st.header("📈 Performance Summary")

if stats.empty:
    st.info("No performance stats available")
else:
    st.dataframe(stats)
