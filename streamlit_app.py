import pandas as pd
import streamlit as st
import requests 

@st.cache_data(ttl=300)
def load_today():
    return pd.read_parquet("artifacts/today_C.parquet")

st.title("Momentum — Bucket C (80/20)")
df = load_today()

st.dataframe(df, use_container_width=True)

def trigger_backtest():
    url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/actions/workflows/run_backtest.yml/dispatches"

    headers = {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "ref": "main"
    }

    r = requests.post(url, headers=headers, json=payload)

    return r.status_code, r.text

st.sidebar.markdown("## 🔄 Backtest Controls")

if st.sidebar.button("Run Full Backtest"):
    status, msg = trigger_backtest()

    if status == 204:
        st.sidebar.success("Backtest triggered successfully 🚀")
        st.sidebar.info("Results will update in ~1–2 minutes.")
    else:
        st.sidebar.error("Failed to trigger backtest")
        st.sidebar.code(msg)
