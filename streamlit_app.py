import pandas as pd
import streamlit as st

@st.cache_data(ttl=300)
def load_today():
    return pd.read_parquet("artifacts/today_C.parquet")

st.title("Momentum — Bucket C (80/20)")
df = load_today()

st.dataframe(df, use_container_width=True)
