import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Momentum — Bucket C", layout="wide")

@st.cache_data
def load_artifacts():
    df = pd.read_parquet("artifacts/today_C.parquet")
    meta_path = Path("artifacts/metadata.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return df, meta

st.title("Momentum Strategy — Bucket C")

df, meta = load_artifacts()

if meta:
    st.caption(f"As of: {meta.get('as_of', '-')}")
    w = meta.get("weights", {})
    if w:
        st.caption(f"Weights: A={w.get('A', '-')}, B={w.get('B', '-')}")

st.dataframe(df, use_container_width=True)
