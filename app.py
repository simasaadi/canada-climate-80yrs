import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Canada Climate — Smoke Test", layout="wide")

REPO_DIR = Path(__file__).resolve().parent
RAW_DIR = REPO_DIR / "data" / "raw"
PROCESSED_DIR = REPO_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TIDY_PATH = PROCESSED_DIR / "climate_daily_tidy.parquet"
MONTHLY_PATH = PROCESSED_DIR / "climate_monthly.parquet"
YEARLY_PATH = PROCESSED_DIR / "climate_yearly.parquet"

st.title("Streamlit Smoke Test")
st.write("Repo:", str(REPO_DIR))
st.write("RAW exists:", RAW_DIR.exists(), str(RAW_DIR))
st.write("PROCESSED exists:", PROCESSED_DIR.exists(), str(PROCESSED_DIR))

st.write("TIDY exists:", TIDY_PATH.exists(), str(TIDY_PATH))
st.write("MONTHLY exists:", MONTHLY_PATH.exists(), str(MONTHLY_PATH))
st.write("YEARLY exists:", YEARLY_PATH.exists(), str(YEARLY_PATH))

if TIDY_PATH.exists():
    df = pd.read_parquet(TIDY_PATH)
    st.success(f"Loaded tidy parquet: {df.shape}")
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.warning("TIDY parquet not found. Run Notebook 02/03/04/05 build cell once, then rerun Streamlit.")

