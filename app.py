# app.py — Canada Climate Dashboard (1940–Present)
# Streamlit app designed to run on Streamlit Community Cloud with minimal dependencies.
# Uses: streamlit, pandas, numpy, matplotlib, pydeck (bundled with Streamlit)

import os
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk


# -----------------------------
# App config + styling
# -----------------------------
st.set_page_config(
    page_title="Canada Climate Dashboard",
    page_icon="🇨🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Layout tightening */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Sidebar headings */
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { margin-top: 0.6rem; }

/* Card-like metric boxes */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 12px 12px 10px 12px;
    border-radius: 14px;
}

/* Make tables feel cleaner */
thead tr th { background: rgba(255,255,255,0.03) !important; }

/* Small help text tone */
.small-note { color: rgba(255,255,255,0.65); font-size: 0.9rem; }

/* Section titles */
.section-title { font-size: 1.2rem; font-weight: 700; margin: 0.4rem 0 0.8rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Paths + city geocoding
# -----------------------------
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_CANDIDATES = [
    RAW_DIR / "canadian_climate_daily.csv",
    RAW_DIR / "canadian_climate_daily.csv",  # keep as fallback if naming changes
]

TIDY_PATH = PROCESSED_DIR / "climate_daily_tidy.parquet"
MONTHLY_PATH = PROCESSED_DIR / "climate_monthly.parquet"
YEARLY_PATH = PROCESSED_DIR / "climate_yearly.parquet"

CITY_COORDS = {
    "CALGARY":    (51.0447, -114.0719),
    "EDMONTON":   (53.5461, -113.4938),
    "HALIFAX":    (44.6488,  -63.5752),
    "MONCTON":    (46.0878,  -64.7782),
    "MONTREAL":   (45.5019,  -73.5674),
    "OTTAWA":     (45.4215,  -75.6972),
    "QUEBEC":     (46.8139,  -71.2080),   # Québec City
    "SASKATOON":  (52.1332, -106.6700),
    "STJOHNS":    (47.5615,  -52.7126),   # St. John's
    "TORONTO":    (43.6532,  -79.3832),
    "VANCOUVER":  (49.2827, -123.1207),
    "WHITEHORSE": (60.7212, -135.0568),
    "WINNIPEG":   (49.8951,  -97.1384),
}

# -----------------------------
# Utility functions
# -----------------------------
def _find_raw_csv() -> Path:
    for p in RAW_CANDIDATES:
        if p.exists():
            return p
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No raw CSV found in {RAW_DIR}. Expected one of: {RAW_CANDIDATES}")

def _standardize_city(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)

def _ols_slope_per_decade(year: np.ndarray, y: np.ndarray) -> float:
    # simple OLS slope (y ~ a + b*year), return b*10
    year = np.asarray(year, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(year) & np.isfinite(y)
    year = year[m]
    y = y[m]
    if year.size < 10:
        return np.nan
    x = year - year.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return np.nan
    b = np.sum(x * (y - y.mean())) / denom
    return float(b * 10.0)

def _diverging_color(val: float, vmin: float, vmax: float) -> list:
    # returns [r,g,b,alpha] for pydeck, simple diverging around 0
    if not np.isfinite(val):
        return [160, 160, 160, 140]
    if vmax <= 0 and vmin <= 0:
        t = 0.0 if vmin == vmax else (val - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0, 1))
        return [40, int(80 + 120 * t), int(120 + 110 * t), 200]
    if vmin >= 0 and vmax >= 0:
        t = 0.0 if vmin == vmax else (val - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0, 1))
        return [int(120 + 110 * t), int(80 + 90 * (1 - t)), 40, 200]

    # diverging around zero: negative -> blue, positive -> red
    max_abs = max(abs(vmin), abs(vmax), 1e-9)
    t = float(np.clip(val / max_abs, -1, 1))
    if t < 0:
        u = abs(t)
        return [60, int(110 + 80 * (1 - u)), int(180 + 50 * u), 210]
    else:
        u = t
        return [int(180 + 50 * u), int(110 + 80 * (1 - u)), 60, 210]

def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception:
        # On Streamlit Cloud, filesystem is writable but not guaranteed. Failing to write is OK.
        pass

# -----------------------------
# Build processed datasets (if missing)
# -----------------------------
def build_processed(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(raw_path)

    # Date
    if "LOCAL_DATE" not in df.columns:
        raise ValueError("Expected column LOCAL_DATE in raw CSV.")
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"], errors="coerce")
    if df["LOCAL_DATE"].isna().any():
        bad = int(df["LOCAL_DATE"].isna().sum())
        raise ValueError(f"LOCAL_DATE parsing failed for {bad} rows.")

    # Wide -> tidy (expects MEAN_TEMPERATURE_CITY, TOTAL_PRECIPITATION_CITY)
    pat = re.compile(r"^(MEAN_TEMPERATURE|TOTAL_PRECIPITATION)_(.+)$")
    value_cols = [c for c in df.columns if c != "LOCAL_DATE" and pat.match(c)]
    if not value_cols:
        raise ValueError("No metric columns found. Expected patterns like MEAN_TEMPERATURE_CITYNAME.")

    long = df.melt(
        id_vars=["LOCAL_DATE"],
        value_vars=value_cols,
        var_name="series",
        value_name="value",
    )
    long[["metric", "city"]] = long["series"].str.extract(r"^(MEAN_TEMPERATURE|TOTAL_PRECIPITATION)_(.+)$")
    long = long.drop(columns=["series"])

    tidy = (
        long.pivot_table(
            index=["LOCAL_DATE", "city"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "LOCAL_DATE": "local_date",
                "MEAN_TEMPERATURE": "mean_temperature",
                "TOTAL_PRECIPITATION": "total_precipitation",
            }
        )
    )

    tidy["city"] = _standardize_city(tidy["city"])
    tidy["local_date"] = pd.to_datetime(tidy["local_date"])
    tidy["mean_temperature"] = pd.to_numeric(tidy["mean_temperature"], errors="coerce")
    tidy["total_precipitation"] = pd.to_numeric(tidy["total_precipitation"], errors="coerce")

    tidy["year"] = tidy["local_date"].dt.year
    tidy["month"] = tidy["local_date"].dt.month
    tidy["dayofyear"] = tidy["local_date"].dt.dayofyear

    monthly = (
        tidy.groupby(["city", "year", "month"], as_index=False)
        .agg(
            mean_temperature=("mean_temperature", "mean"),
            total_precipitation=("total_precipitation", "sum"),
            n_obs_temp=("mean_temperature", "count"),
            n_obs_precip=("total_precipitation", "count"),
        )
    )

    yearly = (
        tidy.groupby(["city", "year"], as_index=False)
        .agg(
            mean_temperature=("mean_temperature", "mean"),
            total_precipitation=("total_precipitation", "sum"),
            n_obs_temp=("mean_temperature", "count"),
            n_obs_precip=("total_precipitation", "count"),
        )
    )

    _safe_write_parquet(tidy, TIDY_PATH)
    _safe_write_parquet(monthly, MONTHLY_PATH)
    _safe_write_parquet(yearly, YEARLY_PATH)

    return tidy, monthly, yearly

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    status = {
        "repo_dir": str(REPO_DIR),
        "raw_dir": str(RAW_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "raw_path": None,
        "used_parquet": False,
        "built_processed": False,
        "tidy_path": str(TIDY_PATH),
        "monthly_path": str(MONTHLY_PATH),
        "yearly_path": str(YEARLY_PATH),
    }

    raw_path = _find_raw_csv()
    status["raw_path"] = str(raw_path)

    if TIDY_PATH.exists() and MONTHLY_PATH.exists() and YEARLY_PATH.exists():
        tidy = pd.read_parquet(TIDY_PATH)
        monthly = pd.read_parquet(MONTHLY_PATH)
        yearly = pd.read_parquet(YEARLY_PATH)
        status["used_parquet"] = True
        return tidy, monthly, yearly, status

    tidy, monthly, yearly = build_processed(raw_path)
    status["built_processed"] = True
    return tidy, monthly, yearly, status

# -----------------------------
# Derived analytics
# -----------------------------
def compute_annual_metrics(df_daily: pd.DataFrame) -> pd.DataFrame:
    d = df_daily.copy()
    d["city"] = _standardize_city(d["city"])
    d["local_date"] = pd.to_datetime(d["local_date"])
    d["year"] = d["local_date"].dt.year

    annual = (
        d.groupby(["city", "year"], as_index=False)
        .agg(
            mean_temp=("mean_temperature", "mean"),
            total_precip=("total_precipitation", "sum"),
            n_temp=("mean_temperature", "count"),
            n_precip=("total_precipitation", "count"),
            wet_days=("total_precipitation", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
    )
    return annual

def compute_extremes(df_daily: pd.DataFrame, baseline: tuple[int, int], min_days_per_year: int) -> pd.DataFrame:
    d = df_daily.copy()
    d["city"] = _standardize_city(d["city"])
    d["local_date"] = pd.to_datetime(d["local_date"])
    d["year"] = d["local_date"].dt.year

    base = d[(d["year"] >= baseline[0]) & (d["year"] <= baseline[1])].copy()
    thresholds = (
        base.groupby("city")["mean_temperature"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(columns={0.05: "t_p05", 0.95: "t_p95"})
        .reset_index()
    )

    d2 = d.merge(thresholds, on="city", how="left")
    d2["cold_extreme"] = d2["mean_temperature"] <= d2["t_p05"]
    d2["hot_extreme"] = d2["mean_temperature"] >= d2["t_p95"]

    ex = (
        d2.groupby(["city", "year"], as_index=False)
        .agg(
            cold_days=("cold_extreme", "sum"),
            hot_days=("hot_extreme", "sum"),
            n_temp=("mean_temperature", "count"),
        )
    )
    ex["temp_ok"] = ex["n_temp"] >= int(min_days_per_year)
    return ex

def attach_coords(df_city: pd.DataFrame) -> pd.DataFrame:
    out = df_city.copy()
    out["city"] = _standardize_city(out["city"])
    out["lat"] = out["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
    out["lon"] = out["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])
    return out

def compute_map_layers(
    annual: pd.DataFrame,
    extremes: pd.DataFrame,
    year_range: tuple[int, int],
    min_days_per_year: int,
    baseline: tuple[int, int],
    recent: tuple[int, int],
) -> pd.DataFrame:
    a = annual.copy()
    a = a[(a["year"] >= year_range[0]) & (a["year"] <= year_range[1])].copy()
    a["temp_ok"] = a["n_temp"] >= int(min_days_per_year)

    # Trend per city (using temp_ok only)
    rows = []
    for city, g in a[a["temp_ok"]].groupby("city"):
        g = g.sort_values("year")
        slope_dec = _ols_slope_per_decade(g["year"].to_numpy(), g["mean_temp"].to_numpy())
        rows.append({"city": city, "warming_trend_decade": slope_dec})
    trend = pd.DataFrame(rows)

    # Extreme frequency change: baseline vs recent (avg days/year)
    e = extremes.copy()
    e = e[(e["year"] >= year_range[0]) & (e["year"] <= year_range[1])].copy()
    e_ok = e[e["temp_ok"]].copy()

    def _period_mean(df, y0, y1, col):
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        return sub.groupby("city", as_index=False)[col].mean().rename(columns={col: f"{col}_mean_{y0}_{y1}"})

    hot_b = _period_mean(e_ok, baseline[0], baseline[1], "hot_days")
    hot_r = _period_mean(e_ok, recent[0], recent[1], "hot_days")
    cold_b = _period_mean(e_ok, baseline[0], baseline[1], "cold_days")
    cold_r = _period_mean(e_ok, recent[0], recent[1], "cold_days")

    delta = hot_b.merge(hot_r, on="city", how="outer").merge(cold_b, on="city", how="outer").merge(cold_r, on="city", how="outer")
    delta["delta_hot_days"] = delta[f"hot_days_mean_{recent[0]}_{recent[1]}"] - delta[f"hot_days_mean_{baseline[0]}_{baseline[1]}"]
    delta["delta_cold_days"] = delta[f"cold_days_mean_{recent[0]}_{recent[1]}"] - delta[f"cold_days_mean_{baseline[0]}_{baseline[1]}"]

    out = trend.merge(delta[["city", "delta_hot_days", "delta_cold_days"]], on="city", how="outer")
    out = attach_coords(out)
    return out

# -----------------------------
# Map rendering (pydeck)
# -----------------------------
def make_pydeck_map(df_points: pd.DataFrame, value_col: str, title: str, selected_cities: list[str] | None):
    d = df_points.dropna(subset=["lat", "lon"]).copy()
    d["city"] = _standardize_city(d["city"])
    if selected_cities:
        sel = [_standardize_city(pd.Series(selected_cities)).iloc[i] for i in range(len(selected_cities))]
        d_view = d[d["city"].isin(sel)].copy()
    else:
        d_view = d.copy()

    # fallback: show everything if filter empties
    if d_view.empty:
        d_view = d.copy()

    vals = d_view[value_col].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin = float(np.min(finite)) if finite.size else -1.0
    vmax = float(np.max(finite)) if finite.size else 1.0

    # marker size: emphasize magnitude, but keep readable
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    d_view["radius"] = 22000 * (0.35 + 0.65 * (np.abs(d_view[value_col].fillna(0.0)) / (max_abs + 1e-9)))
    d_view["color"] = d_view[value_col].apply(lambda v: _diverging_color(float(v) if pd.notna(v) else np.nan, vmin, vmax))

    # view state (auto-fit-ish)
    center_lat = float(d_view["lat"].mean())
    center_lon = float(d_view["lon"].mean())
    # rough zoom heuristic
    lon_span = float(d_view["lon"].max() - d_view["lon"].min()) if len(d_view) > 1 else 6.0
    lat_span = float(d_view["lat"].max() - d_view["lat"].min()) if len(d_view) > 1 else 4.0
    span = max(lon_span, lat_span, 2.0)
    zoom = float(np.clip(3.8 - math.log(span + 1e-9, 2), 2.2, 6.5))

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0, bearing=0)

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=d_view,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        get_line_color=[20, 20, 20, 200],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    labels = pdk.Layer(
        "TextLayer",
        data=d_view,
        get_position=["lon", "lat"],
        get_text="city",
        get_size=14,
        get_color=[255, 255, 255, 220],
        get_angle=0,
        get_text_anchor="start",
        get_alignment_baseline="bottom",
        pickable=False,
    )

    tooltip = {
        "html": f"<b>{{city}}</b><br/>{title}: <b>{{{value_col}}}</b>",
        "style": {"backgroundColor": "rgba(20,20,20,0.92)", "color": "white"},
    }

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=view_state,
        layers=[scatter, labels],
        tooltip=tooltip,
    )
    return deck

# -----------------------------
# Load data
# -----------------------------
df_daily, df_monthly, df_yearly, status = load_data()

# defensive typing
df_daily = df_daily.copy()
df_daily["city"] = _standardize_city(df_daily["city"])
df_daily["local_date"] = pd.to_datetime(df_daily["local_date"])

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Canada Climate Dashboard")
st.sidebar.caption("1940–Present • City-level daily → monthly/yearly rollups")

with st.sidebar.expander("Data status", expanded=False):
    st.write("Repo:", status["repo_dir"])
    st.write("Raw:", status["raw_path"])
    st.write("Processed dir:", status["processed_dir"])
    st.write("Loaded parquet:", status["used_parquet"])
    st.write("Built processed:", status["built_processed"])
    st.write("Tidy parquet exists:", TIDY_PATH.exists())
    st.write("Monthly parquet exists:", MONTHLY_PATH.exists())
    st.write("Yearly parquet exists:", YEARLY_PATH.exists())

st.sidebar.markdown("### Navigate")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Trends", "Extremes", "Maps", "Download"],
    index=0,
    label_visibility="collapsed",
)

all_cities = sorted(df_daily["city"].unique().tolist())
default_cities = all_cities  # important: show many points by default

selected_cities = st.sidebar.multiselect(
    "Cities",
    options=all_cities,
    default=default_cities,
)

min_year = int(df_daily["local_date"].dt.year.min())
max_year = int(df_daily["local_date"].dt.year.max())
year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)

min_days_per_year = st.sidebar.slider("Coverage threshold (days/year)", min_value=200, max_value=365, value=300, step=5)

baseline_label = st.sidebar.selectbox("Baseline period", ["1961–1990", "1951–1980", "1971–2000"], index=0)
baseline_map = {"1961–1990": (1961, 1990), "1951–1980": (1951, 1980), "1971–2000": (1971, 2000)}
BASELINE = baseline_map[baseline_label]
RECENT = (1991, 2020)

# filter daily for pages
d = df_daily.copy()
d = d[(d["local_date"].dt.year >= year_range[0]) & (d["local_date"].dt.year <= year_range[1])]
if selected_cities:
    d = d[d["city"].isin([c.upper().strip() for c in selected_cities])]

annual = compute_annual_metrics(d)
extremes = compute_extremes(d, baseline=BASELINE, min_days_per_year=min_days_per_year)

# -----------------------------
# Top header
# -----------------------------
st.markdown("## Canada Climate (1940–Present)")
st.markdown('<div class="small-note">Daily mean temperature and total precipitation, aggregated to annual metrics with coverage checks.</div>', unsafe_allow_html=True)

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    left, mid, right = st.columns(3)

    n_cities = int(d["city"].nunique())
    n_rows = int(len(d))
    date_min = d["local_date"].min().date() if len(d) else None
    date_max = d["local_date"].max().date() if len(d) else None

    with left:
        st.metric("Cities in view", f"{n_cities}")
    with mid:
        st.metric("Daily rows in view", f"{n_rows:,}")
    with right:
        st.metric("Date range", f"{date_min} → {date_max}")

    st.markdown('<div class="section-title">Quick health checks</div>', unsafe_allow_html=True)

    city_summary = (
        d.groupby("city", as_index=False)
        .agg(
            start=("local_date", "min"),
            end=("local_date", "max"),
            n_days=("local_date", "count"),
            missing_temp=("mean_temperature", lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
            missing_precip=("total_precipitation", lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
        )
    )
    city_summary["temp_missing_pct"] = (city_summary["missing_temp"] / city_summary["n_days"] * 100).round(2)
    city_summary["precip_missing_pct"] = (city_summary["missing_precip"] / city_summary["n_days"] * 100).round(2)

    st.dataframe(city_summary.sort_values(["temp_missing_pct", "precip_missing_pct"], ascending=False), use_container_width=True)

    st.markdown('<div class="section-title">Annual snapshot (mean temp + total precip)</div>', unsafe_allow_html=True)

    # small multiple-like: choose one city in overview for detail
    focus_city = st.selectbox("Focus city", options=sorted(d["city"].unique().tolist()), index=0 if n_cities else None)

    if focus_city and not annual.empty:
        sub = annual[annual["city"] == focus_city].sort_values("year").copy()
        sub["temp_ok"] = sub["n_temp"] >= int(min_days_per_year)

        c1, c2 = st.columns(2)

        with c1:
            fig = plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.plot(sub["year"], sub["mean_temp"])
            ax.set_title(f"{focus_city} — Annual Mean Temperature")
            ax.set_xlabel("Year")
            ax.set_ylabel("°C")
            st.pyplot(fig, clear_figure=True)

        with c2:
            fig = plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.plot(sub["year"], sub["total_precip"])
            ax.set_title(f"{focus_city} — Annual Total Precipitation")
            ax.set_xlabel("Year")
            ax.set_ylabel("mm")
            st.pyplot(fig, clear_figure=True)

# -----------------------------
# Trends
# -----------------------------
elif page == "Trends":
    st.markdown('<div class="section-title">Temperature trends (coverage-aware)</div>', unsafe_allow_html=True)

    annual2 = annual.copy()
    annual2["temp_ok"] = annual2["n_temp"] >= int(min_days_per_year)

    # plot all selected cities on one axis (readable when <= ~10)
    cities_in_view = sorted(annual2["city"].unique().tolist())
    if not cities_in_view:
        st.warning("No data in view for the chosen filters.")
    else:
        fig = plt.figure(figsize=(12, 5))
        ax = plt.gca()
        for c in cities_in_view:
            s = annual2[(annual2["city"] == c) & (annual2["temp_ok"])].sort_values("year")
            if len(s) >= 2:
                ax.plot(s["year"], s["mean_temp"], linewidth=1, label=c)
        ax.set_title("Annual Mean Temperature (temp-eligible years only)")
        ax.set_xlabel("Year")
        ax.set_ylabel("°C")
        if len(cities_in_view) <= 12:
            ax.legend(ncol=3, fontsize=8)
        st.pyplot(fig, clear_figure=True)

        # Trend table
        rows = []
        for c in cities_in_view:
            s = annual2[(annual2["city"] == c) & (annual2["temp_ok"])].sort_values("year")
            slope_dec = _ols_slope_per_decade(s["year"].to_numpy(), s["mean_temp"].to_numpy())
            rows.append({"city": c, "warming_trend_°C_per_decade": slope_dec, "n_years_used": int(len(s))})
        trends = pd.DataFrame(rows).sort_values("warming_trend_°C_per_decade", ascending=False)

        st.markdown('<div class="section-title">OLS warming slope (°C/decade)</div>', unsafe_allow_html=True)
        st.dataframe(trends, use_container_width=True)

# -----------------------------
# Extremes
# -----------------------------
elif page == "Extremes":
    st.markdown('<div class="section-title">Temperature extremes (baseline quantiles)</div>', unsafe_allow_html=True)
    st.caption(f"Cold extreme = ≤ p05 of {BASELINE[0]}–{BASELINE[1]}; Hot extreme = ≥ p95 of {BASELINE[0]}–{BASELINE[1]}. Only years meeting coverage threshold are included.")

    ex = extremes[extremes["temp_ok"]].copy()
    if ex.empty:
        st.warning("No extreme indices available under current filters (coverage threshold may be too strict).")
    else:
        cities_in_view = sorted(ex["city"].unique().tolist())

        # Hot extremes plot
        fig = plt.figure(figsize=(12, 5))
        ax = plt.gca()
        for c in cities_in_view:
            s = ex[ex["city"] == c].sort_values("year")
            ax.plot(s["year"], s["hot_days"], linewidth=1, label=c)
        ax.set_title(f"Hot extreme days per year (≥ p95 of {BASELINE[0]}–{BASELINE[1]})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Days")
        if len(cities_in_view) <= 12:
            ax.legend(ncol=3, fontsize=8)
        st.pyplot(fig, clear_figure=True)

        # Cold extremes plot
        fig = plt.figure(figsize=(12, 5))
        ax = plt.gca()
        for c in cities_in_view:
            s = ex[ex["city"] == c].sort_values("year")
            ax.plot(s["year"], s["cold_days"], linewidth=1, label=c)
        ax.set_title(f"Cold extreme days per year (≤ p05 of {BASELINE[0]}–{BASELINE[1]})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Days")
        if len(cities_in_view) <= 12:
            ax.legend(ncol=3, fontsize=8)
        st.pyplot(fig, clear_figure=True)

        # Summary table
        baseline_mean = ex[(ex["year"] >= BASELINE[0]) & (ex["year"] <= BASELINE[1])].groupby("city", as_index=False)[["hot_days", "cold_days"]].mean()
        recent_mean = ex[(ex["year"] >= RECENT[0]) & (ex["year"] <= RECENT[1])].groupby("city", as_index=False)[["hot_days", "cold_days"]].mean()

        summary = baseline_mean.merge(recent_mean, on="city", suffixes=("_baseline", "_recent"))
        summary["Δ hot days (recent-baseline)"] = summary["hot_days_recent"] - summary["hot_days_baseline"]
        summary["Δ cold days (recent-baseline)"] = summary["cold_days_recent"] - summary["cold_days_baseline"]

        st.markdown('<div class="section-title">Baseline vs recent (mean days/year)</div>', unsafe_allow_html=True)
        st.dataframe(summary.sort_values("Δ hot days (recent-baseline)", ascending=False), use_container_width=True)

# -----------------------------
# Maps
# -----------------------------
elif page == "Maps":
    st.markdown('<div class="section-title">Spatial patterns</div>', unsafe_allow_html=True)
    st.caption("These are true maps (georeferenced points) rendered with pydeck. Labels are included; zoom adjusts to your city selection.")

    # Build map layers from current view
    map_df = compute_map_layers(
        annual=annual,
        extremes=extremes,
        year_range=year_range,
        min_days_per_year=min_days_per_year,
        baseline=BASELINE,
        recent=RECENT,
    )

    # Keep only cities we can geocode
    map_df = map_df.dropna(subset=["lat", "lon"]).copy()

    layer = st.selectbox(
        "Map layer",
        [
            "Warming trend (°C/decade)",
            f"Δ hot extreme days (avg/yr): {RECENT[0]}–{RECENT[1]} vs {BASELINE[0]}–{BASELINE[1]}",
            f"Δ cold extreme days (avg/yr): {RECENT[0]}–{RECENT[1]} vs {BASELINE[0]}–{BASELINE[1]}",
        ],
        index=0,
    )

    if layer.startswith("Warming trend"):
        value_col = "warming_trend_decade"
        title = "Warming trend (°C/decade)"
    elif "Δ hot" in layer:
        value_col = "delta_hot_days"
        title = "Change in hot extreme days (avg/yr)"
    else:
        value_col = "delta_cold_days"
        title = "Change in cold extreme days (avg/yr)"

    if map_df.empty:
        st.warning("No mappable points available (cities missing coordinates).")
    else:
        deck = make_pydeck_map(
            df_points=map_df.rename(columns={"warming_trend_decade": "warming_trend_decade", "delta_hot_days": "delta_hot_days", "delta_cold_days": "delta_cold_days"}),
            value_col=value_col,
            title=title,
            selected_cities=selected_cities,
        )
        st.pydeck_chart(deck, use_container_width=True)

        with st.expander("Show mapped metrics table", expanded=False):
            show_cols = ["city", "warming_trend_decade", "delta_hot_days", "delta_cold_days", "lat", "lon"]
            st.dataframe(map_df[show_cols].sort_values("city"), use_container_width=True)

# -----------------------------
# Download
# -----------------------------
elif page == "Download":
    st.markdown('<div class="section-title">Download data</div>', unsafe_allow_html=True)
    st.caption("Exports reflect your current filters (cities + year range + coverage threshold affects derived tables).")

    st.download_button(
        "Download filtered daily (CSV)",
        data=d.to_csv(index=False).encode("utf-8"),
        file_name="canada_climate_filtered_daily.csv",
        mime="text/csv",
        use_container_width=True,
    )

    annual_export = annual.copy()
    annual_export["temp_ok"] = annual_export["n_temp"] >= int(min_days_per_year)
    st.download_button(
        "Download annual metrics (CSV)",
        data=annual_export.to_csv(index=False).encode("utf-8"),
        file_name="canada_climate_annual_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    ex_export = extremes.copy()
    st.download_button(
        "Download extremes indices (CSV)",
        data=ex_export.to_csv(index=False).encode("utf-8"),
        file_name="canada_climate_extremes_indices.csv",
        mime="text/csv",
        use_container_width=True,
    )
