import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk


# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(
    page_title="Canada Climate Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      [data-testid="stSidebar"] {min-width: 320px; max-width: 380px;}
      .kpi {border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px;}
      .muted {color: rgba(0,0,0,0.55); font-size: 0.95rem;}
      .small {font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Robust paths
# -----------------------------
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Expected (from your notebooks)
TIDY_PARQUET = PROCESSED_DIR / "climate_daily_tidy.parquet"
MONTHLY_PARQUET = PROCESSED_DIR / "climate_monthly.parquet"
YEARLY_PARQUET = PROCESSED_DIR / "climate_yearly.parquet"

# Raw fallback
RAW_CANDIDATES = [
    RAW_DIR / "canadian_climate_daily.csv",
    RAW_DIR / "canadian_climate_daily.csv",  # keeping your naming pattern
]
RAW_PATH = next((p for p in RAW_CANDIDATES if p.exists()), None)
if RAW_PATH is None and RAW_DIR.exists():
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if csvs:
        RAW_PATH = csvs[0]


# -----------------------------
# Utilities
# -----------------------------
CITY_COORDS = {
    "CALGARY": (51.0447, -114.0719),
    "EDMONTON": (53.5461, -113.4938),
    "HALIFAX": (44.6488, -63.5752),
    "MONCTON": (46.0878, -64.7782),
    "MONTREAL": (45.5019, -73.5674),
    "OTTAWA": (45.4215, -75.6972),
    "QUEBEC": (46.8139, -71.2080),
    "SASKATOON": (52.1332, -106.6700),
    "STJOHNS": (47.5615, -52.7126),
    "TORONTO": (43.6532, -79.3832),
    "VANCOUVER": (49.2827, -123.1207),
    "WHITEHORSE": (60.7212, -135.0568),
    "WINNIPEG": (49.8951, -97.1384),
}

def _safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=False)

def _ols_slope_ci(x, y):
    """Return slope per year and ~95% CI (normal approx)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 10:
        return None

    x0 = x - x.mean()
    b = np.sum(x0 * (y - y.mean())) / np.sum(x0 ** 2)
    a = y.mean() - b * x.mean()

    yhat = a + b * x
    resid = y - yhat
    s2 = np.sum(resid ** 2) / (n - 2)
    se_b = np.sqrt(s2 / np.sum(x0 ** 2))

    z = 1.96
    lo, hi = b - z * se_b, b + z * se_b
    return {"n": n, "slope_per_year": b, "lo": lo, "hi": hi}

def _get_city_latlon(df_daily):
    # If dataset has coordinates, use them; otherwise fall back to CITY_COORDS.
    lat_cols = [c for c in df_daily.columns if c.lower() in ("lat", "latitude")]
    lon_cols = [c for c in df_daily.columns if c.lower() in ("lon", "lng", "longitude")]
    if lat_cols and lon_cols:
        latc, lonc = lat_cols[0], lon_cols[0]
        coords = (
            df_daily.groupby("city", as_index=False)
            .agg(lat=(latc, "median"), lon=(lonc, "median"))
        )
        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
        return coords

    coords = pd.DataFrame(
        [{"city": k, "lat": v[0], "lon": v[1]} for k, v in CITY_COORDS.items()]
    )
    return coords


# -----------------------------
# Load / build data (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data():
    # Preferred: processed parquet
    if TIDY_PARQUET.exists() and MONTHLY_PARQUET.exists() and YEARLY_PARQUET.exists():
        df_daily = pd.read_parquet(TIDY_PARQUET)
        df_monthly = pd.read_parquet(MONTHLY_PARQUET)
        df_yearly = pd.read_parquet(YEARLY_PARQUET)
        source = "processed_parquet"
    else:
        if RAW_PATH is None:
            raise FileNotFoundError(
                "Could not find raw CSV in data/raw. Ensure the repo contains data/raw/*.csv "
                "or commit processed parquet files to data/processed."
            )
        df = pd.read_csv(RAW_PATH)
        if "LOCAL_DATE" not in df.columns:
            raise ValueError("Expected LOCAL_DATE column in raw CSV.")

        df["LOCAL_DATE"] = _safe_to_datetime(df["LOCAL_DATE"])
        if df["LOCAL_DATE"].isna().any():
            raise ValueError("LOCAL_DATE parsing failed for some rows.")

        # Melt wide -> tidy for metrics
        # expected column patterns: MEAN_TEMPERATURE_CITY, TOTAL_PRECIPITATION_CITY
        value_cols = [c for c in df.columns if c != "LOCAL_DATE" and "_" in c]
        # Keep only two metrics if present
        keep = []
        for c in value_cols:
            uc = c.upper()
            if uc.startswith("MEAN_TEMPERATURE_") or uc.startswith("TOTAL_PRECIPITATION_"):
                keep.append(c)

        if not keep:
            raise ValueError(
                "Raw CSV does not look like expected wide city columns "
                "(MEAN_TEMPERATURE_* / TOTAL_PRECIPITATION_*)."
            )

        long = df.melt(
            id_vars=["LOCAL_DATE"], value_vars=keep,
            var_name="series", value_name="value"
        )
        # Split series -> metric + city
        parts = long["series"].str.split("_", n=2, expand=True)
        # Rebuild robustly because city names can contain underscores in some datasets
        # series like: MEAN_TEMPERATURE_TORONTO
        metric = long["series"].str.extract(r"^(MEAN_TEMPERATURE|TOTAL_PRECIPITATION)")[0]
        city = long["series"].str.replace(r"^(MEAN_TEMPERATURE|TOTAL_PRECIPITATION)_", "", regex=True)

        long["metric"] = metric
        long["city"] = city

        tidy = (
            long.pivot_table(index=["LOCAL_DATE", "city"], columns="metric", values="value", aggfunc="first")
            .reset_index()
            .rename(columns={
                "LOCAL_DATE": "local_date",
                "MEAN_TEMPERATURE": "mean_temperature",
                "TOTAL_PRECIPITATION": "total_precipitation",
            })
        )

        tidy["city"] = tidy["city"].astype(str).str.upper().str.strip()
        tidy["local_date"] = _safe_to_datetime(tidy["local_date"]).dt.tz_localize(None)
        tidy["mean_temperature"] = pd.to_numeric(tidy["mean_temperature"], errors="coerce")
        tidy["total_precipitation"] = pd.to_numeric(tidy["total_precipitation"], errors="coerce")

        tidy["year"] = tidy["local_date"].dt.year
        tidy["month"] = tidy["local_date"].dt.month
        tidy["dayofyear"] = tidy["local_date"].dt.dayofyear

        df_daily = tidy

        df_monthly = (
            df_daily.groupby(["city", "year", "month"], as_index=False)
            .agg(
                mean_temperature=("mean_temperature", "mean"),
                total_precipitation=("total_precipitation", "sum"),
                n_obs_temp=("mean_temperature", "count"),
                n_obs_precip=("total_precipitation", "count"),
            )
        )

        df_yearly = (
            df_daily.groupby(["city", "year"], as_index=False)
            .agg(
                mean_temperature=("mean_temperature", "mean"),
                total_precipitation=("total_precipitation", "sum"),
                n_obs_temp=("mean_temperature", "count"),
                n_obs_precip=("total_precipitation", "count"),
            )
        )
        source = "raw_csv"

    # Standardize types
    df_daily = df_daily.copy()
    df_daily["local_date"] = _safe_to_datetime(df_daily["local_date"]).dt.tz_localize(None)
    df_daily["city"] = df_daily["city"].astype(str).str.upper().str.strip()

    # Ensure derived columns exist
    if "year" not in df_daily.columns:
        df_daily["year"] = df_daily["local_date"].dt.year
    if "month" not in df_daily.columns:
        df_daily["month"] = df_daily["local_date"].dt.month
    if "dayofyear" not in df_daily.columns:
        df_daily["dayofyear"] = df_daily["local_date"].dt.dayofyear

    # Metadata
    meta = {
        "source": source,
        "date_min": df_daily["local_date"].min(),
        "date_max": df_daily["local_date"].max(),
        "cities": sorted(df_daily["city"].dropna().unique().tolist()),
    }
    return df_daily, df_monthly, df_yearly, meta


@st.cache_data(show_spinner=False)
def compute_city_year_metrics(df_daily: pd.DataFrame, min_days_per_year: int):
    annual = (
        df_daily.groupby(["city", "year"], as_index=False)
        .agg(
            mean_temp=("mean_temperature", "mean"),
            p50_temp=("mean_temperature", "median"),
            sd_temp=("mean_temperature", "std"),
            total_precip=("total_precipitation", "sum"),
            wet_days=("total_precipitation", lambda s: int((s > 0).sum())),
            n_temp=("mean_temperature", "count"),
            n_precip=("total_precipitation", "count"),
        )
    )
    annual["temp_ok"] = annual["n_temp"] >= min_days_per_year
    annual["precip_ok"] = annual["n_precip"] >= min_days_per_year
    annual["wet_day_ratio"] = (annual["wet_days"] / annual["n_precip"]).replace([np.inf, -np.inf], np.nan)
    return annual


@st.cache_data(show_spinner=False)
def compute_trends(annual: pd.DataFrame):
    cities = sorted(annual["city"].unique())
    rows_t = []
    rows_p = []
    for c in cities:
        sub_t = annual[(annual["city"] == c) & (annual["temp_ok"])].sort_values("year")
        r = _ols_slope_ci(sub_t["year"], sub_t["mean_temp"])
        if r:
            rows_t.append({
                "city": c,
                "n": r["n"],
                "slope_c_per_decade": r["slope_per_year"] * 10,
                "ci_lo": r["lo"] * 10,
                "ci_hi": r["hi"] * 10,
            })

        sub_p = annual[(annual["city"] == c) & (annual["precip_ok"])].sort_values("year")
        r2 = _ols_slope_ci(sub_p["year"], sub_p["total_precip"])
        if r2:
            rows_p.append({
                "city": c,
                "n": r2["n"],
                "slope_mm_per_decade": r2["slope_per_year"] * 10,
                "ci_lo": r2["lo"] * 10,
                "ci_hi": r2["hi"] * 10,
            })
    return pd.DataFrame(rows_t), pd.DataFrame(rows_p)


@st.cache_data(show_spinner=False)
def compute_extremes(df_daily: pd.DataFrame, baseline=(1961, 1990), q_lo=0.05, q_hi=0.95, min_days_per_year=300):
    y0, y1 = baseline
    base = df_daily[(df_daily["year"] >= y0) & (df_daily["year"] <= y1)].copy()

    thresholds = (
        base.groupby("city")["mean_temperature"]
        .quantile([q_lo, q_hi])
        .unstack()
        .rename(columns={q_lo: "t_lo", q_hi: "t_hi"})
        .reset_index()
    )
    d2 = df_daily.merge(thresholds, on="city", how="left")
    d2["cold_extreme"] = d2["mean_temperature"] <= d2["t_lo"]
    d2["hot_extreme"] = d2["mean_temperature"] >= d2["t_hi"]

    ex = (
        d2.groupby(["city", "year"], as_index=False)
        .agg(
            cold_days=("cold_extreme", "sum"),
            hot_days=("hot_extreme", "sum"),
            n_temp=("mean_temperature", "count"),
        )
    )
    ex["year_ok"] = ex["n_temp"] >= min_days_per_year
    return thresholds, ex


# -----------------------------
# Sidebar controls
# -----------------------------
df_daily, df_monthly, df_yearly, meta = load_data()

st.sidebar.markdown("## Canada Climate Dashboard")
st.sidebar.markdown(
    f"<div class='muted'>"
    f"{meta['date_min'].date()} → {meta['date_max'].date()} • City-level daily → monthly/yearly rollups"
    f"</div>",
    unsafe_allow_html=True
)

with st.sidebar.expander("Data status", expanded=False):
    st.write(f"**Source:** {meta['source']}")
    st.write(f"**Repo path:** `{REPO_DIR}`")
    st.write(f"**Raw path:** `{RAW_PATH}`" if RAW_PATH else "**Raw path:** not found")
    st.write(f"**Processed parquet present:** {TIDY_PARQUET.exists() and MONTHLY_PARQUET.exists() and YEARLY_PARQUET.exists()}")

page = st.sidebar.radio("Navigate", ["Overview", "Trends", "Extremes", "Maps", "Download"], index=0)

cities_all = meta["cities"]
default_cities = [c for c in ["TORONTO", "VANCOUVER", "CALGARY", "MONTREAL"] if c in cities_all] or cities_all[:3]
cities_sel = st.sidebar.multiselect("Cities", cities_all, default=default_cities)

year_min, year_max = int(df_daily["year"].min()), int(df_daily["year"].max())
year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(max(year_min, 1940), year_max))

min_days = st.sidebar.slider("Coverage threshold (days/year)", min_value=200, max_value=365, value=300, step=5)

baseline_label = st.sidebar.selectbox("Baseline period", ["1961-1990", "1991-2020"], index=0)
baseline = (1961, 1990) if baseline_label == "1961-1990" else (1991, 2020)

# Filtered daily
d = df_daily[
    (df_daily["city"].isin(cities_sel)) &
    (df_daily["year"] >= year_range[0]) &
    (df_daily["year"] <= year_range[1])
].copy()

annual = compute_city_year_metrics(df_daily, min_days_per_year=min_days)
annual_f = annual[
    (annual["city"].isin(cities_sel)) &
    (annual["year"] >= year_range[0]) &
    (annual["year"] <= year_range[1])
].copy()

temp_trends, precip_trends = compute_trends(annual)
thresholds, extremes = compute_extremes(df_daily, baseline=baseline, min_days_per_year=min_days)


# -----------------------------
# Header KPIs
# -----------------------------
top_left, top_right = st.columns([1.35, 1])

with top_left:
    st.title("Canada Climate Dashboard")
    st.markdown(
        "<div class='muted'>A compact analytics layer for long-run temperature, precipitation, and extremes, using coverage-aware rollups.</div>",
        unsafe_allow_html=True
    )

with top_right:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        st.metric("Cities selected", len(cities_sel))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        st.metric("Years", f"{year_range[0]}–{year_range[1]}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        st.metric("Coverage gate", f"{min_days} days/yr")
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()


# -----------------------------
# Altair theme defaults
# -----------------------------
alt.data_transformers.disable_max_rows()

def a_line(df, x, y, color, tooltip, title):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            color=alt.Color(color, legend=alt.Legend(orient="bottom", columns=4)),
            tooltip=tooltip,
        )
        .properties(title=title, height=330)
    )

def a_heatmap(df, x, y, v, title):
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            color=alt.Color(v, title=None),
            tooltip=[x, y, v],
        )
        .properties(title=title, height=350)
    )


# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.subheader("Overview")
    st.markdown(
        "<div class='muted'>High-signal views first: annual rollups, seasonality, and distribution shifts.</div>",
        unsafe_allow_html=True
    )

    # Annual mean temp (coverage-aware)
    temp_ok = annual_f[annual_f["temp_ok"]].copy()
    if temp_ok.empty:
        st.warning("No temp-eligible years under the current coverage threshold. Lower the threshold or widen the year range.")
    else:
        chart = a_line(
            temp_ok,
            x="year:Q",
            y="mean_temp:Q",
            color="city:N",
            tooltip=["city:N", "year:Q", "mean_temp:Q", "n_temp:Q"],
            title="Annual Mean Temperature (eligible years only)",
        )
        st.altair_chart(chart, use_container_width=True)

    # Seasonality: monthly normals comparison (baseline vs recent within selected years)
    d2 = d.copy()
    d2["month"] = d2["local_date"].dt.month

    # Use baseline vs recent within overall selection, but compute on full series for stability
    base_y0, base_y1 = (1961, 1990)
    rec_y0, rec_y1 = (1991, 2020)
    sub_base = df_daily[(df_daily["city"].isin(cities_sel)) & (df_daily["year"].between(base_y0, base_y1))].copy()
    sub_rec = df_daily[(df_daily["city"].isin(cities_sel)) & (df_daily["year"].between(rec_y0, rec_y1))].copy()
    sub_base["period"] = f"{base_y0}-{base_y1}"
    sub_rec["period"] = f"{rec_y0}-{rec_y1}"
    sub_base["month"] = sub_base["local_date"].dt.month
    sub_rec["month"] = sub_rec["local_date"].dt.month

    norms = pd.concat([sub_base, sub_rec], ignore_index=True)
    norms_m = (
        norms.groupby(["city", "period", "month"], as_index=False)
        .agg(mean_temp=("mean_temperature", "mean"), n=("mean_temperature", "count"))
    )

    st.markdown("### Seasonality (Normals)")
    cA, cB = st.columns([1.2, 1])
    with cA:
        chart2 = (
            alt.Chart(norms_m)
            .mark_line()
            .encode(
                x=alt.X("month:Q", title="Month", scale=alt.Scale(domain=[1, 12])),
                y=alt.Y("mean_temp:Q", title="Mean temperature (°C)"),
                color=alt.Color("period:N", legend=alt.Legend(orient="bottom")),
                facet=alt.Facet("city:N", columns=2),
                tooltip=["city:N", "period:N", "month:Q", "mean_temp:Q", "n:Q"],
            )
            .properties(height=170)
        )
        st.altair_chart(chart2, use_container_width=True)

    with cB:
        st.markdown("### Distribution shift (selected years)")
        if d.empty:
            st.info("No data in current filter.")
        else:
            # Temperature distribution by city
            dist = d[["city", "mean_temperature"]].dropna()
            chart3 = (
                alt.Chart(dist)
                .transform_density(
                    "mean_temperature",
                    as_=["mean_temperature", "density"],
                    groupby=["city"],
                )
                .mark_area(opacity=0.25)
                .encode(
                    x=alt.X("mean_temperature:Q", title="Temperature (°C)"),
                    y=alt.Y("density:Q", title=None),
                    color=alt.Color("city:N", legend=alt.Legend(orient="bottom", columns=2)),
                    tooltip=["city:N", "mean_temperature:Q", "density:Q"],
                )
                .properties(height=380)
            )
            st.altair_chart(chart3, use_container_width=True)


elif page == "Trends":
    st.subheader("Trends")
    st.markdown("<div class='muted'>OLS slopes computed on coverage-eligible annual series; includes uncertainty bands.</div>", unsafe_allow_html=True)

    t = temp_trends[temp_trends["city"].isin(cities_sel)].copy()
    p = precip_trends[precip_trends["city"].isin(cities_sel)].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Warming rate (°C/decade)")
        if t.empty:
            st.info("No trend results available for selected cities (coverage gate may be too strict).")
        else:
            bar = (
                alt.Chart(t)
                .mark_bar()
                .encode(
                    x=alt.X("slope_c_per_decade:Q", title="°C/decade"),
                    y=alt.Y("city:N", sort="-x", title=None),
                    tooltip=["city:N", "n:Q", "slope_c_per_decade:Q", "ci_lo:Q", "ci_hi:Q"],
                )
                .properties(height=360)
            )
            err = (
                alt.Chart(t)
                .mark_rule()
                .encode(
                    x="ci_lo:Q",
                    x2="ci_hi:Q",
                    y=alt.Y("city:N", sort="-x"),
                )
            )
            st.altair_chart((bar + err), use_container_width=True)

    with c2:
        st.markdown("### Precipitation trend (mm/decade)")
        if p.empty:
            st.info("No precip trend results available for selected cities (coverage gate may be too strict).")
        else:
            bar2 = (
                alt.Chart(p)
                .mark_bar()
                .encode(
                    x=alt.X("slope_mm_per_decade:Q", title="mm/decade"),
                    y=alt.Y("city:N", sort="-x", title=None),
                    tooltip=["city:N", "n:Q", "slope_mm_per_decade:Q", "ci_lo:Q", "ci_hi:Q"],
                )
                .properties(height=360)
            )
            err2 = (
                alt.Chart(p)
                .mark_rule()
                .encode(
                    x="ci_lo:Q",
                    x2="ci_hi:Q",
                    y=alt.Y("city:N", sort="-x"),
                )
            )
            st.altair_chart((bar2 + err2), use_container_width=True)

    st.markdown("### Rolling 10-year mean (temperature)")
    temp_ok = annual_f[annual_f["temp_ok"]].copy()
    if temp_ok.empty:
        st.warning("No temp-eligible years under the current coverage threshold.")
    else:
        tmp = temp_ok.sort_values(["city", "year"]).copy()
        tmp["roll10"] = (
            tmp.groupby("city")["mean_temp"]
            .transform(lambda s: s.rolling(10, min_periods=7).mean())
        )
        chart = a_line(
            tmp.dropna(subset=["roll10"]),
            x="year:Q",
            y="roll10:Q",
            color="city:N",
            tooltip=["city:N", "year:Q", "roll10:Q"],
            title="Rolling 10-year mean temperature",
        )
        st.altair_chart(chart, use_container_width=True)


elif page == "Extremes":
    st.subheader("Extremes")
    st.markdown(
        f"<div class='muted'>Extremes are defined relative to the baseline quantiles (≤p05 and ≥p95) for {baseline[0]}–{baseline[1]}.</div>",
        unsafe_allow_html=True
    )

    ex = extremes[(extremes["city"].isin(cities_sel)) & (extremes["year"].between(year_range[0], year_range[1])) & (extremes["year_ok"])].copy()

    if ex.empty:
        st.warning("No extreme-day series available under current filters. Try lowering the coverage threshold.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            chart_hot = a_line(
                ex,
                x="year:Q",
                y="hot_days:Q",
                color="city:N",
                tooltip=["city:N", "year:Q", "hot_days:Q", "n_temp:Q"],
                title=f"Hot extreme days/year (≥ p95 of {baseline[0]}–{baseline[1]})",
            )
            st.altair_chart(chart_hot, use_container_width=True)

        with c2:
            chart_cold = a_line(
                ex,
                x="year:Q",
                y="cold_days:Q",
                color="city:N",
                tooltip=["city:N", "year:Q", "cold_days:Q", "n_temp:Q"],
                title=f"Cold extreme days/year (≤ p05 of {baseline[0]}–{baseline[1]})",
            )
            st.altair_chart(chart_cold, use_container_width=True)

        st.markdown("### Extremes regime shift (baseline vs recent)")
        # Compare mean annual extremes for baseline vs recent
        def _period(y):
            if 1961 <= y <= 1990:
                return "1961–1990"
            if 1991 <= y <= 2020:
                return "1991–2020"
            return None

        ex2 = extremes[extremes["city"].isin(cities_sel)].copy()
        ex2["period"] = ex2["year"].apply(_period)
        ex2 = ex2.dropna(subset=["period"])
        shift = (
            ex2.groupby(["city", "period"], as_index=False)
            .agg(hot_days=("hot_days", "mean"), cold_days=("cold_days", "mean"))
        )
        shift_long = shift.melt(id_vars=["city", "period"], var_name="metric", value_name="days")

        chart_shift = (
            alt.Chart(shift_long)
            .mark_bar()
            .encode(
                x=alt.X("days:Q", title="Avg days/year"),
                y=alt.Y("city:N", title=None),
                color=alt.Color("period:N", legend=alt.Legend(orient="bottom")),
                column=alt.Column("metric:N", title=None),
                tooltip=["city:N", "period:N", "metric:N", "days:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_shift, use_container_width=True)


elif page == "Maps":
    st.subheader("Spatial patterns")
    st.caption(
        "These are true maps (georeferenced city points). Because this dataset is city-level (not gridded), "
        "the map will always show a limited number of locations — the value is in the styling, tooltips, and layers."
    )

    # --- Base map styles that DO NOT require a Mapbox token ---
    # (These are public styles; they work on Streamlit Cloud without secrets.)
    BASEMAPS = {
        "Light (CARTO Positron)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Dark (CARTO Dark Matter)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    }
    basemap_choice = st.selectbox("Basemap", list(BASEMAPS.keys()), index=0)
    map_style = BASEMAPS[basemap_choice]

    coords = _get_city_latlon(df_daily)
    coords = coords[coords["city"].isin(cities_sel)].copy()

    # Attach metrics already computed in the app
    m = coords.merge(temp_trends[["city", "slope_c_per_decade"]], on="city", how="left")
    m = m.merge(precip_trends[["city", "slope_mm_per_decade"]], on="city", how="left")

    # Extremes shift (recent - baseline)
    ex = extremes.copy()
    ex["period"] = np.where(ex["year"].between(1961, 1990), "1961–1990",
                    np.where(ex["year"].between(1991, 2020), "1991–2020", None))
    ex = ex.dropna(subset=["period"])
    exm = (
        ex.groupby(["city", "period"], as_index=False)
          .agg(hot=("hot_days", "mean"), cold=("cold_days", "mean"))
    )
    exw = exm.pivot(index="city", columns="period", values=["hot", "cold"])

    def _get(w, metric, period):
        try:
            return w[(metric, period)]
        except Exception:
            return pd.Series(index=w.index, dtype=float)

    ex_shift = pd.DataFrame({"city": exw.index})
    ex_shift["d_hot"]  = (_get(exw, "hot",  "1991–2020") - _get(exw, "hot",  "1961–1990")).values
    ex_shift["d_cold"] = (_get(exw, "cold", "1991–2020") - _get(exw, "cold", "1961–1990")).values

    m = m.merge(ex_shift, on="city", how="left")

    # ---------- Better auto-zoom ----------
    # Compute bounds and choose a zoom that actually frames the selected cities.
    def _auto_view(df):
        if len(df) == 0:
            return pdk.ViewState(latitude=56.1304, longitude=-106.3468, zoom=2.6, pitch=35)

        lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
        lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
        lat_c = (lat_min + lat_max) / 2
        lon_c = (lon_min + lon_max) / 2

        span = max(abs(lat_max - lat_min), abs(lon_max - lon_min))
        # Heuristic zoom: smaller span -> higher zoom
        if span < 2:   zoom = 6.0
        elif span < 5: zoom = 4.6
        elif span < 10: zoom = 3.8
        else:          zoom = 3.0

        return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, pitch=40)

    view_state = _auto_view(m)

    # ---------- Layer selector ----------
    layer = st.selectbox(
        "Map layer",
        [
            "Warming trend (°C/decade) — 3D columns + points",
            "Precip trend (mm/decade) — 3D columns + points",
            "Hot extremes shift (days/year) — sized points + heat",
            "Cold extremes shift (days/year) — sized points + heat",
        ],
        index=0,
    )

    # Make sure numeric
    for col in ["slope_c_per_decade", "slope_mm_per_decade", "d_hot", "d_cold"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    # Helper for color bins (keeps pydeck JSON simple and reliable)
    def _color_bin(v, neg_rgb, pos_rgb):
        if pd.isna(v): return [160, 160, 160, 120]
        return (pos_rgb if v >= 0 else neg_rgb) + [190]

    layers = []

    # City label layer (always on)
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=m,
            get_position="[lon, lat]",
            get_text="city",
            get_size=14,
            get_color=[20, 20, 20],
            get_alignment_baseline="'bottom'",
        )
    )

    # A “base” point layer so the map never looks empty
    m["_base_r"] = 14000  # meters
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=m,
            get_position="[lon, lat]",
            get_radius="_base_r",
            radius_min_pixels=5,
            radius_max_pixels=40,
            get_fill_color=[80, 80, 80, 80],
            pickable=False,
        )
    )

    tooltip = {"html": "<b>{city}</b>", "style": {"backgroundColor": "white"}}
    title = ""

    if "Warming trend" in layer:
        title = "Warming trend (°C/decade)"
        m2 = m.copy()
        m2["val"] = m2["slope_c_per_decade"].fillna(0.0)

        # Precompute elevation + colors (no functions in JSON)
        m2["elevation"] = (m2["val"].abs() * 120000).astype(float)
        m2["fill"] = m2["val"].apply(lambda x: _color_bin(x, [25, 120, 255], [255, 80, 30]))

        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=m2,
                get_position="[lon, lat]",
                get_elevation="elevation",
                elevation_scale=1,
                radius=26000,              # bigger so it reads as “map”
                get_fill_color="fill",
                pickable=True,
                auto_highlight=True,
            )
        )
        tooltip = {
            "html": "<b>{city}</b><br/>Warming: {slope_c_per_decade} °C/decade",
            "style": {"backgroundColor": "white"},
        }

    elif "Precip trend" in layer:
        title = "Precipitation trend (mm/decade)"
        m2 = m.copy()
        m2["val"] = m2["slope_mm_per_decade"].fillna(0.0)
        m2["elevation"] = (m2["val"].abs() * 2500).astype(float)
        m2["fill"] = m2["val"].apply(lambda x: _color_bin(x, [30, 144, 255], [46, 184, 92]))

        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=m2,
                get_position="[lon, lat]",
                get_elevation="elevation",
                elevation_scale=1,
                radius=26000,
                get_fill_color="fill",
                pickable=True,
                auto_highlight=True,
            )
        )
        tooltip = {
            "html": "<b>{city}</b><br/>Precip trend: {slope_mm_per_decade} mm/decade",
            "style": {"backgroundColor": "white"},
        }

    elif "Hot extremes shift" in layer:
        title = "Change in hot extremes (1991–2020 vs 1961–1990)"
        m2 = m.copy()
        m2["val"] = m2["d_hot"].fillna(0.0)
        m2["radius"] = (12000 + m2["val"].abs() * 3500).astype(float)
        m2["fill"] = m2["val"].apply(lambda x: _color_bin(x, [30, 144, 255], [255, 60, 0]))
        m2["weight"] = m2["val"].abs().astype(float)

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=m2,
                get_position="[lon, lat]",
                get_radius="radius",
                radius_min_pixels=6,
                radius_max_pixels=70,
                get_fill_color="fill",
                pickable=True,
                auto_highlight=True,
            )
        )
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=m2,
                get_position="[lon, lat]",
                get_weight="weight",
                radiusPixels=110,
            )
        )
        tooltip = {
            "html": "<b>{city}</b><br/>Δ hot extremes: {d_hot} days/year",
            "style": {"backgroundColor": "white"},
        }

    else:
        title = "Change in cold extremes (1991–2020 vs 1961–1990)"
        m2 = m.copy()
        m2["val"] = m2["d_cold"].fillna(0.0)
        m2["radius"] = (12000 + m2["val"].abs() * 3500).astype(float)
        # Typically cold extremes decline (negative) — color negatives as blue, positives as red
        m2["fill"] = m2["val"].apply(lambda x: _color_bin(x, [30, 144, 255], [255, 60, 0]))
        m2["weight"] = m2["val"].abs().astype(float)

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=m2,
                get_position="[lon, lat]",
                get_radius="radius",
                radius_min_pixels=6,
                radius_max_pixels=70,
                get_fill_color="fill",
                pickable=True,
                auto_highlight=True,
            )
        )
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=m2,
                get_position="[lon, lat]",
                get_weight="weight",
                radiusPixels=110,
            )
        )
        tooltip = {
            "html": "<b>{city}</b><br/>Δ cold extremes: {d_cold} days/year",
            "style": {"backgroundColor": "white"},
        }

    st.markdown(f"### {title}")

    deck = pdk.Deck(
        map_style=map_style,  # CARTO style -> basemap loads without token
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

    # Optional: show the underlying metrics driving the map (small, not table-heavy)
    with st.expander("Show map metrics"):
        show_cols = ["city", "lat", "lon", "slope_c_per_decade", "slope_mm_per_decade", "d_hot", "d_cold"]
        st.dataframe(m[show_cols].sort_values("city"), use_container_width=True)



elif page == "Download":
    st.subheader("Download")
    st.markdown("<div class='muted'>Export filtered datasets for reporting or reuse.</div>", unsafe_allow_html=True)

    # Filtered extracts
    d_out = d.copy()
    annual_out = annual_f.copy()

    st.download_button(
        "Download filtered daily (CSV)",
        data=d_out.to_csv(index=False).encode("utf-8"),
        file_name="filtered_daily.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download filtered annual rollup (CSV)",
        data=annual_out.to_csv(index=False).encode("utf-8"),
        file_name="filtered_annual.csv",
        mime="text/csv",
    )

    ex_out = extremes[(extremes["city"].isin(cities_sel)) & extremes["year"].between(year_range[0], year_range[1])].copy()
    st.download_button(
        "Download extremes (CSV)",
        data=ex_out.to_csv(index=False).encode("utf-8"),
        file_name="filtered_extremes.csv",
        mime="text/csv",
    )
