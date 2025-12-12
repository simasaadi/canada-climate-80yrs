import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# App config + minimal styling
# -----------------------------
st.set_page_config(
    page_title="Canada Climate (1940–Present)",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      [data-testid="stSidebar"] { padding-top: 1rem; }
      .kpi { padding: 0.9rem 1rem; border-radius: 14px; border: 1px solid rgba(120,120,120,0.25); }
      .kpi .label { font-size: 0.85rem; opacity: 0.8; }
      .kpi .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
      .kpi .note { font-size: 0.8rem; opacity: 0.75; margin-top: 0.25rem; }
      .muted { opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Paths (works on Streamlit Cloud)
# -----------------------------
REPO_DIR = Path(__file__).resolve().parent
RAW_DIR = REPO_DIR / "data" / "raw"
PROCESSED_DIR = REPO_DIR / "data" / "processed"
REPORT_DIR = REPO_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TIDY_PATH = PROCESSED_DIR / "climate_daily_tidy.parquet"
MONTHLY_PATH = PROCESSED_DIR / "climate_monthly.parquet"
YEARLY_PATH = PROCESSED_DIR / "climate_yearly.parquet"


# -----------------------------
# Reference city coordinates
# -----------------------------
CITY_COORDS = {
    "CALGARY": (51.0447, -114.0719),
    "EDMONTON": (53.5461, -113.4938),
    "HALIFAX": (44.6488, -63.5752),
    "MONCTON": (46.0878, -64.7782),
    "MONTREAL": (45.5019, -73.5674),
    "OTTAWA": (45.4215, -75.6972),
    "QUEBEC": (46.8139, -71.2080),
    "SASKATOON": (52.1579, -106.6702),
    "STJOHNS": (47.5615, -52.7126),
    "TORONTO": (43.6532, -79.3832),
    "VANCOUVER": (49.2827, -123.1207),
    "WHITEHORSE": (60.7212, -135.0568),
    "WINNIPEG": (49.8951, -97.1384),
}


# -----------------------------
# Utilities
# -----------------------------
def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_raw_csv() -> Path:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    # common candidates
    candidates = [
        RAW_DIR / "canadian_climate_daily.csv",
        RAW_DIR / "canadian_climate_daily.csv",
    ]
    raw_path = next((p for p in candidates if p.exists()), None)
    if raw_path is not None:
        return raw_path

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}.")
    return csvs[0]


def build_processed_from_raw(raw_path: Path) -> None:
    df = pd.read_csv(raw_path)

    if "LOCAL_DATE" not in df.columns:
        raise ValueError("Expected LOCAL_DATE column not found in raw CSV.")

    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"], errors="coerce")
    bad = int(df["LOCAL_DATE"].isna().sum())
    if bad > 0:
        raise ValueError(f"LOCAL_DATE parsing failed for {bad} rows.")

    pat = re.compile(r"^(MEAN_TEMPERATURE|TOTAL_PRECIPITATION)_(.+)$")
    value_cols = [c for c in df.columns if c != "LOCAL_DATE" and pat.match(c)]
    if len(value_cols) == 0:
        raise ValueError("No MEAN_TEMPERATURE_* or TOTAL_PRECIPITATION_* columns found.")

    long = df.melt(
        id_vars=["LOCAL_DATE"],
        value_vars=value_cols,
        var_name="series",
        value_name="value",
    )
    long[["metric", "city"]] = long["series"].str.extract(r"^(MEAN_TEMPERATURE|TOTAL_PRECIPITATION)_(.+)$")
    long.drop(columns=["series"], inplace=True)

    tidy = (
        long.pivot_table(index=["LOCAL_DATE", "city"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename(
            columns={
                "LOCAL_DATE": "local_date",
                "MEAN_TEMPERATURE": "mean_temperature",
                "TOTAL_PRECIPITATION": "total_precipitation",
            }
        )
    )

    tidy["city"] = tidy["city"].astype(str).str.upper().str.strip()
    tidy["local_date"] = pd.to_datetime(tidy["local_date"])
    for col in ["mean_temperature", "total_precipitation"]:
        tidy[col] = pd.to_numeric(tidy[col], errors="coerce")

    tidy["year"] = tidy["local_date"].dt.year
    tidy["month"] = tidy["local_date"].dt.month
    tidy["doy"] = tidy["local_date"].dt.dayofyear

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

    tidy.to_parquet(TIDY_PATH, index=False)
    monthly.to_parquet(MONTHLY_PATH, index=False)
    yearly.to_parquet(YEARLY_PATH, index=False)


@st.cache_data(show_spinner=False)
def load_data():
    # Ensure processed exists; rebuild if missing
    if not (TIDY_PATH.exists() and MONTHLY_PATH.exists() and YEARLY_PATH.exists()):
        raw_path = resolve_raw_csv()
        build_processed_from_raw(raw_path)

    daily = pd.read_parquet(TIDY_PATH)
    monthly = pd.read_parquet(MONTHLY_PATH)
    yearly = pd.read_parquet(YEARLY_PATH)

    daily["local_date"] = pd.to_datetime(daily["local_date"])
    daily["city"] = daily["city"].astype(str).str.upper().str.strip()
    daily["year"] = daily["local_date"].dt.year
    daily["month"] = daily["local_date"].dt.month

    return daily, monthly, yearly


def ols_slope_ci(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 15:
        return None

    x0 = x - x.mean()
    b = np.sum(x0 * (y - y.mean())) / np.sum(x0**2)
    a = y.mean() - b * x.mean()

    yhat = a + b * x
    resid = y - yhat
    s2 = np.sum(resid**2) / (n - 2)
    se_b = np.sqrt(s2 / np.sum(x0**2))

    z = 1.96
    lo, hi = b - z * se_b, b + z * se_b

    return {
        "n": n,
        "slope_per_decade": b * 10,
        "ci95_lo_decade": lo * 10,
        "ci95_hi_decade": hi * 10,
    }


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Canada Climate Dashboard")
st.sidebar.caption("1940–Present • City-level daily → monthly/yearly rollups")

with st.sidebar.expander("Data status", expanded=False):
    st.write("Repo:", str(REPO_DIR))
    st.write("RAW:", str(RAW_DIR), "exists:", RAW_DIR.exists())
    st.write("PROCESSED:", str(PROCESSED_DIR), "exists:", PROCESSED_DIR.exists())
    st.write("TIDY:", str(TIDY_PATH), "exists:", TIDY_PATH.exists())
    st.write("MONTHLY:", str(MONTHLY_PATH), "exists:", MONTHLY_PATH.exists())
    st.write("YEARLY:", str(YEARLY_PATH), "exists:", YEARLY_PATH.exists())

daily, monthly, yearly = load_data()

all_cities = sorted(daily["city"].unique())
default_city = "TORONTO" if "TORONTO" in all_cities else all_cities[0]

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Trends", "Extremes", "Maps", "Download"],
    index=0,
)

cities_selected = st.sidebar.multiselect(
    "Cities",
    options=all_cities,
    default=[default_city],
)

min_year = int(daily["year"].min())
max_year = int(daily["year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

MIN_DAYS_PER_YEAR = st.sidebar.slider("Coverage threshold (days/year)", 250, 365, 300)
BASELINE = st.sidebar.selectbox("Baseline period", ["1961–1990", "1971–2000", "1981–2010"], index=0)
RECENT = st.sidebar.selectbox("Recent period", ["1991–2020", "2001–2020", "2011–2020"], index=0)

def parse_period(s):
    a, b = s.split("–")
    return int(a), int(b)

baseline_y0, baseline_y1 = parse_period(BASELINE)
recent_y0, recent_y1 = parse_period(RECENT)


# -----------------------------
# Shared derived tables
# -----------------------------
d = daily[(daily["city"].isin(cities_selected)) & (daily["year"].between(year_range[0], year_range[1]))].copy()

cov = (
    daily.groupby(["city", "year"], as_index=False)
    .agg(n_temp=("mean_temperature", "count"), n_precip=("total_precipitation", "count"))
)
cov["temp_ok"] = cov["n_temp"] >= MIN_DAYS_PER_YEAR
cov["precip_ok"] = cov["n_precip"] >= MIN_DAYS_PER_YEAR

annual = (
    daily.groupby(["city", "year"], as_index=False)
    .agg(
        mean_temp=("mean_temperature", "mean"),
        total_precip=("total_precipitation", "sum"),
        wet_days=("total_precipitation", lambda s: int((s > 0).sum())),
        n_temp=("mean_temperature", "count"),
        n_precip=("total_precipitation", "count"),
    )
)
annual["temp_ok"] = annual["n_temp"] >= MIN_DAYS_PER_YEAR
annual["precip_ok"] = annual["n_precip"] >= MIN_DAYS_PER_YEAR
annual["wet_day_ratio"] = (annual["wet_days"] / annual["n_precip"]).replace([np.inf, -np.inf], np.nan)

# thresholds for extremes (baseline quantiles by city)
base = daily[daily["year"].between(baseline_y0, baseline_y1)].copy()
thresholds = (
    base.groupby("city")["mean_temperature"]
    .quantile([0.05, 0.95])
    .unstack()
    .rename(columns={0.05: "t_p05", 0.95: "t_p95"})
    .reset_index()
)
thresholds["lat"] = thresholds["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
thresholds["lon"] = thresholds["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])

dd = daily.merge(thresholds[["city", "t_p05", "t_p95"]], on="city", how="left")
dd["is_hot_extreme"] = dd["mean_temperature"] >= dd["t_p95"]
dd["is_cold_extreme"] = dd["mean_temperature"] <= dd["t_p05"]

ext_annual = (
    dd.groupby(["city", "year"], as_index=False)
    .agg(
        hot_days=("is_hot_extreme", "sum"),
        cold_days=("is_cold_extreme", "sum"),
        n_temp=("mean_temperature", "count"),
    )
)
ext_annual["temp_ok"] = ext_annual["n_temp"] >= MIN_DAYS_PER_YEAR


def period_means_ext(df, y0, y1):
    sub = df[(df["year"].between(y0, y1)) & (df["temp_ok"])]
    return (
        sub.groupby("city", as_index=False)
        .agg(hot_days=("hot_days", "mean"), cold_days=("cold_days", "mean"))
    )

ext_base = period_means_ext(ext_annual, baseline_y0, baseline_y1).rename(
    columns={"hot_days": "base_hot_days", "cold_days": "base_cold_days"}
)
ext_recent = period_means_ext(ext_annual, recent_y0, recent_y1).rename(
    columns={"hot_days": "recent_hot_days", "cold_days": "recent_cold_days"}
)

ext_delta = ext_base.merge(ext_recent, on="city", how="inner")
ext_delta["delta_hot_days"] = ext_delta["recent_hot_days"] - ext_delta["base_hot_days"]
ext_delta["delta_cold_days"] = ext_delta["recent_cold_days"] - ext_delta["base_cold_days"]
ext_delta["lat"] = ext_delta["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
ext_delta["lon"] = ext_delta["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])

# temp trend slopes by city
trend_rows = []
for c in sorted(daily["city"].unique()):
    tmp = annual[(annual["city"] == c) & (annual["temp_ok"])].sort_values("year")
    res = ols_slope_ci(tmp["year"], tmp["mean_temp"])
    if res:
        trend_rows.append({"city": c, **res})
temp_trends = pd.DataFrame(trend_rows)
temp_trends["lat"] = temp_trends["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
temp_trends["lon"] = temp_trends["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])


# -----------------------------
# Header
# -----------------------------
st.title("Canada Climate (1940–Present)")
st.caption("Senior-style EDA dashboard: trends, extremes, and spatial patterns across Canadian cities.")


# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Cities", f"{len(all_cities)}", "in the dataset")
    with c2:
        kpi("Date range", f"{min_year}–{max_year}", "daily coverage varies by city")
    with c3:
        kpi("Daily rows", f"{len(daily):,}", "tidy city-date observations")
    with c4:
        kpi("Baseline → Recent", f"{BASELINE} → {RECENT}", "used for extreme thresholds + deltas")

    st.markdown("### Quick preview")
    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(daily.head(30), use_container_width=True)
    with right:
        cov_rate = (
            cov.groupby("city", as_index=False)
            .agg(temp_years_ok=("temp_ok", "sum"), total_years=("year", "nunique"))
        )
        cov_rate["temp_ok_pct"] = (cov_rate["temp_years_ok"] / cov_rate["total_years"] * 100).round(1)
        fig = px.bar(
            cov_rate.sort_values("temp_ok_pct", ascending=True),
            x="temp_ok_pct",
            y="city",
            orientation="h",
            title="Coverage Quality (Temperature): % of years meeting threshold",
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What this dashboard focuses on")
    st.markdown(
        """
        - **Trends:** annual mean temperature and precipitation, including per-city OLS slopes.
        - **Extremes:** baseline-quantile thresholds (p05/p95) and how hot/cold extreme-day frequency shifts over time.
        - **Maps:** city-level dot maps for warming and extreme deltas (fast, clean, and easy to interpret).
        """
    )


elif page == "Trends":
    st.markdown("## Trends")

    metric = st.radio("Metric", ["Mean temperature (annual)", "Total precipitation (annual)", "Wet-day ratio (annual)"], horizontal=True)

    sub = annual[annual["city"].isin(cities_selected) & annual["year"].between(year_range[0], year_range[1])].copy()
    if metric == "Mean temperature (annual)":
        ycol = "mean_temp"
        ok_col = "temp_ok"
        ylab = "°C"
        title = "Annual Mean Temperature"
    elif metric == "Total precipitation (annual)":
        ycol = "total_precip"
        ok_col = "precip_ok"
        ylab = "mm (sum)"
        title = "Annual Total Precipitation"
    else:
        ycol = "wet_day_ratio"
        ok_col = "precip_ok"
        ylab = "ratio"
        title = "Wet-day ratio (wet days / observed precip days)"

    sub = sub[sub[ok_col]]

    fig = px.line(
        sub,
        x="year",
        y=ycol,
        color="city",
        markers=False,
        title=f"{title} (coverage-filtered)",
        labels={ycol: ylab, "year": "Year"},
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Trend table (OLS slope per decade)")
    if metric != "Mean temperature (annual)":
        st.info("OLS slope table is shown for temperature only in this dashboard build.")
    else:
        t = temp_trends.sort_values("slope_per_decade", ascending=False).copy()
        st.dataframe(t, use_container_width=True)

        fig2 = px.scatter(
            t,
            x="slope_per_decade",
            y="city",
            title="Warming trend by city (°C/decade)",
        )
        fig2.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, use_container_width=True)


elif page == "Extremes":
    st.markdown("## Extremes (baseline quantiles → annual counts)")
    st.caption(f"Hot extremes are days ≥ p95 of {BASELINE}; cold extremes are days ≤ p05 of {BASELINE} (per city).")

    sub = ext_annual[ext_annual["city"].isin(cities_selected) & ext_annual["year"].between(year_range[0], year_range[1])].copy()
    sub = sub[sub["temp_ok"]]

    tabs = st.tabs(["Hot extremes", "Cold extremes", "Baseline vs Recent delta"])
    with tabs[0]:
        fig = px.line(
            sub,
            x="year",
            y="hot_days",
            color="city",
            title="Hot extreme days per year (TX95p-style, simplified)",
            labels={"hot_days": "days/year"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig = px.line(
            sub,
            x="year",
            y="cold_days",
            color="city",
            title="Cold extreme days per year (TN05p-style, simplified)",
            labels={"cold_days": "days/year"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        dsub = ext_delta[ext_delta["city"].isin(cities_selected)].copy()
        if len(dsub) == 0:
            st.warning("No delta rows available (check city selection and coverage threshold).")
        else:
            fig = px.bar(
                dsub.sort_values("delta_hot_days", ascending=False),
                x="city",
                y=["delta_hot_days", "delta_cold_days"],
                barmode="group",
                title=f"Change in extreme-day frequency (Recent − Baseline): {RECENT} vs {BASELINE}",
                labels={"value": "days/year", "variable": "metric"},
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dsub, use_container_width=True)


elif page == "Maps":
    st.markdown("## Maps")
    st.caption("These are true interactive map visualizations (pan/zoom/hover), rendered as point layers.")

    map_metric = st.selectbox(
        "Map layer",
        ["Warming trend (°C/decade)", "Δ hot extremes (days/year)", "Δ cold extremes (days/year)"],
        index=0,
    )

    if map_metric == "Warming trend (°C/decade)":
        m = temp_trends.copy()
        m = m[m["city"].isin(cities_selected)]
        val = "slope_per_decade"
        title = "Warming trend (OLS slope, °C/decade)"
        label = "°C/decade"
    elif map_metric == "Δ hot extremes (days/year)":
        m = ext_delta.copy()
        m = m[m["city"].isin(cities_selected)]
        val = "delta_hot_days"
        title = f"Change in hot extreme days (Recent − Baseline): {RECENT} vs {BASELINE}"
        label = "days/year"
    else:
        m = ext_delta.copy()
        m = m[m["city"].isin(cities_selected)]
        val = "delta_cold_days"
        title = f"Change in cold extreme days (Recent − Baseline): {RECENT} vs {BASELINE}"
        label = "days/year"

    m = m.dropna(subset=["lat", "lon", val]).copy()
    if len(m) == 0:
        st.warning("No mappable rows. Check city coords, selection, and coverage threshold.")
    else:
        m["abs_size"] = np.clip(np.abs(m[val]), 0, np.nanpercentile(np.abs(m[val]), 95))
        # keep size visually stable
        m["abs_size"] = (m["abs_size"] / (m["abs_size"].max() if m["abs_size"].max() else 1)) * 40 + 12

        fig = px.scatter_mapbox(
            m,
            lat="lat",
            lon="lon",
            size="abs_size",
            color=val,
            hover_name="city",
            hover_data={val: True, "lat": False, "lon": False, "abs_size": False},
            zoom=2.4,
            height=620,
            title=title,
        )
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, t=60, b=10))
        fig.update_coloraxes(colorbar_title=label)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### What to look for")
        st.markdown(
            """
            - **Color** encodes direction and magnitude of change/trend.
            - **Marker size** scales with absolute magnitude to make strong signals pop without hiding the smaller ones.
            """
        )


elif page == "Download":
    st.markdown("## Download")
    st.caption("Export derived tables for reproducibility and GitHub artifacts.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Annual climate metrics")
        sub = annual[annual["year"].between(year_range[0], year_range[1])].copy()
        st.dataframe(sub.head(20), use_container_width=True)
        csv = sub.to_csv(index=False).encode("utf-8")
        st.download_button("Download annual_metrics.csv", data=csv, file_name="annual_metrics.csv", mime="text/csv")

    with c2:
        st.markdown("### Extreme indices (annual)")
        sub = ext_annual[ext_annual["year"].between(year_range[0], year_range[1])].copy()
        st.dataframe(sub.head(20), use_container_width=True)
        csv = sub.to_csv(index=False).encode("utf-8")
        st.download_button("Download extremes_annual.csv", data=csv, file_name="extremes_annual.csv", mime="text/csv")

    st.markdown("### Trend table")
    st.dataframe(temp_trends, use_container_width=True)
    csv = temp_trends.to_csv(index=False).encode("utf-8")
    st.download_button("Download temp_trends_ols_city.csv", data=csv, file_name="temp_trends_ols_city.csv", mime="text/csv")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div class='muted'>Notes: Trends and extremes are coverage-filtered. Extreme thresholds are city-specific and computed from the selected baseline period.</div>",
    unsafe_allow_html=True,
)
