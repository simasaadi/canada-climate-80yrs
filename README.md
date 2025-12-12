🇨🇦 Canada Climate (1940–2020): Long-Run Trends, Extremes, and Spatial Patterns

An interactive climate analytics project examining 80 years of city-level climate data across Canada, with a focus on long-run temperature trends, seasonality shifts, extremes, and spatial patterns.
The project combines rigorous climate metrics with modern interactive visualization to support exploratory analysis and communication.

## 🚀 Live Interactive Dashboard

👉 **[Open the Streamlit Dashboard](https://canada-climate-80yrs-6u55vcsxoekyqoq4jh4mzk.streamlit.app/)**


The dashboard allows users to:

Select cities and time ranges (1940–2020)

Compare baseline climate normals (1961–1990 vs 1991–2020)

Explore trends, extremes, distributions, and spatial patterns interactively

📊 What This Project Shows
1. Long-Run Temperature Trends

City-level warming trends estimated over multiple decades

Coverage-aware aggregation to avoid biased annual estimates

Clear comparison across major Canadian cities

2. Seasonality and Climate Normals

Monthly temperature normals by city

Direct comparison between historical and recent baseline periods

Visual evidence of systematic warming across seasons, not just annual means

3. Distribution Shifts (Beyond Averages)

Full temperature distributions for selected periods

Visualization of rightward shifts and tail behavior

Highlights how warming affects the entire temperature profile, not only extremes

4. Climate Extremes

Hot and cold extreme day counts using percentile-based thresholds

Consistent baseline definition (1961–1990)

Annual aggregation suitable for climate diagnostics

5. Spatial Patterns

Georeferenced city-level visualizations

Scaled and layered representations of warming trends

Designed to complement, not replace, analytical charts

🧠 Why This Is Different

This project is intentionally analysis-first, not map-first.

Visualizations are chosen to answer climate questions, not just look impressive

Distributional change and seasonality are emphasized over single summary metrics

Spatial views are used where geography adds insight, not noise

The result is a dashboard that aligns with how climate scientists, policy analysts, and applied researchers actually reason about warming.

🗂️ Repository Structure
├── data/
│   ├── raw/                 # Original daily climate records
│   └── processed/           # Coverage-aware monthly & yearly rollups
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_trends_analysis.ipynb
│   ├── 03_seasonality_normals.ipynb
│   ├── 04_extremes_indices.ipynb
│   └── 05_spatial_patterns.ipynb
├── app.py                   # Streamlit dashboard application
├── requirements.txt
└── README.md

🛠️ Tools & Methods

Python: pandas, numpy, scipy

Visualization: Plotly, Matplotlib, Streamlit

Spatial: city-level georeferencing (lat/lon)

Methods:

Coverage-aware rollups

Percentile-based extreme indices

Baseline-consistent climate normals

Distributional analysis (not only means)

📌 Intended Use

This project is suitable for:

Climate and environmental data portfolios

Policy-oriented climate analysis

Urban and regional climate comparisons

Demonstrating applied data science with real-world longitudinal data
