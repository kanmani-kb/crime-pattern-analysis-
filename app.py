"""
app.py
------
Main Streamlit entry point for the Crime Pattern Analysis dashboard.

Run with:
    streamlit run app.py

This file is intentionally "thin": it wires together the modules in
modules/ (preprocessing, analysis, hotspot_analysis, visualization,
ml_model) and lays out the UI. Business logic lives in the modules so it
can be tested independently of Streamlit.
"""

import os
import io
import pandas as pd
import streamlit as st

from modules.data_preprocessing import load_and_clean_data
from modules import analysis as A
from modules import hotspot_analysis as H
from modules import visualization as V
from modules import ml_model as M
from modules.report_generator import generate_pdf_report

DEFAULT_DATA_PATH = os.path.join("data", "crime_data.csv")


# ---------------------------------------------------------------------------
# Page config + "command center" dark styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crime Pattern Analysis | Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        color: #1f2937;
    }
    section[data-testid="stSidebar"] {
        background: #fff5f5;
        border-right: 1px solid #fecaca;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(185, 28, 28, 0.08);
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #b91c1c;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }
    h1, h2, h3, h4 {
        color: #991b1b !important;
    }
    p, label, .stMarkdown, [data-testid="stMetricValue"] {
        color: #1f2937;
    }
    .disclaimer-box {
        background: #fff1f2;
        border-left: 4px solid #dc2626;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #7f1d1d;
    }
    .stButton > button {
        background: #dc2626;
        color: #ffffff;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background: #b91c1c;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Crime Pattern Analysis — Big Data Analytics Command Center")
st.caption(
    "Academic project | Aggregate historical crime-pattern analysis, "
    "hotspot detection, and pattern clustering. Not a predictive-policing "
    "or individual-risk-scoring tool."
)


# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a crime CSV file", type=["csv"])

data_source = uploaded_file if uploaded_file is not None else DEFAULT_DATA_PATH

if uploaded_file is None:
    st.sidebar.info(
        "No file uploaded — using the sample dataset at `data/crime_data.csv`.\n\n"
        "⚠️ This sample dataset is SYNTHETIC and randomly generated for "
        "demonstration only. Replace it with a real dataset for real analysis."
    )

try:
    df, info = load_and_clean_data(data_source)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading the dataset: {e}")
    st.stop()

for w in info["warnings"]:
    st.sidebar.warning(w)

with st.sidebar.expander("🔍 Preprocessing report"):
    st.write(f"Raw shape: {info['raw_shape']}")
    st.write(f"Clean shape: {info['clean_shape']}")
    st.write(f"Raw columns: {info['raw_columns']}")


# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ Filters")
filtered_df = df.copy()

if "state" in df.columns:
    states = sorted(df["state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("State", states, default=[])
    if selected_states:
        filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]

if "city" in df.columns:
    cities = sorted(filtered_df["city"].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect("City / District", cities, default=[])
    if selected_cities:
        filtered_df = filtered_df[filtered_df["city"].isin(selected_cities)]

if "crime_type" in df.columns:
    crime_types = sorted(df["crime_type"].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect("Crime Type", crime_types, default=[])
    if selected_types:
        filtered_df = filtered_df[filtered_df["crime_type"].isin(selected_types)]

if "year" in df.columns:
    years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
    if years:
        selected_years = st.sidebar.multiselect("Year", years, default=[])
        if selected_years:
            filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]

if "month_name" in df.columns:
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    available_months = [m for m in months if m in df["month_name"].unique()]
    selected_months = st.sidebar.multiselect("Month", available_months, default=[])
    if selected_months:
        filtered_df = filtered_df[filtered_df["month_name"].isin(selected_months)]

if "date" in df.columns and df["date"].notna().any():
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                        min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered_df = filtered_df[(filtered_df["date"].isna()) |
                                   ((filtered_df["date"] >= start) & (filtered_df["date"] <= end))]

if filtered_df.empty:
    st.warning("No records match the selected filters. Try widening your filter selection.")
    st.stop()


# ---------------------------------------------------------------------------
# Home — KPI cards
# ---------------------------------------------------------------------------
st.markdown("## 📊 Dashboard Home")
kpis = A.summary_kpis(filtered_df)

cols = st.columns(5)
kpi_items = [
    ("Total Records", kpis["total_records"]),
    ("Crime Types", kpis["num_crime_types"]),
    ("Locations", kpis["num_locations"]),
    ("Most Common Crime", kpis["most_common_crime"]),
    ("Highest Crime Location", kpis["highest_crime_location"]),
]
for col, (label, value) in zip(cols, kpi_items):
    col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")

with st.expander("📄 View filtered dataset"):
    st.dataframe(filtered_df, use_container_width=True, height=280)
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Download filtered data as CSV", csv_buffer.getvalue(),
                        file_name="filtered_crime_data.csv", mime="text/csv")

st.markdown("---")


# ---------------------------------------------------------------------------
# Crime type & location analysis
# ---------------------------------------------------------------------------
st.markdown("## 🔎 Crime Type & Location Analysis")
c1, c2 = st.columns(2)

top_crimes = A.get_top_crimes(filtered_df, 10)
fig = V.crime_type_distribution_chart(top_crimes)
if fig:
    c1.plotly_chart(fig, use_container_width=True)
else:
    c1.info("Crime type column not available.")

top_locations = A.get_top_locations(filtered_df, 10)
fig = V.crime_by_location_chart(top_locations)
if fig:
    c2.plotly_chart(fig, use_container_width=True)
else:
    c2.info("City/location column not available.")

pivot = A.crime_type_location_analysis(filtered_df)
fig = V.crime_type_location_heatmap(pivot)
if fig:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Time-based analysis
# ---------------------------------------------------------------------------
st.markdown("## ⏱️ Time-Based Analysis")
c1, c2 = st.columns(2)

by_year = A.crime_by_year(filtered_df)
fig = V.crime_trend_over_time_chart(by_year)
if fig:
    c1.plotly_chart(fig, use_container_width=True)
else:
    c1.info("Year data not available.")

by_month = A.crime_by_month(filtered_df)
fig = V.crime_by_month_chart(by_month)
if fig:
    c2.plotly_chart(fig, use_container_width=True)
else:
    c2.info("Month data not available.")

c3, c4 = st.columns(2)
by_weekday = A.crime_by_weekday(filtered_df)
fig = V.crime_by_weekday_chart(by_weekday)
if fig:
    c3.plotly_chart(fig, use_container_width=True)
else:
    c3.info("Day-of-week data not available.")

by_hour = A.crime_by_time(filtered_df)
fig = V.crime_by_hour_chart(by_hour)
if fig:
    c4.plotly_chart(fig, use_container_width=True)
else:
    c4.info("Hour/time data not available.")

heatmap_pivot = A.crime_time_weekday_heatmap(filtered_df)
fig = V.time_weekday_heatmap(heatmap_pivot)
if fig:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Hotspot analysis
# ---------------------------------------------------------------------------
st.markdown("## 📍 Crime Hotspot Analysis")
st.markdown(
    '<div class="disclaimer-box">"Hotspot" here means a location with a '
    'relatively high crime COUNT in the current filtered data — a '
    'descriptive statistic, not an official designation or a forecast.</div>',
    unsafe_allow_html=True
)
st.write("")

if H.has_coordinates(filtered_df):
    map_data = H.map_ready_data(filtered_df)
    fig = V.hotspot_map(map_data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough valid coordinate data to render the map.")
else:
    freq_table = H.location_frequency_table(filtered_df)
    fig = V.hotspot_bar_fallback(freq_table)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("City/location column not available for hotspot analysis.")

hc1, hc2 = st.columns(2)
with hc1:
    st.markdown("#### Top High-Risk Locations")
    hotspots = A.identify_hotspots(filtered_df, top_n=10)
    if not hotspots.empty:
        st.dataframe(hotspots, use_container_width=True)
    else:
        st.info("City/location column not available.")

with hc2:
    st.markdown("#### Location Risk-Level Buckets")
    risk_df = H.classify_risk_level(filtered_df)
    if not risk_df.empty:
        st.dataframe(risk_df, use_container_width=True)
    else:
        st.info("City/location column not available.")

st.markdown("---")


# ---------------------------------------------------------------------------
# Machine Learning: clustering + optional classification
# ---------------------------------------------------------------------------
st.markdown("## 🤖 Machine Learning: Crime-Pattern Analysis")
st.markdown(
    '<div class="disclaimer-box">These models group and statistically '
    'associate AGGREGATE crime records by time/location/type frequency. '
    'They do not identify, profile, or predict the behavior of any '
    'individual person.</div>',
    unsafe_allow_html=True
)
st.write("")

tab1, tab2 = st.tabs(["K-Means Clustering", "Crime-Type Classification (optional)"])

with tab1:
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3)
    clustered_df, cluster_metrics = M.run_kmeans_clustering(filtered_df, n_clusters=n_clusters)

    if clustered_df is None:
        st.info(cluster_metrics.get("error", "Clustering not available for this data."))
    else:
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Silhouette Score", cluster_metrics["silhouette_score"],
                      help="Ranges -1 to 1; higher means better-separated clusters.")
        mcol2.metric("Davies-Bouldin Index", cluster_metrics["davies_bouldin_index"],
                      help="Lower is better; measures cluster compactness vs. separation.")
        mcol3.metric("Records Clustered", cluster_metrics["n_records_clustered"])

        fig = V.cluster_scatter(clustered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster meaning (simple terms):**")
        for label, size in cluster_metrics["cluster_sizes"].items():
            st.write(f"- **{label}**: {size} records")
        st.caption(f"Features used for clustering: {', '.join(cluster_metrics['features_used'])}")

with tab2:
    st.caption(
        "This trains a Random Forest to see how well crime CATEGORY can be "
        "statistically associated with time/location patterns in this "
        "dataset. Accuracy will be modest on random/synthetic data — that's "
        "expected and honest, not a bug."
    )
    model, cls_metrics = M.run_crime_type_classification(filtered_df)

    if model is None:
        st.info(cls_metrics.get("error", "Classification not available for this data."))
    else:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Accuracy", cls_metrics["accuracy"])
        mcol2.metric("Precision (weighted)", cls_metrics["precision_weighted"])
        mcol3.metric("Recall (weighted)", cls_metrics["recall_weighted"])
        mcol4.metric("F1 (weighted)", cls_metrics["f1_weighted"])

        cm_fig = V.confusion_matrix_chart(cls_metrics["confusion_matrix"], cls_metrics["classes"])
        st.plotly_chart(cm_fig, use_container_width=True)

        fi_fig = V.feature_importance_chart(cls_metrics["feature_importance"])
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)

st.markdown("---")

# Downloadable analysis report
st.markdown("## 📄 Download Analysis Report")
st.write("Generate a PDF report for the current dataset and active filters. The report includes analysis results, findings, tables, and graphs.")
filter_summary = {}
if "state" in df.columns: filter_summary["State"] = ", ".join(selected_states) if selected_states else "All"
if "city" in df.columns: filter_summary["City / District"] = ", ".join(selected_cities) if selected_cities else "All"
if "crime_type" in df.columns: filter_summary["Crime Type"] = ", ".join(selected_types) if selected_types else "All"
if "year" in df.columns: filter_summary["Year"] = ", ".join(map(str, selected_years)) if selected_years else "All"
if "month_name" in df.columns: filter_summary["Month"] = ", ".join(selected_months) if selected_months else "All"
if st.button("📑 Generate PDF Analysis Report", type="primary"):
    with st.spinner("Generating your analysis report and graphs..."):
        try:
            pdf_bytes = generate_pdf_report(info, filtered_df, top_crimes, top_locations, by_year, by_month, by_weekday, by_hour, hotspots, risk_df, filter_summary)
            st.success("Report generated successfully.")
            st.download_button("⬇️ Download Crime Analysis Report (PDF)", data=pdf_bytes, file_name="crime_pattern_analysis_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Could not generate the report: {e}")

st.markdown("---")
st.caption(
    "Crime Pattern Analysis Using Big Data Analytics — academic demonstration project. "
    "Historical pattern analysis ≠ statistical risk analysis ≠ ML clustering ≠ real-world crime prediction."
)
