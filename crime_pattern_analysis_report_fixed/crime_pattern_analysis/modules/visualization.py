"""
modules/visualization.py
-------------------------
Builds all Plotly figures used by the Streamlit dashboard. Kept separate
from analysis.py so that "compute the numbers" and "draw the chart" are
independent, testable concerns.

Every function returns either a plotly.graph_objects Figure, or None if
the required columns/data aren't available -- app.py checks for None and
shows an informational message instead of crashing.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# White/red theme shared by every chart
LIGHT_TEMPLATE = "plotly_white"
ACCENT_COLORS = ["#dc2626", "#ef4444", "#b91c1c", "#f87171", "#991b1b",
                  "#e11d48", "#f43f5e", "#be123c", "#fb7185", "#fecaca"]


def _style(fig, height=420):
    fig.update_layout(
        template=LIGHT_TEMPLATE,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1f2937"),
        margin=dict(l=30, r=20, t=50, b=30),
        height=height,
        legend=dict(bgcolor="#ffffff"),
    )
    return fig


def crime_type_distribution_chart(top_crimes_df: pd.DataFrame):
    if top_crimes_df.empty:
        return None
    fig = px.bar(top_crimes_df, x="crime_type", y="count",
                 color="crime_type", color_discrete_sequence=ACCENT_COLORS,
                 title="Crime Distribution by Type")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of records")
    return _style(fig)


def crime_by_location_chart(top_locations_df: pd.DataFrame):
    if top_locations_df.empty:
        return None
    fig = px.bar(top_locations_df.sort_values("count"), x="count", y="city",
                 orientation="h", color="count", color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"],
                 title="Crime Count by City / Location")
    fig.update_layout(xaxis_title="Number of records", yaxis_title="")
    return _style(fig)


def crime_trend_over_time_chart(by_year_df: pd.DataFrame):
    if by_year_df.empty:
        return None
    fig = px.line(by_year_df, x="year", y="count", markers=True,
                   title="Crime Trend Over Time (Yearly)")
    fig.update_traces(line_color=ACCENT_COLORS[0], line_width=3)
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of records")
    return _style(fig)


def crime_type_location_heatmap(pivot_df: pd.DataFrame):
    if pivot_df.empty:
        return None
    fig = px.imshow(pivot_df, color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"], aspect="auto",
                     labels=dict(x="City", y="Crime Type", color="Count"),
                     title="Crime Type vs. Location")
    return _style(fig, height=460)


def crime_by_month_chart(by_month_df: pd.DataFrame):
    if by_month_df.empty:
        return None
    fig = px.bar(by_month_df, x="month", y="count", color="count",
                 color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"], title="Crime by Month")
    fig.update_layout(xaxis_title="", yaxis_title="Number of records")
    return _style(fig)


def crime_by_weekday_chart(by_weekday_df: pd.DataFrame):
    if by_weekday_df.empty:
        return None
    fig = px.bar(by_weekday_df, x="weekday", y="count", color="count",
                 color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"], title="Crime by Day of Week")
    fig.update_layout(xaxis_title="", yaxis_title="Number of records")
    return _style(fig)


def crime_by_hour_chart(by_hour_df: pd.DataFrame):
    if by_hour_df.empty:
        return None
    fig = px.area(by_hour_df, x="hour", y="count",
                   title="Crime by Hour of Day")
    fig.update_traces(line_color=ACCENT_COLORS[0], fillcolor="rgba(220,38,38,0.18)")
    fig.update_layout(xaxis_title="Hour (0-23)", yaxis_title="Number of records")
    return _style(fig)


def time_weekday_heatmap(pivot_df: pd.DataFrame):
    if pivot_df.empty:
        return None
    fig = px.imshow(pivot_df, color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"], aspect="auto",
                     labels=dict(x="Hour of Day", y="Day of Week", color="Count"),
                     title="Heatmap: Crime Frequency by Day & Time")
    return _style(fig, height=420)


def hotspot_map(map_df: pd.DataFrame):
    """Interactive bubble map -- only called when latitude/longitude exist."""
    if map_df.empty:
        return None
    fig = px.scatter_map(
        map_df, lat="latitude", lon="longitude",
        size="crime_count", color="crime_count",
        hover_name="city",
        hover_data={"crime_count": True, "latitude": False, "longitude": False},
        color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"], size_max=40, zoom=3.5,
        title="Crime Hotspot Map",
    )
    fig.update_layout(map_style="carto-positron")
    return _style(fig, height=520)


def hotspot_bar_fallback(freq_df: pd.DataFrame):
    """Used instead of a map when no coordinates are available."""
    if freq_df.empty:
        return None
    fig = px.bar(freq_df.head(15).sort_values("crime_count"),
                 x="crime_count", y="city", orientation="h",
                 color="crime_count", color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"],
                 title="Top Locations by Crime Frequency (no coordinates available)")
    fig.update_layout(xaxis_title="Number of records", yaxis_title="")
    return _style(fig)


def cluster_scatter(clustered_df: pd.DataFrame):
    """
    2D visualization of clusters using location_frequency vs. crime_type_frequency
    (falls back gracefully if those exact columns aren't present).
    """
    if clustered_df is None or clustered_df.empty:
        return None
    x_col = "location_frequency" if "location_frequency" in clustered_df.columns else None
    y_col = "crime_type_frequency" if "crime_type_frequency" in clustered_df.columns else None
    if x_col is None or y_col is None:
        return None
    fig = px.scatter(clustered_df, x=x_col, y=y_col, color="cluster_label",
                      color_discrete_sequence=ACCENT_COLORS,
                      title="Crime-Pattern Clusters (K-Means)",
                      opacity=0.75)
    return _style(fig)


def confusion_matrix_chart(cm, class_labels):
    fig = px.imshow(cm, x=class_labels, y=class_labels,
                     color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"], text_auto=True,
                     labels=dict(x="Predicted", y="Actual", color="Count"),
                     title="Confusion Matrix (Crime-Type Classification)")
    return _style(fig, height=460)


def feature_importance_chart(importance_dict: dict):
    if not importance_dict:
        return None
    imp_df = pd.DataFrame(list(importance_dict.items()),
                           columns=["feature", "importance"]).sort_values("importance")
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#fff5f5", "#fecaca", "#ef4444", "#b91c1c", "#7f1d1d"],
                 title="Feature Importance (Random Forest)")
    return _style(fig, height=320)
