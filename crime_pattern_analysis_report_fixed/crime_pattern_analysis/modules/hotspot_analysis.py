"""
modules/hotspot_analysis.py
----------------------------
Geographic / location-based hotspot analysis.

Two paths, chosen automatically based on what's in the data:

1. If latitude/longitude are present -> build data for an interactive
   scatter-map (used by app.py with plotly.express.scatter_mapbox).
2. If coordinates are NOT present -> fall back to a location-frequency
   bar chart. We never invent coordinates for a city; the spec explicitly
   forbids fabricating location data.

This module only computes/returns DataFrames -- the actual Plotly figure
objects are built in modules/visualization.py, keeping "data" and
"presentation" concerns separate.
"""

import pandas as pd


def has_coordinates(df: pd.DataFrame) -> bool:
    return "latitude" in df.columns and "longitude" in df.columns


def location_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crime count and dominant crime type per city -- used regardless of
    whether coordinates are available."""
    if "city" not in df.columns:
        return pd.DataFrame()

    counts = df.groupby("city").size().rename("crime_count")

    if "crime_type" in df.columns:
        dominant = (df.groupby("city")["crime_type"]
                    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "N/A")
                    .rename("dominant_crime_type"))
        result = pd.concat([counts, dominant], axis=1)
    else:
        result = counts.to_frame()

    result = result.sort_values("crime_count", ascending=False).reset_index()
    return result


def map_ready_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates crimes to one row per (city, lat, lon) with a count, suitable
    for plotting as bubble size on a map. Uses the MEAN reported coordinate
    per city from the actual data -- not a fabricated/looked-up coordinate.
    """
    if not has_coordinates(df) or "city" not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=["latitude", "longitude"])
    if d.empty:
        return pd.DataFrame()

    agg = d.groupby("city").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        crime_count=("city", "size"),
    ).reset_index()

    if "crime_type" in d.columns:
        dominant = (d.groupby("city")["crime_type"]
                    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "N/A")
                    .rename("dominant_crime_type"))
        agg = agg.merge(dominant, on="city", how="left")

    return agg.sort_values("crime_count", ascending=False)


def classify_risk_level(df: pd.DataFrame, low_pct: float = 33, high_pct: float = 66) -> pd.DataFrame:
    """
    Buckets each location into Low / Moderate / High relative crime-frequency
    risk based on percentile thresholds WITHIN the current filtered data.
    This is a descriptive statistical bucketing of AGGREGATE location data,
    not a prediction about any individual or event.
    """
    freq = location_frequency_table(df)
    if freq.empty:
        return freq

    low_cut = freq["crime_count"].quantile(low_pct / 100)
    high_cut = freq["crime_count"].quantile(high_pct / 100)

    def bucket(count):
        if count >= high_cut:
            return "High"
        elif count >= low_cut:
            return "Moderate"
        return "Low"

    freq["risk_level"] = freq["crime_count"].apply(bucket)
    return freq
