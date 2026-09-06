"""
modules/analysis.py
--------------------
Aggregation / pattern-analysis functions used by the dashboard.

Every function here is written to be "filter-safe": you pass it the CURRENT
filtered DataFrame and it returns fresh results, so the whole dashboard
recomputes automatically whenever the user changes a sidebar filter.

Every function also checks whether the columns it needs actually exist
before using them, and returns an empty DataFrame (not an error) if they
don't -- this is what lets the app "not crash if optional columns are
missing" as required by the project spec.
"""

import pandas as pd


def _has_cols(df: pd.DataFrame, cols: list) -> bool:
    return all(c in df.columns for c in cols)


def get_top_crimes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if not _has_cols(df, ["crime_type"]):
        return pd.DataFrame()
    result = (df["crime_type"].value_counts()
              .head(n).rename_axis("crime_type")
              .reset_index(name="count"))
    return result


def get_top_locations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if not _has_cols(df, ["city"]):
        return pd.DataFrame()
    result = (df["city"].value_counts()
              .head(n).rename_axis("city")
              .reset_index(name="count"))
    return result


def crime_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, ["month"]):
        return pd.DataFrame()
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    d = df.dropna(subset=["month"])
    if "month_name" in d.columns:
        result = d["month_name"].value_counts().reindex(month_order).fillna(0)
        result = result.rename_axis("month").reset_index(name="count")
    else:
        result = d["month"].value_counts().sort_index().rename_axis("month").reset_index(name="count")
    return result


def crime_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, ["year"]):
        return pd.DataFrame()
    d = df.dropna(subset=["year"])
    result = d["year"].value_counts().sort_index().rename_axis("year").reset_index(name="count")
    result["year"] = result["year"].astype(int)
    return result


def crime_by_location(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, ["city", "crime_type"]):
        return pd.DataFrame()
    result = (df.groupby("city")["crime_type"]
              .count().sort_values(ascending=False)
              .rename_axis("city").reset_index(name="count"))
    return result


def crime_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Crime frequency by hour of day (0-23)."""
    if not _has_cols(df, ["hour"]):
        return pd.DataFrame()
    d = df.dropna(subset=["hour"])
    result = d["hour"].value_counts().sort_index().rename_axis("hour").reset_index(name="count")
    result["hour"] = result["hour"].astype(int)
    return result


def crime_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, ["weekday"]):
        return pd.DataFrame()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
    d = df.dropna(subset=["weekday"])
    result = d["weekday"].value_counts().reindex(weekday_order).fillna(0)
    result = result.rename_axis("weekday").reset_index(name="count")
    return result


def crime_type_location_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of crime type vs. city -- used for a heatmap."""
    if not _has_cols(df, ["city", "crime_type"]):
        return pd.DataFrame()
    pivot = pd.crosstab(df["crime_type"], df["city"])
    return pivot


def crime_time_weekday_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of weekday vs. hour -- used for the time-pattern heatmap."""
    if not _has_cols(df, ["weekday", "hour"]):
        return pd.DataFrame()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
    d = df.dropna(subset=["weekday", "hour"])
    pivot = pd.crosstab(d["weekday"], d["hour"])
    pivot = pivot.reindex(weekday_order)
    return pivot


def identify_hotspots(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    A location is treated as a 'hotspot' when its crime COUNT is high
    relative to other locations in the current (filtered) data -- this is
    a descriptive, aggregate statistic, not a prediction about any
    individual or a claim of official/statutory 'hotspot' designation.
    """
    if not _has_cols(df, ["city"]):
        return pd.DataFrame()
    counts = df["city"].value_counts()
    top = counts.head(top_n)
    total = counts.sum()
    result = top.rename_axis("city").reset_index(name="count")
    result["share_of_total_%"] = (result["count"] / total * 100).round(2)
    return result


def summary_kpis(df: pd.DataFrame) -> dict:
    """Top-of-dashboard KPI numbers."""
    kpis = {
        "total_records": len(df),
        "num_crime_types": df["crime_type"].nunique() if "crime_type" in df.columns else 0,
        "num_locations": df["city"].nunique() if "city" in df.columns else 0,
        "most_common_crime": (df["crime_type"].mode().iloc[0]
                               if "crime_type" in df.columns and not df.empty else "N/A"),
        "highest_crime_location": (df["city"].mode().iloc[0]
                                    if "city" in df.columns and not df.empty else "N/A"),
    }
    return kpis
