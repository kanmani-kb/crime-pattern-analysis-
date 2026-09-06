"""
modules/data_preprocessing.py
------------------------------
Responsible for the "Big Data Analytics" ingestion & cleaning stage:

    Raw CSV -> Column standardization -> Cleaning -> Feature engineering
    -> Clean, analysis-ready DataFrame

This module is deliberately defensive: real-world Indian crime datasets
(NCRB-style exports, state police open-data portals, Kaggle datasets, etc.)
use inconsistent column names. Instead of hard-coding one dataset's schema,
we map many possible raw column names onto a small set of STANDARD names
that the rest of the app relies on. Any column that isn't found is simply
skipped -- the app degrades gracefully instead of crashing.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Column standardization
# ---------------------------------------------------------------------------

# Maps STANDARD_NAME -> list of raw column names it might appear as.
# Matching is case-insensitive and ignores spaces/underscores.
COLUMN_ALIASES = {
    "state": ["state", "state_ut", "state/ut"],
    "city": ["city", "district", "city_district", "town", "location_name"],
    "crime_type": ["crime_type", "crimetype", "offence", "offense",
                   "crime_head", "crime_category", "type_of_crime", "crime"],
    "date": ["date", "date_of_occurrence", "occurrence_date", "incident_date",
             "date_reported"],
    "time": ["time", "time_of_occurrence", "occurrence_time", "incident_time"],
    "year": ["year"],
    "month": ["month"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "victim_age": ["victim_age", "age", "victim_age_years"],
    "victim_gender": ["victim_gender", "gender", "sex", "victim_sex"],
}

REQUIRED_STANDARD_COLUMNS = ["crime_type"]  # the one column we truly need


def _normalize(name: str) -> str:
    """Lowercase and strip spaces/underscores for loose matching."""
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename whatever columns exist in `df` onto our standard names, based on
    COLUMN_ALIASES. Columns that don't match anything are left untouched
    (and simply ignored by later analysis functions).
    """
    normalized_lookup = {_normalize(c): c for c in df.columns}
    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize(alias)
            if key in normalized_lookup:
                original_col = normalized_lookup[key]
                rename_map[original_col] = standard_name
                break  # stop at first match for this standard name

    df = df.rename(columns=rename_map)
    return df


def validate_columns(df: pd.DataFrame) -> list:
    """
    Returns a list of human-readable warning strings for missing optional
    columns, and raises a ValueError if a truly required column is absent.
    """
    warnings = []
    missing_required = [c for c in REQUIRED_STANDARD_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"The dataset is missing required column(s): {missing_required}. "
            f"Please check your CSV headers, or update COLUMN_ALIASES in "
            f"modules/data_preprocessing.py to match your dataset."
        )

    optional_useful = ["state", "city", "date", "time", "latitude", "longitude"]
    for col in optional_useful:
        if col not in df.columns:
            warnings.append(
                f"Column '{col}' not found -- related charts/features will be skipped."
            )
    return warnings


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[preprocessing] Removed {removed} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing-value strategy (kept simple and transparent for an academic project):
      - crime_type missing        -> row dropped (crime type is the core field)
      - state / city missing      -> filled with 'Unknown'
      - date missing               -> row kept, but excluded from time-based charts
      - latitude/longitude missing -> row kept, but excluded from map view
      - victim_age missing         -> filled with the column median
      - victim_gender missing      -> filled with 'Unknown'
    """
    if "crime_type" in df.columns:
        df = df.dropna(subset=["crime_type"]).reset_index(drop=True)

    for col in ["state", "city", "victim_gender"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "victim_age" in df.columns:
        df["victim_age"] = pd.to_numeric(df["victim_age"], errors="coerce")
        median_age = df["victim_age"].median()
        df["victim_age"] = df["victim_age"].fillna(median_age)

    return df


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize casing for text categorical columns."""
    for col in ["state", "city", "crime_type", "victim_gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df


# ---------------------------------------------------------------------------
# 3. Date/time feature engineering
# ---------------------------------------------------------------------------

def engineer_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses `date` (and `time` if present) into a real datetime, then derives
    year / month / month_name / day / hour / day_of_week / weekday_name.
    Rows with unparseable dates keep NaT and are simply excluded from
    date-based aggregations later (they are NOT dropped from the dataset).
    """
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

        # If year/month weren't already provided as separate columns, derive them
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month

        df["month_name"] = df["date"].dt.month_name()
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.day_name()
        df["weekday_num"] = df["date"].dt.dayofweek  # 0 = Monday

    if "time" in df.columns:
        parsed_time = pd.to_datetime(df["time"].astype(str), errors="coerce",
                                      format="mixed")
        df["hour"] = parsed_time.dt.hour

    return df


def encode_crime_type(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a numeric-encoded version of crime_type for use by ML models."""
    if "crime_type" in df.columns:
        df["crime_type_code"] = df["crime_type"].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------

def load_and_clean_data(file_path_or_buffer):
    """
    Full pipeline: load CSV -> standardize columns -> validate -> clean ->
    engineer features -> return (clean_df, info_dict).

    `info_dict` carries shape/columns/warnings so the Streamlit UI can show
    the user exactly what happened during preprocessing.
    """
    info = {"warnings": [], "errors": []}

    try:
        raw_df = pd.read_csv(file_path_or_buffer)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find the dataset file. Make sure your CSV is placed "
            "at data/crime_data.csv, or upload one from the sidebar."
        )
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty.")
    except Exception as e:
        raise ValueError(f"Could not read the CSV file: {e}")

    info["raw_shape"] = raw_df.shape
    info["raw_columns"] = list(raw_df.columns)

    df = standardize_columns(raw_df)
    warnings = validate_columns(df)
    info["warnings"].extend(warnings)

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_categoricals(df)
    df = engineer_datetime_features(df)
    df = encode_crime_type(df)

    info["clean_shape"] = df.shape
    info["clean_columns"] = list(df.columns)

    return df, info
