"""
modules/ml_model.py
--------------------
Machine-learning component for crime-PATTERN analysis.

IMPORTANT SCOPE NOTE (please read before presenting this project):
This module analyzes and groups AGGREGATE crime patterns -- e.g. "which
combinations of location, time and crime-type frequency tend to occur
together" -- and evaluates whether crime TYPE can be statistically
associated with time/location features. It does NOT identify, profile, or
make predictions about any individual person, and it does not claim to
predict whether or where a specific future crime will occur. Any
"prediction" language below refers strictly to predicting a crime
CATEGORY or a location's crime-RISK LEVEL from historical, aggregate data.

Two models are provided:

1. K-MEANS CLUSTERING (primary / always available)
   Groups crime RECORDS into pattern clusters using frequency- and
   time-based numeric features. No labeled target is required, which is
   why clustering is the default -- most public crime CSVs don't have a
   clean "ground truth" label to predict.

2. RANDOM FOREST CLASSIFICATION (optional, only if useful)
   If the dataset has a usable target (we use crime_type as an example
   classification target: "given the time/location pattern, which crime
   category is statistically most associated?"), a Random Forest is
   trained and evaluated. This is a demonstration of supervised ML on
   crime-category association -- not a real-world deployable predictor.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


CLUSTER_LABELS = {
    0: "Low-frequency pattern",
    1: "Moderate-frequency pattern",
    2: "High-frequency pattern",
}


def _build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a numeric feature table, ONE ROW PER RECORD, describing how
    "busy" that record's location/time/crime-type context is. Clustering
    on these frequency features groups records into low/moderate/high
    activity patterns rather than clustering on raw, meaningless IDs.
    """
    feats = pd.DataFrame(index=df.index)

    if "city" in df.columns:
        city_freq = df["city"].map(df["city"].value_counts())
        feats["location_frequency"] = city_freq

    if "crime_type" in df.columns:
        crime_freq = df["crime_type"].map(df["crime_type"].value_counts())
        feats["crime_type_frequency"] = crime_freq
        feats["crime_type_encoded"] = LabelEncoder().fit_transform(df["crime_type"].astype(str))

    if "hour" in df.columns:
        feats["hour"] = df["hour"]

    if "weekday_num" in df.columns:
        feats["weekday_num"] = df["weekday_num"]

    feats = feats.dropna()
    return feats


def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 3):
    """
    Returns (df_with_clusters, metrics_dict) or (None, error_dict) if there
    isn't enough usable numeric feature data to cluster.
    """
    feats = _build_cluster_features(df)

    if feats.empty or len(feats) < n_clusters * 2:
        return None, {"error": "Not enough data / numeric features available for clustering."}

    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = model.fit_predict(X)

    result_df = df.loc[feats.index].copy()
    result_df["cluster"] = cluster_ids
    result_df["cluster_label"] = result_df["cluster"].map(
        lambda c: CLUSTER_LABELS.get(c, f"Cluster {c}")
    )

    metrics = {
        "n_clusters": n_clusters,
        "n_records_clustered": len(feats),
        "silhouette_score": round(float(silhouette_score(X, cluster_ids)), 4),
        "davies_bouldin_index": round(float(davies_bouldin_score(X, cluster_ids)), 4),
        "cluster_sizes": result_df["cluster_label"].value_counts().to_dict(),
        "features_used": list(feats.columns),
    }

    return result_df, metrics


def run_crime_type_classification(df: pd.DataFrame, min_class_count: int = 10):
    """
    Trains a Random Forest to statistically associate time/location
    features with crime_type category (an illustrative supervised-learning
    demo, not a deployable prediction system).

    Classes with too few examples (< min_class_count) are excluded so the
    train/test split and metrics are meaningful.
    """
    required = ["crime_type"]
    if not all(c in df.columns for c in required):
        return None, {"error": "crime_type column not available for classification."}

    feature_cols = [c for c in ["hour", "weekday_num", "month"] if c in df.columns]
    if "city" in df.columns:
        df = df.copy()
        df["city_encoded"] = LabelEncoder().fit_transform(df["city"].astype(str))
        feature_cols.append("city_encoded")

    if len(feature_cols) < 1:
        return None, {"error": "Not enough time/location features available for classification."}

    working = df.dropna(subset=feature_cols + ["crime_type"]).copy()

    class_counts = working["crime_type"].value_counts()
    valid_classes = class_counts[class_counts >= min_class_count].index
    working = working[working["crime_type"].isin(valid_classes)]

    if working["crime_type"].nunique() < 2 or len(working) < 30:
        return None, {"error": "Not enough labeled data per crime type to train/evaluate a classifier."}

    X = working[feature_cols]
    y = working["crime_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": sorted(y.unique().tolist()),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=sorted(y.unique())).tolist(),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.round(4))),
    }

    return model, metrics
