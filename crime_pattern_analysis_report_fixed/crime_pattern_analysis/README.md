# Crime Pattern Analysis Using Big Data Analytics

An interactive, beginner-friendly final-year engineering project that analyzes
historical crime records to surface **patterns** — not to predict or profile
individual people. Built with Python, Pandas, NumPy, Scikit-learn, Plotly,
and Streamlit.

> ⚠️ **About the bundled dataset**: `data/crime_data.csv` is a **small,
> randomly generated synthetic sample** created only to demonstrate the
> project's structure and column layout. It does **not** contain real crime
> statistics for any real city, state, or year. **Replace it with your own
> real dataset** (e.g. an NCRB open-data export or a state police portal
> CSV) before drawing any real conclusions.

---

## 1. What this project does

- Cleans and standardizes messy, inconsistently-named crime CSV columns
- Aggregates crime data by type, location, and time
- Detects location-based hotspots (map-based if lat/long exist, bar-chart
  based otherwise — coordinates are never invented)
- Groups crime records into pattern clusters using K-Means
- Optionally trains a Random Forest to see how well crime *category*
  statistically associates with time/location features
- Presents everything in an interactive, filterable, dark "command center"
  Streamlit dashboard

### What it deliberately does NOT do
- It does not predict that a specific person will commit a crime
- It does not claim official/statutory "hotspot" status
- It does not fabricate GPS coordinates or crime statistics
- Classification accuracy on the sample synthetic data will be modest —
  that's expected, since the sample data is random by design

---

## 2. Folder structure

```
crime_pattern_analysis/
│
├── app.py                      # Streamlit dashboard entry point
├── data/
│   └── crime_data.csv          # Sample/demo dataset (replace with real data)
│
├── modules/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Column mapping, cleaning, feature engineering
│   ├── analysis.py             # Aggregation functions (top crimes, trends, etc.)
│   ├── visualization.py        # All Plotly chart builders
│   ├── hotspot_analysis.py     # Location frequency + hotspot/risk logic
│   └── ml_model.py             # K-Means clustering + optional Random Forest
│
├── models/                     # (created at runtime if you choose to save a model)
├── requirements.txt
└── README.md
```

### Purpose of each file
| File | Purpose |
|---|---|
| `app.py` | Wires all modules together into the Streamlit UI: sidebar filters, KPI cards, charts, ML tabs. |
| `modules/data_preprocessing.py` | Maps varied raw column names to standard ones, removes duplicates, handles missing values, parses dates/times, engineers year/month/day/hour/weekday features. |
| `modules/analysis.py` | Pure aggregation functions (`get_top_crimes`, `crime_by_month`, `identify_hotspots`, etc.) that work on whatever DataFrame (filtered or not) is passed in. |
| `modules/visualization.py` | Builds every Plotly figure with a consistent dark theme; returns `None` gracefully when required data is missing. |
| `modules/hotspot_analysis.py` | Computes location-frequency tables, map-ready aggregates (only if lat/long exist), and simple Low/Moderate/High risk-level buckets. |
| `modules/ml_model.py` | K-Means clustering (primary) and an optional Random Forest crime-type classifier, with proper metrics. |

---

## 3. Dataset column mapping

You don't need your CSV to use exact column names. `data_preprocessing.py`
recognizes many common variants and maps them to standard internal names:

| Standard name | Recognized raw column names (case/space/underscore-insensitive) |
|---|---|
| `state` | State, State_UT, State/UT |
| `city` | City, District, City_District, Town, Location_Name |
| `crime_type` | Crime_Type, CrimeType, Offence, Offense, Crime_Head, Crime_Category, Type_of_Crime, Crime |
| `date` | Date, Date_of_Occurrence, Occurrence_Date, Incident_Date, Date_Reported |
| `time` | Time, Time_of_Occurrence, Occurrence_Time, Incident_Time |
| `latitude` / `longitude` | Latitude/Lat, Longitude/Lon/Lng/Long |
| `victim_age` | Victim_Age, Age, Victim_Age_Years |
| `victim_gender` | Victim_Gender, Gender, Sex, Victim_Sex |

If your dataset uses a completely different name, just add it to the
`COLUMN_ALIASES` dictionary at the top of `modules/data_preprocessing.py`.

Only `crime_type` is strictly required — every other column is optional;
the dashboard simply skips charts that need a column you don't have.

---

## 4. Installation (VS Code / Windows / macOS / Linux)

### Step 1 — Create a virtual environment
```bash
python -m venv venv
```

### Step 2 — Activate it

**Windows PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```
If you get an execution-policy error, run PowerShell as Administrator once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your dataset
Place your CSV at `data/crime_data.csv`, **or** just upload it from the
sidebar once the app is running (no file replacement needed).

### Step 5 — Run the app
```bash
streamlit run app.py
```
Streamlit will open the dashboard at `http://localhost:8501`.

---

## 5. Using the dashboard

1. **Upload** your CSV from the sidebar (or use the bundled sample).
2. Check the **Preprocessing report** expander to confirm your columns
   were recognized correctly.
3. Use the **sidebar filters** (State, City, Crime Type, Year, Month,
   Date range) — every chart and KPI recalculates automatically.
4. Scroll through **Dashboard Home → Crime Type & Location → Time-Based
   Analysis → Hotspot Analysis → Machine Learning**.
5. Expand **"View filtered dataset"** to inspect rows and **download**
   the currently filtered data as CSV.
6. In the **Machine Learning** section, adjust the number of clusters and
   read the plain-language cluster meanings; check the optional
   classification tab for a supervised-learning demo.

---

## 6. Machine learning explained simply

### K-Means Clustering (primary method)
K-Means groups records that "look similar" numerically. Here, each crime
record is described by how frequent its location is, how frequent its
crime type is, what hour it happened, and what day of week — then K-Means
groups records into clusters of similar activity level:

- **Cluster 0 → Low-frequency pattern** — happens in less busy contexts
- **Cluster 1 → Moderate-frequency pattern**
- **Cluster 2 → High-frequency pattern** — happens in the busiest contexts

This is descriptive grouping of **aggregate patterns**, not a judgment
about any person or place.

**Evaluation metrics used:**
- **Silhouette Score** (–1 to 1, higher is better) — measures how well
  separated the clusters are.
- **Davies–Bouldin Index** (lower is better) — measures cluster
  compactness relative to separation. Both are standard, appropriate
  metrics for unsupervised clustering where there's no ground-truth label.

### Random Forest Classification (optional)
If the dataset has enough examples per crime type, a Random Forest is
trained to see how well crime *category* can be statistically associated
with hour/day/month/city features. Metrics reported: **Accuracy,
Precision, Recall, F1-score, Confusion Matrix, Feature Importance** — the
standard, appropriate metrics for a multi-class classification problem.

### What a "crime hotspot" means here
A hotspot is simply a location whose crime **count** is high relative to
other locations in your **currently filtered** data — a statistical
observation about historical records, not an official designation and
not a forecast of future events.

---

## 7. Big Data Analytics: scaling to PySpark

The current implementation uses Pandas, which comfortably handles
datasets from a few thousand up to a few million rows on a typical
laptop. For **true big-data scale** (tens of millions+ rows, distributed
storage like HDFS/S3), the same architecture maps cleanly onto
**Apache Spark / PySpark**:

| Pandas (current) | PySpark (scaled version) |
|---|---|
| `pd.read_csv(path)` | `spark.read.csv(path, header=True, inferSchema=True)` |
| `df.drop_duplicates()` | `df.dropDuplicates()` |
| `df.fillna(...)` | `df.fillna({...})` |
| `df.groupby(col).size()` | `df.groupBy(col).count()` |
| `pd.crosstab(...)` | `df.groupBy(col1).pivot(col2).count()` |
| In-memory, single machine | Distributed across a cluster of machines |

### Example PySpark workflow (optional / for large datasets)
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("CrimePatternAnalysis").getOrCreate()

# 1. Ingest
df = spark.read.csv("data/crime_data.csv", header=True, inferSchema=True)

# 2. Clean
df = df.dropDuplicates()
df = df.na.fill({"State": "Unknown", "City": "Unknown"})

# 3. Transform: parse date, extract year/month
df = df.withColumn("Date", F.to_date("Date"))
df = df.withColumn("Year", F.year("Date")).withColumn("Month", F.month("Date"))

# 4. Aggregate
top_crimes = (df.groupBy("Crime_Type")
                .count()
                .orderBy(F.desc("count")))
top_crimes.show(10)

crime_by_city_year = (df.groupBy("City", "Year")
                        .count()
                        .orderBy("City", "Year"))
crime_by_city_year.show(20)

spark.stop()
```

**Why distributed processing helps at scale:** Spark partitions data
across multiple machines/cores, so operations like grouping, filtering,
and aggregation run in parallel instead of loading everything into one
machine's RAM — this is what makes "Big Data Analytics" genuinely
different from single-machine spreadsheet-style analysis once row counts
reach tens of millions.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` (or pandas/plotly/sklearn) | Activate your virtual environment, then re-run `pip install -r requirements.txt`. |
| `FileNotFoundError` for the dataset | Ensure `data/crime_data.csv` exists, or use the sidebar file uploader instead. |
| App says a required column is missing | Your CSV's `crime_type`-equivalent column wasn't recognized — check the exact header name and add it to `COLUMN_ALIASES` in `modules/data_preprocessing.py`. |
| Charts look empty / say "not available" | That's expected if your CSV lacks that particular optional column (e.g. no lat/long → map is replaced by a bar chart). |
| Date parsing errors / dates showing as blank | Confirm your date column format is consistent (e.g. `YYYY-MM-DD`); mixed formats in the same column can cause some rows to fail parsing and be excluded from time-based charts only (not deleted from the data). |
| Missing latitude/longitude | This is handled automatically — the dashboard shows a location-frequency bar chart instead of a map. |
| `streamlit: command not found` / "streamlit is not recognized" | Your virtual environment isn't activated, or Streamlit didn't install correctly — re-activate the venv and reinstall. |
| PowerShell won't activate the venv | Run PowerShell as Administrator once and execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry `venv\Scripts\Activate.ps1`. |

---

## 9. Project workflow (plain-language summary)

```
Dataset → Data Cleaning → Feature Engineering → Big Data Analytics
→ Pattern Detection → Hotspot Analysis → Machine Learning
→ Interactive Dashboard → Results
```

- **Why Big Data Analytics?** Crime datasets from real police records can
  be large and messy; the pipeline here (cleaning → transformation →
  aggregation) mirrors real analytics-engineering practice and is built
  to scale to PySpark.
- **Why machine learning?** Simple charts show *what happened*; clustering
  and classification reveal *how records group together* and *how well
  categories associate with time/location patterns* — a step beyond
  plain visualization.
- **What makes this different from a simple visualization project?**
  It includes a real preprocessing/validation pipeline that adapts to
  different column names, genuine unsupervised + supervised ML with
  proper evaluation metrics, and an explicit scale-up path to distributed
  computing — not just static charts on a fixed dataset.

---

## 10. Possible future enhancements

- Add real-time data ingestion from a live police open-data API
- Add DBSCAN or HDBSCAN for density-based hotspot detection instead of
  simple frequency counts
- Add a proper PySpark execution mode toggle for very large CSVs
- Add user authentication and role-based dashboard views
- Add trend forecasting (e.g. Prophet or ARIMA) for monthly crime counts
- Export dashboard sections as a PDF report

---

## 11. References (IEEE style)

[1] S. Kim, P. Joshi, P. S. Kalsi, and P. Taheri, "Crime Analysis Through Machine Learning," in *Proc. IEEE 9th Annual Information Technology, Electronics and Mobile Communication Conf. (IEMCON)*, Vancouver, BC, Canada, 2018, pp. 415–420.

[2] R. Iqbal, M. A. A. Murad, A. Mustapha, P. H. S. Panahy, and N. Khanahmadliravi, "An Experimental Study of Classification Algorithms for Crime Prediction," *Indian J. Science and Technology*, vol. 6, no. 3, pp. 4219–4225, 2013.

[3] S. Sivaranjani, S. Sivakumari, and M. Aasha, "Crime Prediction and Forecasting in Tamil Nadu Using Clustering Approaches," in *Proc. IEEE Int. Conf. on Emerging Technological Trends (ICETT)*, Kollam, India, 2016, pp. 1–6.

[4] A. Almanie, R. Mirza, and E. Lor, "Crime Prediction Based on Crime Types and Using Spatial and Temporal Criminal Hotspots," *Int. J. Data Mining & Knowledge Management Process*, vol. 5, no. 4, pp. 1–19, 2015.

[5] M. Zaharia, R. S. Xin, P. Wendell, et al., "Apache Spark: A Unified Engine for Big Data Processing," *Commun. ACM*, vol. 59, no. 11, pp. 56–65, Nov. 2016.

[6] J. Dean and S. Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters," *Commun. ACM*, vol. 51, no. 1, pp. 107–113, Jan. 2008.

[7] L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical Density Based Clustering," *J. Open Source Software*, vol. 2, no. 11, p. 205, 2017.

[8] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[9] W. McKinney, "Data Structures for Statistical Computing in Python," in *Proc. 9th Python in Science Conf. (SciPy)*, Austin, TX, USA, 2010, pp. 56–61.

[10] National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India, "Crime in India" statistical publications. [Online]. Available: https://ncrb.gov.in

---

## 12. Academic integrity note

This project analyzes **aggregate, historical crime patterns**. It
distinguishes clearly between:
- **Historical pattern analysis** (what happened, described statistically)
- **Statistical risk-level bucketing** (relative Low/Moderate/High labels
  for locations, based on historical counts)
- **Machine-learning clustering/classification** (grouping/associating
  records, evaluated with standard metrics)
- **Actual crime prediction or individual profiling** — which this
  project does **not** perform and does not claim to perform.


## Downloadable PDF Reports
Use **Generate PDF Analysis Report** after uploading a dataset and applying filters. The PDF contains preprocessing summary, active filters, key results, analysis tables, graphs, hotspot results, findings, and conclusion.
