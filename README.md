# Visitor Forecast FTI Pipeline – MLOps Project

Forecasting the number of visitors at a fitness and wellness park in Lucerne for the next 1–3 hours at 15-minute resolution. Implemented as an FTI architecture (Feature /
Training / Inference Pipeline) using the Hopsworks Feature Store as the
central feature repository and MLflow as the model registry.

---

## 1. Use Case and Data

### What is being predicted?
The number of visitors (`visitors`) inside the location for each
upcoming 15-minute bucket — i.e. a regression problem.

### Data Source 1 – Visitor counts (historical)
A CSV file (`data/crawled_content.csv`) containing ~28,800 observations
spanning ~27 months (February 2024 – May 2026). Each row holds a
timestamp and the number of visitors present at that moment. The data was
originally crawled from a publicly visible occupancy display of the park
and is sampled at irregular intervals (~5–15 min). For this project the
samples are aggregated to 15-minute means.

The park is closed between **23:00 and 07:00** (Europe/Zurich). The
feature pipeline drops observations made during closed hours and the
inference pipeline skips horizon steps that fall outside the operating
window — both bounds are configured via `OPEN_HOUR` / `CLOSE_HOUR` in
[pipelines/config.py](pipelines/config.py).

Raw CSV (`pd.read_csv(...).head()`):

```
                          date  visitors
0  Sun, 18 Feb 2024 14:17:32 GMT       263
1  Sun, 18 Feb 2024 14:17:38 GMT       263
2  Sun, 18 Feb 2024 14:21:23 GMT       263
3  Sun, 18 Feb 2024 14:22:54 GMT       263
4  Sun, 18 Feb 2024 14:28:43 GMT       263
```


The first rows of the dataset (~5 days from 2024-02-18) are dropped
because the 7-day rolling window needs at least 24 h of history.

### Data Source 2 – Weather ([Open-Meteo](https://open-meteo.com/))
Hourly weather data for Lucerne (47.05° N, 8.31° E), upsampled to the
15-minute grid via forward fill (temperature/humidity don't change
meaningfully within an hour):
- **Historical** via `archive-api.open-meteo.com` (used for training)
- **Forecast** via `api.open-meteo.com` (used for inference)

Open-Meteo does not require an API key.

---

## 2. Features
After running the feature pipeline — the resulting feature DataFrame
that is written to Hopsworks (`build_features(...).head()`):

```
               event_time  visitors  visitors_avg_24h  visitors_avg_7d  hour  day_of_week  month  is_weekend  temperature_2m  precipitation  cloud_cover  relative_humidity_2m
2024-02-23 12:00:00+00:00       198            186.38           164.68    12            4      2           0             7.4            0.1         97.0                  78.0
2024-02-23 12:15:00+00:00       190            186.84           165.01    12            4      2           0             7.4            0.1         97.0                  78.0
2024-02-23 12:30:00+00:00       170            186.96           165.25    12            4      2           0             7.4            0.1         97.0                  78.0
2024-02-23 12:45:00+00:00       160            186.33           165.30    12            4      2           0             7.4            0.1         97.0                  78.0
2024-02-23 13:00:00+00:00       162            185.39           165.25    13            4      2           0             7.4            1.1         79.0                  70.0
```

| Feature | Type | Description |
|---|---|---|
| `visitors_avg_24h` | **aggregated** (rolling 96 steps = 24 h) | Mean visitor count over the last 24 hours |
| `visitors_avg_7d` | **aggregated** (rolling 672 steps = 7 days) | Mean over the last week — captures weekly seasonality |
| `hour` | calendar | Hour of day (0–23) |
| `day_of_week` | calendar | Day of week (0=Mon … 6=Sun) |
| `month` | calendar | Month (1–12) |
| `is_weekend` | calendar | 1 for Sat/Sun, 0 otherwise |
| `temperature_2m` | **RT** (weather) | Current temperature 2 m above ground |
| `precipitation` | **RT** (weather) | Current precipitation amount |
| `cloud_cover` | **RT** (weather) | Current cloud cover |
| `relative_humidity_2m` | **RT** (weather) | Current relative humidity |

this includes following feature types :
- **aggregated feature spanning multiple timesteps**:
  `visitors_avg_24h` and `visitors_avg_7d` are computed via
  `pandas.Series.rolling(...).mean()` over previous 15-minute values
  (96 / 672 steps respectively). A preceding `shift(1)` ensures that
  the label cannot leak into its own features.
- **real-time feature**: the weather variables are only
  available at inference time and are pulled live from Open-Meteo's
  forecast endpoint.

**Label**: `visitors` — 15-minute mean of the observed visitor count.
Buckets without any observation are dropped rather than filled, so the
model never trains on synthetic targets.

---

## 3. Model

`sklearn.ensemble.RandomForestRegressor` configured with:

- `n_estimators = 200`
- `max_depth = 12`
- `min_samples_leaf = 5`
- `random_state = 42`

Deliberately kept small — the goal is a clean FTI
pipeline, not predictive accuracy. Train and test sets are split
**chronologically** (train: everything before 2026-01-01, test: from
2026-01-01 onwards), because a random split on time-series data would
introduce leakage.

---

## 4. Architecture (FTI)

```
┌──────────────────────┐    ┌──────────────────────┐
│  CSV (visitors)      │    │  Open-Meteo Archive  │
│  ~28k raw samples    │    │  (historical         │
│                      │    │   weather, hourly)   │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           └─────────────┬─────────────┘
                         ▼
              ┌──────────────────────┐
              │  Feature Pipeline    │
              │  - 15-min aggreg.    │
              │  - Weather ffill     │
              │  - Rolling windows   │
              │  - Merge weather     │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Hopsworks Feature   │
              │  Store               │
              │  (Feature Group +    │
              │   Feature View)      │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Training Pipeline   │
              │  - Load Feature View │
              │  - Chrono. split     │
              │  - RandomForest      │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  MLflow Model        │
              │  Registry (local,    │
              │  ./mlruns/)          │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Inference Pipeline  │
              │  - Load model        │
              │  - Aggregates from FS│
              │  - Weather forecast  │
              │  - Predict           │
              └──────────────────────┘
```

### Feature Pipeline (`pipelines/feature_pipeline.py`)
1. Read CSV, parse timestamps, filter implausible values, drop observations
   outside the operating window (07:00–23:00 local)
2. Aggregate to 15-minute means
3. Fetch historical weather from the Open-Meteo Archive API and upsample
   it to the 15-minute grid via `ffill`/`bfill`
4. Join both sources on `event_time`
5. Feature engineering: rolling windows (96 / 672 steps) + calendar features
6. Insert into the Hopsworks feature group `wellnesspark_features` (v2)

### Training Pipeline (`pipelines/training_pipeline.py`)
1. Pull the feature group from Hopsworks
2. Create / load the feature view `wellnesspark_view` (v2)
3. Read training data from the feature view
4. Chronological train/test split
5. Train RandomForest, compute metrics (MAE, RMSE, R²)
6. Log model to MLflow + persist locally as joblib

### Inference Pipeline (`pipelines/inference_pipeline.py`)
1. Load model from the MLflow registry (fallback: local joblib)
2. Read the latest aggregated features from the feature store
3. Pull a live weather forecast for the next hours via Open-Meteo
4. Build feature vectors for the next 12 × 15-min steps (= 3 h), skipping
   any step that falls into the closure window (23:00–07:00 local), then
   run prediction and print the results

---

## 5. Repository Layout

```
.
├── pipelines/
│   ├── __init__.py
│   ├── config.py                # central configuration
│   ├── feature_pipeline.py      # FTI - F
│   ├── training_pipeline.py     # FTI - T
│   ├── inference_pipeline.py    # FTI - I
│   ├── hopsworks_client.py      # login wrapper
│   └── weather_api.py           # Open-Meteo wrapper
├── data/
│   └── crawled_content.csv      # historical visitor data
├── models/                      # local model artifacts
├── mlruns/                      # MLflow tracking
├── requirements.txt
├── .env.example                 # template for API keys
├── .gitignore
└── README.md
```

---

## 6. Setup and Usage

### Prerequisites
- Python ≥ 3.12 and < 3.14 (Hopsworks requirement)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- An account on [hopsworks.ai](https://www.hopsworks.ai/) (free tier is enough)

### Installation
```bash
git clone <your-repo-url>
cd visitor_forecast_fti_pipeline

# Create the virtual environment and install locked dependencies
uv sync
```

`uv sync` reads [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock), creates a `.venv/` in the project root, and installs the exact pinned versions. Prefix subsequent commands with `uv run` (e.g. `uv run python -m pipelines.feature_pipeline`) or activate the venv manually with `source .venv/bin/activate`.

### Hopsworks API key
1. Sign in on [app.hopsworks.ai](https://app.hopsworks.ai)
2. Account Settings → API Keys → create a new key with the scopes
   `featurestore`, `project`, `job`
3. Copy `.env.example` to `.env` and paste in your key:

```bash
cp .env.example .env
# then edit .env and set HOPSWORKS_API_KEY
```

### Run the pipelines (in this order)

```bash
# 1) Feature Pipeline – ingest historical data into the feature store
uv run -m pipelines.feature_pipeline

# 2) Training Pipeline – train the model and register it in MLflow
uv run -m pipelines.training_pipeline

# 3) Inference Pipeline – forecast the next few hours
uv run -m pipelines.inference_pipeline
```

### View MLflow runs (optional)
```bash
mlflow ui --backend-store-uri ./mlruns
# Browser: http://127.0.0.1:5000
```

---

## 7. Reflection and Limitations

The solution is intentionally minimal and has a few known shortcomings:

1. **Aggregated features are not recomputed live at inference time.**
   The inference pipeline reads the most recent value of
   `visitors_avg_24h` / `visitors_avg_7d` from the feature group and uses
   it as a stand-in for "now". A production pipeline would either
   re-aggregate the last 7 days of raw visitor data on demand or use an
   online feature group with continuously updated aggregates.

2. **No real live visitor data**: the CSV is a historical crawl; the
   pipeline simulates "current" aggregates via the latest stored data
   point. A production version would need to integrate the crawler into
   the feature pipeline and trigger it hourly.

3. **Weather gaps are filled with `ffill`/`bfill`** (also used to upsample
   the hourly weather feed to the 15-min visitor grid). This is robust
   for isolated missing hours but not ideal for longer outages. A
   production system would benefit from data validation using Great
   Expectations.

4. **No hyperparameter tuning.** The RandomForest uses pragmatically
   chosen values; neither cross-validation nor grid search are
   implemented.

5. **Gaps in the source CSV**: the original CSV contains visible periods
   where no observations exist for hours or days. Those 15-minute buckets
   simply do not appear in the final training set (the label is never
   forward-filled) — they don't distort the aggregates because the rolling
   windows use a `min_periods` parameter.

6. **The chronological test set covers only ~4.5 months (Jan–May 2026).**
   Seasonal effects (e.g. summer operations) are therefore not represented
   in the test split. A blocked seasonal split (one week per month, with
   buffer zones around each test block to avoid rolling-window leakage)
   would cover this but was deliberately left out to keep the pipeline
   simple.

7. **No containerization, no online serving.** Extension stages such as
   Docker, Hopsworks model deployment, or a web service were intentionally
   left out.

8. **Operating hours are treated as a fixed 07:00–23:00 window.** In
   reality, opening and closing times can shift by roughly ±1 h on
   different days, and the park is occasionally closed for a full day
   (holidays, maintenance). These variations are ignored — observations
   on fully-closed days simply appear as gaps in the source data, and
   the day-level shifts contribute a small amount of noise at the
   window edges. A production system would consume an official opening
   calendar instead of hardcoded constants.

---

## 8. References

- Open-Meteo Historical Weather API – <https://open-meteo.com/en/docs/historical-weather-api>
- Open-Meteo Forecast API – <https://open-meteo.com/en/docs>
- Hopsworks Feature Store user guide – <https://docs.hopsworks.ai/latest/user_guides/fs/>
- Hopsworks Python SDK – <https://docs.hopsworks.ai/hopsworks-api/latest/>
- MLflow Tracking – <https://mlflow.org/docs/latest/tracking.html>
