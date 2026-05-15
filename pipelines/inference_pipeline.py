"""
Inference Pipeline
==================

Predicts the visitor count at the wellness park for the next 1–3 hours
at 15-minute resolution.

Steps:
1. Load the model from the MLflow Model Registry (or the local joblib file)
2. Fetch the aggregated features (visitors_avg_24h, visitors_avg_7d) from the
   feature store for the most recent available timestamps
3. Pull the weather forecast (RT feature) for the upcoming hours via Open-Meteo
4. Build a feature vector per 15-min step in the forecast horizon (steps
   that fall into the closure window OPEN_HOUR–CLOSE_HOUR are skipped)
5. Run prediction and print the result as a table

Run with:
    python -m pipelines.inference_pipeline
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from pipelines.config import (
    CLOSE_HOUR,
    FG_NAME,
    FG_VERSION,
    FORECAST_HORIZON_STEPS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_LOCAL_PATH,
    MODEL_NAME,
    OPEN_HOUR,
    SAMPLE_FREQ,
    STEPS_PER_HOUR,
    TIMEZONE,
)
from pipelines.hopsworks_client import login_to_hopsworks
from pipelines.weather_api import fetch_weather_forecast


FEATURE_COLUMNS = [
    "visitors_avg_24h",
    "visitors_avg_7d",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "relative_humidity_2m",
]


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model():
    """
    Tries to load the production model from the MLflow registry first; if that
    fails it falls back to the local joblib file.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Latest registered version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise RuntimeError("No model versions found in MLflow.")
        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{MODEL_NAME}/{latest.version}"
        print(f"Loading model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        return model, FEATURE_COLUMNS
    except Exception as exc:
        print(f"  MLflow load failed ({exc}). Falling back to joblib.")
        bundle = joblib.load(MODEL_LOCAL_PATH)
        return bundle["model"], bundle["feature_columns"]


# ---------------------------------------------------------------------------
# Fetch aggregated features from Hopsworks
# ---------------------------------------------------------------------------
def get_latest_aggregated_features(fs) -> dict:
    """
    Fetches the latest stored visitors_avg_24h / visitors_avg_7d from the
    feature group as an approximation of "now". A real live pipeline would
    poll the past 7 days of visitor data and re-aggregate it on the fly —
    that is out of scope for this project.
    """
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.read()
    df = df.sort_values("event_time")
    last = df.iloc[-1]
    print(f"  Latest data point: {last['event_time']} "
          f"(visitors={last['visitors']}, "
          f"avg24h={last['visitors_avg_24h']:.1f}, "
          f"avg7d={last['visitors_avg_7d']:.1f})")
    return {
        "visitors_avg_24h": float(last["visitors_avg_24h"]),
        "visitors_avg_7d": float(last["visitors_avg_7d"]),
    }


# ---------------------------------------------------------------------------
# Build feature vector per forecast hour
# ---------------------------------------------------------------------------
def build_feature_vectors(weather_fc: pd.DataFrame, agg: dict) -> pd.DataFrame:
    """Combines weather forecast (RT) + aggregated features + calendar fields."""
    now = pd.Timestamp.now(tz="UTC").floor(SAMPLE_FREQ)
    horizon = [now + pd.Timedelta(minutes=15 * i)
               for i in range(1, FORECAST_HORIZON_STEPS + 1)]

    # Skip steps that fall into the closure window — the model was never
    # trained on those hours and would emit nonsense.
    horizon = [
        ts for ts in horizon
        if OPEN_HOUR <= ts.tz_convert(TIMEZONE).hour < CLOSE_HOUR
    ]
    if not horizon:
        print("  Forecast horizon falls entirely outside operating hours — "
              "no predictions to make.")
        return pd.DataFrame()

    rows = []
    for ts in horizon:
        # Pick the weather row for the matching (or closest) hour — Open-Meteo
        # only provides hourly forecasts, so every 4 consecutive 15-min steps
        # snap to the same weather row.
        diffs = (weather_fc["event_time"] - ts).abs()
        if diffs.empty:
            raise RuntimeError("Weather forecast empty — check the Open-Meteo response.")
        wx_row = weather_fc.loc[diffs.idxmin()]

        rows.append({
            "event_time": ts,
            "visitors_avg_24h": agg["visitors_avg_24h"],
            "visitors_avg_7d": agg["visitors_avg_7d"],
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "is_weekend": int(ts.weekday() >= 5),
            "temperature_2m": float(wx_row["temperature_2m"]),
            "precipitation": float(wx_row["precipitation"]),
            "cloud_cover": float(wx_row["cloud_cover"]),
            "relative_humidity_2m": float(wx_row["relative_humidity_2m"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading model...")
    model, feature_cols = load_model()

    print("Connecting to Hopsworks...")
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    print("Fetching latest aggregated features from the feature store...")
    agg = get_latest_aggregated_features(fs)

    print("Fetching weather forecast (RT feature) from Open-Meteo...")
    horizon_hours = FORECAST_HORIZON_STEPS // STEPS_PER_HOUR + 1
    weather_fc = fetch_weather_forecast(hours_ahead=horizon_hours)
    print(f"  {len(weather_fc)} hourly weather values received")

    print("Building feature vectors for the forecast horizon...")
    fvecs = build_feature_vectors(weather_fc, agg)
    if fvecs.empty:
        return

    print("Running predictions...")
    preds = model.predict(fvecs[feature_cols])
    fvecs["predicted_visitors"] = np.round(preds).astype(int)

    print(f"\n=== Visitor count forecast — Wellnesspark Lucerne ({TIMEZONE}) ===")
    print(f"    Horizon: {FORECAST_HORIZON_STEPS} steps of {SAMPLE_FREQ}")
    display = fvecs[[
        "event_time", "predicted_visitors",
        "temperature_2m", "precipitation", "cloud_cover"
    ]].copy()
    display["event_time"] = (
        pd.to_datetime(display["event_time"], utc=True)
        .dt.tz_convert(TIMEZONE)
        .dt.strftime("%Y-%m-%d %H:%M %Z")
    )
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
