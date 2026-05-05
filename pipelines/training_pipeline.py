"""
Training Pipeline
=================

Steps:
1. Fetch the feature group from Hopsworks
2. Create / retrieve the feature view (query: all features + label)
3. Build the training dataset from the feature view
4. Train and evaluate a RandomForestRegressor (chronological split)
5. Log model + metrics to MLflow; also persist the model locally as a joblib file

Run with:
    python -m pipelines.training_pipeline
"""
from __future__ import annotations

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from pipelines.config import (
    FG_NAME,
    FG_VERSION,
    FV_NAME,
    FV_VERSION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_LOCAL_PATH,
    MODEL_NAME,
    TRAIN_TEST_SPLIT_DATE,
)
from pipelines.hopsworks_client import login_to_hopsworks


# Columns used as model features (not the label, not event_time)
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
LABEL_COLUMN = "visitors"


def get_feature_view(fs):
    """Creates the feature view (if needed) and returns it."""
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    query = fg.select_all()

    fv = fs.get_or_create_feature_view(
        name=FV_NAME,
        version=FV_VERSION,
        description="Feature view for visitor count forecasting",
        labels=[LABEL_COLUMN],
        query=query,
    )
    return fv


def load_training_data(fs) -> pd.DataFrame:
    """Loads features + label + event_time from the feature group."""
    print("Loading training dataset from feature group...")
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.read()
    print(f"  {len(df)} rows loaded")
    return df


def chronological_split(df: pd.DataFrame, split_date: str):
    """Chronological split — important for time series, no shuffle."""
    df = df.sort_values("event_time").reset_index(drop=True)
    cutoff = pd.Timestamp(split_date, tz="UTC")
    train = df[df["event_time"] < cutoff]
    test = df[df["event_time"] >= cutoff]
    print(f"  Train: {len(train)} rows (up to {split_date})")
    print(f"  Test:  {len(test)} rows (from {split_date})")
    return train, test


def main() -> None:
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    # Materialize the feature view so it exists for downstream consumers
    # (inference pipeline, lineage), then load training data from the FG —
    # fv.get_batch_data() strips the label column, which we need here.
    get_feature_view(fs)
    df = load_training_data(fs)

    train_df, test_df = chronological_split(df, TRAIN_TEST_SPLIT_DATE)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[LABEL_COLUMN]

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)

        print("Training RandomForestRegressor...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        print("Evaluating...")
        preds = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(root_mean_squared_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        for k, v in metrics.items():
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        mlflow.log_metrics(metrics)

        # Persist model locally
        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS},
                    MODEL_LOCAL_PATH)
        print(f"  Model saved locally: {MODEL_LOCAL_PATH}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )
        mlflow.log_artifact(str(MODEL_LOCAL_PATH))
        print(f"  Model registered in MLflow (run_id={run.info.run_id})")

    print("\nTraining pipeline finished.")


if __name__ == "__main__":
    main()
