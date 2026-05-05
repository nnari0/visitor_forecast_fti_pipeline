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


def main() -> None:
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    fv = get_feature_view(fs)

    # Chronological split via the feature view — Hopsworks materializes a
    # versioned training dataset and tracks lineage against the model.
    print(f"Creating chronological train/test split at {TRAIN_TEST_SPLIT_DATE}...")
    X_train, X_test, y_train, y_test = fv.train_test_split(
        train_end=TRAIN_TEST_SPLIT_DATE,
        test_start=TRAIN_TEST_SPLIT_DATE,
        description=f"Chronological split, cutoff={TRAIN_TEST_SPLIT_DATE}",
    )
    X_train, X_test = X_train[FEATURE_COLUMNS], X_test[FEATURE_COLUMNS]
    print(f"  Train: {len(X_train)} rows (event_time < {TRAIN_TEST_SPLIT_DATE})")
    print(f"  Test:  {len(X_test)} rows (event_time >= {TRAIN_TEST_SPLIT_DATE})")

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
