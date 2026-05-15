"""
Training Pipeline
=================

Steps:
1. Fetch the feature group from Hopsworks
2. Create / retrieve the feature view (query: all features + label)
3. Build the training dataset from the feature view
4. Tune & train an XGBRegressor via RandomizedSearchCV with TimeSeriesSplit
   (chronological folds, with a 7-day gap to prevent rolling-window leakage)
5. Log model + metrics + best hyperparameters to MLflow;
   also persist the model locally as a joblib file

Run with:
    python -m pipelines.training_pipeline
"""
from __future__ import annotations

import joblib
import mlflow
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from pipelines.config import (
    FG_NAME,
    FG_VERSION,
    FV_NAME,
    FV_VERSION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_LOCAL_PATH,
    MODEL_NAME,
    ROLLING_7D,
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

# ---------------------------------------------------------------------------
# Hyperparameter search configuration
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "n_estimators": randint(200, 1000),
    "max_depth": randint(3, 12),
    "learning_rate": loguniform(1e-2, 3e-1),
    "subsample": uniform(0.6, 0.4),            # 0.6 .. 1.0
    "colsample_bytree": uniform(0.6, 0.4),     # 0.6 .. 1.0
    "min_child_weight": randint(1, 10),
    "reg_alpha": loguniform(1e-3, 1.0),
    "reg_lambda": loguniform(1e-3, 1.0),
    "gamma": loguniform(1e-3, 1.0),
}
SEARCH_N_ITER = 25
SEARCH_N_SPLITS = 3
# 7-day gap between train and validation folds — prevents the 7-day rolling
# feature on a validation row from peeking into the training fold.
SEARCH_GAP = ROLLING_7D


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


def _sort_train_chronologically(X_train: pd.DataFrame, y_train: pd.Series):
    """TimeSeriesSplit requires rows in event_time order; the FV doesn't guarantee one."""
    order = X_train["event_time"].argsort().values
    return (
        X_train.iloc[order].reset_index(drop=True),
        y_train.iloc[order].reset_index(drop=True),
    )


def main() -> None:
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    fv = get_feature_view(fs)

    print(f"Creating chronological train/test split at {TRAIN_TEST_SPLIT_DATE}...")
    X_train, X_test, y_train, y_test = fv.train_test_split(
        train_end=TRAIN_TEST_SPLIT_DATE,
        test_start=TRAIN_TEST_SPLIT_DATE,
        description=f"Chronological split, cutoff={TRAIN_TEST_SPLIT_DATE}",
    )

    # Sort train rows so TimeSeriesSplit folds are chronologically meaningful
    X_train, y_train = _sort_train_chronologically(X_train, y_train)
    X_train_feat = X_train[FEATURE_COLUMNS]
    X_test_feat = X_test[FEATURE_COLUMNS]
    print(f"  Train: {len(X_train_feat)} rows (event_time < {TRAIN_TEST_SPLIT_DATE})")
    print(f"  Test:  {len(X_test_feat)} rows (event_time >= {TRAIN_TEST_SPLIT_DATE})")

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        base = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )
        tscv = TimeSeriesSplit(n_splits=SEARCH_N_SPLITS, gap=SEARCH_GAP)
        search = RandomizedSearchCV(
            base,
            param_distributions=PARAM_DIST,
            n_iter=SEARCH_N_ITER,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            refit=True,
            verbose=1,
        )

        print(f"Running RandomizedSearchCV "
              f"({SEARCH_N_ITER} candidates × {SEARCH_N_SPLITS} folds, gap={SEARCH_GAP})...")
        search.fit(X_train_feat, y_train)

        best_params = search.best_params_
        cv_best_mae = -search.best_score_
        print(f"  CV best MAE: {cv_best_mae:.3f}")
        print(f"  Best params: {best_params}")
        mlflow.log_params(best_params)
        mlflow.log_params({
            "search_n_iter": SEARCH_N_ITER,
            "search_n_splits": SEARCH_N_SPLITS,
            "search_gap": SEARCH_GAP,
            "model_type": "XGBRegressor",
        })
        mlflow.log_metric("cv_best_mae", cv_best_mae)

        model = search.best_estimator_

        print("Evaluating on hold-out test set...")
        preds = model.predict(X_test_feat)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(root_mean_squared_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "n_train": len(X_train_feat),
            "n_test": len(X_test_feat),
        }
        for k, v in metrics.items():
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        mlflow.log_metrics(metrics)

        # Persist model locally
        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS},
                    MODEL_LOCAL_PATH)
        print(f"  Model saved locally: {MODEL_LOCAL_PATH}")

        # Log model to MLflow (xgboost flavor)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train_feat.head(3),
        )
        mlflow.log_artifact(str(MODEL_LOCAL_PATH))
        print(f"  Model registered in MLflow (run_id={run.info.run_id})")

    print("\nTraining pipeline finished.")


if __name__ == "__main__":
    main()
