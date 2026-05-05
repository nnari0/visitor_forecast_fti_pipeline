"""
Central project configuration.
Values are deliberately not exposed via environment variables to keep the
project easy to follow. Only the Hopsworks API secret comes from .env.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "crawled_content.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Location: Lucerne
# ---------------------------------------------------------------------------
LOCATION_NAME = "Lucerne"
LATITUDE = 47.0502
LONGITUDE = 8.3093
TIMEZONE = "Europe/Zurich"

# ---------------------------------------------------------------------------
# Open-Meteo API endpoints
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Weather variables we use (RT features)
WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "relative_humidity_2m",
]

# ---------------------------------------------------------------------------
# Hopsworks Feature Store: naming
# ---------------------------------------------------------------------------
FG_NAME = "wellnesspark_features"
FG_VERSION = 1
FG_DESCRIPTION = (
    "Hourly aggregated visitor counts for a wellness/fitness park in Lucerne, "
    "joined with weather data and derived aggregated features."
)
FG_PRIMARY_KEY = ["event_time"]
FG_EVENT_TIME = "event_time"

FV_NAME = "wellnesspark_view"
FV_VERSION = 1

# ---------------------------------------------------------------------------
# MLflow / model
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}"
MLFLOW_EXPERIMENT_NAME = "wellnesspark_visitor_forecast"
MODEL_NAME = "wellnesspark_visitor_rf"
MODEL_LOCAL_PATH = MODELS_DIR / "model.joblib"

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
# Train: everything up to this date
# Test:  everything from this date onwards (chronological split — important for time series!)
TRAIN_TEST_SPLIT_DATE = "2026-01-01"

# Data quality threshold: hourly values above this are treated as outliers
VISITORS_MAX_PLAUSIBLE = 1000

# Forecast horizon (hours into the future)
FORECAST_HORIZON_HOURS = 2
