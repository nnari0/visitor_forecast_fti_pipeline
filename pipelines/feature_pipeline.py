"""
Feature Pipeline
================

Steps:
1. Read the raw data (CSV with visitor counts), drop observations outside
   the operating window (OPEN_HOUR–CLOSE_HOUR local time) and aggregate
   to a 15-min grid
2. Fetch historical weather data from Open-Meteo (hourly, upsampled to 15-min)
3. Merge both data sources
4. Compute features:
     - aggregated features (rolling windows over multiple timesteps)
     - calendar features (hour, day of week, month, weekend)
     - RT features (weather)
5. Create / update the feature group in Hopsworks and insert the data

Run with:
    python -m pipelines.feature_pipeline
"""
from __future__ import annotations

import pandas as pd

from pipelines.config import (
    CLOSE_HOUR,
    FG_DESCRIPTION,
    FG_EVENT_TIME,
    FG_NAME,
    FG_PRIMARY_KEY,
    FG_VERSION,
    OPEN_HOUR,
    RAW_CSV,
    ROLLING_24H,
    ROLLING_7D,
    SAMPLE_FREQ,
    TIMEZONE,
    VISITORS_MAX_PLAUSIBLE,
)
from pipelines.hopsworks_client import login_to_hopsworks
from pipelines.weather_api import fetch_weather_archive


# ---------------------------------------------------------------------------
# 1) Load raw data + aggregate to a 15-min grid
# ---------------------------------------------------------------------------
def load_visitor_data() -> pd.DataFrame:
    """Loads the CSV and aggregates the raw samples to SAMPLE_FREQ buckets."""
    print(f"Reading raw data from {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, sep=";")

    # Date format: 'Sun, 18 Feb 2024 14:17:32 GMT'
    df["date"] = pd.to_datetime(df["date"], format="%a, %d %b %Y %H:%M:%S GMT", utc=True)

    # Drop very implausible values (e.g. crawler errors, spikes)
    df = df[(df["visitors"] >= 0) & (df["visitors"] <= VISITORS_MAX_PLAUSIBLE)]

    # Drop observations made while the park is closed. Operating hours are
    # defined in local time, so convert before filtering.
    local_hour = df["date"].dt.tz_convert(TIMEZONE).dt.hour
    df = df[(local_hour >= OPEN_HOUR) & (local_hour < CLOSE_HOUR)]

    df["event_time"] = df["date"].dt.floor(SAMPLE_FREQ)
    binned = (
        df.groupby("event_time", as_index=False)["visitors"]
        .mean()
    )
    binned["visitors"] = binned["visitors"].round().astype(int)

    print(f"  {len(binned)} {SAMPLE_FREQ} buckets between "
          f"{binned['event_time'].min()} and {binned['event_time'].max()}")
    return binned


# ---------------------------------------------------------------------------
# 2) Fetch weather data
# ---------------------------------------------------------------------------
def load_weather_for_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Fetches weather data from Open-Meteo for the visitor data time range."""
    print(f"Fetching weather data from Open-Meteo "
          f"({start_ts.date()} to {end_ts.date()})")
    weather = fetch_weather_archive(start_ts.date(), end_ts.date())
    print(f"  {len(weather)} hourly weather values received")
    return weather


# ---------------------------------------------------------------------------
# 3) Feature engineering
# ---------------------------------------------------------------------------
def build_features(visitors: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the final feature DataFrame.

    Main feature categories:
    - aggregated features (rolling windows): visitors_avg_24h, visitors_avg_7d
    - calendar features: hour, day_of_week, month, is_weekend
    - RT features (weather, known at inference time):
        temperature_2m, precipitation, cloud_cover, relative_humidity_2m
    - Label: visitors (= visitor count for that hour)
    """
    print("Building features...")

    # Expand to a complete SAMPLE_FREQ index so rolling windows compute consistently
    full_index = pd.date_range(
        start=visitors["event_time"].min(),
        end=visitors["event_time"].max(),
        freq=SAMPLE_FREQ,
        tz="UTC",
    )

    # Weather is hourly — upsample to SAMPLE_FREQ via ffill (temperature/humidity
    # don't change meaningfully within an hour).
    weather_upsampled = (
        weather.set_index("event_time")
        .reindex(full_index)
        .ffill()
        .bfill()
        .rename_axis("event_time")
        .reset_index()
    )

    df = (
        visitors.set_index("event_time")
        .reindex(full_index)
        .rename_axis("event_time")
        .reset_index()
        .merge(weather_upsampled, on="event_time", how="left")
    )

    # Calendar features
    df["hour"] = df["event_time"].dt.hour.astype("int32")
    df["day_of_week"] = df["event_time"].dt.dayofweek.astype("int32")  # 0=Mon, 6=Sun
    df["month"] = df["event_time"].dt.month.astype("int32")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int32")

    # ---- AGGREGATED FEATURES (rolling windows over multiple timesteps) ----
    #
    # shift(1) is required so the feature window points STRICTLY into the past —
    # otherwise the label would leak into its own features.
    df = df.sort_values("event_time").reset_index(drop=True)

    df["visitors_avg_24h"] = (
        df["visitors"].shift(1).rolling(window=ROLLING_24H, min_periods=ROLLING_24H // 4).mean()
    )
    df["visitors_avg_7d"] = (
        df["visitors"].shift(1).rolling(window=ROLLING_7D, min_periods=ROLLING_24H).mean()
    )

    # ---- Fill residual weather NaNs (defensive — gaps at the series edges) ----
    weather_cols = ["temperature_2m", "precipitation", "cloud_cover", "relative_humidity_2m"]
    for c in weather_cols:
        df[c] = df[c].astype("float64").ffill().bfill()

    # Drop rows without a label (hours without visitor observations)
    df = df.dropna(subset=["visitors"])
    df["visitors"] = df["visitors"].astype("int64")

    # Drop rows without rolling features (at the very beginning of the series)
    df = df.dropna(subset=["visitors_avg_24h", "visitors_avg_7d"])

    # Pin the column order — for consistency with the feature view
    df = df[
        [
            "event_time",
            "visitors",
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
    ].reset_index(drop=True)

    print(f"  Feature DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# 4) Write to the Hopsworks feature group
# ---------------------------------------------------------------------------
def write_to_feature_store(features: pd.DataFrame) -> None:
    """Creates the feature group (if needed) and inserts the data."""
    print("Connecting to Hopsworks...")
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    print(f"Creating / fetching feature group '{FG_NAME}' v{FG_VERSION}")
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description=FG_DESCRIPTION,
        primary_key=FG_PRIMARY_KEY,
        event_time=FG_EVENT_TIME,
        online_enabled=False,
    )

    print(f"Inserting {len(features)} rows into the feature group...")
    fg.insert(features, wait=True)
    print("  Insert complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    visitors = load_visitor_data()
    weather = load_weather_for_range(
        visitors["event_time"].min(),
        visitors["event_time"].max(),
    )
    features = build_features(visitors, weather)
    write_to_feature_store(features)
    print("\nFeature pipeline finished.")


if __name__ == "__main__":
    main()
