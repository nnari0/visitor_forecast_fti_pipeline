"""
Open-Meteo wrapper — fetches weather data without needing an API key.

Two modes:
  - fetch_weather_archive(): historical hourly data (for training)
  - fetch_weather_forecast(): forecast for the upcoming hours (for inference)

Both return a DataFrame with the columns:
    event_time (UTC), temperature_2m, precipitation, cloud_cover, relative_humidity_2m
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from pipelines.config import (
    LATITUDE,
    LONGITUDE,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    WEATHER_VARS,
)


def _request_with_retry(url: str, params: dict, retries: int = 3) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo request failed: {last_exc}")


def _hourly_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["event_time"] + WEATHER_VARS)

    df = pd.DataFrame(hourly)
    # The payload's time field is in UTC. We parse it as UTC-aware so the
    # merge with the visitor data (also UTC) works cleanly.
    df["event_time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"])

    # Make sure all expected columns exist
    for col in WEATHER_VARS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[["event_time"] + WEATHER_VARS]


def fetch_weather_archive(start: date, end: date) -> pd.DataFrame:
    """Fetches hourly historical weather data for Lucerne between start and end (inclusive)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
    }
    payload = _request_with_retry(OPEN_METEO_ARCHIVE_URL, params)
    return _hourly_to_df(payload)


def fetch_weather_forecast(hours_ahead: int = 6) -> pd.DataFrame:
    """
    Fetches current and future weather data (for inference).
    Returns at least hours_ahead hours into the future.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_VARS),
        "forecast_days": 2,        # today + tomorrow, more than enough
        "past_days": 1,            # in case we also need the last few hours
        "timezone": "UTC",
    }
    payload = _request_with_retry(OPEN_METEO_FORECAST_URL, params)
    df = _hourly_to_df(payload)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cutoff_end = now + timedelta(hours=hours_ahead + 1)
    df = df[(df["event_time"] >= now - timedelta(hours=1)) & (df["event_time"] <= cutoff_end)]
    return df.reset_index(drop=True)
