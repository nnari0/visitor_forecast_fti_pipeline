"""
Thin wrapper around the Hopsworks connection.
The API key is loaded from .env (or prompted for interactively at login).
"""
from __future__ import annotations

import os

import hopsworks
from dotenv import load_dotenv


def login_to_hopsworks():
    """Logs into Hopsworks and returns the project object."""
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")

    # If HOPSWORKS_API_KEY is set, hopsworks.login() picks it up automatically.
    # Otherwise the user is prompted interactively for the key.
    if api_key:
        project = hopsworks.login(api_key_value=api_key)
    else:
        project = hopsworks.login()

    print(f"Logged into Hopsworks project: {project.name}")
    return project
