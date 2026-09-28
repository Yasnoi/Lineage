"""Explicit, credential-free run configuration; no project-specific profiles."""
from dataclasses import fields
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from lineage_agent.config.settings import ModelSettings

MODEL_KEYS = {item.name for item in fields(ModelSettings)} - {"project_root", "log_dir"}

def checked_model(values):
    if not isinstance(values, dict) or set(values) - MODEL_KEYS:
        raise ValueError("unknown model configuration fields")
    settings = ModelSettings(**values)
    url = urlsplit(settings.base_url)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("endpoint must not contain credentials or query parameters")
    return {key: getattr(settings, key) for key in sorted(MODEL_KEYS)}

def checked_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "1":
        raise ValueError("explicit versioned run configuration required")
    allowed = {"schema_version", "enabled", "source_repository", "model", "credential_env", "repeats", "limits"}
    if set(config) - allowed or config.get("enabled") is not True:
        raise ValueError("run configuration is disabled or has unknown fields")
    if not isinstance(config.get("source_repository"), str) or not config["source_repository"]:
        raise ValueError("source repository required")
    if type(config.get("repeats")) is not int or not 1 <= config["repeats"] <= 20:
        raise ValueError("explicit bounded repetitions required")
    limits = config.get("limits", {})
    maxima = {"requests": 400, "execution_seconds": 7200, "sample_seconds": 300}
    if set(limits) != set(maxima):
        raise ValueError("explicit request and execution limits required")
    for key, maximum in maxima.items():
        value = limits[key]
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("invalid execution limit")
    if type(limits["requests"]) is not int:
        raise ValueError("request limit must be an integer")
    name = config.get("credential_env")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("credential environment variable name required")
    return {**config, "model": checked_model(config["model"])}

def load_connection(config, environment=None):
    environment = os.environ if environment is None else environment
    key = environment.get(config["credential_env"], "").strip()
    if not key:
        raise ValueError("missing_credentials")
    return {"API_KEY": key, "BASE_URL": config["model"]["base_url"], "MODEL_NAME": config["model"]["model_name"]}
