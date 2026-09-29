"""Explicit, credential-free run configuration; no project-specific profiles."""
from dataclasses import fields
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch
from lineage_agent.config.settings import ModelSettings, load_settings

MODEL_KEYS = {item.name for item in fields(ModelSettings)} - {"project_root", "log_dir"}
CONNECTION_KEYS = {"API_KEY", "BASE_URL", "MODEL_NAME"}
TRACE_ENV_KEYS = {"LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT",
                  "LANGSMITH_PROJECT", "LANGSMITH_WORKSPACE_ID"}


def configuration_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

def checked_model(values):
    if not isinstance(values, dict) or set(values) - MODEL_KEYS:
        raise ValueError("unknown model configuration fields")
    settings = ModelSettings(**values)
    url = urlsplit(settings.base_url)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("endpoint must not contain credentials or query parameters")
    return {key: getattr(settings, key) for key in sorted(MODEL_KEYS)}

def checked_config(config):
    if not isinstance(config, dict) or config.get("schema_version") not in {"1", "2"}:
        raise ValueError("explicit versioned run configuration required")
    version = config["schema_version"]
    allowed = {"schema_version", "enabled", "source_repository", "repeats", "limits"}
    allowed |= {"model", "credential_env"} if version == "1" else {
        "agent_source", "model_overrides", "units", "gate_first_sample", "tracing",
    }
    if set(config) - allowed or config.get("enabled") is not True:
        raise ValueError("run configuration is disabled or has unknown fields")
    if not isinstance(config.get("source_repository"), str) or not config["source_repository"]:
        raise ValueError("source repository required")
    if type(config.get("repeats")) is not int or not 1 <= config["repeats"] <= 20:
        raise ValueError("explicit bounded repetitions required")
    limits = config.get("limits", {})
    maxima = {"requests": 400, "execution_seconds": 7200, "sample_seconds": 1200}
    if not isinstance(limits, dict) or set(limits) != set(maxima):
        raise ValueError("explicit request and execution limits required")
    for key, maximum in maxima.items():
        value = limits[key]
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("invalid execution limit")
    if type(limits["requests"]) is not int:
        raise ValueError("request limit must be an integer")
    if version == "1":
        name = config.get("credential_env")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("credential environment variable name required")
        return {**config, "model": checked_model(config["model"])}
    if config.get("agent_source") != "project_env":
        raise ValueError("v2 agent_source must be project_env")
    overrides = config.get("model_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - MODEL_KEYS:
        raise ValueError("unknown model override fields")
    units = config.get("units")
    if (not isinstance(units, list) or not units or any(not isinstance(item, str) or not item for item in units)
            or len(set(units)) != len(units)):
        raise ValueError("explicit distinct ordered units required")
    gate = config.get("gate_first_sample", False)
    if type(gate) is not bool:
        raise ValueError("gate_first_sample must be boolean")
    tracing = config.get("tracing", {})
    if not isinstance(tracing, dict) or set(tracing) - {"enabled", "unit_ids"}:
        raise ValueError("unknown tracing configuration fields")
    enabled, traced = tracing.get("enabled", False), tracing.get("unit_ids", [])
    if (type(enabled) is not bool or not isinstance(traced, list)
            or any(not isinstance(item, str) or item not in units for item in traced)
            or len(set(traced)) != len(traced) or enabled and not traced):
        raise ValueError("tracing requires distinct selected unit IDs")
    return {**deepcopy(config), "model_overrides": deepcopy(overrides), "gate_first_sample": gate,
            "tracing": {"enabled": enabled, "unit_ids": list(traced)}}

def load_connection(config, environment=None):
    environment = os.environ if environment is None else environment
    key = environment.get(config["credential_env"], "").strip()
    if not key:
        raise ValueError("missing_credentials")
    return {"API_KEY": key, "BASE_URL": config["model"]["base_url"], "MODEL_NAME": config["model"]["model_name"]}


def resolve_run_config(config, root, environment=None):
    """Freeze production-derived settings once; return secrets separately.

    The temporary environment preserves load_settings' normal process-over-file
    precedence without leaking dotenv additions into later runs or the Judge.
    """
    config = checked_config(config)
    if config["schema_version"] == "1":
        return config, load_connection(config, environment)
    initial = dict(os.environ if environment is None else environment)
    path = Path(root).resolve() / ".env"
    with patch.dict(os.environ, initial, clear=True):
        settings = load_settings(path)
        production = checked_model({key: getattr(settings, key) for key in MODEL_KEYS})
        sources = {key: "environment" if key.upper() in initial else
                   "project_env" if key.upper() in os.environ else "default" for key in MODEL_KEYS}
        profile = checked_model({**production, **config["model_overrides"]})
        connection = load_connection({"model": profile, "credential_env": "API_KEY"})
        trace = {"enabled": config["tracing"]["enabled"], "unit_ids": config["tracing"]["unit_ids"],
                 "field_sources": {"enabled": "benchmark", "unit_ids": "benchmark"}}
        if trace["enabled"]:
            # Validate before filtering: unsupported SDK destination/disk knobs
            # must be rejected, not silently dropped to appear safe.
            from lineage_agent.recording.tracing import _read_settings
            os.environ["LANGSMITH_TRACING"] = "true"
            traced = _read_settings()
            if traced is None:
                raise ValueError("tracing configuration unavailable")
            trace.update(endpoint=traced.endpoint, project=traced.project, workspace_id=traced.workspace_id)
            trace["field_sources"].update({
                key: "environment" if name in initial else "project_env" if name in os.environ else "default"
                for key, name in (("endpoint", "LANGSMITH_ENDPOINT"), ("project", "LANGSMITH_PROJECT"),
                                  ("workspace_id", "LANGSMITH_WORKSPACE_ID"))
            })
            connection.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=traced.api_key,
                              LANGSMITH_ENDPOINT=traced.endpoint, LANGSMITH_PROJECT=traced.project,
                              LANGSMITH_WORKSPACE_ID=traced.workspace_id)
    snapshot = {"agent_source": "project_env", "path": str(path), "settings": production,
                "field_sources": sources, "model_overrides": deepcopy(config["model_overrides"]),
                "tracing": deepcopy(trace)}
    final_sources = {key: "model_override" if key in config["model_overrides"] else source
                     for key, source in sources.items()}
    return {**config, "model": profile, "tracing": trace, "source_snapshot": snapshot,
            "source_snapshot_sha256": configuration_hash(snapshot), "field_sources": final_sources}, connection
