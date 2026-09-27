"""Explicit, trusted definitions; files are enumerated here, never discovered."""

from copy import deepcopy
import json

from benchmarks.common import ROOT


DEFINITION_FILES = ("execution.json", "sessions_context.json", "recording.json")
PRIMARY_SUITES = ("control-execution", "control-filesystem", "control-sessions",
                  "control-context", "control-recording")


def load_catalog():
    units = {}
    for filename in DEFINITION_FILES:
        data = json.loads((ROOT / "benchmarks/definitions" / filename).read_text(encoding="utf-8"))
        for item in data["units"]:
            unit = {"category": "control_logic", "definition_version": "2", "granularity": "script",
                    "execution_mode": "deterministic_stub", "timeout_seconds": 120, **item}
            unit["suite_id"] = unit["primary_suite"]
            if unit["unit_id"] in units or unit["primary_suite"] not in PRIMARY_SUITES:
                raise ValueError("invalid or duplicate trusted unit")
            ids = [group["group_id"] for group in unit["groups"]]
            if len(ids) != len(set(ids)) or not ids:
                raise ValueError("invalid group plan")
            units[unit["unit_id"]] = unit
    suites = {name: [key for key, unit in units.items() if unit["primary_suite"] == name]
              for name in PRIMARY_SUITES}
    suites["control-react"] = ["react-script"]
    suites["control-all"] = list(units)
    return units, suites


def resolve_selection(selected, units, suites):
    if not selected or any(name not in suites for name in selected):
        raise ValueError("unknown or empty suite selection")
    selected = list(dict.fromkeys(selected))
    resolved = []
    for unit_id, definition in units.items():
        selected_by = [name for name in selected if unit_id in suites[name]]
        if selected_by:
            resolved.append({**deepcopy(definition), "selected_by": selected_by})
    return resolved
