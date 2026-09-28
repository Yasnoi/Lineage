"""Frozen engineering contract selection, independent of project materials."""
from copy import deepcopy
import json
from pathlib import Path

def load_catalog():
    definitions = json.loads(Path(__file__).with_name("contracts.json").read_text(encoding="utf-8"))
    units = {}
    for value in definitions["units"]:
        unit = {"suite_id": "agent-contracts", "category": "control_logic", "definition_version": "3",
                "granularity": "contract_chain", "execution_mode": "deterministic_stub", "timeout_seconds": 120, **value}
        if unit["unit_id"] in units:
            raise ValueError("duplicate contract chain")
        units[unit["unit_id"]] = unit
    return units, {"agent-contracts": list(units)}

def definition():
    units, _ = load_catalog()
    return {"suite_id": "agent-contracts", "version": "3", "title": "Agent 工程约定",
            "kind": "engineering", "executor": "contracts", "network": False,
            "units": list(deepcopy(units).values())}
