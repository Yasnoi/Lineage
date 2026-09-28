"""Explicit trusted registry. Importing or listing never starts a trial."""
from importlib import import_module

SUITES = {
    "agent-contracts": "benchmarks.suites.agent_contracts.suite:definition",
    "wechat-reporting": "benchmarks.suites.wechat_reporting.suite:definition",
}
EXECUTORS = {
    "contracts": "benchmarks.core.executors.control",
    "agent": "benchmarks.core.executors.agent",
}

def load_suite(identifier, *, registry=None):
    registry = SUITES if registry is None else registry
    if identifier not in registry:
        raise ValueError("unknown suite")
    loader = registry[identifier]
    if isinstance(loader, str):
        module, symbol = loader.split(":")
        loader = getattr(import_module(module), symbol)
    suite = loader()
    if (suite.get("suite_id") != identifier or suite.get("executor") not in EXECUTORS
            or not suite.get("version") or not suite.get("units")):
        raise ValueError("invalid suite definition")
    ids = [unit["unit_id"] for unit in suite["units"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task identity")
    return suite
