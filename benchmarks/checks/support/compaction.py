"""Run shared Compact assertions as individually recorded engineering groups."""
from dataclasses import replace
import inspect
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from benchmarks.checks.support.events import observe_state, observe_harness, evidence
from benchmarks.core.files import ROOT
from lineage_agent.agent import runtime

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

async def run_case(case_type, method):
    case = case_type(method)
    case.setUp()
    fixture = getattr(case, "fixture", case)
    if hasattr(fixture, "settings"):
        fixture.settings = replace(fixture.settings, log_dir=CHECK_DIR)
    native = runtime._run_agent
    async def observed(*args, **kwargs):
        result = await native(*args, **kwargs)
        state, checkpoint_id, error = result
        observe_state(state, session_id=kwargs.get("session_id"), checkpoint_id=checkpoint_id,
                      checkpoint_complete=checkpoint_id is not None and error is None, checkpoint_error=error)
        evidence(state["log_path"], kind="agent_log", run_id=state["run_id"])
        return result
    try:
        with patch.object(runtime, "_run_agent", side_effect=observed), \
                patch("lineage_agent.agent.sessions._run_agent", side_effect=observed):
            value = getattr(case, method)()
            if inspect.isawaitable(value):
                await value
        observe_harness("compact_contract", scenario=method)
    finally:
        case.tearDown()
        while case._cleanups:
            callback, args, kwargs = case._cleanups.pop()
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                await result
