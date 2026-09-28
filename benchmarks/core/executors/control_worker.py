"""Execute one allowlisted contract chain in an isolated offline process."""

import argparse
from importlib import import_module
from pathlib import Path
import sys
import traceback

from benchmarks.suites.agent_contracts.suite import load_catalog
from benchmarks.core.files import ROOT, atomic_write_json, sha256_file, utc_now
from benchmarks.core.events import GroupJournal, plan_digest


from benchmarks.core.processes import require_local_agent as _require_local_agent, offline_network
from benchmarks.checks.support import events as support


def _execute_entry(definition):
    module_name, function_name = definition["entrypoint"].split(":")
    module = import_module(module_name)
    if Path(module.__file__).resolve() != (ROOT / definition["script"]).resolve():
        raise ImportError("check source does not belong to checkout")
    return getattr(module, function_name)()


def execute(run_id, unit_id, receipt_path):
    units, _ = load_catalog()
    definition = units[unit_id]
    script = ROOT / definition["script"]
    receipt_path = Path(receipt_path).absolute()
    events_path = receipt_path.with_name("groups.jsonl")
    if receipt_path.exists() or events_path.exists():
        print("Refusing existing worker artifacts.", file=sys.stderr)
        return 2
    receipt = {
        "schema_version": "3", "adapter_version": "3", "run_id": run_id, "unit_id": unit_id,
        "script_path": str(script), "script_sha256": None,
        "plan_digest": plan_digest(definition["groups"]), "phase": "preflight", "outcome": "started",
        "error": None, "started_at": utc_now(), "ended_at": None,
        "events_sha256": None, "events_count": 0,
    }
    journal = None
    code = 2
    try:
        atomic_write_json(receipt_path, receipt)
        if not __debug__:
            raise RuntimeError("Assertions must be enabled")
        receipt["script_sha256"] = sha256_file(script)
        receipt["phase"] = "import"
        _require_local_agent()
        with offline_network():
            receipt["phase"] = "group_recording"
            journal = GroupJournal(events_path, run_id, definition)
            receipt["phase"] = "check_execution"
            atomic_write_json(receipt_path, receipt)
            try:
                with support.activate(journal):
                    _execute_entry(definition)
                if journal.active is not None or journal.index != len(definition["groups"]):
                    raise RuntimeError("entrypoint did not execute every declared group")
                if sha256_file(script) != receipt["script_sha256"]:
                    raise RuntimeError("check source changed during execution")
                receipt["outcome"] = "coverage_gap" if journal.coverage_gaps else "passed"
                code = 2 if journal.coverage_gaps else 0
            except support.CheckStopped as exc:
                original = exc.cause
                receipt["outcome"] = ("assertion_failed" if isinstance(original, AssertionError)
                                      else "interrupted" if isinstance(original, KeyboardInterrupt) else "error")
                receipt["error"] = {"stage": "check_execution", "code": "check_error", "exception_type": type(original).__name__}
                traceback.print_exception(original, file=sys.stderr)
                code = 1 if receipt["outcome"] == "assertion_failed" else 2
            except support.RecordingFailure as exc:
                receipt.update(outcome="recording_failed", error={"stage": "group_recording", "code": "recording_failed",
                                                                  "exception_type": type(exc).__name__})
                traceback.print_exception(exc, file=sys.stderr)
    except BaseException as exc:
        outcome = ("recording_failed" if receipt["phase"] == "group_recording"
                   else "interrupted" if isinstance(exc, KeyboardInterrupt) else "error")
        receipt.update(outcome=outcome, error={"stage": receipt["phase"], "code": "worker_error", "exception_type": type(exc).__name__})
        traceback.print_exception(exc, file=sys.stderr)
        code = 2
    finally:
        if journal is not None:
            try:
                journal.close()
                receipt["events_count"] = journal.sequence
                receipt["events_sha256"] = sha256_file(events_path)
            except BaseException as exc:
                receipt.update(outcome="recording_failed", error={"stage": "group_recording", "code": "recording_failed",
                                                                  "exception_type": type(exc).__name__})
                code = 2
        receipt.update(phase="finished", ended_at=utc_now())
        try:
            atomic_write_json(receipt_path, receipt)
        except BaseException as exc:
            print(f"Could not save final worker receipt ({type(exc).__name__}).", file=sys.stderr)
            code = 2
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--unit-id", required=True, choices=tuple(load_catalog()[0]))
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    return execute(args.run_id, args.unit_id, args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
