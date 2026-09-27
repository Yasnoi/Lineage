"""Isolated adapter for the existing deterministic ReAct check script."""

import argparse
from importlib import import_module
import importlib.util
from pathlib import Path
import sys
import traceback
from types import ModuleType

from benchmarks.common import ROOT, atomic_write_json, sha256_file, utc_now


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_react.py"
ADAPTER_VERSION = "1"


def _save_receipt(path: Path, receipt: dict) -> bool:
    try:
        atomic_write_json(path, receipt)
    except Exception as exc:
        print("Benchmark worker could not save its receipt.", file=sys.stderr)
        traceback.print_exception(exc, file=sys.stderr)
        return False
    return True


def _load_script(path: Path) -> ModuleType:
    name = "_lineage_benchmark_react_check"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot construct a loader for the fixed ReAct script")
    module = importlib.util.module_from_spec(spec)
    # Some Python decorators consult sys.modules while the script is imported.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def _require_local_agent() -> None:
    """Do not attach this checkout's Git baseline to another installed Agent."""
    sources = {
        "lineage_agent": ROOT / "src/lineage_agent/__init__.py",
        "lineage_agent.agent.runtime": ROOT / "src/lineage_agent/agent/runtime.py",
    }
    for name, expected in sources.items():
        module = import_module(name)
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != expected.resolve():
            raise ImportError(f"Benchmark Agent module is outside the current repository: {name}")


def execute(
    run_id: str,
    unit_id: str,
    receipt_path: Path,
    *,
    script_path: Path = SCRIPT_PATH,
) -> int:
    """Run the fixed script; script_path is an internal fault-injection seam."""
    receipt_path = Path(receipt_path)
    script_path = Path(script_path).resolve()
    if receipt_path.exists() or receipt_path.is_symlink():
        print("Benchmark worker refuses an existing receipt.", file=sys.stderr)
        return 2
    receipt_path = receipt_path.resolve()

    receipt = {
        "schema_version": "1",
        "adapter_version": ADAPTER_VERSION,
        "run_id": run_id,
        "unit_id": unit_id,
        "script_path": str(script_path),
        "script_sha256": None,
        "phase": "preflight",
        "outcome": "started",
        "check_id": None,
        "check_dir": None,
        "error": None,
        "started_at": utc_now(),
        "ended_at": None,
    }
    if not _save_receipt(receipt_path, receipt):
        return 2

    exit_code = 2
    original_search_path = list(sys.path)
    fixed_script = script_path == SCRIPT_PATH.resolve()
    try:
        # Do not use assert here: this protection must also work under python -O.
        if not __debug__:
            raise RuntimeError("Benchmark checks require Python assertions enabled")
        receipt["script_sha256"] = sha256_file(script_path)
        receipt["phase"] = "import"
        if not _save_receipt(receipt_path, receipt):
            return 2
        if fixed_script:
            sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
            _require_local_agent()
        module = _load_script(script_path)
        if fixed_script:
            _require_local_agent()
        receipt["check_id"] = str(module.CHECK_ID)
        receipt["check_dir"] = str(Path(module.CHECK_DIR).resolve())
        receipt["phase"] = "check_execution"
        if not _save_receipt(receipt_path, receipt):
            return 2
        module.main()
        receipt["outcome"] = "passed"
        exit_code = 0
    except BaseException as exc:
        is_assertion = (
            receipt["phase"] == "check_execution" and isinstance(exc, AssertionError)
        )
        receipt["outcome"] = "assertion_failed" if is_assertion else "error"
        receipt["error"] = {
            "stage": receipt["phase"],
            "code": "assertion_failed" if is_assertion else "worker_error",
            "exception_type": type(exc).__name__,
        }
        exit_code = 1 if is_assertion else 2
        traceback.print_exception(exc, file=sys.stderr)
    finally:
        sys.path[:] = original_search_path
    receipt["phase"] = "finished"
    receipt["ended_at"] = utc_now()
    if not _save_receipt(receipt_path, receipt):
        return 2
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    return execute(args.run_id, args.unit_id, args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
