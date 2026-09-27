"""Run from the repository root: python -m benchmarks [--suite control-react]."""

import argparse
import sys

from benchmarks.results import exit_code
from benchmarks.runner import SUITES, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行本地 Benchmark；默认仅执行确定性控制逻辑检查。")
    parser.add_argument("--list", action="store_true", help="列出已接入的评测集，不运行检查")
    parser.add_argument("--suite", action="append", choices=tuple(SUITES), help="选择评测集，可重复指定")
    args = parser.parse_args(argv)
    if args.list:
        for name, definition in SUITES.items():
            print(f"{name}: {definition['unit_id']} (script, deterministic_stub, v{definition['definition_version']})")
        return 0
    try:
        result, saved = run_benchmark(args.suite or ["control-react"])
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Benchmark 框架失败（{type(exc).__name__}）；未确认报告保存成功。", file=sys.stderr)
        return 2
    print(f"Benchmark run: {result['run_id']}")
    for unit in result["units"]:
        print(f"{unit['suite_id']}/{unit['unit_id']}: {unit['execution_status']} / {unit['verdict']} (script)")
    for path in saved:
        print(f"Saved: {path}")
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
