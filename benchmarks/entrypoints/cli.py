"""Small public interface; no project-specific branches or connection options."""
import argparse
import json
from pathlib import Path
import sys
from benchmarks.core.contracts import exit_code
from benchmarks.core.registry import SUITES, load_suite
from benchmarks.core.runner import run_benchmark

def main(argv=None):
    parser = argparse.ArgumentParser(description="选择已定义的 Benchmark，执行并保留证据。")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("list", help="列出评测集，不执行或调用模型")
    run = commands.add_parser("run", help="执行一个评测集")
    run.add_argument("suite", choices=tuple(SUITES))
    run.add_argument("--config", type=Path, help="显式运行配置，不包含凭据")
    review = commands.add_parser("review", help="导入人工评审或执行已配置的模型评审")
    review.add_argument("directory", type=Path)
    source = review.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="人工填写的评分 JSON")
    source.add_argument("--config", type=Path, help="独立的评判模型与预算配置")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "list":
            for identifier in SUITES:
                suite = load_suite(identifier)
                print(f"{identifier}: {suite['title']} ({len(suite['units'])} definitions; "
                      f"{'explicit model configuration required' if suite['network'] else 'offline'})")
            return 0
        if args.command == "review":
            from benchmarks.evaluators.review import import_review
            if args.file:
                reviewed = import_review(args.directory / "result.json", args.file)
            else:
                from benchmarks.evaluators.rubric import grade_run
                reviewed = grade_run(args.directory, json.loads(args.config.read_text(encoding="utf-8")))
            for path in reviewed["saved"].values():
                print(f"Saved: {path}")
            return reviewed["exit_code"]
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        result, saved = run_benchmark(args.suite, config=config)
        print(f"Benchmark run: {result['run_id']}")
        for unit in result["units"]:
            print(f"{unit['unit_id']}: {unit['execution_status']} / {unit['verdict']}")
        for path in saved:
            print(f"Saved: {path}")
        return exit_code(result)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Benchmark 失败（{type(exc).__name__}）；未确认完整保存。", file=sys.stderr)
        for path in getattr(exc, "saved", {}).values():
            print(f"Partial artifact: {path}", file=sys.stderr)
        return 2
