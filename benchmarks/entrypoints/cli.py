"""Small public interface; no project-specific branches or connection options."""
import argparse
import json
from pathlib import Path
import sys
from benchmarks.core.contracts import exit_code
from benchmarks.core.diagnostics import ConfigurationError
from benchmarks.core.registry import SUITES, load_suite
from benchmarks.core.runner import run_benchmark
from benchmarks.evaluators.review import ReviewSaveError


def read_json(path, *, field):
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ConfigurationError(field, "file_unreadable", "必须是可读取的 UTF-8 JSON 文件。",
                                 "检查命令参数所指的文件、读取权限及编码。") from None
    try:
        return json.loads(source)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(field, "invalid_json", f"JSON 语法无效（第 {exc.lineno} 行，第 {exc.colno} 列）。",
                                 "检查该位置的引号、逗号和括号；不支持 JSON 注释。") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description="选择已定义的 Benchmark，执行并保留证据。")
    commands = parser.add_subparsers(dest="command")
    listing = commands.add_parser("list", help="列出评测集，或指定评测集列出任务 ID")
    listing.add_argument("suite", nargs="?", choices=tuple(SUITES))
    run = commands.add_parser("run", help="执行一个评测集")
    run.add_argument("suite", choices=tuple(SUITES))
    run.add_argument("--config", type=Path, help="显式运行配置，不包含凭据")
    review = commands.add_parser("review", help="导入人工评审或执行已配置的模型评审")
    review.add_argument("directory", type=Path)
    source = review.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="人工填写的评分 JSON")
    source.add_argument("--config", type=Path, help="独立的评判模型与预算配置")
    source.add_argument("--prepare", action="store_true", help="根据原结果重新生成独立人工评审包")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "list":
            for identifier in ([args.suite] if args.suite else SUITES):
                suite = load_suite(identifier)
                print(f"{identifier}: {suite['title']} ({len(suite['units'])} definitions; "
                      f"{'explicit model configuration required' if suite['network'] else 'offline'})")
                if args.suite:
                    for unit in suite["units"]:
                        if "turns" in unit:
                            print(f"  {unit['unit_id']}：{unit['title']}（{len(unit['turns'])} 轮）")
                        elif "groups" in unit:
                            print(f"  {unit['unit_id']}：行为链（{len(unit['groups'])} 个行为组）")
                        else:
                            print(f"  {unit['unit_id']}")
            return 0
        if args.command == "review":
            from benchmarks.evaluators.review import import_review
            if args.prepare:
                from benchmarks.evaluators.review import prepare_review_packet
                reviewed = prepare_review_packet(args.directory)
            elif args.file:
                reviewed = import_review(args.directory / "result.json", args.file)
            else:
                from benchmarks.evaluators.rubric import grade_run
                reviewed = grade_run(args.directory, read_json(args.config, field="--config"))
            for path in reviewed["saved"].values():
                print(f"Saved: {path}")
            return reviewed["exit_code"]
        config = read_json(args.config, field="--config") if args.config else None
        result, saved = run_benchmark(args.suite, config=config)
        print(f"Benchmark run: {result['run_id']}")
        for unit in result["units"]:
            print(f"{unit['unit_id']}: {unit['execution_status']} / {unit['verdict']}")
        shown = set()
        for problem in [*result.get("framework_errors", []), *(p for unit in result["units"] for p in unit["issues"])]:
            diagnostic = problem.get("diagnostic")
            if diagnostic:
                key = tuple(diagnostic[name] for name in ("path", "code", "constraint", "hint"))
                if key not in shown:
                    print(f"配置错误：{ConfigurationError(*key)}", file=sys.stderr)
                    shown.add(key)
        for path in saved:
            print(f"Saved: {path}")
        return exit_code(result)
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, ConfigurationError):
            print(f"Benchmark 配置错误：{exc}", file=sys.stderr)
        elif isinstance(exc, ReviewSaveError):
            hint = ("可对原运行目录重新执行 benchmarks review <运行目录> --file <填写文件>。"
                    if args.command == "review" and args.file else
                    "可运行 benchmarks review <运行目录> --prepare 生成新评审包。")
            print("评审产物未完整保存；已保存的执行结果和评审产物保留。" + hint, file=sys.stderr)
        else:
            print(f"Benchmark 失败（{type(exc).__name__}）；未确认完整保存。", file=sys.stderr)
        for path in getattr(exc, "saved", {}).values():
            print(f"Partial artifact: {path}", file=sys.stderr)
        return 2
