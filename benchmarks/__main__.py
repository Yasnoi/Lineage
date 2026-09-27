"""Run from the repository root: python -m benchmarks [--suite control-react]."""

import argparse
import sys

from benchmarks.results import exit_code
from benchmarks.runner import SUITES, UNITS, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行本地 Benchmark；默认仅执行确定性控制逻辑检查。")
    parser.add_argument("--list", action="store_true", help="列出已接入的评测集，不运行检查")
    parser.add_argument("--suite", action="append", choices=(*SUITES, "research-wechat"), help="选择评测集，可重复指定")
    parser.add_argument("--research-profile", choices=("wechat-v1",), help="真实研究必须显式选择批准的配置")
    parser.add_argument("--review-result", help="人工评分对应的原 result.json；不调用模型")
    parser.add_argument("--review-file", help="已填写的人工评分 JSON；与 --review-result 同用")
    args = parser.parse_args(argv)
    if args.list:
        for name, members in SUITES.items():
            print(f"{name}: {', '.join(members)} ({sum(len(UNITS[key]['groups']) for key in members)} groups, v2)")
        print("research-wechat: 6 cases × (1 pilot + 3 formal); real model; requires --research-profile wechat-v1")
        return 0
    if args.review_result or args.review_file:
        if not (args.review_result and args.review_file) or args.suite or args.research_profile:
            parser.error("人工评分需同时指定 --review-result 与 --review-file，且不得混选运行。")
        from pathlib import Path
        from benchmarks.research.review import import_review
        try:
            reviewed = import_review(Path(args.review_result), Path(args.review_file))
        except (Exception, KeyboardInterrupt) as exc:
            print(f"人工评分导入失败（{type(exc).__name__}）；原执行记录保持不变。", file=sys.stderr)
            for path in getattr(exc, "saved", {}).values():
                print(f"Partial artifact: {path}", file=sys.stderr)
            return 2
        for path in reviewed["saved"].values():
            print(f"Saved: {path}")
        return reviewed["exit_code"]
    research = "research-wechat" in (args.suite or [])
    if research and (set(args.suite) != {"research-wechat"} or not args.research_profile):
        parser.error("研究须显式提供 --research-profile，且不能与控制回归混选。")
    if args.research_profile and not research:
        parser.error("--research-profile 只能与 --suite research-wechat 使用。")
    try:
        if research:
            from benchmarks.research.batch import run_research
            result, saved = run_research(args.research_profile)
        else:
            result, saved = run_benchmark(args.suite or ["control-react"])
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Benchmark 框架失败（{type(exc).__name__}）；未确认报告保存成功。", file=sys.stderr)
        return 2
    print(f"Benchmark run: {result['run_id']}")
    for unit in result["units"]:
        print(f"{unit['suite_id']}/{unit['unit_id']}: {unit['execution_status']} / {unit['verdict']} ({unit['granularity']})")
    for path in saved:
        print(f"Saved: {path}")
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
