"""Compatibility entry point for the bounded, versioned Compact Benchmarks.

Prefer ``python -m benchmarks run wechat-reporting --config <path>``. This
wrapper requires the same explicit credential-free configuration and delegates
all execution, request accounting and evidence to that implementation.
"""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.core.diagnostics import ConfigurationError
from benchmarks.entrypoints import cli


COMPACT_UNITS = frozenset({"compaction-continuation", "history-recall"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="显式 Benchmark 运行配置；units 选择跨轮 Compact 或单 Run 历史回读")
    args = parser.parse_args(argv)
    try:
        config = cli.read_json(args.config, field="--config")
        units = config.get("units") if isinstance(config, dict) else None
        if (not isinstance(units, list) or not units
                or any(not isinstance(unit, str) or unit not in COMPACT_UNITS for unit in units)):
            raise ConfigurationError("units", "unsupported_compaction_units",
                                     "只能选择 compaction-continuation 和 history-recall。",
                                     "其他研究任务请使用 python -m benchmarks run wechat-reporting。")
    except ConfigurationError as exc:
        print(f"Compact Benchmark 配置错误：{exc}", file=sys.stderr)
        return 2
    return cli.main(["run", "wechat-reporting", "--config", str(args.config)])


if __name__ == "__main__":
    raise SystemExit(main())
