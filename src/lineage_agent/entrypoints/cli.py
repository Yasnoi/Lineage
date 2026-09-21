"""Command-line input and output for a single agent run."""

import argparse
import asyncio
import sys

from lineage_agent.agent.runtime import run_agent
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.models.responses import ModelError


def main() -> int:
    """Parse one prompt, run the agent, and display its answer."""
    parser = argparse.ArgumentParser(
        prog="lineage-agent",
        description="运行一次最小 Agent Loop。",
    )
    parser.add_argument("prompt", help="本次发送给模型的用户输入")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("用户输入不能为空或仅包含空白字符。")

    try:
        settings = load_settings()
        state = asyncio.run(run_agent(args.prompt, settings))
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except ModelError as exc:
        print(f"模型调用失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130

    print(state["result"]["text"])
    return 0
