"""Command-line input and output for a single agent run."""

import argparse
import asyncio
import sys

from lineage_agent.agent.runtime import run_agent
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.tools.filesystem import directory
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


def format_run_summary(state: AgentState) -> str:
    """Show program-recorded facts independently of the model's answer."""
    outcomes = state["tool_outcomes"]
    completeness = "完整" if state["log_complete"] else "不完整"
    return (
        f"运行标识：{state['run_id']}\n"
        f"工具处理：成功 {outcomes['succeeded']}，拒绝 {outcomes['rejected']}，"
        f"失败 {outcomes['failed']}，未执行 {outcomes['skipped']}\n"
        f"运行日志（{completeness}）：{state['log_path']}"
    )


def main() -> int:
    """Parse one prompt, run the agent, and display its answer."""
    parser = argparse.ArgumentParser(
        prog="lineage-agent",
        description="运行一次 Agent 请求。",
    )
    parser.add_argument("prompt", help="本次发送给模型的用户输入")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("用户输入不能为空或仅包含空白字符。")

    try:
        settings = load_settings()
        if settings.project_root is None:
            state = asyncio.run(run_agent(args.prompt, settings))
        else:
            execute_tool = create_filesystem_executor(
                settings.project_root, max_entries=settings.directory_max_entries,
                max_bytes=settings.text_max_bytes,
            )
            state = asyncio.run(
                run_agent(
                    args.prompt,
                    settings,
                    tools=[directory.DIRECTORY_TOOL, TEXT_TOOL],
                    execute_tool=execute_tool,
                )
            )
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130

    print(format_run_summary(state), file=sys.stderr)
    if state["stop_reason"] == "completed" and state["result"] is not None:
        print(state["result"]["text"])
        return 0
    if state["stop_reason"] == "max_model_calls":
        print(
            f"任务未完成：已达到模型调用次数上限（{settings.max_model_calls}）。",
            file=sys.stderr,
        )
        return 3

    print(f"运行失败：{state['error'] or '未获得最终回答。'}", file=sys.stderr)
    return 1
