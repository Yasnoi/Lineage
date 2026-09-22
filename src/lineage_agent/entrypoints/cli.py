"""Command-line input and output for single requests and in-process sessions."""

import argparse
import asyncio
import sys

from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.runtime import CLIENT_CLOSE_MESSAGE, run_agent
from lineage_agent.agent.sessions import SessionError, SessionManager, SessionTurn
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings
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


def display_run(state: AgentState, settings: ModelSettings) -> int:
    """Display one run's answer and program-recorded facts."""
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

    if (state.get("cleanup_error") or {}).get("previous_stop_reason") == "completed" and state["result"] is not None:
        print(state["result"]["text"])
    if state.get("cleanup_error") and state.get("error_code") != "client_close_error":
        print(f"收尾失败：{CLIENT_CLOSE_MESSAGE}", file=sys.stderr)
    print(f"运行失败：{state['error'] or '未获得最终回答。'}", file=sys.stderr)
    return 1


def display_turn(turn: SessionTurn, settings: ModelSettings) -> None:
    print(f"会话标识：{turn.session_id}", file=sys.stderr)
    display_run(turn.state, settings)
    if turn.checkpoint_complete:
        print(f"状态已保存（仅本进程）：{turn.checkpoint_id}", file=sys.stderr)
    else:
        print(
            f"状态保存未确认：{turn.checkpoint_error or 'checkpoint_incomplete'}；"
            "此会话不可继续，请使用 /new 或 /switch。",
            file=sys.stderr,
        )


def _tool_dependencies(settings: ModelSettings) -> dict:
    if settings.project_root is None:
        return {}
    return {
        "tools": [directory.DIRECTORY_TOOL, TEXT_TOOL],
        "execute_tool": create_filesystem_executor(
            settings.project_root, max_entries=settings.directory_max_entries,
            max_bytes=settings.text_max_bytes,
            max_total_bytes=settings.text_total_max_bytes,
        ),
    }


async def interactive(settings: ModelSettings, dependencies: dict) -> int:
    """Handle terminal interaction; session state belongs to SessionManager."""
    sessions = SessionManager()
    current = sessions.create_session(settings, **dependencies)
    known_sessions = {current}
    print(
        "交互会话仅保存在当前进程，退出后丢失。\n"
        "命令：/new 新建；/switch <会话ID> 切换；/exit 退出。\n"
        f"当前会话：{current}",
        file=sys.stderr,
    )
    while True:
        print(f"[{current}] > ", end="", file=sys.stderr, flush=True)
        try:
            prompt = input().strip()
        except EOFError:
            return 0
        if not prompt:
            continue
        if prompt == "/exit":
            return 0
        if prompt == "/new":
            try:
                current = sessions.create_session(settings, **dependencies)
            except SessionError as exc:
                print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
                continue
            known_sessions.add(current)
            print(f"当前会话：{current}", file=sys.stderr)
            continue
        if prompt.startswith("/switch"):
            parts = prompt.split()
            if len(parts) != 2 or parts[0] != "/switch":
                print("用法：/switch <会话ID>", file=sys.stderr)
            elif parts[1] not in known_sessions:
                print("会话不存在；只能切换到本次交互已创建的会话。", file=sys.stderr)
            else:
                current = parts[1]
                print(f"当前会话：{current}", file=sys.stderr)
            continue
        if prompt.startswith("/"):
            print("未知命令；可使用 /new、/switch <会话ID> 或 /exit。", file=sys.stderr)
            continue
        try:
            turn = await sessions.run_turn(current, prompt)
        except SessionError as exc:
            print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
            continue
        display_turn(turn, settings)


def main() -> int:
    """Parse an optional prompt and select single-run or interactive mode."""
    parser = argparse.ArgumentParser(
        prog="lineage-agent",
        description="运行一次 Agent 请求；省略问题时进入进程内交互会话。",
    )
    parser.add_argument("prompt", nargs="?", help="本次用户输入；省略则进入交互会话")
    args = parser.parse_args()
    if args.prompt is not None and not args.prompt.strip():
        parser.error("用户输入不能为空或仅包含空白字符。")
    try:
        settings = load_settings()
        dependencies = _tool_dependencies(settings)
        if args.prompt is None:
            return asyncio.run(interactive(settings, dependencies))
        state = asyncio.run(run_agent(args.prompt, settings, **dependencies))
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except (SessionError, CheckpointError) as exc:
        print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    return display_run(state, settings)
