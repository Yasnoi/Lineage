"""Command-line input and output for single requests and in-process sessions."""

import argparse
import asyncio
import sys

from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.runtime import (
    ARTIFACT_CLOSE_MESSAGE, CLIENT_CLOSE_MESSAGE, PreExecutionConfigurationError, run_agent,
)
from lineage_agent.agent.sessions import SessionError, SessionManager, SessionTurn
from lineage_agent.agent.state import AgentState, CompactionStatus
from lineage_agent.artifacts.store import ArtifactError
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem import directory
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


_COMPACTION_STAGE_LABELS: dict[CompactionStatus, str] = {
    "preparing": "准备摘要", "summary_budget_rejected": "摘要预算拒绝",
    "candidate": "候选已生成", "retry_pending": "等待再次压缩", "pending_save": "版本等待保存确认",
    "confirmed": "版本保存已确认", "used": "新版本已用于业务请求",
    "stopped": "未切换并停止", "failed": "压缩失败",
    "history_ready": "历史摘要已准备", "current_ready": "当前前缀摘要已准备", "validated": "候选检查通过",
}


def format_run_summary(state: AgentState) -> str:
    """Show program-recorded facts independently of the model's answer."""
    outcomes = state["tool_outcomes"]
    completeness = "完整" if state["log_complete"] else "不完整"
    summary = (
        f"运行标识：{state['run_id']}\n"
        f"工具处理：成功 {outcomes['succeeded']}，拒绝 {outcomes['rejected']}，"
        f"失败 {outcomes['failed']}，未执行 {outcomes['skipped']}\n"
        f"运行日志（{completeness}）：{state['log_path']}"
    )
    summary += (
        f"\n模型调用：业务 {state.get('model_calls', 0)}，摘要 {state.get('summary_calls', 0)}"
    )
    active = TraceSanitizer.active_context(state.get("active_context"))
    if active.get("version_id"):
        summary += f"\n活动上下文版本：{active['version_id']}"
    compaction = TraceSanitizer.compaction(state.get("compaction"))
    if compaction:
        stage = _COMPACTION_STAGE_LABELS.get(compaction.get("status"), "未确认")
        summary += f"\n上下文压缩：{stage}"
        if compaction.get("reason"):
            summary += f"（{compaction['reason']}）"
        if compaction.get("summary_id"):
            summary += f"，摘要 {compaction['summary_id']}"
        if compaction.get("attempt_no"):
            summary += f"，第 {compaction['attempt_no']} 次尝试"
        lanes = compaction.get("lanes", {})
        if lanes:
            details = []
            labels = {"empty": "无来源", "prepared": "待生成", "reused": "复用", "generated": "已生成", "failed": "失败"}
            for lane, value in lanes.items():
                details.append(f"{lane}：{labels.get(value.get('status'), '未确认')}")
            summary += "\n摘要分路：" + "，".join(details)
        if "net_reclaimed" in compaction and "recovery_margin" in compaction:
            summary += (f"\n上下文净回收：{compaction['net_reclaimed']}；"
                        f"恢复余量：{compaction['recovery_margin']}（本地预算估算单位）")
        if compaction.get("confirmed_checkpoint_id"):
            summary += f"\n版本确认检查点：{compaction['confirmed_checkpoint_id']}"
    budget = TraceSanitizer.context_budget(state.get("context_budget"))
    if budget:
        decision = {"allow": "放行", "exceeded": "超限", "error": "计量失败"}.get(budget.get("decision"), "未知")
        sent = "已发起" if budget.get("request_sent") is True else "未发起"
        summary += (
            f"\n上下文预算（估算；非端点精确 Token）：{decision}，请求{sent}"
            f"，步骤 {budget.get('step_id', '未知')}\n"
            f"透明内容 {budget.get('transparent_tokens', '未知')} + "
            f"opaque 风险预留 {budget.get('opaque_risk_tokens', '未知')}"
            f"（{budget.get('opaque_items', '未知')} 项） + "
            f"输出预留 {budget.get('output_reserve_tokens', '未知')} + "
            f"安全余量 {budget.get('safety_margin_tokens', '未知')} = "
            f"所需 {budget.get('required_tokens', '未知')}；"
            f"触发阈值 {budget.get('trigger_tokens', '未知')}"
            f"（总预算 {budget.get('limit_tokens', '未知')} 的 {budget.get('trigger_percent', '未知')}%）。"
        )
    return summary


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
    if state["stop_reason"] == "context_budget_exceeded":
        print("任务未完成：上下文预算未通过，未发起对应业务模型请求；请查看压缩状态与运行记录。", file=sys.stderr)
        return 4

    if (state.get("cleanup_error") or {}).get("previous_stop_reason") == "completed" and state["result"] is not None:
        print(state["result"]["text"])
    cleanup_code = (state.get("cleanup_error") or {}).get("code")
    if cleanup_code and state.get("error_code") != cleanup_code:
        message = ARTIFACT_CLOSE_MESSAGE if cleanup_code == "artifact_cleanup_error" else CLIENT_CLOSE_MESSAGE
        print(f"收尾失败：{message}", file=sys.stderr)
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
    try:
        with sessions:
            return await _interactive_loop(settings, dependencies, sessions)
    except BaseException as exc:
        # A primary interruption keeps propagating, while its suppressed cleanup
        # failure still gets a safe report. A standalone cleanup error is shown
        # by main(), so reporting it here would duplicate the same failure.
        if sessions.cleanup_error and not (
            isinstance(exc, ArtifactError) and exc.code == sessions.cleanup_error
        ):
            print(f"收尾失败：{ARTIFACT_CLOSE_MESSAGE}", file=sys.stderr)
        raise


async def _interactive_loop(settings: ModelSettings, dependencies: dict, sessions: SessionManager) -> int:
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
        except PreExecutionConfigurationError as exc:
            print(f"配置错误：{exc}", file=sys.stderr)
            continue
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
    except (SessionError, CheckpointError, ArtifactError) as exc:
        print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    return display_run(state, settings)
