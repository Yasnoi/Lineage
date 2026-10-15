"""Command-line input and output for requests and persistent conversations."""

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path
import sys

from lineage_agent.agent import sessions as session_api
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.runtime import (
    ARTIFACT_CLOSE_MESSAGE, CLIENT_CLOSE_MESSAGE, PreExecutionConfigurationError, run_agent,
)
from lineage_agent.agent.sessions import SessionError, SessionManager, SessionTurn
from lineage_agent.agent.state import AgentState, CompactionStatus
from lineage_agent.archive import ArchiveError
from lineage_agent.artifacts.store import ArtifactError
from lineage_agent.config.settings import (
    ConfigurationError, ModelSettings, load_current_limits, load_local_paths, load_settings,
)
from lineage_agent.entrypoints.console_input import ConsoleInput, ConsoleInputError
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.storage import StorageError
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


def _needs_log_diagnostic(state: AgentState) -> bool:
    return state.get("stop_reason") not in (None, "completed") or not state.get("log_complete", False)


def _format_log_diagnostic(state: AgentState) -> str:
    """Report recorded completeness separately from current file availability."""
    completeness = "记录完整" if state.get("log_complete") else "记录不完整"
    path = state.get("log_path")
    if not path:
        return f"运行日志（{completeness}）：未记录可用路径。"
    try:
        with Path(path).open("rb"):
            pass
    except (OSError, ValueError, TypeError):
        availability = "文件不可读取"
    else:
        availability = "文件可读取"
    return f"运行日志（{completeness}；{availability}）：{path}"


def format_run_summary(state: AgentState) -> str:
    """Show program-recorded facts independently of the model's answer."""
    outcomes = state["tool_outcomes"]
    summary = (
        f"工具处理：成功 {outcomes['succeeded']}，拒绝 {outcomes['rejected']}，"
        f"失败 {outcomes['failed']}，未执行 {outcomes['skipped']}"
    )
    if _needs_log_diagnostic(state):
        summary += "\n" + _format_log_diagnostic(state)
    summary += (
        f"\n模型调用：业务 {state.get('model_calls', 0)}，摘要 {state.get('summary_calls', 0)}"
    )
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
        if "net_reclaimed" in compaction and "remaining_budget" in compaction:
            summary += (f"\n上下文净回收：{compaction['net_reclaimed']}；"
                        f"剩余额度：{compaction['remaining_budget']}（观测值；本地预算估算单位）")
        if compaction.get("confirmed_checkpoint_id"):
            summary += f"\n版本确认检查点：{compaction['confirmed_checkpoint_id']}"
    budget = TraceSanitizer.context_budget(state.get("context_budget"))
    if budget:
        decision = {"allow": "放行", "exceeded": "超限", "error": "计量失败"}.get(budget.get("decision"), "未知")
        sent = "已发起" if budget.get("request_sent") is True else "未发起"
        summary += (
            f"\n上下文预算（本地估算；非端点精确 Token）：{decision}，请求{sent}"
            f"，步骤 {budget.get('step_id', '未知')}\n"
            f"透明内容 {budget.get('transparent_tokens', '未知')} + "
            f"opaque 风险预留 {budget.get('opaque_risk_tokens', '未知')}"
            f"（{budget.get('opaque_items', '未知')} 项） + "
            f"输出预留 {budget.get('output_reserve_tokens', '未知')} + "
            f"安全余量 {budget.get('safety_margin_tokens', '未知')} = "
            f"所需 {budget.get('required_tokens', '未知')}／上限 {budget.get('limit_tokens', '未知')}。"
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
    display_run(turn.state, settings)
    if turn.snapshot_status != "published" and not _needs_log_diagnostic(turn.state):
        print(_format_log_diagnostic(turn.state), file=sys.stderr)
    if turn.snapshot_status == "published" and turn.snapshot_id is not None:
        print(f"对话快照已发布：{turn.snapshot_id}；可在新进程中打开。", file=sys.stderr)
    elif turn.snapshot_status == "unknown":
        print("对话快照发布结果不确定；此实例已阻断，请关闭后重新打开核对。", file=sys.stderr)
    else:
        print(
            f"状态保存未确认：{turn.checkpoint_error or 'checkpoint_incomplete'}；"
            "此会话不可继续；可关闭后重新打开已发布快照，或使用 /new、/switch。",
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


async def interactive(settings: ModelSettings | None = None, dependencies: dict | None = None, *,
                      local_paths: dict | None = None, open_id: str | None = None,
                      prompt: str | None = None) -> int:
    """Handle terminal interaction; session state belongs to SessionManager."""
    sessions = SessionManager()
    try:
        with sessions, ConsoleInput() as input_reader:
            if local_paths is None and open_id is None and prompt is None:
                return await _interactive_loop(settings, dependencies, sessions, input_reader=input_reader)
            return await _interactive_loop(settings, dependencies, sessions,
                                           local_paths=local_paths, open_id=open_id, initial_prompt=prompt,
                                           input_reader=input_reader)
    except BaseException as exc:
        # A primary interruption keeps propagating, while its suppressed cleanup
        # failure still gets a safe report. A standalone cleanup error is shown
        # by main(), so reporting it here would duplicate the same failure.
        if sessions.cleanup_error and not (
            isinstance(exc, ArtifactError) and exc.code == sessions.cleanup_error
        ):
            print(f"收尾失败：{ARTIFACT_CLOSE_MESSAGE}", file=sys.stderr)
        raise


def _show_sessions(sessions: SessionManager, application_root: Path) -> None:
    rows = sessions.list_sessions(application_root)
    if not rows:
        print("没有已登记的对话。", file=sys.stderr)
        return
    for row in rows:
        identity = row["conversation_id"]
        snapshot = row.get("current_snapshot_id") or "无可恢复快照"
        unfinished = row.get("unfinished_runs", [])
        run_ids = [item["run_id"] if isinstance(item, dict) else item for item in unfinished]
        suffix = f"；未封存轮次：{', '.join(run_ids)}" if run_ids else ""
        if row.get("error_code"):
            suffix += f"；不可用：{row['error_code']}"
        status = {"ready": "可打开", "no_snapshot": "无恢复点", "recovery_required": "有未封存轮次",
                  "unsupported": "格式不兼容", "damaged": "归档不可用"}.get(row.get("status"), "未知")
        print(f"{identity}；状态：{status}；快照：{snapshot}{suffix}", file=sys.stderr)


async def _open_session(sessions: SessionManager, identity: str, local_paths: dict, *,
                        input_reader: ConsoleInput | None = None):
    if input_reader is None:
        with ConsoleInput() as reader:
            return await _open_session(sessions, identity, local_paths, input_reader=reader)
    try:
        limits = load_current_limits()
    except ConfigurationError:
        limits = None
        print("当前额度配置无效，无法进行差异比较；仍将校验并采用会话保存的配置。", file=sys.stderr)
    arguments = {"application_root": local_paths["application_root"],
                 "current_project_root": local_paths["project_root"],
                 "log_dir": local_paths["log_dir"], "current_limits": limits}
    try:
        opened = await sessions.open_session(identity, **arguments)
    except SessionError as error:
        if error.code != "recovery_required":
            raise
        details = error.details
        unfinished = tuple(details["unfinished_run_ids"])
        print(f"未封存轮次：{', '.join(unfinished)}。", file=sys.stderr)
        print(f"最后已发布快照：{details['snapshot_id']}。旧轮次及产物将保留，不会续跑。", file=sys.stderr)
        confirmed = await input_reader.confirm("输入 yes 从此快照开始新轮次；其他输入取消：")
        if not confirmed:
            print("已取消打开。", file=sys.stderr)
            return None
        choice = session_api.RecoveryChoice(details["snapshot_id"], unfinished)
        opened = await sessions.open_session(identity, **arguments, recovery=choice)
    for warning in opened.warnings:
        print(f"配置提示：{warning}", file=sys.stderr)
    print(f"已打开会话：{opened.session_id}；快照：{opened.snapshot_id}", file=sys.stderr)
    return opened


async def _interactive_loop(settings: ModelSettings | None, dependencies: dict | None,
                            sessions: SessionManager, *, local_paths: dict | None = None,
                            open_id: str | None = None, initial_prompt: str | None = None,
                            input_reader: ConsoleInput | None = None) -> int:
    if input_reader is None:
        with ConsoleInput() as reader:
            return await _interactive_loop(settings, dependencies, sessions, local_paths=local_paths,
                                           open_id=open_id, initial_prompt=initial_prompt, input_reader=reader)
    if local_paths is None:
        local_paths = ({"application_root": settings.application_root, "project_root": settings.project_root,
                        "log_dir": settings.log_dir} if settings is not None else load_local_paths())
    current = None
    known_sessions = set()

    def create():
        chosen = settings if settings is not None else load_settings()
        chosen = replace(chosen, application_root=local_paths["application_root"], log_dir=local_paths["log_dir"])
        bound = dependencies if dependencies is not None else _tool_dependencies(chosen)
        return sessions.create_session(chosen, **bound)

    if open_id is not None:
        opened = await _open_session(sessions, open_id, local_paths, input_reader=input_reader)
        if opened is None:
            return 1
        current = opened.session_id
        known_sessions.add(current)
    print(
        "命令：/new 新建；/list 列出；/open <会话ID> 打开；/switch <会话ID> 切换；"
        "/close [会话ID] 关闭；/exit 退出。\n"
        f"当前会话：{current or '未选择'}",
        file=sys.stderr,
    )
    while True:
        one_shot = initial_prompt is not None
        if one_shot:
            prompt, initial_prompt = initial_prompt, None
        else:
            try:
                prompt = await input_reader.read(f"[{current or '未选择'}] > ")
            except EOFError:
                return 0
        if not prompt.strip():
            continue
        command = prompt.strip() if not one_shot and "\n" not in prompt and "\r" not in prompt else ""
        if command == "/exit":
            return 0
        if command == "/new":
            try:
                current = create()
            except (SessionError, ConfigurationError, CheckpointError, ArtifactError, ArchiveError, StorageError) as exc:
                if isinstance(exc, ConfigurationError):
                    print(f"配置错误：{exc}", file=sys.stderr)
                    continue
                print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
                continue
            known_sessions.add(current)
            print(f"当前会话：{current}", file=sys.stderr)
            continue
        if command == "/list":
            _show_sessions(sessions, local_paths["application_root"])
            continue
        if command and command.split()[0] == "/open":
            parts = command.split()
            if len(parts) != 2:
                print("用法：/open <会话ID>", file=sys.stderr)
                continue
            if parts[1] in known_sessions:
                print("此会话已打开，可用 /switch 切换。", file=sys.stderr)
                continue
            try:
                opened = await _open_session(sessions, parts[1], local_paths, input_reader=input_reader)
            except ConfigurationError as exc:
                print(f"配置错误：{exc}", file=sys.stderr)
                continue
            except (SessionError, CheckpointError, ArtifactError, ArchiveError, StorageError) as exc:
                print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
                continue
            if opened is not None:
                current = opened.session_id
                known_sessions.add(current)
            continue
        if command and command.split()[0] == "/close":
            parts = command.split()
            if len(parts) > 2:
                print("用法：/close [会话ID]", file=sys.stderr)
                continue
            identity = parts[1] if len(parts) == 2 else current
            if identity not in known_sessions:
                print("没有对应的已打开会话。", file=sys.stderr)
                continue
            try:
                sessions.close_session(identity)
            except (SessionError, CheckpointError, ArtifactError, ArchiveError, StorageError) as exc:
                print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
                continue
            known_sessions.remove(identity)
            if current == identity:
                current = None
            print(f"已关闭会话：{identity}；持久记录保留。", file=sys.stderr)
            continue
        if command.startswith("/switch"):
            parts = command.split()
            if len(parts) != 2 or parts[0] != "/switch":
                print("用法：/switch <会话ID>", file=sys.stderr)
            elif parts[1] not in known_sessions:
                print("会话尚未打开；请先使用 /open <会话ID>。", file=sys.stderr)
            else:
                current = parts[1]
                print(f"当前会话：{current}", file=sys.stderr)
            continue
        if command.startswith("/"):
            print("未知命令；可使用 /new、/list、/open、/switch、/close 或 /exit。", file=sys.stderr)
            continue
        try:
            if current is None:
                current = create()
                known_sessions.add(current)
            turn = await sessions.run_turn(current, prompt)
        except (PreExecutionConfigurationError, ConfigurationError) as exc:
            print(f"配置错误：{exc}", file=sys.stderr)
            if one_shot:
                return 2
            continue
        except (SessionError, CheckpointError, ArtifactError, ArchiveError, StorageError) as exc:
            print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
            if one_shot:
                return 1
            continue
        if one_shot:
            print(f"会话标识：{current}", file=sys.stderr)
        display_turn(turn, sessions.settings(current))
        if one_shot:
            if not turn.checkpoint_complete:
                return 1
            return {"completed": 0, "max_model_calls": 3, "context_budget_exceeded": 4}.get(turn.state["stop_reason"], 1)


def main() -> int:
    """Parse an optional prompt and select single-run or interactive mode."""
    parser = argparse.ArgumentParser(
        prog="lineage-agent",
        description="运行一次 Agent 请求，或列出、打开持久保存的对话。",
    )
    parser.add_argument("prompt", nargs="?", help="本次用户输入；省略则进入交互会话")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", dest="list_sessions", help="列出当前存储根目录的对话")
    modes.add_argument("--open", metavar="ID", dest="open_id", help="打开已保存的对话")
    parser.add_argument("--application-root", type=Path, help="会话存储根目录；不改变 .env 的查找位置")
    args = parser.parse_args()
    if args.prompt is not None and not args.prompt.strip():
        parser.error("用户输入不能为空或仅包含空白字符。")
    if args.list_sessions and args.prompt is not None:
        parser.error("--list 不能同时提供问题。")
    try:
        if args.list_sessions or args.open_id is not None or args.prompt is None:
            local_paths = load_local_paths(application_root=args.application_root)
            if args.list_sessions:
                with SessionManager() as sessions:
                    _show_sessions(sessions, local_paths["application_root"])
                return 0
            return asyncio.run(interactive(local_paths=local_paths, open_id=args.open_id, prompt=args.prompt))
        settings = load_settings()
        if args.application_root is not None:
            settings = replace(settings, application_root=args.application_root)
        dependencies = _tool_dependencies(settings)
        state = asyncio.run(run_agent(args.prompt, settings, **dependencies))
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except ConsoleInputError as exc:
        print(f"输入失败：{exc}", file=sys.stderr)
        return 1
    except (SessionError, CheckpointError, ArtifactError, ArchiveError, StorageError) as exc:
        print(f"会话错误（{exc.code}）：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    return display_run(state, settings)
