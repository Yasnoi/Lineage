"""Verify two turns against the real endpoint using non-sensitive file facts."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


def events_for(turn):
    path = Path(turn.state["log_path"])
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def report(turn, number):
    state = turn.state
    events = events_for(turn)
    print(json.dumps({
        "scenario": f"real_session_turn_{number}",
        "session_id": turn.session_id, "run_id": state["run_id"],
        "checkpoint_id": turn.checkpoint_id,
        "checkpoint_complete": turn.checkpoint_complete,
        "checkpoint_error": turn.checkpoint_error,
        "log_path": state["log_path"], "log_complete": state["log_complete"],
        "stop_reason": state["stop_reason"],
        "model_calls": state["model_calls"], "tool_calls": state["tool_calls"],
        "tool_requests": state["tool_requests"], "tool_outcomes": state["tool_outcomes"],
        "last_usage": state["result"]["usage"] if state["result"] else None,
        "usage_totals": state["usage_totals"], "history_end": state["history_end"],
        "steps": list(dict.fromkeys(item["step_id"] for item in events if "step_id" in item)),
        "call_ids": [item["call_id"] for item in events if item["event"] == "tool.result_created"],
    }, ensure_ascii=False))
    if state["result"] is not None and state["stop_reason"] == "completed":
        print(state["result"]["text"])
    elif state["error"]:
        print(state["error"], file=sys.stderr)


async def verify():
    settings = load_settings(Path(__file__).resolve().parents[1] / ".env")
    if settings.max_model_calls < 3:
        raise ConfigurationError("会话资料验证需要至少三次模型调用额度。")
    actual_inputs = {1: [], 2: []}
    actual_results = {1: [], 2: []}
    current_turn = 1
    original_create_client = runtime.create_client

    @asynccontextmanager
    async def observed_client(config):
        async with original_create_client(config) as client:
            original_create = client.responses.create

            async def observed_create(**kwargs):
                actual_inputs[current_turn].append(deepcopy(kwargs["input"]))
                return await original_create(**kwargs)

            with patch.object(client.responses, "create", side_effect=observed_create):
                yield client

    class ObservedExecutor:
        def __init__(self, executor):
            self.executor = executor

        def bind_run(self):
            return ObservedExecutor(self.executor.bind_run())

        @property
        def read_budget(self):
            return self.executor.read_budget

        async def __call__(self, name, arguments):
            result = await self.executor(name, arguments)
            actual_results[current_turn].append({
                "name": name, "arguments": deepcopy(arguments), "result": result,
            })
            return result

    cwd = Path.cwd()
    with TemporaryDirectory(prefix="lineage-session-verify-") as temporary:
        root = Path(temporary)
        filename = f"观察记录 {uuid4().hex}.md"
        fact = f"SESSION_OBSERVATION_{uuid4().hex}"
        body = f"# 非敏感会话资料\n样例结论：纸船的颜色是蓝色。\n观察识别码：{fact}\n"
        source = root / filename
        original = body.encode("utf-8")
        source.write_bytes(original)
        if min(settings.text_max_bytes, settings.text_total_max_bytes) < len(original):
            raise ConfigurationError("读取额度小于真实会话验证样例，请提高额度后重试。")
        executor = ObservedExecutor(create_filesystem_executor(
            root, max_entries=settings.directory_max_entries,
            max_bytes=settings.text_max_bytes, max_total_bytes=settings.text_total_max_bytes,
        ))
        sessions = SessionManager()
        session_id = sessions.create_session(
            replace(settings, project_root=root),
            tools=[DIRECTORY_TOOL, TEXT_TOOL], execute_tool=executor,
        )
        first_question = (
            "请先调用 list_directory 列举根目录 '.'，再对实际发现的 Markdown 文件"
            "调用 read_text_file。读完后只说明已读取以及来源文件的相对路径，"
            "暂时不展开样例结论和观察识别码。不要猜测文件名或正文内容。"
        )
        second_question = (
            "根据刚才已经读取的记录，样例结论是什么？请原样引用观察识别码，"
            "并明确注明来源文件的相对路径。本轮使用对话中已有资料，不重新读取文件。"
        )
        assert filename not in first_question and fact not in first_question + second_question
        with patch.object(runtime, "create_client", side_effect=observed_client):
            first = await sessions.run_turn(session_id, first_question)
            report(first, 1)
            assert first.state["stop_reason"] == "completed", "第一轮未完成发现、读取及回答。"
            assert first.checkpoint_complete and first.checkpoint_id, "第一轮状态未确认保存。"
            current_turn = 2
            second = await sessions.run_turn(session_id, second_question)
            report(second, 2)
        files_unchanged = list(root.iterdir()) == [source] and source.read_bytes() == original

    assert files_unchanged and Path.cwd() == cwd, "文件或工作目录被修改。"
    assert first.session_id == second.session_id == session_id
    assert first.state["run_id"] != second.state["run_id"], "新请求没有独立运行标识。"
    assert first.checkpoint_id != second.checkpoint_id and second.checkpoint_complete
    assert second.state["stop_reason"] == "completed", "第二轮未完成追问。"
    assert second.state["result"] is not None
    assert fact in second.state["result"]["text"] and filename in second.state["result"]["text"], (
        "真实模型没有在追问回答中正确引用历史正文事实和来源。"
    )

    first_history = first.state["input"][:first.state["history_end"]]
    assert actual_inputs[2][0] == first_history + [{"role": "user", "content": second_question}], (
        "第二轮实际 SDK 输入没有恰好继承有效历史并追加新问题。"
    )
    first_calls = {
        item["call_id"]: item for item in first_history if item.get("type") == "function_call"
    }
    first_outputs = [item for item in first_history if item.get("type") == "function_call_output"]
    successful = [item for item in first_outputs if json.loads(item["output"])["ok"]]
    assert [
        {"name": first_calls[item["call_id"]]["name"],
         "arguments": json.loads(first_calls[item["call_id"]]["arguments"]),
         "result": json.loads(item["output"])["result"]}
        for item in successful
    ] == actual_results[1], "已保存工具结果与实际执行器返回值不一致。"
    reads = [
        item for item in successful if first_calls[item["call_id"]]["name"] == TEXT_TOOL["name"]
        and json.loads(json.loads(item["output"])["result"])["path"] == filename
    ]
    directories = [
        item for item in successful if first_calls[item["call_id"]]["name"] == DIRECTORY_TOOL["name"]
        and json.loads(json.loads(item["output"])["result"])["path"] == "."
    ]
    assert reads and directories, "第一轮未实际列举并读取随机文件。"
    assert json.loads(json.loads(reads[0]["output"])["result"])["content"] == body
    assert reads[0] in actual_inputs[1][-1] and reads[0] in actual_inputs[2][0]
    first_read_seen = next(index for index, items in enumerate(actual_inputs[1]) if reads[0] in items)
    assert first_read_seen > 0 and directories[0] in actual_inputs[1][first_read_seen - 1], (
        "读取请求不是在模型实际收到目录结果之后产生的。"
    )
    for number, turn in enumerate((first, second), 1):
        state = turn.state
        events = events_for(turn)
        assert state["log_complete"] and turn.checkpoint_complete
        assert state["model_calls"] == len(actual_inputs[number]), "模型计数不是本轮独立计量。"
        assert state["tool_outcomes"]["succeeded"] == len(actual_results[number])
        assert all(event["session_id"] == session_id for event in events)
        started = [event for event in events if event["event"] == "model.started"]
        assert len(started) == len(actual_inputs[number])
        by_id = {
            item["call_id"]: item for item in state["input"]
            if item.get("type") == "function_call_output"
        }
        for event, items in zip(started, actual_inputs[number]):
            results = [item for item in items if item.get("type") == "function_call_output"]
            ids = [item["call_id"] for item in results]
            assert len(ids) == len(set(ids)), "实际模型输入重复追加了历史工具结果。"
            assert event["input_tool_result_call_ids"] == ids
            assert results == [by_id[call_id] for call_id in ids]
    print("PASS 两轮实际请求保留同会话资料、原工具关联及来源；实际模型追问引用正确。")
    print("本验证仅覆盖同进程连续对话，不验证跨进程加载或任意中断恢复。")


def main():
    try:
        asyncio.run(verify())
    except (ConfigurationError, SessionError, CheckpointError, AssertionError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
