"""Verify the real endpoint against a non-sensitive temporary directory."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.agent import runtime
from lineage_agent.agent.runtime import run_agent
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor


async def verify():
    settings = load_settings(Path(__file__).resolve().parents[1] / ".env")
    if settings.max_model_calls < 3:
        raise ConfigurationError("根目录、子目录及最终回答验证需要至少三次模型调用额度。")

    actual_inputs = []
    original_create_client = runtime.create_client

    @asynccontextmanager
    async def observed_client(config):
        # Forward the real SDK request unchanged; retain only local protocol input
        # for acceptance assertions, never connection settings or credentials.
        async with original_create_client(config) as client:
            original_create = client.responses.create

            async def observed_create(**kwargs):
                actual_inputs.append(deepcopy(kwargs["input"]))
                return await original_create(**kwargs)

            with patch.object(client.responses, "create", side_effect=observed_create):
                yield client

    with TemporaryDirectory(prefix="lineage-directory-verify-") as temporary:
        root = Path(temporary)
        child_name = f"folder_{uuid4().hex}"
        child = root / child_name
        child.mkdir()
        marker = f"found_{uuid4().hex}.txt"
        (child / marker).touch()
        (root / "example.txt").touch()
        bound_executor = create_directory_executor(root)
        actual_results = []

        async def execute_tool(name, arguments):
            result = await bound_executor(name, arguments)
            actual_results.append(result)
            return result

        with patch.object(runtime, "create_client", side_effect=observed_client):
            state = await run_agent(
                "请先调用 list_directory 列举项目根目录 '.'，"
                "再列举其中实际发现的子目录。最后原样列出实际看到的所有文件相对路径。"
                "只依据工具返回，不猜测文件名，不读取正文，也不要推断代码功能。",
                settings, tools=[DIRECTORY_TOOL], execute_tool=execute_tool,
            )

    log_path = Path(state["log_path"])
    events = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ] if log_path.exists() else []
    print(json.dumps({
        "scenario": "real_directory_loop", "run_id": state["run_id"],
        "log_path": state["log_path"], "log_complete": state["log_complete"],
        "stop_reason": state["stop_reason"], "model_calls": state["model_calls"],
        "tool_requests": state["tool_requests"], "tool_calls": state["tool_calls"],
        "tool_outcomes": state["tool_outcomes"],
        "last_usage": state["result"]["usage"] if state["result"] else None,
        "usage_totals": state["usage_totals"],
        "steps": list(dict.fromkeys(event["step_id"] for event in events if "step_id" in event)),
        "call_ids": [event["call_id"] for event in events if event["event"] == "tool.result_created"],
    }, ensure_ascii=False))
    if state["stop_reason"] == "completed":
        print(state["result"]["text"])
    else:
        print(state["error"] or "未获得最终回答。", file=sys.stderr)

    assert state["stop_reason"] == "completed", "真实端点未完成目录工具循环。"
    assert state["log_complete"], "真实运行的日志不完整。"
    assert state["tool_outcomes"]["succeeded"] >= 2, "未成功列举根目录和子目录。"
    outputs = [item for item in state["input"] if item.get("type") == "function_call_output"]
    successful = [item for item in outputs if json.loads(item["output"])["ok"]]
    assert [json.loads(item["output"])["result"] for item in successful] == actual_results, (
        "上下文中的成功结果与实际目录执行器返回值不同。"
    )
    listings = [json.loads(result) for result in actual_results]
    assert {".", child_name}.issubset({listing["path"] for listing in listings}), (
        "未根据根目录结果继续列举子目录。"
    )
    assert any(
        entry["path"] == f"{child_name}/{marker}"
        for listing in listings for entry in listing["entries"]
    ), "实际目录结果中缺少临时文件。"
    assert f"{child_name}/{marker}" in state["result"]["text"], (
        "最终回答未引用仅存在于实际目录结果中的随机文件路径。"
    )
    created = [event["call_id"] for event in events if event["event"] == "tool.result_created"]
    assert created == [item["call_id"] for item in outputs], "日志结果生成关联不匹配。"
    started = [event for event in events if event["event"] == "model.started"]
    assert len(started) == len(actual_inputs) == state["model_calls"]
    by_id = {item["call_id"]: item for item in outputs}
    for event, actual_input in zip(started, actual_inputs):
        returned = [item for item in actual_input if item.get("type") == "function_call_output"]
        assert event["input_tool_result_call_ids"] == [item["call_id"] for item in returned]
        requested_ids = [item["call_id"] for item in actual_input if item.get("type") == "function_call"]
        assert returned == [by_id[call_id] for call_id in requested_ids], (
            "实际 SDK 请求未按原顺序携带完整工具结果。"
        )
    root_result = next(item for item in successful if json.loads(json.loads(item["output"])["result"])["path"] == ".")
    child_result = next(item for item in successful if json.loads(json.loads(item["output"])["result"])["path"] == child_name)
    child_first_seen = next(index for index, items in enumerate(actual_inputs) if child_result in items)
    assert child_first_seen > 0 and root_result in actual_inputs[child_first_seen - 1], (
        "子目录请求不是在模型实际收到根目录结果后产生的。"
    )
    for event in (item for item in events if item["event"] == "tool.result_created"):
        assert any(
            later["event"] == "model.started"
            and later["event_seq"] > event["event_seq"]
            and event["call_id"] in later["input_tool_result_call_ids"]
            for later in events
        ), "已生成的工具结果缺少后续模型输入关联。"
    print("PASS 真实目录结果进入后续模型输入并用于回答；调用次数不作固定要求。")


def main():
    try:
        asyncio.run(verify())
    except (ConfigurationError, AssertionError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
