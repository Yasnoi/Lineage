"""Verify discovery, bounded text reading and answering against the real endpoint."""

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
from lineage_agent.agent.runtime import run_agent
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL

if __package__:
    from .verification_observations import business_arguments, inline_result
else:
    from verification_observations import business_arguments, inline_result


async def verify():
    settings = load_settings(Path(__file__).resolve().parents[1] / ".env")
    if settings.max_model_calls < 3:
        raise ConfigurationError("列举、读取及最终回答验证需要至少三次模型调用额度。")

    actual_inputs = []
    original_create_client = runtime.create_client

    @asynccontextmanager
    async def observed_client(config):
        # Observe local protocol input while forwarding the actual SDK call.
        # Never capture credentials, connection settings or a filtered trace view.
        async with original_create_client(config) as client:
            original_create = client.responses.create

            async def observed_create(**kwargs):
                actual_inputs.append(deepcopy(kwargs["input"]))
                return await original_create(**kwargs)

            with patch.object(client.responses, "create", side_effect=observed_create):
                yield client

    working_directory = Path.cwd()
    with TemporaryDirectory(prefix="lineage-text-verify-") as temporary:
        root = Path(temporary)
        filename = f"说明 {uuid4().hex}.Md"
        fact = f"OBSERVATION_{uuid4().hex}"
        body = (
            "# 非敏感临时观察记录\n\n"
            "样例结论：本次观察中的纸船颜色为蓝色。\n"
            f"观察识别码：{fact}\n"
        )
        original_bytes = body.encode("utf-8")
        source = root / filename
        source.write_bytes(original_bytes)
        if min(settings.text_max_bytes, settings.text_total_max_bytes) < len(original_bytes):
            raise ConfigurationError("读取限额小于真实验证样例大小，请提高配置后再验证。")

        bound_executor = create_filesystem_executor(
            root, max_entries=settings.directory_max_entries,
            max_bytes=settings.text_max_bytes,
            max_total_bytes=settings.text_total_max_bytes,
        )
        actual_results = []

        async def execute_tool(name, arguments):
            result = await bound_executor(name, arguments)
            actual_results.append({
                "name": name, "arguments": deepcopy(arguments), "result": result,
            })
            return result

        # Match tracing's bound root to this fixture; preserve tracing enablement
        # and all other settings supplied by the caller's existing configuration.
        run_settings = replace(settings, project_root=root)
        with patch.object(runtime, "create_client", side_effect=observed_client):
            state = await run_agent(
                "请先调用 list_directory 列举项目根目录 '.'，"
                "然后对其中实际发现的 Markdown 或 TXT 文件调用 read_text_file。"
                "最后依据实际读取的正文，说明样例结论，原样引用观察识别码，"
                "并明确列出来源文件的相对路径。不要猜测文件名、内容或识别码；"
                "目录名称不能代替正文证据，读取一个文件也不能证明已理解整个项目。",
                run_settings, tools=[DIRECTORY_TOOL, TEXT_TOOL], execute_tool=execute_tool,
            )
        files_unchanged = (
            list(root.iterdir()) == [source] and source.read_bytes() == original_bytes
        )

    log_path = Path(state["log_path"])
    events = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ] if log_path.exists() else []
    print(json.dumps({
        "scenario": "real_discover_read_answer", "run_id": state["run_id"],
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

    assert state["stop_reason"] == "completed", "真实端点未完成列举、读取及回答。"
    assert state["log_complete"], "真实运行的日志不完整。"
    assert files_unchanged and Path.cwd() == working_directory, "输入目录或工作目录被修改。"
    assert state["tool_outcomes"]["succeeded"] >= 2, "未成功完成目录列举和文本读取。"

    outputs = [item for item in state["input"] if item.get("type") == "function_call_output"]
    calls = {
        item["call_id"]: item for item in state["input"]
        if item.get("type") == "function_call"
    }
    successful = [item for item in outputs if json.loads(item["output"])["ok"]
                  and calls[item["call_id"]]["name"] in {"list_directory", "read_text_file"}]
    assert [
        {"name": calls[item["call_id"]]["name"],
         "arguments": business_arguments(calls[item["call_id"]]),
         "result": inline_result(item)}
        for item in successful
    ] == actual_results, "上下文中的成功结果与实际执行器收到的请求及返回值不同。"

    def result_value(item):
        return json.loads(inline_result(item))

    root_results = [
        item for item in successful
        if calls[item["call_id"]]["name"] == DIRECTORY_TOOL["name"]
        and result_value(item)["path"] == "."
    ]
    read_results = [
        item for item in successful
        if calls[item["call_id"]]["name"] == TEXT_TOOL["name"]
        and result_value(item)["path"] == filename
    ]
    assert root_results and read_results, "未实际列举根目录并读取随机文件。"
    root_result, read_result = root_results[0], read_results[0]
    assert any(
        entry["path"] == filename and entry["type"] == "file"
        for entry in result_value(root_result)["entries"]
    ), "实际目录结果未包含随机文件。"
    assert result_value(read_result) == {
        "path": filename, "content": body, "encoding": "utf-8",
        "bytes_read": len(original_bytes), "complete": True,
    }, "实际文本结果的正文或读取元数据不正确。"
    answer = state["result"]["text"]
    assert filename in answer and fact in answer, "最终回答未引用实际正文识别码和来源路径。"

    created = [event["call_id"] for event in events if event["event"] == "tool.result_created"]
    assert created == [item["call_id"] for item in outputs], "日志结果生成关联不匹配。"
    started = [event for event in events if event["event"] == "model.started"]
    assert len(started) == len(actual_inputs) == state["model_calls"]
    by_id = {item["call_id"]: item for item in outputs}
    assert len(by_id) == len(outputs), "工具结果 call_id 重复。"
    for event, actual_input in zip(started, actual_inputs):
        returned = [item for item in actual_input if item.get("type") == "function_call_output"]
        requested_ids = [item["call_id"] for item in actual_input if item.get("type") == "function_call"]
        assert event["input_tool_result_call_ids"] == [item["call_id"] for item in returned]
        assert returned == [by_id[call_id] for call_id in requested_ids], (
            "实际 SDK 请求未按原顺序携带完整工具结果。"
        )
    read_first_seen = next(index for index, items in enumerate(actual_inputs) if read_result in items)
    assert read_first_seen > 0 and root_result in actual_inputs[read_first_seen - 1], (
        "读取请求不是在模型实际收到目录结果后产生的。"
    )
    assert read_result in actual_inputs[-1], "最终回答前的实际模型请求未携带读取结果。"
    for event in (item for item in events if item["event"] == "tool.result_created"):
        assert any(
            later["event"] == "model.started"
            and later["event_seq"] > event["event_seq"]
            and event["call_id"] in later["input_tool_result_call_ids"]
            for later in events
        ), "已生成的工具结果缺少后续模型输入关联。"
    print("PASS 真实端点先发现文件再读取正文，并在回答中引用实际内容和来源；不固定调用次数。")


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
