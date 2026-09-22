"""Real endpoint: sorted paging, source/config reads, and final-call tool choice."""

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
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor

from lineage_agent.tools.filesystem.text import TEXT_TOOL


class ObservedExecutor:
    """Preserve run binding and observe unmodified results without logging bodies."""

    def __init__(self, executor, records):
        self.executor = executor
        self.records = records

    def bind_run(self):
        return ObservedExecutor(self.executor.bind_run(), self.records)

    @property
    def read_budget(self):
        return self.executor.read_budget

    async def __call__(self, name, arguments):
        value = await self.executor(name, arguments)
        self.records.append((name, deepcopy(arguments), value))
        return value


def report(scenario, state, requests):
    events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
    print(json.dumps({
        "scenario": scenario, "run_id": state["run_id"], "log_path": state["log_path"],
        "stop_reason": state["stop_reason"], "error_code": state["error_code"],
        "log_complete": state["log_complete"], "model_calls": state["model_calls"],
        "tool_calls": state["tool_calls"], "tool_outcomes": state["tool_outcomes"],
        "usage_totals": state["usage_totals"], "read_budget": events[-1].get("read_budget"),
        "steps": list(dict.fromkeys(event["step_id"] for event in events if "step_id" in event)),
        "call_ids": [event["call_id"] for event in events if event["event"] == "tool.result_created"],
        "tool_choices": [request.get("tool_choice") for request in requests],
    }, ensure_ascii=False), flush=True)
    return events


async def verify():
    settings = load_settings()
    if settings.max_model_calls < 4:
        raise ConfigurationError("分页、读取及回答的真实验证需要至少四次模型调用额度。")
    original_create = runtime.create_client
    requests = []

    @asynccontextmanager
    async def observed_client(config):
        async with original_create(config) as client:
            original_request = client.responses.create

            async def request(**kwargs):
                requests.append(deepcopy(kwargs))
                return await original_request(**kwargs)

            with patch.object(client.responses, "create", side_effect=request):
                yield client

    cwd = Path.cwd()
    with TemporaryDirectory(prefix="lineage-review-verify-") as temporary:
        root = Path(temporary)
        source, config = f"a_{uuid4().hex}.py", f"z_{uuid4().hex}.toml"
        source_fact, config_fact = f"SOURCE_{uuid4().hex}", f"CONFIG_{uuid4().hex}"
        files = {
            source: f'OBSERVATION = "{source_fact}"\n'.encode(),
            config: f'observation = "{config_fact}"\n'.encode(),
        }
        for name, data in files.items():
            (root / name).write_bytes(data)
        if settings.text_max_bytes < max(map(len, files.values())) or settings.text_total_max_bytes < sum(map(len, files.values())):
            raise ConfigurationError("真实验证样例超过当前读取限额。")
        records = []
        executor = ObservedExecutor(create_filesystem_executor(
            root, max_entries=1, max_bytes=settings.text_max_bytes,
            max_total_bytes=settings.text_total_max_bytes,
        ), records)
        with patch.object(runtime, "create_client", side_effect=observed_client):
            state = await runtime.run_agent(
                "请先调用 list_directory 分页发现根目录 '.' 的所有文件，第一页 after=null；"
                "如果 truncated=true，就用 next_after 继续，直到完整发现。随后读取发现的 Python 源码和 TOML 配置。"
                "请在最终回答逐项原样引用各文件正文中的 observation 值和来源相对路径。"
                "不要猜测名称或正文；目录发现不能代替读取。",
                replace(settings, project_root=root), tools=[DIRECTORY_TOOL, TEXT_TOOL], execute_tool=executor,
            )
        events = report("real_review_paging_read_answer", state, requests)
        assert state["stop_reason"] == "completed" and state["log_complete"], "真实分页读取未完成。"
        calls = {item["call_id"]: item for item in state["input"] if item.get("type") == "function_call"}
        outputs = [item for item in state["input"] if item.get("type") == "function_call_output"]
        success = [item for item in outputs if json.loads(item["output"])["ok"]]
        assert [(calls[item["call_id"]]["name"], json.loads(calls[item["call_id"]]["arguments"]), json.loads(item["output"])["result"]) for item in success] == records
        directory_results = [json.loads(value) for name, _, value in records if name == "list_directory"]
        assert any(value["truncated"] and value["next_after"] == source for value in directory_results), "未观察到真实截断第一页。"
        assert any(value["after"] == source and not value["truncated"] for value in directory_results), "模型未使用游标继续列举。"
        for name, data in files.items():
            assert any(tool == "read_text_file" and json.loads(value)["path"] == name and json.loads(value)["content"].encode() == data for tool, _, value in records)
        assert all(value in state["result"]["text"] for value in (source, config, source_fact, config_fact)), "回答未引用真实正文及来源。"
        assert {item.name: item.read_bytes() for item in root.iterdir()} == files and Path.cwd() == cwd
        started = [event for event in events if event["event"] == "model.started"]
        assert len(started) == len(requests) == state["model_calls"]
        for event, request in zip(started, requests):
            received = [item for item in request["input"] if item.get("type") == "function_call_output"]
            assert received == outputs[:len(received)]
            assert event["input_tool_result_call_ids"] == [item["call_id"] for item in received]
        assert [item for item in requests[-1]["input"] if item.get("type") == "function_call_output"] == outputs
        print(state["result"]["text"], flush=True)

        requests.clear()
        records.clear()
        with patch.object(runtime, "create_client", side_effect=observed_client):
            final = await runtime.run_agent(
                "请调用工具检查项目文件，然后回答。若当前不允许工具，请明确说明无法检查。",
                replace(settings, project_root=root, max_model_calls=1),
                tools=[DIRECTORY_TOOL, TEXT_TOOL], execute_tool=executor,
            )
        report("real_review_final_tool_choice_none", final, requests)
        assert final["model_calls"] == 1 and final["tool_calls"] == 0 and not records
        assert requests[0]["tool_choice"] == "none"
        assert final["stop_reason"] == "completed" and final["tool_requests"] == 0, "端点未按 tool_choice=none 返回最终回答。"
    print("PASS 真实分页、源码/配置读取和来源引用；单独验证最后回答轮 tool_choice=none。")


def main():
    try:
        asyncio.run(verify())
    except (ConfigurationError, AssertionError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
