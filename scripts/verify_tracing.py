"""Opt-in real model/LangSmith acceptance using only a synthetic directory."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from tempfile import TemporaryDirectory
import threading
import time
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.models.responses import parse_response
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.recording.tracing import _create_client, _read_settings, _WorkerLogFilter
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor


async def _remote_runs(config, run_id, expected):
    """Bound both polling and interpreter exit, including a stalled DNS lookup."""
    result = Queue(maxsize=1)
    stop = threading.Event()

    def query():
        client = None
        logger = logging.getLogger("langsmith.client")
        safe_logs = _WorkerLogFilter()
        logger.addFilter(safe_logs)
        try:
            client = _create_client(config)
            for _ in range(8):
                if stop.is_set():
                    return
                try:
                    runs = list(client.list_runs(trace_id=run_id, limit=expected + 1))
                    if len(runs) >= expected and all(run.end_time for run in runs):
                        root = next(run for run in runs if str(run.id) == run_id)
                        url = root.url or client.get_run_url(run=root)
                        result.put((runs, url))
                        return
                except Exception:
                    pass
                if stop.wait(1):
                    return
            result.put(None)
        except Exception:
            result.put(None)
        finally:
            if client is not None:
                try:
                    client.close(timeout=0)
                except Exception:
                    pass
            logger.removeFilter(safe_logs)

    threading.Thread(target=query, name="lineage-trace-verify", daemon=True).start()
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            try:
                found = result.get_nowait()
                assert found is not None, "远端查询失败或追踪仍不完整。"
                return found
            except Empty:
                await asyncio.sleep(0.05)
        raise AssertionError("远端查询超过三十秒，停止等待。")
    finally:
        stop.set()


async def verify():
    settings = load_settings(Path(__file__).resolve().parents[1] / ".env")
    config = _read_settings()
    if config is None or not os.environ.get("API_KEY", "").strip():
        raise ConfigurationError("请在本机配置模型和 LangSmith 密钥，并开启 LANGSMITH_TRACING。")
    if settings.max_model_calls < 3:
        raise ConfigurationError("验收需要 MAX_MODEL_CALLS 至少为 3。")
    observations, actual_results = [], []
    original_client = runtime.create_client

    @asynccontextmanager
    async def observe(config):
        async with original_client(config) as client:
            create = client.responses.create

            async def observed(**kwargs):
                response = await create(**kwargs)
                observations.append((deepcopy(kwargs), response.model_dump(mode="json")))
                return response

            with patch.object(client.responses, "create", side_effect=observed):
                yield client

    with TemporaryDirectory(prefix="lineage-tracing-verify-") as temporary:
        root = Path(temporary)
        child = root / ("folder_" + uuid4().hex)
        child.mkdir()
        marker = child / ("found_" + uuid4().hex + ".txt")
        marker.touch()
        (root / "example.txt").touch()
        settings = replace(settings, project_root=root)
        bound_executor = create_directory_executor(root, max_entries=settings.directory_max_entries)

        async def execute(name, arguments):
            output = await bound_executor(name, arguments)
            actual_results.append(output)
            return output

        with patch.object(runtime, "create_client", side_effect=observe):
            state = await runtime.run_agent(
                "进行目录工具验收。请先在同一轮分别调用 list_directory 列举根目录 '.' 和文件 'example.txt'。"
                "文件不是目录，后一个请求应被拒绝；收到结果后继续列举根目录中实际发现的子目录。"
                "最后原样列出实际看到的所有文件相对路径。只依据工具返回，不读取正文，不猜测。",
                settings, tools=[DIRECTORY_TOOL], execute_tool=execute,
            )
        print(json.dumps({"run_id": state["run_id"], "log_path": state["log_path"]}, ensure_ascii=False))
        assert state["stop_reason"] == "completed" and state["log_complete"], "运行或本地日志未正常完成。"
        assert state["tool_outcomes"]["succeeded"] >= 2, "未完成至少两次目录列举。"
        listings = [json.loads(value) for value in actual_results]
        assert {".", child.name}.issubset({value["path"] for value in listings}), "根目录及实际发现的子目录未全部列举。"
        assert marker.relative_to(root).as_posix() in state["result"]["text"], "回答未引用实际临时目录结果。"

        events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
        started = [event for event in events if event["event"] == "model.started"]
        created = [event for event in events if event["event"] == "tool.result_created"]
        outputs = [item for item in state["input"] if item.get("type") == "function_call_output"]
        assert [event["call_id"] for event in created] == [item["call_id"] for item in outputs], "本地 call_id 关联不一致。"
        envelopes = {item["call_id"]: json.loads(item["output"]) for item in outputs}
        assert [value["result"] for value in envelopes.values() if value["ok"]] == actual_results, "上下文与实际执行器结果不一致。"
        assert len(started) == len(observations) == state["model_calls"], "实际模型调用统计不一致。"
        for event, (request, _) in zip(started, observations):
            returned = [item for item in request["input"] if item.get("type") == "function_call_output"]
            assert [item["call_id"] for item in returned] == event["input_tool_result_call_ids"], "实际 SDK 输入关联不一致。"
            assert all(json.loads(item["output"]) == envelopes[item["call_id"]] for item in returned), "实际 SDK 未携带原工具结果。"

        expected = 1 + state["model_calls"] + state["tool_requests"]
        runs, url = await _remote_runs(config, state["run_id"], expected)
        assert len(runs) == expected, "远端 span 数量与模型/工具请求统计不一致。"
        root_run = next(run for run in runs if str(run.id) == state["run_id"])
        cleaner = TraceSanitizer(
            [os.environ.get("API_KEY", ""), os.environ.get("API_KEY", "").strip(), config.api_key], root,
        )
        assert root_run.outputs == cleaner.root_outputs(state) and not root_run.error, "远端根运行结果不一致。"
        calls = {}
        for index, (request, raw_response) in enumerate(observations, 1):
            parsed = parse_response(raw_response)
            calls.update({call["call_id"]: call for call in parsed["tool_calls"]})
            run = next(run for run in runs if run.run_type == "llm" and run.extra["metadata"].get("model_call") == index)
            projected = cleaner.cap_payload({"inputs": cleaner.model_inputs(request["model"], request["input"], request.get("tools")), "outputs": cleaner.model_outputs(parsed)})
            assert run.inputs == projected["inputs"] and run.outputs == projected["outputs"], "模型参数、输出或 usage 与实际调用不一致。"
            assert run.extra["metadata"].get("step_id") == started[index - 1]["step_id"] and not run.error, "模型 span 状态或 step_id 不一致。"
        for event in created:
            call_id = event["call_id"]
            run = next(run for run in runs if run.run_type == "tool" and run.extra["metadata"].get("call_id") == call_id)
            projected = cleaner.cap_payload({"inputs": cleaner.tool_inputs(calls[call_id]), "outputs": cleaner.tool_outputs(envelopes[call_id], name=calls[call_id]["name"])})
            assert run.inputs == projected["inputs"] and run.outputs == projected["outputs"], "工具参数或结果与实际请求不一致。"
            metadata = run.extra["metadata"]
            assert all(metadata.get(key) == event[key] for key in ("step_id", "request_index", "outcome")), "工具 span 关联或结果状态不一致。"
            assert bool(run.error) == (event["outcome"] != "succeeded"), "失败/拒绝被错误显示为成功。"
            if "reason_code" in event:
                assert metadata.get("reason_code") == event["reason_code"] and metadata.get("reason_message") == event["reason_message"], "拒绝诊断与本地日志不一致。"
        for run in runs:
            assert str(run.trace_id) == state["run_id"] and run.extra["metadata"]["run_id"] == state["run_id"], "trace_id 或本地 run_id 不一致。"
            assert run.end_time and run.end_time >= run.start_time, "远端缺少有效起止时间。"
            assert type(run.extra["metadata"].get("duration_ms")) is int and run.extra["metadata"]["duration_ms"] >= 0, "远端缺少有效耗时。"
            if run is not root_run:
                assert str(run.parent_run_id) == state["run_id"] and run.dotted_order.startswith(root_run.dotted_order + "."), "远端层级关联不一致。"
        print(json.dumps({
            "trace_url": url, "run_id": state["run_id"], "remote_spans": len(runs),
            "same_batch_covered": any(event.get("tool_request_count", 0) > 1 for event in events),
            "rejection_covered": state["tool_outcomes"]["rejected"] > 0,
            "usage_covered": any(parse_response(raw)["usage"] is not None for _, raw in observations),
        }, ensure_ascii=False))
        print("PASS 真实模型与远端数据核对；同批/拒绝未覆盖项不因模型随机性判失败。")
        print("待人工核对：在上方私有 Trace 链接打开网页，检查参数、结果、失败状态、层级与耗时；脚本成功不等于网页验收完成。")


def main():
    try:
        asyncio.run(verify())
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception:
        # Never display raw SDK/network errors, response bodies, or credentials.
        print("FAIL 真实追踪验收未通过；请检查本机配置、模型结果及 LangSmith 中对应 run_id。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
