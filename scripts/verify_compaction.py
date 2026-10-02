"""Real endpoint compaction and follow-up using only non-sensitive temporary data.

Uses a deliberately modest application budget, not a model-window stress test.
Only captured SDK requests establish that the follow-up used a saved summary;
the model's semantic preservation is reported separately from that protocol fact.
"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
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
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.summarization import is_summary_request
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL

if __package__:
    from .verification_observations import runtime_tools
else:
    from verification_observations import runtime_tools


def summary_request(request):
    return is_summary_request(request)


def report(turn, scenario, requests):
    state = turn.state
    path = Path(state["log_path"])
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    compaction = state.get("compaction") or {}
    print(json.dumps({
        "scenario": scenario, "session_id": turn.session_id, "run_id": state["run_id"],
        "checkpoint_id": turn.checkpoint_id, "checkpoint_complete": turn.checkpoint_complete,
        "checkpoint_error": turn.checkpoint_error, "log_path": str(path),
        "log_complete": state["log_complete"], "stop_reason": state["stop_reason"],
        "summary_id": compaction.get("summary_id"),
        "version_id": state["active_context"]["version_id"],
        "compaction_status": compaction.get("status"), "compaction_reason": compaction.get("reason"),
        "model_calls": state["model_calls"], "summary_calls": state["summary_calls"],
        "tool_calls": state["tool_calls"], "tool_outcomes": state["tool_outcomes"],
        "usage_totals": state["usage_totals"], "summary_usage_totals": state["summary_usage_totals"],
        "steps": list(dict.fromkeys(item["step_id"] for item in events if "step_id" in item)),
        "call_ids": [item["call_id"] for item in events if item["event"] == "tool.result_created"],
        "captured_business_calls": sum(not summary_request(item) for item in requests),
        "captured_summary_calls": sum(summary_request(item) for item in requests),
    }, ensure_ascii=False))
    assert state["log_complete"] and turn.checkpoint_complete
    assert state["model_calls"] == sum(not summary_request(item) for item in requests)
    assert state["summary_calls"] == sum(summary_request(item) for item in requests)
    assert state["stop_reason"] == "completed", state.get("error")
    return events


async def verify():
    configured = load_settings(Path(__file__).resolve().parents[1] / ".env")
    settings = replace(
        configured, context_budget_tokens=32768, max_output_tokens=1024,
        context_safety_margin_tokens=256, context_keep_recent_turns=2,
        summary_target_percent=40,
        max_model_calls=max(5, configured.max_model_calls),
    )
    print(json.dumps({
        "scenario": "validation_limits", "context_budget_tokens": settings.context_budget_tokens,
        "summary_target_percent": settings.summary_target_percent,
        "context_keep_recent_turns": settings.context_keep_recent_turns,
        "max_output_tokens": settings.max_output_tokens,
        "context_safety_margin_tokens": settings.context_safety_margin_tokens,
    }))
    actual_requests = {}
    current = 1
    original_factory = runtime.create_client

    @asynccontextmanager
    async def observed_factory(config):
        async with original_factory(config) as client:
            original_create = client.responses.create

            async def observed_create(**kwargs):
                actual_requests.setdefault(current, []).append(deepcopy(kwargs))
                return await original_create(**kwargs)

            with patch.object(client.responses, "create", side_effect=observed_create):
                yield client

    cwd = Path.cwd()
    with TemporaryDirectory(prefix="lineage-compaction-verify-") as temporary, SessionManager() as manager:
        root = Path(temporary)
        filename = f"观察来源 {uuid4().hex}.md"
        fact = f"COMPACTION_FACT_{uuid4().hex}"
        body = (
            "# 非敏感压缩验证资料\n"
            f"唯一观察识别码：{fact}\n"
            "有效观察结论：样例纸船颜色为蓝色。红色是已否决的旧说法。\n"
            "来源可信范围：只代表本文件中的虚构验证样例。\n"
            + "普通背景记录：这是没有额外事实的重复校验材料。\n" * 130
        )
        source = root / filename
        original_bytes = body.encode("utf-8")
        source.write_bytes(original_bytes)
        if min(settings.text_max_bytes, settings.text_total_max_bytes) < len(original_bytes):
            raise ConfigurationError("当前读取额度小于非敏感压缩验证样例。")
        tools = [DIRECTORY_TOOL, TEXT_TOOL]
        sid = manager.create_session(
            replace(settings, project_root=root), tools=tools,
            execute_tool=create_filesystem_executor(
                root, max_entries=settings.directory_max_entries,
                max_bytes=settings.text_max_bytes, max_total_bytes=settings.text_total_max_bytes,
            ),
        )
        questions = [
            "请先 list_directory 列举 '.'，再 read_text_file 读取实际找到的 Markdown 文件。"
            "同时尝试读取不存在的 missing-case.txt，明确记录失败，不把它说成成功。"
            "只回复已处理及真实来源的相对路径，不展开正文结论、旧说法或识别码。",
            "补充本轮要求：后续引用资料必须注明来源，不能把已否决说法恢复成有效结论。"
            "这轮只回复收到，不调用工具，不复述之前正文。" + "无新增事实。" * 80,
            "继续确认：工具失败应保留为失败，不得把未取得的文件当成已读。"
            "本轮只回复收到，不调用工具，不复述正文。" + "无新增事实。" * 80,
        ]
        assert fact not in "".join(questions) and filename not in questions[0]
        turns = []
        with patch.object(runtime, "create_client", side_effect=observed_factory):
            for index, question in enumerate(questions, 1):
                current = index
                turn = await manager.run_turn(sid, question)
                report(turn, f"real_setup_{index}", actual_requests.get(index, []))
                assert turn.state["summary_calls"] == 0, "setup unexpectedly compressed"
                turns.append(turn)

            before = turns[-1]
            old_record = await manager.get_record(sid, turns[0].state["run_id"])
            history = before.state["input"][:before.state["history_end"]]
            base_question = (
                "以下校验填充没有新增事实，请仅回复已保留此前有效要求和资料。"
                "不要调用工具，不要重复正文结论或识别码。填充："
            )
            previous_request = actual_requests[3][-1]
            base_input = history + [{"role": "user", "content": base_question}]
            prepared = prepare_request(
                settings.model_name, base_input, tools=runtime_tools(tools),
                instructions=previous_request.get("instructions"),
                max_output_tokens=settings.max_output_tokens,
            )
            risks = before.state["context_risks"][:before.state["history_end"]] + [None]
            measured = check_context_budget(prepared, risks, settings)
            assert measured["decision"] != "error"
            padding = max(1, measured["limit_tokens"] - measured["required_tokens"] + 512)
            pressure_question = base_question + "p" * padding
            current = 4
            compacted = await manager.run_turn(sid, pressure_question)
            events = report(compacted, "real_compaction", actual_requests.get(4, []))
            assert 1 <= compacted.state["summary_calls"] <= 2
            compacted_record = await manager.get_record(sid, compacted.state["run_id"])
            assert compacted.state["compaction"]["status"] == "used"
            assert compacted.state["active_context"]["version_id"] != before.state["active_context"]["version_id"]
            assert await manager.get_record(sid, turns[0].state["run_id"]) == old_record

            current = 5
            follow_question = (
                "请根据此前已有资料回答：有效观察结论是什么，哪种旧说法被否决？"
                "原样引用唯一观察识别码并注明来源相对路径，另说明 missing-case.txt 是否成功读取。"
                "不要重新调用工具。"
            )
            assert fact not in follow_question and filename not in follow_question
            following = await manager.run_turn(sid, follow_question)
            report(following, "real_post_compaction_followup", actual_requests.get(5, []))
            assert following.state["summary_calls"] == following.state["tool_calls"] == 0

        assert list(root.iterdir()) == [source] and source.read_bytes() == original_bytes
        assert cwd == Path.cwd()
    summary_calls = [item for item in actual_requests[4] if summary_request(item)]
    business_calls = [item for item in actual_requests[4] if not summary_request(item)]
    archived = compacted_record["summary_results"]
    assert len(summary_calls) == len(archived) == compacted.state["summary_calls"] and business_calls
    for index, (request, saved) in enumerate(zip(summary_calls, archived, strict=True)):
        attempt = saved["compaction"]
        assert attempt["pass_no"] == index + 1
        assert attempt["request_no"] == compacted.state["compaction"]["request_no"]
        assert attempt["summary_request_sha256"] == sha256(json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if index:
            first = archived[index - 1]
            assert attempt["candidate_input"] == {
                "summary_id": first["compaction"]["summary_id"], "checkpoint_id": first["checkpoint_id"],
            }
            assert attempt["sources"] == first["compaction"]["sources"]
    assert archived[-1]["compaction"]["summary_id"] == compacted.state["active_context"]["summary"]["summary_id"]
    summary_input = json.loads(summary_calls[0]["input"][0]["content"])
    assert fact in json.dumps(summary_input, ensure_ascii=False)
    assert pressure_question not in json.dumps(summary_input, ensure_ascii=False)
    assert business_calls[0]["input"][-1]["content"] == pressure_question
    summary_item = business_calls[0]["input"][0]
    assert summary_item["role"] == "assistant" and "derived_context_summary" in summary_item["content"]
    assert not any(item.get("type") == "function_call_output" for item in business_calls[0]["input"])
    assert actual_requests[5][0]["input"] == compacted.state["input"] + [{"role": "user", "content": follow_question}]
    assert actual_requests[5][0]["input"][0] == summary_item
    answer = following.state["result"]["text"]
    semantic = {
        "summary_preserved_fact": fact in summary_item["content"],
        "summary_preserved_source": filename in summary_item["content"],
        "answer_preserved_fact": fact in answer,
        "answer_preserved_source": filename in answer,
        "answer_mentions_effective_color": "蓝" in answer,
        "answer_mentions_rejected_color": "红" in answer,
        "answer_mentions_failed_source": "missing-case.txt" in answer and any(word in answer for word in ("失败", "不存在", "未成功", "未读取")),
    }
    print(json.dumps({"scenario": "real_semantic_observation", **semantic}, ensure_ascii=False))
    print(answer)
    assert all(semantic.values()), "真实摘要/追问未保留全部验收事实，请分别检查模型语义与协议证据。"
    print("PASS 实际摘要后追问使用已确认的新活动版本；关键事实来自被压缩旧记录。")
    print("本次真实表现不证明摘要无损或精确窗口适配；未验证跨进程恢复或远端追踪网页。")


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
