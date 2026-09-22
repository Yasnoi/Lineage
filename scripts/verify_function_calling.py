"""Check the configured real endpoint with a temporary, fixed-result tool."""

import asyncio
import json
from pathlib import Path
import sys

from lineage_agent.agent.runtime import run_agent
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.tools.types import ToolArgumentError


TOOL = {
    "type": "function",
    "name": "get_verification_value",
    "description": "Read the verification value of the sample key.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string", "enum": ["sample"]}},
        "required": ["key"],
        "additionalProperties": False,
    },
    "strict": True,
}


async def verify():
    settings = load_settings(Path(__file__).resolve().parents[1] / ".env")
    if settings.max_model_calls < 2:
        raise ConfigurationError("完整工具循环需要 MAX_MODEL_CALLS 至少为 2。")

    async def execute_tool(name, arguments):
        if name != "get_verification_value" or arguments != {"key": "sample"}:
            raise ToolArgumentError("只接受 get_verification_value(key='sample')。")
        return "LINEAGE_FIXED_SAMPLE_7C92"

    def report(label, state):
        print(f"{label}: {json.dumps({
            'run_id': state['run_id'],
            'log_path': state['log_path'],
            'log_complete': state['log_complete'],
            'stop_reason': state['stop_reason'],
            'model_calls': state['model_calls'],
            'tool_requests': state['tool_requests'],
            'tool_calls': state['tool_calls'],
            'tool_outcomes': state['tool_outcomes'],
            'last_usage': state['result']['usage'] if state['result'] else None,
            'usage_totals': state['usage_totals'],
        }, ensure_ascii=False)}")
        if state["stop_reason"] == "completed":
            print(state["result"]["text"])
        elif state["error"]:
            print(state["error"], file=sys.stderr)

    plain = await run_agent("请简短回答：一加一等于几？", settings)
    report("无工具回归", plain)
    if (
        plain["stop_reason"] != "completed" or plain["model_calls"] != 1
        or plain["tool_requests"] != 0 or plain["tool_calls"] != 0
        or not plain["log_complete"]
    ):
        raise AssertionError("真实端点无工具回归未通过。")
    print("PASS 真实端点无工具回归")

    state = await run_agent(
        "请调用 get_verification_value 工具读取 key 为 sample 的值。"
        "拿到工具结果后，直接原样报告该值，不要再次调用工具，也不要猜测结果。",
        settings, tools=[TOOL], execute_tool=execute_tool,
    )
    report("Function Calling 验证", state)
    if (
        state["stop_reason"] != "completed" or state["model_calls"] != 2
        or state["tool_requests"] != 1 or state["tool_calls"] != 1
        or not state["log_complete"] or state["tool_outcomes"]["succeeded"] != 1
    ):
        raise AssertionError("真实端点未完成预期的两次模型调用、一次工具执行。")
    tool_outputs = [
        item for item in state["input"] if item.get("type") == "function_call_output"
    ]
    if len(tool_outputs) != 1:
        raise AssertionError("缺少唯一的工具回传结果。")
    envelope = json.loads(tool_outputs[0]["output"])
    if envelope.get("ok") is not True or envelope["result"] not in state["result"]["text"]:
        raise AssertionError("最终回答未使用验证工具的固定返回结果。")
    events = [
        json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()
    ]
    created = [event["call_id"] for event in events if event["event"] == "tool.result_created"]
    model_inputs = [
        event["input_tool_result_call_ids"] for event in events if event["event"] == "model.started"
    ]
    if created != [tool_outputs[0]["call_id"]] or model_inputs != [[], created]:
        raise AssertionError("运行日志未记录工具结果生成与后续模型请求的关联。")
    print("PASS 真实端点 Function Calling 与工具结果回传")


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
