"""Synthetic contract examples; these are not executable Research evaluations."""

from copy import deepcopy

from benchmarks.core.contracts import known, not_applicable, summarize, unknown


EXAMPLE_NOTICE = "契约样例，不是实际 Research 验收；所有轮次和证据引用均为虚构。"


def _evidence(kind: str, path: str) -> dict:
    return {
        "kind": kind,
        "path": path,
        "source": "contract_example",
        "availability": "missing",
    }


def _turn(case: str, number: int, session: str | None) -> dict:
    prefix = f"examples-only/{case}/turn-{number}"
    return {
        "turn": number,
        "run_id": known(f"example-{case}-turn-{number}"),
        "session_id": (
            known(session)
            if session is not None
            else not_applicable("独立单次运行，不继承会话。")
        ),
        "checkpoint_id": (
            known(f"example-{case}-checkpoint-{number}")
            if session is not None
            else not_applicable("独立单次运行不使用会话检查点。")
        ),
        "agent_outcome": known("completed"),
        "answer_evidence": [_evidence("final_answer", f"{prefix}/answer.txt")],
        "process_evidence": [_evidence("process", f"{prefix}/process.jsonl")],
        "active_version": (
            known("example-context-v1")
            if session is not None
            else not_applicable("该样例不使用会话活动版本。")
        ),
    }


def research_examples() -> list[dict]:
    """Return fresh objects so tests cannot contaminate other examples."""
    examples = []
    for case, session, count in (
        ("single-turn", None, 1),
        ("follow-up", "example-follow-up-session", 2),
        ("after-summary", "example-summary-session", 3),
    ):
        turns = [_turn(case, number, session) for number in range(1, count + 1)]
        unit = {
            "suite_id": "contract-examples-only",
            "unit_id": case,
            "category": "research",
            "definition_version": "1",
            "granularity": "case",
            "execution_mode": "contract_example",
            "conditions": {
                "model": unknown("契约样例没有调用模型。"),
                "config": known({"id": "example-config", "version": "1"}),
                "materials": known(
                    {"id": "example-readable-materials", "version": "1"}
                ),
                "scoring": known(
                    {
                        "id": "example-review-rubric",
                        "version": "1",
                        "location": "examples-only/evaluator/rubric.json",
                        "agent_accessible": False,
                    }
                ),
            },
            "execution_status": "completed",
            "verdict": "pending_review",
            "agent_outcome": known("completed"),
            "assessment_basis": f"{EXAMPLE_NOTICE} 执行完成，等待人工评分。",
            "dimensions": [
                {
                    "name": dimension,
                    "verdict": "pending_review",
                    "reason": "尚未人工评分。",
                    "score": unknown("无实际评分。"),
                }
                for dimension in ("factual_support", "goal_coverage")
            ],
            "issues": [],
            "evidence": [
                deepcopy(evidence)
                for turn in turns
                for field in ("answer_evidence", "process_evidence")
                for evidence in turn[field]
            ],
            "metrics": {
                "duration_seconds": unknown("契约样例没有实际执行。"),
                "usage": not_applicable("契约样例未调用真实模型。"),
            },
            "turns": turns,
        }
        examples.append(unit)

    summary_case = examples[-1]
    summary_case["turns"][0]["task"] = (
        "阅读材料中的约束：本轮分析仅限只读研究，并记录资料来源。"
    )
    summary_case["turns"][1]["task"] = "继续比较实现与目标的对应关系。"
    summary_case["turns"][2]["task"] = "沿用先前约束，说明下一步研究方向。"
    summary_case["turns"][2]["active_version"] = known("example-context-v2")
    summary_case["turns"][2]["summary"] = {
        "id": "example-summary-1",
        "source_turns": [1, 2],
        "source_active_version": "example-context-v1",
        "activated_version": "example-context-v2",
        "evidence": _evidence(
            "summary", "examples-only/after-summary/summary.json"
        ),
        "mechanism_evidence": _evidence(
            "summary_activation", "examples-only/after-summary/activation.json"
        ),
    }
    summary_case["evidence"].extend(
        deepcopy(summary_case["turns"][2]["summary"][field])
        for field in ("evidence", "mechanism_evidence")
    )
    return examples


def example_result(units: list[dict] | None = None) -> dict:
    selected = research_examples() if units is None else deepcopy(units)
    return {
        "schema_version": "1",
        "run_id": "contract-example-only",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:00+00:00",
        "selected_suites": list(dict.fromkeys(unit["suite_id"] for unit in selected)),
        "baseline": {
            "agent_commit": unknown("契约样例没有被测 Git 基线。"),
            "workspace_dirty": unknown("契约样例不检查工作区。"),
            "python": unknown("契约样例不记录执行环境。"),
            "dependencies": unknown("契约样例不记录实际依赖版本。"),
            "lockfile_sha256": unknown("契约样例不采集锁文件。"),
        },
        "units": selected,
        "summary": summarize(selected),
        "framework_errors": [],
    }
