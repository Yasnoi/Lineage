"""Mechanical predicates over observed facts; never semantic keyword scoring."""
def control_assessment(unit):
    """Expose the verified assertion judgment without recounting Agent runs."""
    return {"evaluator_id": "control-assertions", "version": "3", "kind": "deterministic",
            "verdict": unit["verdict"], "basis": unit["assessment_basis"],
            "groups": [{"group_id": group["group_id"], "verdict": group["verdict"]}
                       for group in unit["groups"]]}


def mechanism_assessment(unit):
    observations = unit.get("coverage", [])
    states = [item["status"] for item in observations]
    verdict = "failed" if "gap" in states else "pending_review" if "pending_review" in states else "passed"
    return {"evaluator_id": "mechanism-rules", "version": "1", "kind": "deterministic",
            "applicable": bool(observations), "verdict": verdict if observations else "pending_review",
            "basis": "仅检查声明机制的实际过程证据；不判断回答含义。" if observations else "本任务没有特定机制前置条件。",
            "observations": observations}
