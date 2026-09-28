"""Human review packets and immutable, strictly bound assessment revisions.

Only a named human supplies semantic judgments. Evidence existence is never
treated as evidence support: the review form asks for that separate judgment.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from benchmarks.core.files import atomic_write_json, atomic_write_text, sha256_file, utc_now
from benchmarks.core.contracts import summarize, validate_result
from benchmarks.reporting.tasks import (
    render_research_report, research_exit_code, research_units, validate_research_result,
)


class ReviewSaveError(OSError):
    """Expose only artifacts confirmed saved before a later write failed."""

    def __init__(self, message: str, *, saved: dict, result: dict | None = None):
        super().__init__(message)
        self.saved = saved
        self.result = result


def _rubric(value: dict | None, result: dict) -> dict:
    if value is not None:
        return value
    definitions = [u["conditions"]["rubric"] for u in research_units(result)]
    if not definitions or any(v["availability"] != "known" for v in definitions):
        raise ValueError("recorded rubric unavailable")
    rules = definitions[0]["value"]
    if "cases" not in rules or any(v["value"] != rules for v in definitions):
        raise ValueError("one immutable rubric required for this review")
    return rules


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _list(value) -> list:
    return [value] if isinstance(value, dict) else value


def _answer_bindings(unit: dict) -> list[dict]:
    answers = []
    for turn in unit["turns"]:
        for index, evidence in enumerate(_list(turn["answer_evidence"]), 1):
            item = {"turn": turn["turn"], "index": index, "path": evidence["path"],
                    "availability": evidence["availability"], "sha256": None}
            if evidence["availability"] == "available":
                item["sha256"] = sha256_file(Path(evidence["path"]))
                if turn.get("answer_sha256") not in {None, item["sha256"]}:
                    raise ValueError("answer changed since execution")
            answers.append(item)
    return answers


def _bindings(result: dict, result_path: Path, rubric: dict) -> dict:
    if json.loads(result_path.read_text(encoding="utf-8")) != result:
        raise ValueError("saved result differs from supplied execution facts")
    for unit in research_units(result):
        material = unit["conditions"]["materials"]
        if material["availability"] == "known" and isinstance(material["value"], dict) and "material_root" in material["value"]:
            from benchmarks.core.materials import verify_materials
            if verify_materials(Path(material["value"]["material_root"]).parent) != material["value"]:
                raise ValueError("material changed since execution")
        recorded = unit["conditions"]["rubric"]
        if recorded["availability"] == "known":
            definition = recorded["value"]
            if not isinstance(definition, dict) or definition.get("version") != rubric["version"]:
                raise ValueError("recorded scoring version differs from the review rubric")
            if "cases" in definition and _hash(definition) != _hash(rubric):
                raise ValueError("recorded scoring rules differ from the review rubric")
    return {
        "run_id": result["run_id"], "result_sha256": sha256_file(result_path),
        "rubric_version": rubric["version"], "rubric_sha256": _hash(rubric),
        "units": [{"unit_id": unit["unit_id"], "case_id": unit["case_id"],
                   "sample_run_id": unit.get("sample_run_id", result["run_id"]),
                   "definition_version": unit["definition_version"],
                   "conditions_sha256": _hash(unit["conditions"]),
                   "materials_sha256": _hash(unit["conditions"]["materials"]),
                   "answers": _answer_bindings(unit)} for unit in research_units(result)],
    }


def _references(unit: dict, rule: dict) -> dict[str, dict]:
    references = {}
    for turn in unit["turns"]:
        number = turn["turn"]
        references[f"execution:turn:{number}"] = {
            "execution_status": turn["execution_status"], "agent_outcome": turn["agent_outcome"],
            "checkpoint_id": turn["checkpoint_id"],
        }
        for field, kind in (("answer_evidence", "answer"), ("process_evidence", "process")):
            for index, item in enumerate(_list(turn[field]), 1):
                references[f"{kind}:turn:{number}:{index}"] = item
    for index, item in enumerate(unit["evidence"], 1):
        references[f"unit:evidence:{index}"] = item
    for item in unit.get("coverage", []):
        references[f"coverage:{item['name']}"] = item
    for condition in rule["conditions"]:
        for source in condition.get("source_refs", []):
            references[f"source:{source}"] = {"material_path": source}
    return references


def _pending(item: dict) -> dict:
    return {"id": item["id"], "verdict": "pending_review", "basis": "",
            "evidence_refs": [], "evidence_support": "pending"}


def _template(result: dict, result_path: Path, rubric: dict) -> dict:
    assessments = []
    for unit in research_units(result):
        rule = rubric["cases"][unit["case_id"]]
        assessments.append({
            "unit_id": unit["unit_id"],
            "conditions": [_pending(item) for item in rule["conditions"]],
            "severe_errors": [{"id": item["id"], "present": None, "basis": "", "evidence_refs": []}
                              for item in rule["severe_errors"]],
            "mechanism_review": [_pending(item) for item in rule.get("mechanism_review", [])],
        })
    return {"review_schema_version": "1", "binding": _bindings(result, result_path, rubric),
            "reviewer": "", "reviewed_at": "", "assessments": assessments}


def _fence(text: str, language: str = "") -> str:
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _render_packet(result: dict, template: dict, rubric: dict) -> str:
    lines = ["# Research 人工评审包", "", f"运行：`{result['run_id']}`", "",
             "本包尚未进行语义评分。请核对实际回答、材料和过程；引用存在不代表其支持结论。", "",
             "填写同目录 review-template.json 的 reviewer、reviewed_at 及所有判定。"
             "每个非待评审判定填写 basis 和 evidence_refs；条件的 evidence_support 由评审者判断，"
             "只有 sufficient 可以支持 passed。未评条件保持 pending_review，不适用维度已从本题排除。", "",
             "没有保存回答时不能判通过。若已有真实 Agent run_id，可引用对应 execution:turn 记录将目标覆盖判为 failed；"
             "事实维度和严重错误仍待评审。未执行或缺少 Agent 运行证据的样本保持待评审。", "",
             "severe_errors.present 填 true／false；两者都需写判断依据及证据。"
             "mechanism_review 独立核对旧事实是否被近期内容重新提供，不能用正确回答反推摘要机制有效。", "",
             "评审导入会校验原结果、材料／规则版本及回答摘要，另存一个评审修订，不改写原执行结果。", "",
             f"评分规则：`{rubric['version']}`；原结果摘要：`{template['binding']['result_sha256']}`。", ""]
    dimensions = {item["id"]: item["description"] for item in rubric["dimensions"]}
    for unit in research_units(result):
        rule = rubric["cases"][unit["case_id"]]
        lines.extend([f"## {unit['case_id']} / {unit.get('phase', 'samples')} / 重复 {unit['repeat']}", "",
                      f"执行：{unit['execution_status']}；质量：{unit['verdict']}。", "",
                      "### 评分条件", ""])
        for condition in rule["conditions"]:
            lines.extend([f"- `{condition['id']}`（{dimensions[condition['dimension']]}）：{condition['description']}"])
            if condition.get("source_refs"):
                lines.append("  来源定位：" + "；".join(f"`{path}`" for path in condition["source_refs"]))
        lines.extend(["", "### 严重错误", ""])
        lines.extend(f"- `{item['id']}`：{item['description']}" for item in rule["severe_errors"])
        if rule.get("mechanism_review"):
            lines.extend(["", "### 人工机制核对", ""])
            lines.extend(f"- `{item['id']}`：{item['description']}" for item in rule["mechanism_review"])
        lines.extend(["", "### 实际机制与缺口", "",
                      _fence(json.dumps(unit.get("coverage", []), ensure_ascii=False, indent=2), "json"), ""])
        for turn in unit["turns"]:
            lines.extend([f"### 第 {turn['turn']} 轮", "", f"执行：{turn['execution_status']}。", ""])
            if turn.get("prompt_evidence"):
                evidence = turn["prompt_evidence"]
                lines.append(f"输入文件：[{Path(evidence['path']).name}](<{Path(evidence['path']).absolute().as_posix()}>)。")
            for evidence in _list(turn["answer_evidence"]):
                if evidence["availability"] == "available":
                    lines.extend(["", "实际回答（被测输出）：", "",
                                  _fence(Path(evidence["path"]).read_text(encoding="utf-8")), ""])
                else:
                    lines.extend(["", f"回答：{evidence['availability']}；没有可评审的已保存回答。", ""])
            if not turn["answer_evidence"]:
                lines.extend(["尚未取得本轮回答；不能填写通过。已确认 Agent 运行却未交付回答时，可依据实际执行记录判目标覆盖未通过；事实与证据维度继续待评审。", ""])
        lines.extend(["### 可使用的证据引用", "",
                      _fence(json.dumps(_references(unit, rule), ensure_ascii=False, indent=2), "json"), ""])
    return "\n".join(lines)


def make_review_packet(result: dict, output_directory: Path, *, rubric: dict | None = None) -> list[Path]:
    """Create a per-sample packet after result.json has been saved successfully."""
    output_directory = Path(output_directory)
    rubric = _rubric(rubric, result)
    if result.get("schema_version") != "3":
        raise ValueError("historical v1/v2 records are read-only")
    validate_result(result)
    validate_research_result(result)
    if not research_units(result):
        raise ValueError("review packet requires real Research samples")
    paths = {"template": output_directory / "review-template.json", "packet": output_directory / "review.md"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("review packet already exists; existing reviewer work is preserved")
    template = _template(result, output_directory / "result.json", rubric)
    packet = _render_packet(result, template, rubric)
    saved = {}
    try:
        atomic_write_json(paths["template"], template)
        saved["template"] = paths["template"]
        atomic_write_text(paths["packet"], packet)
        saved["packet"] = paths["packet"]
    except (OSError, ValueError, TypeError) as exc:
        raise ReviewSaveError("人工评审包保存失败。", saved=saved, result=result) from exc
    return list(saved.values())


def _indexed(items: list, expected: list[dict], label: str) -> dict[str, dict]:
    if not isinstance(items, list) or any(not isinstance(item, dict) or "id" not in item for item in items):
        raise ValueError(f"invalid review {label}")
    indexed = {item["id"]: item for item in items}
    if len(indexed) != len(items) or set(indexed) != {item["id"] for item in expected}:
        raise ValueError(f"review {label} does not exactly match the rubric")
    return indexed


def _basis(item: dict, references: dict, *, required: bool) -> None:
    if not isinstance(item.get("basis"), str) or not isinstance(item.get("evidence_refs"), list):
        raise ValueError("review basis and evidence references are required")
    refs = item["evidence_refs"]
    if any(not isinstance(reference, str) or reference not in references for reference in refs):
        raise ValueError("unknown review evidence reference")
    if required and (not item["basis"].strip() or not refs):
        raise ValueError("a human judgment requires a reason and evidence references")


def _judgment(item: dict, references: dict) -> None:
    if item.get("verdict") not in {"passed", "failed", "pending_review"}:
        raise ValueError("invalid human verdict")
    support = item.get("evidence_support")
    if support not in {"sufficient", "insufficient", "pending"}:
        raise ValueError("human evidence support judgment is missing")
    if item["verdict"] == "passed" and support != "sufficient":
        raise ValueError("evidence existence or insufficient support cannot establish a pass")
    if item["verdict"] != "pending_review" and support == "pending":
        raise ValueError("unassessed evidence cannot establish a quality verdict")
    _basis(item, references, required=item["verdict"] != "pending_review")


def apply_review(result: dict, review: dict, result_path: Path, *, rubric: dict | None = None) -> dict:
    """Validate human entries and derive a new result without mutating raw facts."""
    rubric = _rubric(rubric, result)
    if result.get("schema_version") != "3":
        raise ValueError("historical v1/v2 records are read-only")
    validate_result(result)
    validate_research_result(result)
    units = research_units(result)
    if not units or any(unit.get("review") for unit in units):
        raise ValueError("review must bind to original, unreviewed Research execution")
    if review.get("review_schema_version") != "1" or review.get("binding") != _bindings(result, Path(result_path), rubric):
        raise ValueError("review binding differs from execution, answers or rubric")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("a named human reviewer is required")
    if not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"].strip():
        raise ValueError("human review time is required")
    from datetime import datetime
    try:
        when = datetime.fromisoformat(review["reviewed_at"])
    except ValueError as exc:
        raise ValueError("human review time must be ISO 8601") from exc
    if when.tzinfo is None:
        raise ValueError("human review time requires a timezone")
    assessments = review.get("assessments")
    if not isinstance(assessments, list) or any(not isinstance(item, dict) for item in assessments):
        raise ValueError("review assessments must be a list")
    by_unit = {item.get("unit_id"): item for item in assessments}
    if len(by_unit) != len(assessments) or set(by_unit) != {unit["unit_id"] for unit in units}:
        raise ValueError("review must preserve all sample identities")
    derived = deepcopy(result)
    for unit in research_units(derived):
        rule = rubric["cases"][unit["case_id"]]
        assessment = by_unit[unit["unit_id"]]
        refs = _references(unit, rule)
        conditions = _indexed(assessment.get("conditions"), rule["conditions"], "conditions")
        severe = _indexed(assessment.get("severe_errors"), rule["severe_errors"], "severe errors")
        mechanism = _indexed(assessment.get("mechanism_review"), rule.get("mechanism_review", []), "mechanism")
        for item in [*conditions.values(), *mechanism.values()]:
            _judgment(item, refs)
        for item in severe.values():
            if item.get("present") is not None and type(item["present"]) is not bool:
                raise ValueError("severe error presence must be true, false or null")
            _basis(item, refs, required=item.get("present") is not None)
        has_answer = any(evidence["availability"] == "available" for turn in unit["turns"]
                         for evidence in _list(turn["answer_evidence"]))
        has_quality_judgment = any(item["verdict"] != "pending_review" for item in conditions.values()) or any(
            item.get("present") is not None for item in severe.values())
        if has_quality_judgment and not has_answer:
            observed_turns = {
                f"execution:turn:{turn['turn']}" for turn in unit["turns"]
                if turn["execution_status"] in {"completed", "failed"}
                and turn["run_id"]["availability"] == "known"
                and bool(turn["run_id"]["value"])
            }
            rules = {item["id"]: item for item in rule["conditions"]}
            judged = [item for item in conditions.values() if item["verdict"] != "pending_review"]
            if (not observed_turns or any(item.get("present") is not None for item in severe.values())
                    or any(item["verdict"] != "failed" or rules[item["id"]]["dimension"] != "target_coverage"
                           or not observed_turns.intersection(item["evidence_refs"]) for item in judged)):
                raise ValueError("semantic review needs an actual saved answer; only observed-run target failure can be judged without one")
        dimensions = []
        for dimension in rubric["dimensions"]:
            criteria = [conditions[item["id"]] for item in rule["conditions"]
                        if item["dimension"] == dimension["id"] and item.get("required", True)]
            if not criteria:
                continue
            verdicts = [item["verdict"] for item in criteria]
            from benchmarks.evaluators.checklist import aggregate
            verdict = aggregate(criteria)
            dimensions.append({"id": dimension["id"], "name": dimension["description"], "verdict": verdict,
                               "conditions": deepcopy(criteria), "assessment_basis": "用户人工条件判定；未以关键词或模型自评代替语义评审。"})
        unit["dimensions"] = dimensions
        unit.setdefault("assessments", []).append({"evaluator_id": "human-checklist", "kind": "checklist", "version": rubric["version"], "verdict": "failed" if any(d["verdict"] == "failed" for d in dimensions) else "passed" if all(d["verdict"] == "passed" for d in dimensions) else "pending_review", "reviewer": review["reviewer"]})
        human_names = {f"human:{item}" for item in mechanism}
        unit["coverage"] = [item for item in unit.get("coverage", []) if item["name"] not in human_names]
        unit["coverage"].extend({"name": f"human:{item['id']}",
                                 "status": {"passed": "verified", "failed": "gap", "pending_review": "pending_review"}[item["verdict"]],
                                 "reason": item["basis"] or "等待用户人工机制核对。", "evidence": item["evidence_refs"]}
                                for item in mechanism.values())
        all_judged = all(item["verdict"] != "pending_review" for item in [*conditions.values(), *mechanism.values()]) and all(
            item.get("present") is not None for item in severe.values())
        unit["review"] = {"reviewer": review["reviewer"], "reviewed_at": review["reviewed_at"],
                          "rubric_version": rubric["version"],
                          "complete": all_judged and unit["execution_status"] == "completed",
                          "conditions": deepcopy(list(conditions.values())),
                          "severe_errors": deepcopy(list(severe.values())),
                          "mechanism_review": deepcopy(list(mechanism.values()))}
        quality_failed = any(item["verdict"] == "failed" for item in dimensions) or any(item.get("present") is True for item in severe.values())
        can_pass = all_judged and has_answer and all(item["status"] == "verified" for item in unit["coverage"]) and all(
            turn["execution_status"] == "completed" and any(evidence["availability"] == "available"
                for evidence in _list(turn["answer_evidence"])) for turn in unit["turns"])
        unit["verdict"] = "pending_review"
        if unit["execution_status"] == "completed":
            unit["verdict"] = "failed" if quality_failed else "passed" if can_pass else "pending_review"
        unit["assessment_basis"] = "用户人工评分；执行事实保持原样。" + (
            "适用条件已完成评审。" if all_judged else "仍有未完成的质量或机制评审。")
    derived["summary"] = summarize(derived["units"])
    derived["review_source"] = {"run_id": result["run_id"], "result_path": str(Path(result_path).absolute()),
                                "result_sha256": review["binding"]["result_sha256"]}
    validate_result(derived)
    validate_research_result(derived)
    return derived


def import_review(result_path: Path, review_path: Path, output_root: Path | None = None, *, rubric: dict | None = None) -> dict:
    result_path, review_path = Path(result_path), Path(review_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    derived = apply_review(result, review, result_path, rubric=rubric)
    revision_id = utc_now().replace(":", "").replace("+", "_") + "-" + uuid4().hex
    root = Path(output_root) if output_root is not None else result_path.parent / "reviews"
    directory = root / revision_id
    directory.mkdir(parents=True, exist_ok=False)
    derived["review_source"]["revision_id"] = revision_id
    saved = {}
    try:
        atomic_write_json(directory / "review.json", review)
        saved["review"] = directory / "review.json"
        atomic_write_json(directory / "result.json", derived)
        saved["result"] = directory / "result.json"
        atomic_write_text(directory / "report.md", render_research_report(derived))
        saved["report"] = directory / "report.md"
    except (OSError, ValueError, TypeError) as exc:
        derived["framework_errors"].append({
            "stage": "saving", "code": "review_artifact_write_failed",
            "message": "人工评分修订的产物保存未全部完成；质量判定保留，运行退出为 2。",
            "exception_type": type(exc).__name__, "evidence": [str(path) for path in saved.values()],
        })
        try:
            atomic_write_json(directory / "result.json", derived)
            saved["result"] = directory / "result.json"
        except (OSError, ValueError, TypeError):
            # A previously saved result is still a real artifact, but the caller
            # must report the nonzero failure when reconciliation also fails.
            pass
        raise ReviewSaveError("人工评审修订保存失败；已保存产物仍保留。", saved=saved, result=derived) from exc
    return {"result": derived, "saved": saved, "exit_code": research_exit_code(derived)}


create_review_packet = make_review_packet
