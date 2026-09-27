"""Frozen WeChat research tasks. Rubrics remain on the evaluation side."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


SUITE_ID = "research-wechat"
DEFINITION_VERSION = "wechat-cases-v1"
PROFILE = {
    "profile_id": "wechat-v1", "version": "wechat-profile-v1",
    "model_name": "deepseek-flash", "base_url": "https://api.deepseek.com",
    "max_model_calls": 8, "timeout_seconds": 60,
    "context_budget_tokens": 32768, "max_output_tokens": 4096,
    "directory_max_entries": 100, "text_max_bytes": 65536,
    "text_total_max_bytes": 262144, "context_safety_margin_tokens": 1024,
    "opaque_fallback_tokens": 4096, "context_trigger_percent": 90,
    "context_keep_recent_turns": 2, "context_target_percent": 70,
    "summary_max_output_tokens": 2048, "summary_max_bytes": 8192,
    "sample_timeout_seconds": 300, "request_limit": 400,
    "execution_time_limit_seconds": 7200, "pilot_repeats": 1, "formal_repeats": 3,
    "automatic_retries": 0,
}

DIMENSIONS = (
    "facts_evidence", "target_coverage", "uncertainty_boundaries",
    "constraints_history", "execution_consistency",
)
_SINGLE = [item for item in DIMENSIONS if item != "constraints_history"]
_INTRO = (
    "请仅依据本次可访问的 WeChat Reporting 材料进行研究。code/ 是固定版本模板，"
    "requirements/ 是另一固定版本的资料；SOURCE_INDEX.md 列出来源。"
    "其中 PDF 文本是评测端提供的带页码转写，并非你直接解析 PDF。"
    "需要查阅相关原文和代码，结论给出可定位的材料路径及函数、段落或原句；"
    "区分已知事实、推断、建议和证据不足。如未执行测试，不得声称测试或压测已通过。\n\n"
)

CASES = (
    {
        "case_id": "wechat-capabilities", "version": DEFINITION_VERSION,
        "title": "离线采集与签名权威性", "coverage": ["project_scope", "offline", "signature", "source_evidence"],
        "dimensions": _SINGLE, "mechanism": None, "settings_overrides": {},
        "turns": [_INTRO + "这个项目现在已经支持离线采集了吗？手写签名是否已实现资料所说的数据权威性验证？请分别说明当前实现、与资料目标的关系、依据和仍待验证或实现的部分。"],
    },
    {
        "case_id": "wechat-security", "version": DEFINITION_VERSION,
        "title": "加密、脱敏要求与实现", "coverage": ["requirements_conflict", "security", "uncertainty"],
        "dimensions": _SINGLE, "mechanism": None, "settings_overrides": {},
        "turns": [_INTRO + "加密与脱敏要求已经确认并得到满足了吗？请查阅 requirements/ 中的 Markdown 和两份 PDF 转写，分别分析数据库、小程序前端展示、拉取到本地的数据，以及当前模板代码中的实现。说明各份资料的表述、确认依据和无法确定之处。"],
    },
    {
        "case_id": "wechat-capacity-ocr", "version": DEFINITION_VERSION,
        "title": "容量证据与 OCR 覆盖", "coverage": ["capacity", "ocr", "follow_up", "open_recommendations"],
        "dimensions": list(DIMENSIONS), "mechanism": None, "settings_overrides": {},
        "turns": [
            _INTRO + "当前版本能满足资料中的数据量和并发目标吗？OCR 是否已经覆盖？请给出依据、检查范围和结论的限制。",
            "沿用上一轮的问题和已有资料。请把关键判断逐项对应到具体源码或原文位置；说明哪些尚未验证，并给出下一步验证办法及为什么需要这些验证。不要把建议说成已完成的工作。",
        ],
    },
    {
        "case_id": "wechat-correction", "version": DEFINITION_VERSION,
        "title": "用户纠正与范围继承", "coverage": ["project_scope", "correction", "history", "constraints"],
        "dimensions": list(DIMENSIONS), "mechanism": None, "settings_overrides": {},
        "turns": [
            _INTRO + "请以 requirements/ 所述最终业务目标为范围，研究当前 code/ 模板距离目标的主要差距和建议。",
            "纠正一下：这次验收仅评估模板自身 README 和开发说明明确承诺的范围，不评估最终业务系统。请重新组织结论，只保留“完成情况／依据／待办”三个栏目；未承诺的后端功能不作为模板缺陷。后续仍沿用这个范围和格式。",
            "请基于这次讨论给接手同事一份简短交接报告，保留目前有效的要求、可定位依据及仍需验证的事项。",
        ],
    },
    {
        "case_id": "wechat-read-limit", "version": DEFINITION_VERSION,
        "title": "读取受限后的证据边界", "coverage": ["read_failure", "evidence_limits", "continuation"],
        "dimensions": list(DIMENSIONS), "mechanism": "read_limit",
        "settings_overrides": {"text_max_bytes": 4096},
        "turns": [
            _INTRO + "请先实际尝试读取 code/tests/collection-service.test.ts，判断它验证了哪些行为、不能证明哪些生产能力。若读取失败，请如实说明，并继续利用可以取得的相关材料，不要编造该文件内容或测试执行结果。",
            "请仅基于实际取得的材料给出最终结论，明确直接查阅到的内容、未能取得的资料、这些限制如何影响判断，以及补足证据的办法。",
        ],
    },
    {
        "case_id": "wechat-compaction", "version": DEFINITION_VERSION,
        "title": "摘要后旧事实、来源与约束", "coverage": ["compaction", "history", "source_retention", "constraints"],
        "dimensions": list(DIMENSIONS), "mechanism": "compaction",
        "settings_overrides": {"summary_max_output_tokens": 8192, "summary_max_bytes": 4096},
        "turns": [
            _INTRO + "请读取 code/miniprogram/domain/form.ts 和 evaluation-fixtures/background.txt。背景文件只是评测构造，不是项目事实。记住源码中的字段修改、签名关联、笔迹判断规则及其来源，供后续追问。本会话的长期要求：只对已取得证据作结论；回答相关问题时给出材料路径与函数名；不把签名关联当作身份核验。从下一轮开始不得再调用工具或重新读取材料。本轮只回复“已阅”，不要复述规则、来源或长期要求。",
            "继续保留此前取得的资料和有效要求。本轮只回复“收到甲”，不得调用工具，也不要复述具体事实或要求。",
            "继续保留此前取得的资料和有效要求。本轮只回复“收到乙”，不得调用工具，也不要复述具体事实或要求。",
            "以下字符仅用于上下文压力，不承载项目事实。继续保留此前资料与要求，不得调用工具；本轮只回复“收到丙”，不要复述具体事实或要求。\n",
            "不调用工具、不重新读取材料，回答：修改一个字段后已有签名会怎样，值未变时是否相同？笔迹判断究竟证明什么、不能证明什么？请给出旧资料中的来源，并说明本会话仍生效的要求。",
        ],
        "pressure": {"turn": 4, "formula_version": "utf8-budget-v1", "formula": "max(1, trigger-required+512)", "max_bytes": 16384},
    },
)


def get_case(case_id: str) -> dict:
    """Return an isolated definition; callers cannot mutate the fixed registry."""
    for case in CASES:
        if case["case_id"] == case_id:
            return deepcopy(case)
    raise ValueError(f"unknown research case: {case_id}")


def load_rubric() -> dict:
    return json.loads(Path(__file__).with_name("rubric.json").read_text(encoding="utf-8"))


def definition_fingerprint() -> str:
    payload = {"cases": CASES, "profile": PROFILE, "rubric": load_rubric()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
