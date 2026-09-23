"""Application-side byte estimates and explicit reserves for opaque protocol data.

These units are a policy, not a provider tokenizer or a context-window guarantee.
The complete request is validated before opaque data is projected to JSON null.
"""

import json
import math
from typing import Any, Literal, TypedDict

from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.types import TokenUsage


class ContextRisk(TypedDict):
    opaque_count: int
    risk_tokens: int
    basis: Literal["input_fallback", "model_fallback", "response_output_usage"]
    source_run_id: str
    source_model_call: int | None


class ContextBudget(TypedDict):
    method: str
    step_id: str | None
    transparent_tokens: int | None
    opaque_items: int | None
    opaque_risk_tokens: int | None
    required_tokens: int | None
    limit_tokens: int
    output_reserve_tokens: int
    safety_margin_tokens: int
    trigger_percent: int
    trigger_tokens: int
    decision: Literal["allow", "exceeded", "error"]
    request_sent: bool
    error_code: str | None


METHOD = "utf8_json_opaque_v1"
_ITEM_TYPES = {"message", "function_call", "function_call_output", "reasoning"}
_TEXT_TYPES = {"input_text", "output_text", "refusal", "reasoning_text", "summary_text", "text"}


def _validate_json(value: Any, ancestors: set[int] | None = None) -> None:
    """Reject lossy JSON coercions as well as non-finite, cyclic or invalid UTF-8 data."""
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        value.encode("utf-8")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return
    if type(value) not in (dict, list):
        raise ValueError("non-JSON value")
    ancestors = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in ancestors:
        raise ValueError("cyclic JSON value")
    ancestors.add(identity)
    try:
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError("non-string JSON key")
                key.encode("utf-8")
                _validate_json(child, ancestors)
        else:
            for child in value:
                _validate_json(child, ancestors)
    finally:
        ancestors.remove(identity)


def _content_projection(value: Any) -> tuple[Any, int]:
    if type(value) is not list:
        return _transparent_projection(value)
    projected: list[Any] = []
    count = 0
    for part in value:
        if type(part) is dict and part.get("type") in _TEXT_TYPES:
            transformed, units = _transparent_projection(part)
        else:
            transformed, units = None, 1
        projected.append(transformed)
        count += units
    return projected, count


def _transparent_projection(value: Any) -> tuple[Any, int]:
    """Find encrypted fields in known protocol data, never parse string payloads."""
    if type(value) is dict:
        result: dict[str, Any] = {}
        count = 0
        for key, child in value.items():
            if key == "encrypted_content":
                result[key] = None
                count += 1
            else:
                result[key], units = _transparent_projection(child)
                count += units
        return result, count
    if type(value) is list:
        result_list: list[Any] = []
        count = 0
        for child in value:
            projected, units = _transparent_projection(child)
            result_list.append(projected)
            count += units
        return result_list, count
    return value, 0


def _item_projection(item: Any) -> tuple[Any, int]:
    if type(item) is not dict:
        return None, 1
    kind = item.get("type")
    message = kind == "message" or (kind is None and "role" in item)
    if not message and kind not in _ITEM_TYPES:
        # Do not also count encrypted fields or content within this opaque unit.
        return None, 1
    projected: dict[str, Any] = {}
    count = 0
    for key, value in item.items():
        if key == "encrypted_content":
            projected[key] = None
            count += 1
        elif (
            (message and key == "content")
            or (kind == "reasoning" and key in {"content", "summary"})
            or (kind == "function_call_output" and key == "output")
        ):
            projected[key], units = _content_projection(value)
            count += units
        else:
            projected[key], units = _transparent_projection(value)
            count += units
    return projected, count


def annotate_context_risks(
    items: list[dict[str, Any]],
    *,
    fallback_tokens: int,
    run_id: str,
    model_call: int | None = None,
    usage: TokenUsage | None = None,
) -> list[ContextRisk | None]:
    """Annotate new facts without making serialization errors erase those facts.

    Invalid input is rejected by measurement or existing protocol validation.
    In particular, errors while inspecting an item never discard the response.
    """
    try:
        output_tokens = usage.get("output_tokens") if model_call is not None and isinstance(usage, dict) else None
        trusted_usage = type(output_tokens) is int and output_tokens > 0
    except Exception:
        output_tokens, trusted_usage = None, False
    result: list[ContextRisk | None] = []
    for item in items:
        try:
            _, count = _item_projection(item)
        except Exception:
            result.append(None)
            continue
        result.append({
            "opaque_count": count,
            "risk_tokens": output_tokens if trusted_usage else fallback_tokens,
            "basis": (
                "response_output_usage" if trusted_usage
                else "model_fallback" if model_call is not None else "input_fallback"
            ),
            "source_run_id": run_id,
            "source_model_call": model_call,
        } if count else None)
    return result


def budget_measurement_error(
    settings: ModelSettings, *, step_id: str | None = None,
) -> ContextBudget:
    return {
        "method": METHOD,
        "step_id": step_id,
        "transparent_tokens": None,
        "opaque_items": None,
        "opaque_risk_tokens": None,
        "required_tokens": None,
        "limit_tokens": settings.context_budget_tokens,
        "output_reserve_tokens": settings.max_output_tokens,
        "safety_margin_tokens": settings.context_safety_margin_tokens,
        "trigger_percent": settings.context_trigger_percent,
        "trigger_tokens": settings.context_budget_tokens * settings.context_trigger_percent // 100,
        "decision": "error",
        "request_sent": False,
        "error_code": "context_measurement_error",
    }


def _risk_tokens(risk: ContextRisk | None, count: int) -> int:
    if count == 0 and risk is None:
        return 0
    if (
        type(risk) is not dict
        or type(risk.get("opaque_count")) is not int
        or risk["opaque_count"] != count or count <= 0
        or type(risk.get("risk_tokens")) is not int or risk["risk_tokens"] <= 0
        or risk.get("basis") not in {"input_fallback", "model_fallback", "response_output_usage"}
        or type(risk.get("source_run_id")) is not str or not risk["source_run_id"]
    ):
        raise ValueError("invalid risk provenance")
    call = risk.get("source_model_call")
    if risk["basis"] == "input_fallback":
        if call is not None:
            raise ValueError("invalid input provenance")
    elif type(call) is not int or call <= 0:
        raise ValueError("invalid model provenance")
    return count * risk["risk_tokens"]


def check_context_budget(
    request: dict[str, Any],
    risks: list[ContextRisk | None],
    settings: ModelSettings,
    *,
    step_id: str | None = None,
) -> ContextBudget:
    """Return a safe decision; ordinary measurement errors never allow a request."""
    diagnostic = budget_measurement_error(settings, step_id=step_id)
    try:
        _validate_json(request)
        if type(request) is not dict or type(risks) is not list:
            raise ValueError("invalid measurement input")
        if (request.get("max_output_tokens") != settings.max_output_tokens
                or type(request.get("max_output_tokens")) is not int
                or request.get("truncation") != "disabled"):
            raise ValueError("request policy differs from budget")
        projected = dict(request)
        items = request["input"]
        opaque_items, reserved = 0, 0
        if type(items) is str:
            if risks:
                raise ValueError("unexpected input provenance")
        elif type(items) is list and len(items) == len(risks):
            projected_items: list[Any] = []
            for item, risk in zip(items, risks, strict=True):
                value, count = _item_projection(item)
                projected_items.append(value)
                opaque_items += count
                reserved += _risk_tokens(risk, count)
            projected["input"] = projected_items
        else:
            raise ValueError("input provenance mismatch")
        transparent = len(json.dumps(
            projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
        required = transparent + reserved + settings.max_output_tokens + settings.context_safety_margin_tokens
        exceeded = required >= diagnostic["trigger_tokens"]
        diagnostic.update(
            transparent_tokens=transparent,
            opaque_items=opaque_items,
            opaque_risk_tokens=reserved,
            required_tokens=required,
            decision="exceeded" if exceeded else "allow",
            error_code="context_budget_exceeded" if exceeded else None,
        )
    except Exception:
        pass
    return diagnostic
