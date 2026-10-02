"""Keep small results inline and disclose bounded previews of saved snapshots."""

from typing import Any

from lineage_agent.artifacts.store import (
    ARTIFACT_ERROR_CODES, CONTEXT_CHARS, ArtifactError, SessionArtifacts, tool_result_view,
)
from lineage_agent.tools.types import (
    MAX_OBSERVATION_BYTES, MAX_RECALL_BYTES, inline_tool_output, serialize_tool_output,
    tool_output_bytes,
)


_ERRORS = {
    "observation_serialize_error": "工具已执行成功，但结果无法在观测限制内序列化。",
    "observation_save_error": "工具已执行成功，但完整结果保存失败，未发布引用。",
    "observation_preview_error": "工具已执行成功，完整结果已保存，但预览生成失败。",
    **{code: str(ArtifactError(code)) for code in ARTIFACT_ERROR_CODES},
}
_BOUNDARIES = frozenset("\r\n。！？；.!?;")


def safe_observation_error(code: object) -> tuple[str, str]:
    """Regenerate fixed diagnostic text for model output and trace projection."""
    safe = code if type(code) is str and code in _ERRORS else "observation_preview_error"
    return safe, _ERRORS[safe]


def _unavailable(stage: str, error: Exception, saved: dict[str, Any] | None = None) -> dict[str, Any]:
    code, message = safe_observation_error(
        error.code if isinstance(error, ArtifactError) else f"observation_{stage}_error",
    )
    observation: dict[str, Any] = {
        "status": "unavailable", "stage": stage, "error": {"code": code, "message": message},
    }
    if saved is not None:
        observation.update({key: saved[key] for key in ("artifact_ref", "view_kind", "total_chars", "raw_bytes")})
    result = {"ok": True, "observation": observation}
    if tool_output_bytes(result) > MAX_OBSERVATION_BYTES:
        # Real Store metadata is fixed and small. Keep failure bounded even if
        # an application substitute violates that contract.
        result["observation"] = {
            "status": "unavailable", "stage": stage, "error": {"code": code, "message": message},
        }
    return result


def _segments(content: str, ranges: list[tuple[int, int]]) -> list[dict[str, Any]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": start, "end": end, "text": content[start:end]} for start, end in merged]


def _artifact_output(saved: dict[str, Any], segments: list[dict[str, Any]], *, strategy: str,
                     keyword_status: str, matches: list[dict[str, int]] | None = None,
                     matches_truncated: bool | None = None) -> dict[str, Any]:
    total = saved["total_chars"]
    complete = total == 0 or (len(segments) == 1 and segments[0]["start"] == 0 and segments[0]["end"] == total)
    observation = {
        "status": "available", "mode": "artifact",
        **{key: saved[key] for key in ("artifact_ref", "view_kind", "total_chars", "raw_bytes")},
        "strategy": strategy, "segments": segments, "complete": complete,
        "keyword_status": keyword_status, "matches_truncated": matches_truncated,
    }
    if "degradation_reason" in saved:
        observation["degradation_reason"] = saved["degradation_reason"]
    if matches is not None:
        observation["matches"] = matches
    return {"ok": True, "observation": observation}


def _fits(output: dict[str, Any]) -> bool:
    return tool_output_bytes(output) <= MAX_OBSERVATION_BYTES


def _head_tail(saved: dict[str, Any], content: str, keyword_status: str) -> dict[str, Any]:
    total = len(content)

    def output(segments: list[dict[str, Any]]) -> dict[str, Any]:
        return _artifact_output(saved, segments, strategy="head_tail", keyword_status=keyword_status)

    if total <= MAX_OBSERVATION_BYTES:
        complete = output(_segments(content, [(0, total)]))
        if _fits(complete):
            return complete
    # Reserve both segment objects using the widest possible range numbers;
    # allocate the remaining escaped UTF-8 text bytes roughly equally. Counting
    # characters instead would heavily favor an ASCII side over an emoji side.
    scaffold = output([{"start": total, "end": total, "text": ""}] * 2)
    text_budget = MAX_OBSERVATION_BYTES - tool_output_bytes(scaffold)
    if text_budget < 12:
        raise ValueError("preview metadata exceeds its budget")

    def fit_side(limit: int, *, tail: bool) -> int:
        low, high = 0, min(total, limit)
        while low < high:
            middle = (low + high + 1) // 2
            fragment = content[total - middle:] if tail else content[:middle]
            # Quotes are already included in the empty scaffold string.
            size = len(serialize_tool_output(fragment).encode("utf-8")) - 2
            if size <= limit:
                low = middle
            else:
                high = middle - 1
        return low

    half = text_budget // 2
    head_count = fit_side(half, tail=False)
    tail_count = fit_side(text_budget - half, tail=True)
    result = output(_segments(content, [(0, head_count), (total - tail_count, total)]))
    # Prefer nearby sentence boundaries after establishing a fitting range.
    segments = result["observation"]["segments"]
    if not result["observation"]["complete"] and len(segments) == 2:
        head, tail = segments
        head_end = head["end"]
        for index in range(head_end - 1, max(0, head_end - 128) - 1, -1):
            if content[index] in _BOUNDARIES:
                head_end = index + 1
                break
        tail_start = tail["start"]
        for index in range(tail_start, min(total - 1, tail_start + 128)):
            if content[index] in _BOUNDARIES:
                tail_start = index + 1
                break
        preferred = output(_segments(content, [(0, head_end), (tail_start, total)]))
        if _fits(preferred):
            result = preferred
    if not _fits(result) or (total and not result["observation"]["segments"]):
        raise ValueError("preview cannot disclose any content")
    return result


def _keyword_preview(saved: dict[str, Any], content: str, page: dict[str, Any]) -> dict[str, Any]:
    hits = page["hits"]

    def candidate(count: int, context: int) -> dict[str, Any]:
        selected = hits[:count]
        matches = [match.copy() for hit in selected for match in hit["matches"]]
        ranges = [(max(0, hit["matches"][0]["start"] - context),
                   min(len(content), max(match["end"] for match in hit["matches"]) + context))
                  for hit in selected]
        return _artifact_output(
            saved, _segments(content, ranges), strategy="keywords", keyword_status="matched",
            matches=matches, matches_truncated=page["has_more"] or count < len(hits),
        )

    # Keep the longest prefix of whole matching-position groups. Context is
    # expendable before a group; matching metadata is never partially returned.
    for count in range(len(hits), 0, -1):
        # Merging newly overlapping intervals can remove metadata, so response
        # size is not monotonic in context length. The bounded 81 values are
        # cheaper and more reliable than assuming a binary-search invariant.
        for context in range(CONTEXT_CHARS, -1, -1):
            result = candidate(count, context)
            if _fits(result):
                return result
    raise ValueError("preview cannot disclose a complete matching group")


class ObservationManager:
    """One session's deterministic pre-context policy, without message mutation."""

    def __init__(self, artifacts: SessionArtifacts) -> None:
        self._artifacts = artifacts

    def observe(self, raw_result: str, *, tool_name: str, keywords=(),
                native_recall: bool = False) -> dict[str, Any]:
        try:
            if type(raw_result) is not str:
                raise ValueError("raw result must be text")
            output = inline_tool_output(raw_result)
            size = tool_output_bytes(output)
            if native_recall:
                if size > MAX_RECALL_BYTES:
                    raise ValueError("native recall exceeded its disclosure budget")
                return output
            if size <= MAX_OBSERVATION_BYTES:
                return output
        except Exception as error:
            return _unavailable("serialize", error)
        try:
            saved = self._artifacts.put(raw_result, tool_name=tool_name)
        except Exception as error:
            return _unavailable("save", error)
        try:
            content, _, _ = tool_result_view(raw_result, tool_name)
            if keywords:
                page = self._artifacts.search(saved["artifact_ref"], list(keywords))
                if page["hits"]:
                    return _keyword_preview(saved, content, page)
                return _head_tail(saved, content, "no_match")
            return _head_tail(saved, content, "not_requested")
        except Exception as error:
            return _unavailable("preview", error, saved)
