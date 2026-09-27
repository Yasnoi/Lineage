"""Offline checks for real-trace usage coverage; no dotenv or endpoint access."""

from copy import deepcopy
from unittest.mock import patch

from lineage_agent.models.responses import parse_response
from lineage_agent.recording.trace_data import TraceSanitizer
from verify_tracing import summarize_usage_coverage
from check_support import check_group, observe_harness


MISSING = object()
TOKEN_FIELDS = ["input_tokens", "output_tokens", "total_tokens"]


def projected_outputs(usage=MISSING):
    raw = {
        "status": "completed",
        "output": [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "synthetic answer"}],
        }],
    }
    if usage is not MISSING:
        raw["usage"] = deepcopy(usage)
    cleaner = TraceSanitizer([], None)
    return cleaner.cap_payload({"outputs": cleaner.model_outputs(parse_response(raw))})["outputs"]


def check_usage_categories():
    cases = (
        ("absent", MISSING, "none", []),
        ("null", None, "none", []),
        ("empty", {}, "none", []),
        ("all_null", dict.fromkeys(TOKEN_FIELDS), "none", []),
        ("boolean", {"input_tokens": True, "output_tokens": False}, "none", []),
        ("negative", {"input_tokens": -1, "output_tokens": -3}, "none", []),
        ("wrong_types", {"input_tokens": "3", "output_tokens": 2.0, "total_tokens": []}, "none", []),
        ("nonfinite", {"input_tokens": float("inf"), "output_tokens": float("nan")}, "none", []),
        ("details_only", {
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 4},
        }, "none", []),
        ("input_zero", {"input_tokens": 0}, "partial", ["input_tokens"]),
        ("output_only", {"output_tokens": 3}, "partial", ["output_tokens"]),
        ("total_only", {"total_tokens": 5}, "partial", ["total_tokens"]),
        ("mixed_validity", {"input_tokens": 0, "output_tokens": 4, "total_tokens": False}, "partial", TOKEN_FIELDS[:2]),
        ("all_zero", dict.fromkeys(TOKEN_FIELDS, 0), "full", TOKEN_FIELDS),
        ("complete", {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}, "full", TOKEN_FIELDS),
    )
    for label, raw_usage, category, fields in cases:
        projected = projected_outputs(raw_usage)
        before = deepcopy(projected)
        summary = summarize_usage_coverage([(projected, deepcopy(projected))])
        assert summary["usage_coverage_counts"] == {
            name: int(name == category) for name in ("none", "partial", "full")
        }, label
        assert summary["usage_coverage_by_call"] == [{
            "model_call": 1, "coverage": category, "reported_fields": fields,
        }], label
        assert summary["usage_covered"] is (category != "none"), label
        assert projected == before, label
        usage = projected.get("usage_metadata", {})
        assert [name for name in TOKEN_FIELDS if name in usage] == fields, label


def check_mixed_run_and_empty_run():
    projected = [projected_outputs(value) for value in (
        None, {"input_tokens": 0}, {}, {"output_tokens": 4},
        {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
    )]
    result = summarize_usage_coverage([(value, deepcopy(value)) for value in projected])
    assert result["usage_coverage_counts"] == {"none": 2, "partial": 2, "full": 1}
    assert [item["model_call"] for item in result["usage_coverage_by_call"]] == [1, 2, 3, 4, 5]
    assert [item["coverage"] for item in result["usage_coverage_by_call"]] == [
        "none", "partial", "none", "partial", "full",
    ]
    assert result["usage_covered"] is True
    assert summarize_usage_coverage([]) == {
        "usage_covered": False,
        "usage_coverage_counts": {"none": 0, "partial": 0, "full": 0},
        "usage_coverage_by_call": [],
    }


def check_remote_mismatches():
    local = projected_outputs({"input_tokens": 0})
    for remote_usage in (MISSING, {}, {"input_tokens": 1}, {"input_tokens": False}, {"input_tokens": 0.0}):
        remote = deepcopy(local)
        if remote_usage is MISSING:
            del remote["usage_metadata"]
        else:
            remote["usage_metadata"] = remote_usage
        try:
            summarize_usage_coverage([(local, remote)])
        except AssertionError:
            pass
        else:
            raise AssertionError("different remote usage was accepted")
    local = projected_outputs({"input_tokens_details": {"cached_tokens": 0}})
    remote = deepcopy(local)
    remote["usage_metadata"]["input_token_details"]["cache_read"] = False
    try:
        summarize_usage_coverage([(local, remote)])
    except AssertionError:
        pass
    else:
        raise AssertionError("different remote usage details were accepted")


def check_final_projection_only():
    # A final payload can omit fields because of its size budget. Source usage
    # must not imply remotely verified coverage when final outputs omit it.
    final_outputs = {"trace_truncated": True, "summary": {"type": "object", "count": 2}}
    assert summarize_usage_coverage([(final_outputs, deepcopy(final_outputs))]) == {
        "usage_covered": False,
        "usage_coverage_counts": {"none": 1, "partial": 0, "full": 0},
        "usage_coverage_by_call": [{"model_call": 1, "coverage": "none", "reported_fields": []}],
    }


def main():
    checks = (
        check_usage_categories, check_mixed_run_and_empty_run,
        check_remote_mismatches, check_final_projection_only,
    )
    with patch("verify_tracing.load_settings", side_effect=AssertionError("dotenv access forbidden")), \
            patch("requests.Session.request", side_effect=AssertionError("network access forbidden")):
        for check in checks:
            with check_group(f"check_trace_usage/{check.__name__}"):
                observe_harness("trace_projection")
                check()
            print(f"PASS {check.__name__}")
    print(f"PASS {len(checks)} offline usage coverage groups; no dotenv or endpoint access")


if __name__ == "__main__":
    main()
