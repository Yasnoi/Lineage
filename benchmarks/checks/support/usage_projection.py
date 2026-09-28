"""Shared usage projection checks for offline and opt-in remote diagnostics."""


import json

def summarize_usage_coverage(projected_remote_outputs):
    """Classify final, remotely verified token fields without filling missing values."""
    fields = ("input_tokens", "output_tokens", "total_tokens")
    counts = {"none": 0, "partial": 0, "full": 0}
    per_call = []
    for index, (projected, remote) in enumerate(projected_remote_outputs, 1):
        expected = projected.get("usage_metadata")
        observed = remote.get("usage_metadata")
        # JSON equality distinguishes a reported zero from False, and ints from floats.
        assert json.dumps(expected, sort_keys=True, allow_nan=False) == json.dumps(
            observed, sort_keys=True, allow_nan=False
        ), "远端 usage_metadata 与最终上传投影不一致。"
        reported = [
            field for field in fields
            if isinstance(expected, dict)
            and type(expected.get(field)) is int and expected[field] >= 0
        ]
        coverage = "full" if len(reported) == len(fields) else "partial" if reported else "none"
        counts[coverage] += 1
        per_call.append({"model_call": index, "coverage": coverage, "reported_fields": reported})
    return {
        "usage_covered": counts["partial"] + counts["full"] > 0,
        "usage_coverage_counts": counts,
        "usage_coverage_by_call": per_call,
    }
