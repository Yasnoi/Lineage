"""Checklist aggregation is independent of who supplies the judgments."""
def aggregate(conditions, severe_errors=()):
    required = [item for item in conditions if item.get("applicable", True) and item.get("required", True)]
    if any(item.get("verdict") == "failed" for item in required) or any(item.get("present") is True for item in severe_errors):
        return "failed"
    if required and all(item.get("verdict") == "passed" for item in required) and all(
            item.get("present") is False for item in severe_errors):
        return "passed"
    return "pending_review"
