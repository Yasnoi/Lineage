"""Reusable fixtures and evidence assertions for usage."""


from copy import deepcopy
from lineage_agent.models.responses import parse_response
from lineage_agent.recording.trace_data import TraceSanitizer

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
