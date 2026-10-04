"""Assert actual observation envelopes without reading or hydrating artifacts."""

from copy import deepcopy
import json

from lineage_agent.artifacts.tools import ARTIFACT_TOOLS
from lineage_agent.tools.history import HISTORY_TOOLS
from lineage_agent.tools.filesystem.text import TEXT_TOOL


def inline_raw(envelope):
    assert set(envelope) == {"ok", "observation"} and envelope["ok"] is True
    observation = envelope["observation"]
    assert observation["status"] == "available" and observation["mode"] == "inline"
    assert isinstance(observation["raw_result"], str)
    return observation["raw_result"]


def assert_tools(actual, business=()):
    expected_names = [item["name"] for item in business] + [item["name"] for item in ARTIFACT_TOOLS + HISTORY_TOOLS]
    assert [item["name"] for item in actual] == expected_names
    for delivered, original in zip(actual, business):
        if original["name"] == TEXT_TOOL["name"] and original["parameters"] == TEXT_TOOL["parameters"]:
            plain = deepcopy(delivered)
            hint = plain["parameters"]["properties"].pop("observation")
            assert "observation" in plain["parameters"]["required"]
            plain["parameters"]["required"].remove("observation")
            assert plain == original
            assert set(hint["properties"]) == {"intent", "keywords"}
            assert set(hint["required"]) == {"intent", "keywords"}
            assert hint["additionalProperties"] is False
        else:
            assert delivered == original
    assert actual[len(business):] == ARTIFACT_TOOLS + HISTORY_TOOLS
    return True


def assert_observed(envelope, raw, tool_name):
    """Compare delivered slices with separately captured original executor output."""
    assert envelope["ok"] is True and "result" not in envelope
    observation = envelope["observation"]
    assert observation["status"] == "available"
    assert len(json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")) <= 4096
    if observation["mode"] == "inline":
        assert inline_raw(envelope) == raw
        return
    assert observation["mode"] == "artifact" and "raw_result" not in observation
    assert observation["artifact_ref"].startswith("artifact://tool/")
    content = json.loads(raw)["content"] if tool_name == "read_text_file" else raw
    assert observation["view_kind"] == ("content" if tool_name == "read_text_file" else "raw_result")
    assert observation["total_chars"] == len(content)
    segments = observation["segments"]
    assert all(segment["text"] == content[segment["start"]:segment["end"]] for segment in segments)
    disclosed = sum(len(segment["text"]) for segment in segments)
    assert (disclosed == len(content)) is observation["complete"]
    assert not content or segments
