"""Versioned project dataset; orchestration and evaluation live elsewhere."""
import json
from pathlib import Path

HERE = Path(__file__).parent

def load_rubric():
    return json.loads((HERE / "rubrics/research.json").read_text(encoding="utf-8"))

def definition():
    index = json.loads((HERE / "task-index.json").read_text(encoding="utf-8"))
    identifiers = index["tasks"] if isinstance(index, dict) else index
    tasks = [json.loads((HERE / "tasks" / f"{name}.json").read_text(encoding="utf-8")) for name in identifiers]
    for task in tasks:
        if task.get("variant"):
            variant = json.loads((HERE / "variants" / f"{task['variant']}.json").read_text(encoding="utf-8"))
            task["material_edits"] = variant["edits"]
    return {"suite_id": "wechat-reporting", "version": "2", "title": "WeChat Reporting 项目研究任务",
            "kind": "project", "executor": "agent", "network": True, "units": tasks,
            "materials": json.loads((HERE / "materials.json").read_text(encoding="utf-8")),
            "rubric": load_rubric()}
