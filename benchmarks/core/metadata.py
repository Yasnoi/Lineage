"""Safe identities and diagnostic facts; never serialize environment/config objects."""
from importlib import metadata
from pathlib import Path
import platform
import subprocess
from benchmarks.core.files import ROOT, sha256_file
from benchmarks.core.contracts import known, unknown
DEPENDENCIES = ("langgraph", "langsmith", "openai", "python-dotenv", "urllib3")

def issue(stage: str, code: str, message: str, exc: BaseException | None = None) -> dict:
    value = {"stage": stage, "code": code, "message": message, "evidence": []}
    if exc is not None:
        value["exception_type"] = type(exc).__name__
    return value

def _digest(path: Path) -> dict:
    try:
        return known(sha256_file(path))
    except OSError:
        return unknown("文件内容摘要无法取得。")

def collect_baseline(root: Path) -> dict:
    def git(*args: str) -> dict:
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=5)
            if result.returncode == 0:
                return known(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return unknown("Git 元数据无法取得；未保存源码快照。")

    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if dirty["availability"] == "known":
        dirty = known(bool(dirty["value"]))
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = known(metadata.version(name))
        except metadata.PackageNotFoundError:
            versions[name] = unknown("当前解释器未找到已安装的分发版本。")
    return {
        "agent_commit": git("rev-parse", "HEAD"), "working_tree_dirty": dirty,
        "python_version": known(platform.python_version()), "dependencies": versions,
        "lock_sha256": _digest(root / "uv.lock"),
        "framework_sources": {str(p.relative_to(root)): _digest(p)
                              for p in sorted((root / "benchmarks").rglob("*.py"))
                              if "__pycache__" not in p.parts},
    }

def _evidence(kind: str, path: Path, source: str) -> dict:
    try:
        availability = "available" if path.exists() else "missing"
    except OSError:
        availability = "unknown"
    return {"kind": kind, "path": str(path.absolute()), "source": source, "availability": availability}