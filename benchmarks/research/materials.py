"""Build the approved material tree from immutable Git blobs, never a checkout.

Manifest and grading data stay outside the Agent's material_root. Preparation
uses only Git and the standard library; PDF transcripts are reviewed, versioned
text files rather than an Agent PDF parsing capability.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

from benchmarks.common import atomic_write_json
from benchmarks.research.definitions import get_case


CODE_COMMIT = "214b38ef4d813c7ae7ab2defb5406bc4165e8b9b"
DOCS_COMMIT = "917efa1c4355f741486265ecd3d33db3cb0982e3"
MATERIAL_DEFINITION_VERSION = "wechat-materials-v1"
TRANSCRIPT_VERSION = "wechat-pdf-text-v1"
CODE_PATHS = (
    "README.md", "docs/development/architecture.md", "docs/development/verification.md",
    "miniprogram/app.json", "miniprogram/app.ts", "miniprogram/app.wxss",
    "miniprogram/components/form-field/index.json", "miniprogram/components/form-field/index.ts",
    "miniprogram/components/form-field/index.wxml", "miniprogram/components/form-field/index.wxss",
    "miniprogram/config/form.ts", "miniprogram/domain/form.ts", "miniprogram/domain/signature-pad.ts",
    "miniprogram/pages/collection/index.json", "miniprogram/pages/collection/index.ts",
    "miniprogram/pages/collection/index.wxml", "miniprogram/pages/collection/index.wxss",
    "miniprogram/pages/result/index.json", "miniprogram/pages/result/index.ts",
    "miniprogram/pages/result/index.wxml", "miniprogram/pages/result/index.wxss",
    "miniprogram/pages/signature/index.json", "miniprogram/pages/signature/index.ts",
    "miniprogram/pages/signature/index.wxml", "miniprogram/pages/signature/index.wxss",
    "miniprogram/pages/subjects/index.json", "miniprogram/pages/subjects/index.ts",
    "miniprogram/pages/subjects/index.wxml", "miniprogram/pages/subjects/index.wxss",
    "miniprogram/pages/welcome/index.json", "miniprogram/pages/welcome/index.ts",
    "miniprogram/pages/welcome/index.wxml", "miniprogram/pages/welcome/index.wxss",
    "miniprogram/platform/signature-canvas.ts", "miniprogram/platform/signature-files.ts",
    "miniprogram/platform/wechat-storage.ts", "miniprogram/ports/platform.ts",
    "miniprogram/services/collection-service.ts", "miniprogram/services/index.ts",
    "miniprogram/services/mock-gateway.ts", "miniprogram/sitemap.json", "miniprogram/storage/repository.ts",
    "miniprogram/types/domain.ts", "miniprogram/utils/display.ts", "package.json",
    "tests/collection-service.test.ts", "tests/fixtures.ts", "tests/form.test.ts",
    "tests/pages.test.ts", "tests/platform.test.ts",
)
DOCS_PATHS = ("README.md", "docs/README.md", "docs/reference/简览.md")
TRANSCRIPTS = (
    {"source_path": "docs/reference/简览.pdf", "file": "简览.md", "pages": 1,
     "sha256": "6eff2f98215be14ef8f8fc158713e00191a386e9a4e1476aa2d2e79ae1570a1b"},
    {"source_path": "docs/reference/信息采集小程序需求简报.pdf", "file": "信息采集小程序需求简报.md", "pages": 2,
     "sha256": "67d45dbcdfae31b04e43b5310bfe2c254c6a9158647a5c969ea008b36569cb37"},
)


class MaterialError(ValueError):
    """A fixed material source is unavailable, changed, or unsafe to expose."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_blob(source_repo: Path, commit: str, path: str) -> tuple[bytes, str]:
    try:
        listing = subprocess.run(
            ["git", "-C", str(source_repo), "ls-tree", "-z", commit, "--", path],
            check=True, capture_output=True, timeout=15,
        ).stdout
        fields = listing.rstrip(b"\0").split(b"\t")
        if len(fields) != 2 or fields[1].decode("utf-8") != path:
            raise MaterialError(f"fixed source missing: {commit}:{path}")
        mode, kind, blob = fields[0].decode("ascii").split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise MaterialError(f"source is not an ordinary file: {path}")
        data = subprocess.run(
            ["git", "-C", str(source_repo), "cat-file", "blob", blob],
            check=True, capture_output=True, timeout=15,
        ).stdout
        return data, blob
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        # No raw Git stderr: source URLs or repository-specific details are not
        # needed to describe a missing fixed object.
        raise MaterialError(f"cannot read fixed source: {commit}:{path} ({type(exc).__name__})") from None


def _destination(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if not relative or posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise MaterialError("material path escapes its root")
    result = root.joinpath(*posix.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise MaterialError("material path resolves outside its root")
    return result


def _write_file(root: Path, relative: str, data: bytes) -> dict:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise MaterialError(f"material is not UTF-8 text: {relative}") from None
    if b"\0" in data:
        raise MaterialError(f"binary material is not allowed: {relative}")
    path = _destination(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": relative, "sha256": _sha(data), "size": len(data)}


def _version(files: list[dict]) -> str:
    payload = {"definition_version": MATERIAL_DEFINITION_VERSION, "files": files}
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def prepare_materials(repo_root: Path, source_repo: Path, destination: Path) -> dict:
    """Prepare one exclusive directory. Failed preparation never publishes a manifest."""
    repo_root, source_repo, destination = Path(repo_root), Path(source_repo), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    material_root = destination / "material"
    material_root.mkdir()
    files = []
    for prefix, commit, paths in (("code", CODE_COMMIT, CODE_PATHS), ("requirements", DOCS_COMMIT, DOCS_PATHS)):
        for path in paths:
            data, blob = _read_blob(source_repo, commit, path)
            entry = _write_file(material_root, f"{prefix}/{path}", data)
            entry.update(kind="git_text", source_commit=commit, source_path=path, source_blob=blob)
            files.append(entry)
    for transcript in TRANSCRIPTS:
        original, blob = _read_blob(source_repo, DOCS_COMMIT, transcript["source_path"])
        if _sha(original) != transcript["sha256"]:
            raise MaterialError(f"PDF differs from the reviewed original: {transcript['source_path']}")
        transcript_path = repo_root / "benchmarks" / "research" / "transcripts" / transcript["file"]
        entry = _write_file(material_root, f"requirements/transcripts/{transcript['file']}", transcript_path.read_bytes())
        entry.update(kind="reviewed_pdf_transcript", source_commit=DOCS_COMMIT,
                     source_path=transcript["source_path"], source_blob=blob,
                     source_sha256=transcript["sha256"], source_pages=transcript["pages"],
                     transcript_version=TRANSCRIPT_VERSION)
        files.append(entry)
    lines = ["# 材料来源索引", "", "本目录只包含固定提交中的选定文本与评测端核对的 PDF 文本转写。",
             "`code/` 对应代码模板；`requirements/` 对应资料提交。索引不表示这些资料已经相互确认。",
             "PDF 原文件不在此目录；转写含原件页码。未列入的文件与未提交修改不属于本次材料。", "",
             "| 当前路径 | 来源提交 | 原路径 | 内容 SHA-256 |", "| --- | --- | --- | --- |"]
    lines.extend(f"| `{entry['path']}` | `{entry['source_commit']}` | `{entry['source_path']}` | `{entry['sha256']}` |" for entry in files)
    # The index is neutral provenance. Rubric/expected facts never enter this tree.
    files.append(_write_file(material_root, "SOURCE_INDEX.md", ("\n".join(lines) + "\n").encode("utf-8")))
    files[-1].update(kind="provenance_index", source="evaluation_preparation")
    manifest = {"schema_version": "1", "definition_version": MATERIAL_DEFINITION_VERSION,
                "material_version": _version(files), "material_root": str(material_root.resolve()),
                "source_commits": {"code": CODE_COMMIT, "requirements": DOCS_COMMIT},
                "source_text_count": len(CODE_PATHS) + len(DOCS_PATHS), "transcript_count": len(TRANSCRIPTS), "files": files}
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def verify_materials(instance_root: Path) -> dict:
    """Verify provenance layout and all bytes without creating or copying files."""
    instance_root = Path(instance_root)
    manifest = json.loads((instance_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("definition_version") != MATERIAL_DEFINITION_VERSION or manifest.get("material_version") != _version(manifest["files"]):
        raise MaterialError("prepared manifest version or contents changed")
    material_root = instance_root / "material"
    if manifest.get("material_root") != str(material_root.resolve()):
        raise MaterialError("manifest material root differs from its actual directory")
    if material_root.is_symlink() or not material_root.is_dir():
        raise MaterialError("prepared material root is missing or linked")
    approved = {f"code/{path}" for path in CODE_PATHS} | {f"requirements/{path}" for path in DOCS_PATHS}
    approved |= {f"requirements/transcripts/{item['file']}" for item in TRANSCRIPTS} | {"SOURCE_INDEX.md"}
    case_id = manifest.get("case_id")
    if case_id is not None:
        get_case(case_id)
    if case_id == "wechat-compaction":
        approved.add("evaluation-fixtures/background.txt")
    # No stale or evaluator files may silently become available through copying.
    listed = {entry["path"] for entry in manifest["files"]}
    actual = {path.relative_to(material_root).as_posix() for path in material_root.rglob("*") if path.is_file()}
    if actual != listed or listed != approved or len(listed) != len(manifest["files"]):
        raise MaterialError("prepared material file set differs from the manifest")
    for entry in manifest["files"]:
        path = _destination(material_root, entry["path"])
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != material_root and parent.is_relative_to(material_root)):
            raise MaterialError("linked material is not allowed")
        data = path.read_bytes()
        if _sha(data) != entry["sha256"] or len(data) != entry["size"]:
            raise MaterialError(f"prepared material changed: {entry['path']}")
    return manifest


def instantiate_materials(prepared_root: Path, destination: Path, case_id: str) -> dict:
    """Copy verified approved files into a fresh, case-specific Agent directory."""
    get_case(case_id)
    prepared_root, destination = Path(prepared_root), Path(destination)
    manifest = verify_materials(prepared_root)
    if "case_id" in manifest:
        raise MaterialError("a sample material tree cannot replace the prepared base")
    source_root = prepared_root / "material"
    destination.mkdir(parents=True, exist_ok=False)
    material_root = destination / "material"
    material_root.mkdir()
    files = []
    for entry in manifest["files"]:
        source = _destination(source_root, entry["path"])
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != source_root and parent.is_relative_to(source_root)):
            raise MaterialError("linked material is not allowed")
        data = source.read_bytes()
        if _sha(data) != entry["sha256"] or len(data) != entry["size"]:
            raise MaterialError(f"prepared material changed: {entry['path']}")
        _write_file(material_root, entry["path"], data)
        files.append(dict(entry))
    if case_id == "wechat-compaction":
        prefix = b"EVALUATION-CONSTRUCTED BACKGROUND. No project facts. Used only for context pressure.\n"
        line = b"This is neutral evaluation padding; it makes no statement about the source project.\n"
        size = 12 * 1024
        padding = prefix + (line * (size // len(line) + 1))[:size - len(prefix)]
        entry = _write_file(material_root, "evaluation-fixtures/background.txt", padding)
        entry.update(kind="evaluation_fixture", fixture_version="neutral-background-v1", source="evaluation_preparation")
        files.append(entry)
    sample_manifest = {**manifest, "case_id": case_id, "material_root": str(material_root.resolve()),
                       "base_material_version": manifest["material_version"], "material_version": _version(files), "files": files}
    atomic_write_json(destination / "manifest.json", sample_manifest)
    return sample_manifest
