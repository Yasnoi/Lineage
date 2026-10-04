"""Immutable Git material preparation; task answers never enter Agent roots."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from benchmarks.core.files import ROOT, atomic_write_json

class MaterialError(ValueError):
    pass

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def _version(files, definition):
    return _sha(json.dumps({"definition": definition, "files": files}, ensure_ascii=False,
                           sort_keys=True, separators=(",", ":")).encode())

def _destination(root, relative):
    value = PurePosixPath(relative)
    if not relative or value.is_absolute() or ".." in value.parts or "\\" in relative or ":" in relative:
        raise MaterialError("material path escapes root")
    path = root.joinpath(*value.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise MaterialError("material path resolves outside root")
    return path

def _read_blob(source_repo, commit, path):
    try:
        listing = subprocess.run(["git", "-C", str(source_repo), "ls-tree", "-z", commit, "--", path],
                                 check=True, capture_output=True, timeout=15).stdout
        fields = listing.rstrip(b"\0").split(b"\t")
        if len(fields) != 2 or fields[1].decode() != path:
            raise MaterialError("fixed source missing")
        mode, kind, blob = fields[0].decode("ascii").split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise MaterialError("source is not an ordinary file")
        data = subprocess.run(["git", "-C", str(source_repo), "cat-file", "blob", blob],
                              check=True, capture_output=True, timeout=15).stdout
        return data, blob
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise MaterialError(f"cannot read fixed source ({type(exc).__name__})") from None

def _write_file(root, relative, data):
    try:
        data.decode("utf-8")
    except UnicodeError:
        raise MaterialError("material is not UTF-8") from None
    if b"\0" in data:
        raise MaterialError("binary material rejected")
    path = _destination(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": relative, "size": len(data), "sha256": _sha(data)}

def prepare_materials(repo_root, source_repo, destination, definition):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / "material"
    root.mkdir()
    files = []
    for source in definition["sources"]:
        for path in source["paths"]:
            data, blob = _read_blob(source_repo, source["commit"], path)
            entry = _write_file(root, source["prefix"] + "/" + path, data)
            entry.update(kind="git_text", source_commit=source["commit"], source_path=path, source_blob=blob)
            files.append(entry)
    for item in definition.get("transcripts", []):
        data, blob = _read_blob(source_repo, item["commit"], item["source_path"])
        if _sha(data) != item["sha256"]:
            raise MaterialError("source differs from reviewed transcript")
        text = _destination(Path(repo_root), item["text_path"]).read_bytes()
        entry = _write_file(root, item["target"], text)
        entry.update(kind="reviewed_transcript", source_commit=item["commit"], source_path=item["source_path"],
                     source_sha256=item["sha256"], source_blob=blob, pages=item["pages"], transcript_version=item["version"])
        files.append(entry)
    files.append({**_write_file(root, "SOURCE_INDEX.md", definition["index"].encode()), "kind": "provenance_index"})
    manifest = {"schema_version": "2", "definition": deepcopy(definition), "files": files,
                "material_version": _version(files, definition), "material_root": str(root.resolve())}
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest

def verify_materials(instance_root):
    instance_root = Path(instance_root)
    manifest = json.loads((instance_root / "manifest.json").read_text(encoding="utf-8"))
    root = instance_root / "material"
    if (manifest.get("schema_version") != "2" or manifest.get("material_root") != str(root.resolve())
            or root.is_symlink() or not root.is_dir()
            or manifest["material_version"] != _version(manifest["files"], manifest["definition"])):
        raise MaterialError("manifest identity changed")
    paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    declared = {item["path"] for item in manifest["files"]}
    if paths != declared or len(declared) != len(manifest["files"]):
        raise MaterialError("material inventory differs from manifest")
    for item in manifest["files"]:
        path = _destination(root, item["path"])
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p.is_relative_to(root)):
            raise MaterialError("linked material rejected")
        data = path.read_bytes()
        if len(data) != item["size"] or _sha(data) != item["sha256"]:
            raise MaterialError("material contents changed")
    return manifest

def instantiate_materials(prepared_root, destination, task):
    if not isinstance(task, dict) or not task.get("case_id"):
        raise MaterialError("explicit task material definition required")
    manifest = verify_materials(prepared_root)
    if manifest.get("task_id") is not None:
        raise MaterialError("a task instance cannot replace the prepared source")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / "material"
    root.mkdir()
    files = []
    edits = {item["path"]: item for item in task.get("material_edits", [])}
    for original in manifest["files"]:
        item = dict(original)
        data = _destination(Path(prepared_root) / "material", item["path"]).read_bytes()
        if _sha(data) != item["sha256"]:
            raise MaterialError("source changed during copying")
        if item["path"] in edits:
            edit = edits.pop(item["path"])
            text = data.decode()
            for replacement in edit["replacements"]:
                if text.count(replacement["before"]) != 1:
                    raise MaterialError("constructed variant no longer matches source")
                text = text.replace(replacement["before"], replacement["after"], 1)
            data = text.encode()
            item.update(kind="evaluation_constructed_variant", original_sha256=item["sha256"], variant_version=edit["version"])
        item.update(_write_file(root, item["path"], data))
        files.append(item)
    if edits:
        raise MaterialError("variant target outside approved material")
    for fixture in task.get("fixtures", []):
        if "text_path" in fixture:
            if "bytes" in fixture:
                raise MaterialError("fixture must have one content source")
            data = _destination(ROOT, fixture["text_path"]).read_bytes()
            if not 1 <= len(data) <= 16384 or _sha(data) != fixture.get("sha256"):
                raise MaterialError("versioned fixture content differs")
        else:
            prefix = b"EVALUATION-CONSTRUCTED BACKGROUND. No project facts.\n"
            size = fixture["bytes"]
            if type(size) is not int or not len(prefix) <= size <= 16384:
                raise MaterialError("invalid bounded fixture")
            data = prefix + b"x" * (size - len(prefix))
        files.append({**_write_file(root, fixture["path"], data),
                      "kind": "evaluation_fixture", "version": fixture["version"]})
    if task.get("material_edits"):
        # Label the copy without giving the injected defect or expected conclusion.
        index = next(item for item in files if item["path"] == "SOURCE_INDEX.md")
        path = root / "SOURCE_INDEX.md"
        data = path.read_bytes() + b"\nThis project copy is an evaluation-constructed variant of the source revision.\n"
        path.write_bytes(data)
        index.update(size=len(data), sha256=_sha(data), kind="evaluation_variant_index")
    definition = {**manifest["definition"], "task_materials": {k: deepcopy(task.get(k, [])) for k in ("material_edits", "fixtures")}}
    instance = {**manifest, "task_id": task["case_id"], "definition": definition, "files": files,
                "base_material_version": manifest["material_version"], "material_version": _version(files, definition),
                "material_root": str(root.resolve())}
    atomic_write_json(destination / "manifest.json", instance)
    return instance
