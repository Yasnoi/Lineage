"""Small filesystem helpers shared by the runner and its child process."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Replace one artifact only after its complete UTF-8 contents are written."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
