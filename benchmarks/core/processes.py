"""Subprocess environment, source validation and bounded tree cleanup."""
import os
import signal
import subprocess
from pathlib import Path
from importlib import import_module
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from ipaddress import ip_address
import socket
from unittest.mock import patch


@contextmanager
def offline_network():
    """Allow local event-loop pipes, forbid connections outside loopback."""
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(original):
        def call(sock, address):
            if not isinstance(address, tuple):
                raise AssertionError("offline socket address rejected")
            try:
                local = ip_address(address[0]).is_loopback
            except ValueError:
                local = address[0] == "localhost"
            if not local:
                raise AssertionError("offline external network access forbidden")
            return original(sock, address)
        return call

    with patch.object(socket.socket, "connect", guarded(connect)), \
         patch.object(socket.socket, "connect_ex", guarded(connect_ex)):
        yield

def safe_environment(source: dict[str, str]) -> dict[str, str]:
    """Only runtime plumbing crosses into this offline, deterministic worker."""
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "TMPDIR",
               "COMSPEC", "SYSTEMDRIVE", "HOME", "USERPROFILE", "LOCALAPPDATA",
               "APPDATA", "VIRTUAL_ENV", "LANG", "LC_ALL"}
    env = {key: value for key, value in source.items() if key.upper() in allowed}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               PYTHONNOUSERSITE="1", LANGSMITH_TRACING="false", LANGCHAIN_TRACING="false",
               LANGCHAIN_TRACING_V2="false")
    return env

class ProcessCleanupError(RuntimeError):
    """The direct child was handled, but its process tree could not be confirmed."""

def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        tree_error = None
        if os.name == "nt" and type(process.pid) is int:
            # The owned process is still alive; kill its descendants before its PID disappears.
            try:
                stopped = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                         capture_output=True, timeout=5)
                if stopped.returncode == 0:
                    process.wait(timeout=5)
                    return
                tree_error = ProcessCleanupError("process tree termination was not confirmed")
            except (OSError, subprocess.SubprocessError) as exc:
                tree_error = exc
        elif os.name != "nt" and type(process.pid) is int:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                tree_error = exc
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if tree_error is not None:
            raise ProcessCleanupError("process tree cleanup remains unconfirmed") from tree_error

def require_local_agent():
    for name, relative in (("lineage_agent", "src/lineage_agent/__init__.py"),
                           ("lineage_agent.agent.runtime", "src/lineage_agent/agent/runtime.py")):
        module = import_module(name)
        if Path(module.__file__).resolve() != (ROOT / relative).resolve():
            raise ImportError("Agent source does not belong to this checkout")
