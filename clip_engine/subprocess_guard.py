# -*- coding: utf-8 -*-
"""Track and terminate child processes (FFmpeg) to prevent orphans."""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import IO

logger = logging.getLogger("clip_engine.subprocess_guard")

_registry_lock = threading.Lock()
_tracked: dict[int, "_TrackedProc"] = {}
_atexit_registered = False


@dataclass
class _TrackedProc:
    proc: subprocess.Popen
    label: str
    cmd_line: str


def _subprocess_kw() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _register(proc: subprocess.Popen, *, label: str, cmd_line: str) -> None:
    global _atexit_registered
    with _registry_lock:
        if not _atexit_registered:
            atexit.register(terminate_all_tracked)
            _atexit_registered = True
        _tracked[proc.pid] = _TrackedProc(proc=proc, label=label, cmd_line=cmd_line)


def _unregister(proc: subprocess.Popen) -> None:
    with _registry_lock:
        _tracked.pop(proc.pid, None)


def terminate_process(proc: subprocess.Popen, *, grace_sec: float = 2.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    deadline = time.time() + grace_sec
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def terminate_all_tracked() -> int:
    """Kill all tracked child processes. Returns count terminated."""
    with _registry_lock:
        items = list(_tracked.values())
    n = 0
    for item in items:
        if item.proc.poll() is None:
            logger.warning("[subprocess] terminating %s pid=%s", item.label, item.proc.pid)
            terminate_process(item.proc)
            n += 1
    with _registry_lock:
        dead = [pid for pid, t in _tracked.items() if t.proc.poll() is not None]
        for pid in dead:
            _tracked.pop(pid, None)
    return n


def list_tracked_pids() -> list[int]:
    with _registry_lock:
        return [pid for pid, t in _tracked.items() if t.proc.poll() is None]


def find_orphan_ffmpeg_pids(*, parent_pid: int | None = None) -> list[int]:
    """
    Return ffmpeg.exe PIDs that are direct children of parent_pid (default: current process).
    Does not kill unrelated user ffmpeg jobs from other apps when parent filter applies.
    """
    if sys.platform != "win32":
        return []
    try:
        import psutil
    except ImportError:
        return []

    root = parent_pid if parent_pid is not None else os.getpid()
    try:
        proc = psutil.Process(root)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []

    orphans: list[int] = []
    tracked = set(list_tracked_pids())
    for child in proc.children(recursive=True):
        try:
            name = (child.name() or "").lower()
            if "ffmpeg" not in name:
                continue
            pid = child.pid
            if pid in tracked:
                continue
            if child.is_running():
                orphans.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return orphans


def terminate_orphan_ffmpeg(*, parent_pid: int | None = None) -> int:
    """Terminate untracked ffmpeg children of this process tree. Returns count killed."""
    if sys.platform != "win32":
        return 0
    try:
        import psutil
    except ImportError:
        return 0

    killed = 0
    for pid in find_orphan_ffmpeg_pids(parent_pid=parent_pid):
        try:
            p = psutil.Process(pid)
            logger.warning("[subprocess] terminating orphan ffmpeg pid=%s", pid)
            p.terminate()
            try:
                p.wait(timeout=3)
            except psutil.TimeoutExpired:
                p.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


def find_orphan_job_process_pids(*, parent_pid: int | None = None) -> list[int]:
    """
    Return untracked ffmpeg.exe or python.exe child PIDs of parent_pid (default: current process).
    Excludes the root process itself.
    """
    if sys.platform != "win32":
        return []
    try:
        import psutil
    except ImportError:
        return []

    root = parent_pid if parent_pid is not None else os.getpid()
    try:
        proc = psutil.Process(root)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []

    orphans: list[int] = []
    tracked = set(list_tracked_pids())
    for child in proc.children(recursive=True):
        try:
            name = (child.name() or "").lower()
            if "ffmpeg" not in name and "python" not in name:
                continue
            pid = child.pid
            if pid == root or pid in tracked:
                continue
            if child.is_running():
                orphans.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return orphans


def terminate_orphan_job_processes(*, parent_pid: int | None = None) -> int:
    """Terminate untracked ffmpeg/python children after job crash or timeout."""
    if sys.platform != "win32":
        return terminate_orphan_ffmpeg(parent_pid=parent_pid)
    try:
        import psutil
    except ImportError:
        return terminate_orphan_ffmpeg(parent_pid=parent_pid)

    killed = 0
    for pid in find_orphan_job_process_pids(parent_pid=parent_pid):
        try:
            p = psutil.Process(pid)
            logger.warning("[subprocess] terminating orphan job child pid=%s name=%s", pid, p.name())
            p.terminate()
            try:
                p.wait(timeout=3)
            except psutil.TimeoutExpired:
                p.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


def run_subprocess_with_input(
    cmd: list[str],
    *,
    input_text: str = "",
    timeout: float | None = None,
    label: str = "subprocess",
    check: bool = False,
    text: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Run tracked subprocess with stdin payload (e.g. Resolve bridge JSON)."""
    from clip_engine.job_control import JobCancelledError, is_cancelled

    if is_cancelled():
        raise JobCancelledError("Cancelled before starting subprocess.")

    cmd_line = " ".join(str(x) for x in cmd)
    logger.info("[%s] start (stdin): %s", label, cmd_line[:500])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE if text else subprocess.DEVNULL,
        stderr=subprocess.PIPE if text else subprocess.DEVNULL,
        text=text,
        cwd=cwd,
        **_subprocess_kw(),
    )
    _register(proc, label=label, cmd_line=cmd_line)
    try:
        try:
            out, err = proc.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process(proc)
            raise
        result = subprocess.CompletedProcess(
            cmd, proc.returncode if proc.returncode is not None else 0, stdout=out, stderr=err
        )
        if check and result.returncode != 0:
            tail = (err or out or "").strip()
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=tail[-4000:] if tail else ""
            )
        return result
    finally:
        _unregister(proc)


def run_subprocess(
    cmd: list[str],
    *,
    timeout: float | None = None,
    label: str = "subprocess",
    check: bool = False,
    text: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a command with tracking, timeout, and cancel support.
    Polls job_control.is_cancelled() while waiting.
    """
    from clip_engine.job_control import JobCancelledError, is_cancelled

    if is_cancelled():
        raise JobCancelledError("Cancelled before starting subprocess.")

    cmd_line = " ".join(str(x) for x in cmd)
    logger.info("[%s] start: %s", label, cmd_line[:500])

    stdout_opt: int | IO = subprocess.PIPE if text else subprocess.DEVNULL
    stderr_opt: int | IO = subprocess.PIPE if text else subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_opt,
        stderr=stderr_opt,
        text=text,
        cwd=cwd,
        **_subprocess_kw(),
    )
    _register(proc, label=label, cmd_line=cmd_line)
    try:
        deadline = time.time() + timeout if timeout is not None else None
        while proc.poll() is None:
            if is_cancelled():
                terminate_process(proc)
                raise JobCancelledError(f"Cancelled during {label}.")
            if deadline is not None and time.time() > deadline:
                terminate_process(proc)
                raise subprocess.TimeoutExpired(cmd, timeout)
            time.sleep(0.25)

        out, err = proc.communicate()
        result = subprocess.CompletedProcess(
            cmd, proc.returncode, stdout=out, stderr=err
        )
        if check and result.returncode != 0:
            tail = (err or out or "").strip()
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=tail[-4000:] if tail else ""
            )
        return result
    finally:
        _unregister(proc)
