# -*- coding: utf-8 -*-
"""Single-instance lock — prevent duplicate Streamlit / Clip Studio processes."""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from config import LOGS_DIR

logger = logging.getLogger("clip_engine.app_lock")

LOCK_PATH = LOGS_DIR / "rt365_app.lock"
DEFAULT_PORT = 8501
_lock_held = False


def _read_lock() -> dict[str, Any] | None:
    if not LOCK_PATH.is_file():
        return None
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    except Exception:
        return False


def _is_clip_studio_process(pid: int) -> bool:
    try:
        import psutil

        p = psutil.Process(pid)
        cmd = " ".join(p.cmdline()).lower()
        return "clip_studio_app" in cmd or (
            "streamlit" in cmd and ("8501" in cmd or "clip_studio" in cmd)
        )
    except Exception:
        return False


def _port_listener_pid(port: int = DEFAULT_PORT) -> int | None:
    """
    Return PID of process LISTENING on 127.0.0.1:port, or None if not listening.
    Ignores TIME_WAIT / CLOSE_WAIT (not LISTEN).
    """
    try:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if not conn.laddr:
                continue
            if getattr(conn.laddr, "port", None) != port:
                continue
            if conn.status != psutil.CONN_LISTEN:
                continue
            if conn.pid:
                return int(conn.pid)
        return None
    except (ImportError, AttributeError, PermissionError):
        pass

    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
        return None
    except OSError:
        return -1


def _another_instance_listening(port: int = DEFAULT_PORT) -> tuple[bool, str]:
    """True if a different live process is listening on port (launcher preflight only)."""
    listener = _port_listener_pid(port)
    if listener is None:
        return False, ""
    if listener == os.getpid():
        return False, ""
    if listener > 0 and _pid_alive(listener) and _is_clip_studio_process(listener):
        return (
            True,
            f"RT365 AI Clip Studio is already running (port {port} held by PID {listener}).",
        )
    if listener == -1:
        return (
            True,
            f"Port {port} is in use. Close the other Streamlit instance and retry.",
        )
    return False, ""


def remove_stale_lock() -> bool:
    """Remove lock file if PID is dead or not Clip Studio. Returns True if removed or no lock."""
    data = _read_lock()
    if data is None:
        return True
    pid = int(data.get("pid", 0))
    if pid == os.getpid():
        return True
    if _pid_alive(pid) and _is_clip_studio_process(pid):
        return False
    try:
        LOCK_PATH.unlink(missing_ok=True)
        logger.info("[app_lock] removed stale lock (pid=%s)", pid)
        return True
    except OSError:
        return False


def preflight_single_instance(*, port: int = DEFAULT_PORT) -> tuple[bool, str]:
    """
    Launcher check before starting Streamlit. Does not acquire the lock.
    Uses LISTEN-only port detection (TIME_WAIT does not block).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    remove_stale_lock()

    data = _read_lock()
    if data:
        pid = int(data.get("pid", 0))
        if pid != os.getpid() and _pid_alive(pid) and _is_clip_studio_process(pid):
            return (
                False,
                "RT365 AI Clip Studio is already running. "
                f"(PID {pid}). Close the other window or end python.exe in Task Manager.",
            )
        remove_stale_lock()

    blocked, msg = _another_instance_listening(port)
    if blocked:
        return False, msg
    return True, ""


def acquire_app_lock(*, port: int = DEFAULT_PORT) -> tuple[bool, str]:
    """
    Acquire lock for the Streamlit server process.
    Does NOT check port 8501 — the server already owns that port when this runs.
    """
    global _lock_held
    if _lock_held:
        return True, ""

    remove_stale_lock()
    data = _read_lock()
    if data:
        pid = int(data.get("pid", 0))
        if pid != os.getpid() and _pid_alive(pid) and _is_clip_studio_process(pid):
            return (
                False,
                "RT365 AI Clip Studio is already running. "
                f"(lock held by PID {pid}).",
            )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "python": sys.executable,
        "version": sys.version.split()[0],
        "port": port,
        "started": time.time(),
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _lock_held = True
    atexit.register(release_app_lock)
    logger.info("[app_lock] acquired pid=%s", os.getpid())
    return True, ""


def release_app_lock() -> None:
    global _lock_held
    data = _read_lock()
    if data and int(data.get("pid", -1)) == os.getpid():
        try:
            LOCK_PATH.unlink(missing_ok=True)
            logger.info("[app_lock] released pid=%s", os.getpid())
        except OSError:
            pass
    _lock_held = False
