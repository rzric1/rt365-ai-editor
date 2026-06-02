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
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
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
            "streamlit" in cmd and "8501" in cmd
        )
    except Exception:
        return False


def _port_in_use(port: int = DEFAULT_PORT) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def remove_stale_lock() -> bool:
    """Remove lock file if PID is dead. Returns True if removed or no lock."""
    data = _read_lock()
    if data is None:
        return True
    pid = int(data.get("pid", 0))
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
    Returns (ok, message).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    data = _read_lock()
    if data:
        pid = int(data.get("pid", 0))
        if _pid_alive(pid) and _is_clip_studio_process(pid):
            return (
                False,
                "RT365 AI Clip Studio is already running. "
                f"(PID {pid}). Close the other window or end python.exe in Task Manager.",
            )
        remove_stale_lock()

    if _port_in_use(port):
        return (
            False,
            f"RT365 AI Clip Studio is already running (port {port} in use). "
            "Close the existing instance first.",
        )
    return True, ""


def acquire_app_lock(*, port: int = DEFAULT_PORT) -> tuple[bool, str]:
    """Acquire lock for current process (call once from Streamlit startup)."""
    global _lock_held
    if _lock_held:
        return True, ""

    ok, msg = preflight_single_instance(port=port)
    if not ok:
        return False, msg

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
