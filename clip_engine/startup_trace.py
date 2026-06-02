# -*- coding: utf-8 -*-
"""Append-only startup lifecycle log — must never raise."""

from __future__ import annotations

import atexit
import os
from datetime import datetime, timezone
from pathlib import Path

from config import LOGS_DIR

TRACE_PATH = LOGS_DIR / "startup_trace.log"
_registered = False


def trace(message: str) -> None:
    """Append one line to logs/startup_trace.log."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        pid = os.getpid()
        line = f"{ts} pid={pid} {message}\n"
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def register_shutdown_trace() -> None:
    """Log shutdown once per process at interpreter exit."""
    global _registered
    if _registered:
        return
    _registered = True

    def _on_exit() -> None:
        trace("shutdown requested (atexit)")
        try:
            from clip_engine.app_lock import release_app_lock

            release_app_lock()
            trace("lock released")
        except Exception:
            pass

    atexit.register(_on_exit)
