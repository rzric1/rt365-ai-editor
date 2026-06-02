# -*- coding: utf-8 -*-
"""Single active job lock and cooperative cancellation for Clip Studio."""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger("clip_engine.job_control")

_lock = threading.Lock()
_active_job: str | None = None
_cancel_requested = threading.Event()
_last_pipeline_step: str = ""


class JobBusyError(RuntimeError):
    """Another long-running job is already active."""


class JobCancelledError(RuntimeError):
    """User requested cancellation."""


def get_active_job() -> str | None:
    with _lock:
        return _active_job


def set_pipeline_step(step: str) -> None:
    global _last_pipeline_step
    _last_pipeline_step = step or ""


def get_pipeline_step() -> str:
    return _last_pipeline_step


def is_cancelled() -> bool:
    return _cancel_requested.is_set()


def clear_cancel() -> None:
    _cancel_requested.clear()


def request_cancel() -> None:
    logger.warning("[job] cancel requested (active=%s)", _active_job)
    _cancel_requested.set()
    try:
        from clip_engine.subprocess_guard import terminate_all_tracked

        terminate_all_tracked()
    except Exception as exc:  # noqa: BLE001
        logger.debug("terminate_all_tracked: %s", exc)


def try_acquire_job(name: str) -> None:
    """Raise JobBusyError if another job holds the lock."""
    if is_cancelled():
        clear_cancel()
    with _lock:
        global _active_job
        if _active_job is not None and _active_job != name:
            raise JobBusyError(
                f"Cannot start '{name}' while '{_active_job}' is running. "
                "Wait for it to finish or click Cancel current job."
            )
        _active_job = name
        logger.info("[job] acquired: %s", name)


def release_job(name: str) -> None:
    with _lock:
        global _active_job
        if _active_job == name:
            _active_job = None
            logger.info("[job] released: %s", name)
        clear_cancel()


def check_cancelled() -> None:
    if is_cancelled():
        raise JobCancelledError("Operation cancelled.")


def run_guarded(job_name: str, fn: Callable[[], object]) -> object:
    """Acquire job, run fn, release on exit; map cancel to JobCancelledError."""
    try_acquire_job(job_name)
    try:
        return fn()
    except JobCancelledError:
        raise
    finally:
        release_job(job_name)
