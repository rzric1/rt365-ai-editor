# -*- coding: utf-8 -*-
"""RT365 AI Clip Studio — main entry point."""
from __future__ import annotations

import os

# CTranslate2 GPU throughput tuning for RTX 4090 — before torch/ctranslate2 import.
if not os.environ.get("CT2_VERBOSE"):
    os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "1")
    os.environ.setdefault("CT2_CUDA_ALLOW_FP16", "1")
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "0,0,0,0")

import traceback
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
load_dotenv(_ROOT / ".env")

# Bundled torch CUDA DLLs must win over system CUDA Toolkit (WinError 127 on cublas64_12.dll).
import clip_engine.whisper_runtime  # noqa: F401

import logging
import streamlit as st

from clip_engine.startup_trace import register_shutdown_trace, trace

trace("app imported")

from config import DEFAULT_WHISPER_MODEL, LOGS_DIR
from clip_engine.telemetry import configure_rotating_logs, get_session_telemetry
from clip_engine.stability import (
    cleanup_temp_artifacts,
    install_exception_hooks,
    log_resource_snapshot,
    run_startup_diagnostics,
    write_crash_report,
)
from clip_engine.subprocess_guard import terminate_orphan_ffmpeg

from ui.session_state import init_session_state, flush_pending_long_defaults
from ui.sidebar import render_sidebar
from ui.clip_cards import render_clips_section
from ui.export_panel import render_export_panel
from ui.resolve_panel import render_resolve_panel

logger = logging.getLogger("clip_studio")
_STARTUP_DONE = False
_ENV_GATE_DONE = False
_SHUTDOWN_REGISTERED = False
_FIRST_RENDER_LOGGED = False
_PREWARM_STARTED = False


def _register_shutdown_once() -> None:
    global _SHUTDOWN_REGISTERED
    if _SHUTDOWN_REGISTERED:
        return
    _SHUTDOWN_REGISTERED = True
    register_shutdown_trace()


def _ensure_environment_gate() -> None:
    """Block UI if Python/venv/deps are unsafe (prevents crash-prone launches)."""
    global _ENV_GATE_DONE
    if _ENV_GATE_DONE:
        return
    trace("environment check started")
    from clip_engine.environment_check import (
        format_streamlit_error,
        validate_startup_environment,
        write_environment_check_log,
    )

    status = validate_startup_environment()
    write_environment_check_log(status)
    _ENV_GATE_DONE = True
    if status.ok:
        trace("environment check passed")
        return
    trace("environment check FAILED")
    st.error(format_streamlit_error(status))
    with st.expander("Environment details", expanded=True):
        for c in status.checks:
            if not c.ok:
                st.text(f"[{'CRITICAL' if c.critical else 'warn'}] {c.name}: {c.detail}")
    st.stop()


def _ensure_app_lock() -> None:
    from clip_engine.app_lock import acquire_app_lock

    ok, msg = acquire_app_lock()
    if not ok:
        trace(f"lock acquire FAILED: {msg[:120]}")
        st.error(msg)
        st.stop()
    trace("lock acquired")


def _prewarm_whisper() -> None:
    try:
        from clip_engine.whisper_runtime import get_whisper_model

        logger.info("[prewarm] loading %s into VRAM at startup", DEFAULT_WHISPER_MODEL)
        get_whisper_model(
            model_size=DEFAULT_WHISPER_MODEL,
            device="cuda",
            compute_type="float16",
        )
        logger.info("[prewarm] %s ready in VRAM", DEFAULT_WHISPER_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[prewarm] failed: %s", exc)


def _start_whisper_prewarm() -> None:
    global _PREWARM_STARTED
    if _PREWARM_STARTED:
        return
    _PREWARM_STARTED = True
    import threading

    threading.Thread(target=_prewarm_whisper, daemon=True, name="whisper_prewarm").start()


def _ensure_startup() -> None:
    global _STARTUP_DONE
    if _STARTUP_DONE:
        return
    _STARTUP_DONE = True
    install_exception_hooks()
    try:
        cleanup_temp_artifacts()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Temp cleanup at startup failed: %s", exc)
    try:
        n = terminate_orphan_ffmpeg()
        if n:
            logger.warning("Terminated %s orphan ffmpeg process(es) at startup", n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Orphan ffmpeg cleanup at startup failed: %s", exc)
    try:
        log_resource_snapshot(label="startup")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Resource snapshot at startup failed: %s", exc)
    try:
        run_startup_diagnostics()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup diagnostics failed: %s", exc)
    _start_whisper_prewarm()


def _log_first_render() -> None:
    global _FIRST_RENDER_LOGGED
    if _FIRST_RENDER_LOGGED:
        return
    _FIRST_RENDER_LOGGED = True
    trace("app rendered first frame")


def main() -> None:
    _register_shutdown_once()
    st.set_page_config(
        page_title="RT365 AI Clip Studio",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    try:
        logging.basicConfig(level=logging.INFO)
        configure_rotating_logs(LOGS_DIR)
        _ensure_environment_gate()
        _ensure_startup()
        _ensure_app_lock()
        init_session_state()
        flush_pending_long_defaults()
        render_sidebar()
        from clip_engine.runtime_debug import render_runtime_debug_panel

        render_runtime_debug_panel()
        render_clips_section()
        render_export_panel()
        render_resolve_panel()
        _log_first_render()
    except Exception as e:
        logger.exception("Clip Studio UI failed")
        try:
            write_crash_report(e, context="clip_studio_main")
        except Exception:
            pass
        print(traceback.format_exc())
        st.error(f"**Clip Studio error:** {e}")
        with st.expander("Technical details", expanded=True):
            st.code(traceback.format_exc())
    finally:
        logger.debug("[lifecycle] Streamlit render cycle complete")


if __name__ == "__main__":
    main()
