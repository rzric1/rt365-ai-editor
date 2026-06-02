# -*- coding: utf-8 -*-
"""RT365 AI Clip Studio — main entry point."""
from __future__ import annotations

import traceback
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
load_dotenv(_ROOT / ".env")

import logging
import streamlit as st

from config import LOGS_DIR
from clip_engine.telemetry import configure_rotating_logs, classify_exception, get_session_telemetry
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


def main() -> None:
    st.set_page_config(
        page_title="RT365 AI Clip Studio",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    try:
        logging.basicConfig(level=logging.INFO)
        configure_rotating_logs(LOGS_DIR)
        _ensure_startup()
        init_session_state()
        flush_pending_long_defaults()
        render_sidebar()
        render_clips_section()
        render_export_panel()
        render_resolve_panel()
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
