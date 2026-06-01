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

from ui.session_state import init_session_state, flush_pending_long_defaults
from ui.sidebar import render_sidebar
from ui.clip_cards import render_clips_section
from ui.export_panel import render_export_panel
from ui.resolve_panel import render_resolve_panel

logger = logging.getLogger("clip_studio")


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
        init_session_state()
        flush_pending_long_defaults()
        render_sidebar()
        render_clips_section()
        render_export_panel()
        render_resolve_panel()
    except Exception as e:
        logger.exception("Clip Studio UI failed")
        print(traceback.format_exc())
        st.error(f"**Clip Studio error:** {e}")
        with st.expander("Technical details", expanded=True):
            st.code(traceback.format_exc())
    finally:
        logger.debug("[lifecycle] Streamlit render cycle complete")


if __name__ == "__main__":
    main()
