# -*- coding: utf-8 -*-
"""
RT365 AI Edit Companion — local Streamlit UI for the RT365 AI Editor workflow.

Run from the project folder:
  streamlit run app.py
or double-click run_app.bat (Windows).

Safety (v1): only timeline markers via marker_writer / Resolve AddMarker.
No cuts, ripple delete, or media pool changes.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Project root on path (Streamlit cwd may vary).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from config import ENV_OPENAI_API_KEY, PROJECT_ROOT, TRANSCRIPTS_DIR, ensure_directories
from transcript_loader import load_transcript

from ai_companion import markers_for_resolve, run_companion_turn
from ui_helpers import (
    apply_markers_to_resolve_safe,
    format_timecode,
    open_folder,
    resolve_transcript_file,
    save_companion_json,
    try_resolve_connection,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rt365.companion")

DEFAULT_PATH_HINT = str((TRANSCRIPTS_DIR / "input.srt").relative_to(PROJECT_ROOT)).replace("/", os.sep)


def _init_session() -> None:
    ensure_directories()
    if "companion_messages" not in st.session_state:
        st.session_state.companion_messages = []
    if "transcript_path_str" not in st.session_state:
        st.session_state.transcript_path_str = DEFAULT_PATH_HINT


def _load_doc_for_path(path_str: str):
    """Return TranscriptDocument or raise."""
    path = resolve_transcript_file(path_str.strip() or DEFAULT_PATH_HINT)
    return load_transcript(path)


def _render_assistant_payload(result: dict, *, button_key_suffix: str) -> None:
    """Turn structured companion output into readable cards + Resolve actions."""
    intent = result.get("intent", "")
    brief = result.get("brief_reason", "")
    data = result.get("data") or {}

    with st.container(border=True):
        st.markdown(f"**What I understood:** `{intent}`")
        if brief:
            st.caption(brief)

    if intent == "GENERAL_EDIT_ADVICE":
        with st.container(border=True):
            st.markdown("### Suggestions for your edit")
            st.markdown(str(data.get("advice", "*No advice returned.*")))
        return

    if intent == "ANALYZE_MARKERS":
        markers = data.get("markers") or []
        with st.container(border=True):
            st.success(f"Found **{len(markers)}** marker suggestion(s).")
            for i, m in enumerate(markers[:40]):
                with st.expander(
                    f"{i + 1}. [{m.get('marker_type', '')}] {m.get('title', '')} @ {format_timecode(float(m.get('timestamp_seconds', 0)))}",
                    expanded=(i < 3 and len(markers) <= 5),
                ):
                    st.write(f"**Time:** {m.get('timestamp_seconds')} s")
                    st.write(f"**Note:** {m.get('note', '')}")
                    st.write(f"**Confidence:** {m.get('confidence', '')}")
            if len(markers) > 40:
                st.info("Showing the first 40 markers. Full list is in the saved JSON log.")

    elif intent == "FIND_CLIP":
        with st.container(border=True):
            st.subheader(data.get("clip_title", "Suggested clip"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Start", f"{float(data.get('start_seconds', 0)):.1f}s")
            c2.metric("End", f"{float(data.get('end_seconds', 0)):.1f}s")
            c3.metric(
                "Length",
                f"{max(0.0, float(data.get('end_seconds', 0)) - float(data.get('start_seconds', 0))):.1f}s",
            )
            st.markdown("**Hook**")
            st.write(data.get("hook", ""))
            st.markdown("**Caption idea**")
            st.write(data.get("caption", ""))
            st.markdown("**Why it works**")
            st.write(data.get("reason", ""))

    elif intent == "CHAPTERS":
        chapters = data.get("chapters") or []
        with st.container(border=True):
            st.success(f"**{len(chapters)}** chapter(s)")
            for i, ch in enumerate(chapters):
                st.markdown(
                    f"**{i + 1}.** {ch.get('title', '')} - _{format_timecode(float(ch.get('start_seconds', 0)))}_"
                )
                st.caption(ch.get("summary", ""))

    elif intent in ("GOOD_QUOTES", "POSSIBLE_CUTS", "AUDIO_ISSUES"):
        items = data.get("items") or []
        label = {"GOOD_QUOTES": "quotes", "POSSIBLE_CUTS": "cut hints", "AUDIO_ISSUES": "audio notes"}.get(
            intent, "items"
        )
        with st.container(border=True):
            st.success(f"**{len(items)}** {label}")
            for i, it in enumerate(items[:50]):
                with st.expander(
                    f"{i + 1}. {it.get('title', '')} @ {format_timecode(float(it.get('timestamp_seconds', 0)))}",
                    expanded=i < 2,
                ):
                    st.write(it.get("note", ""))
                    st.caption(f"Confidence: {it.get('confidence', '')}")
            if len(items) > 50:
                st.info("Showing the first 50 items. See JSON log for the full list.")

    markers = markers_for_resolve(intent, data)
    if markers:
        st.divider()
        if st.button(
            _resolve_button_label(intent),
            key=f"resolve_add_{button_key_suffix}_{intent}",
            type="primary",
        ):
            ok, msg, _count = apply_markers_to_resolve_safe(markers)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


def _resolve_button_label(intent: str) -> str:
    if intent == "FIND_CLIP":
        return "Add START / END markers to Resolve"
    if intent == "CHAPTERS":
        return "Add chapter markers to Resolve"
    if intent == "ANALYZE_MARKERS":
        return "Add all suggested markers to Resolve"
    return "Add markers to Resolve"


def main() -> None:
    _init_session()

    st.set_page_config(
        page_title="RT365 AI Edit Companion",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RT365 AI Edit Companion")
    st.caption(
        "A friendly editing assistant for DaVinci Resolve. "
        "Version 1 only adds **timeline markers** — it never cuts, deletes, ripples, or changes the media pool."
    )

    with st.container(border=True):
        st.markdown(
            "Welcome. Pick your transcript in the sidebar, then ask in plain English what you want "
            "from this episode (chapters, quotes, a short clip idea, and more). "
            "When you are happy with a result, use the blue button to drop markers on your timeline."
        )

    with st.sidebar:
        st.header("Transcript")
        st.text_input(
            "Transcript file path (project or current folder)",
            key="transcript_path_str",
            help=f"Default: `{DEFAULT_PATH_HINT}`. You can use a path relative to this project.",
        )
        uploaded = st.file_uploader(
            "Or upload .srt / .txt / .json",
            type=["srt", "txt", "json"],
            help="Upload replaces the working file path below with a saved copy in transcripts/.",
        )
        if uploaded is not None:
            ext = Path(uploaded.name).suffix.lower() or ".srt"
            dest = TRANSCRIPTS_DIR / f"_streamlit_last_upload{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getbuffer())
            rel = str(dest.relative_to(PROJECT_ROOT)).replace("/", os.sep)
            st.session_state.transcript_path_str = rel
            st.success(f"Saved upload to `{rel}`")

        st.divider()
        st.header("Resolve")
        if st.button("Check Resolve connection", width="stretch"):
            ok, msg, _details = try_resolve_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.divider()
        st.header("Folders")
        if st.button("Open transcripts folder", width="stretch"):
            open_folder(TRANSCRIPTS_DIR)
            st.toast("Opened transcripts folder.", icon="📂")
        if st.button("Open logs folder", width="stretch"):
            from config import LOGS_DIR

            open_folder(LOGS_DIR)
            st.toast("Opened logs folder.", icon="📂")

        st.divider()
        st.markdown(
            "**Tip:** Keep Resolve open with your episode timeline active before "
            "using marker buttons."
        )

    with st.expander("Example requests you can paste or adapt"):
        st.markdown(
            """
- Find me a 30 second short about bullying  
- Give me chapters for this episode  
- Find possible cuts  
- Mark good quotes  
- Find emotional moments  
- Create a YouTube Shorts idea from the fear conversation  
"""
        )

    for msg in st.session_state.companion_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                if "error" in msg:
                    st.error(msg["error"])
                else:
                    _render_assistant_payload(
                        msg["result"],
                        button_key_suffix=str(msg.get("id", uuid.uuid4().hex)),
                    )

    if prompt := st.chat_input("Ask the AI editor what you want..."):
        st.session_state.companion_messages.append({"role": "user", "content": prompt})
        api_key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
        if not api_key:
            st.session_state.companion_messages.append(
                {
                    "role": "assistant",
                    "id": uuid.uuid4().hex,
                    "error": "Your OpenAI API key is missing. Add `OPENAI_API_KEY` to the `.env` file in this project folder, then refresh the page.",
                }
            )
            st.rerun()

        try:
            doc = _load_doc_for_path(st.session_state.get("transcript_path_str", DEFAULT_PATH_HINT))
        except FileNotFoundError as exc:
            st.session_state.companion_messages.append(
                {
                    "role": "assistant",
                    "id": uuid.uuid4().hex,
                    "error": f"**Transcript file problem**\n\n{exc}\n\nFix the path in the sidebar or upload a file.",
                }
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcript load failed")
            st.session_state.companion_messages.append(
                {
                    "role": "assistant",
                    "id": uuid.uuid4().hex,
                    "error": f"Could not read the transcript:\n\n`{exc}`",
                }
            )
            st.rerun()

        with st.spinner("Reading your transcript and talking to the AI…"):
            try:
                result = run_companion_turn(user_message=prompt, doc=doc, api_key=api_key)
                save_companion_json(result)
                st.session_state.companion_messages.append(
                    {"role": "assistant", "id": uuid.uuid4().hex, "result": result}
                )
            except ValueError as exc:
                st.session_state.companion_messages.append(
                    {"role": "assistant", "id": uuid.uuid4().hex, "error": str(exc)}
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Companion turn failed")
                st.session_state.companion_messages.append(
                    {
                        "role": "assistant",
                        "id": uuid.uuid4().hex,
                        "error": f"The AI request failed. Check your network, billing, and model name in `.env`.\n\nDetails: `{exc}`",
                    }
                )
        st.rerun()


if __name__ == "__main__":
    main()
