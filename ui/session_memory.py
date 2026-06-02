# -*- coding: utf-8 -*-
"""Session RAM cleanup — never deletes user video files on disk."""

from __future__ import annotations

import streamlit as st

from config import CLIP_STUDIO_MAX_UPLOAD_MB


def clear_session_heavy_data(*, keep_video_path: bool = True) -> dict[str, bool]:
    """
    Drop large in-memory session payloads (transcript, clips, embeddings state).
    Original uploads under uploads/ are not touched.
    """
    cleared: dict[str, bool] = {}
    keys = {
        "cs_segments": [],
        "cs_formatted": "",
        "cs_clips": [],
        "cs_previews": {},
        "cs_pipeline_stats": {},
        "cs_session_telemetry": {},
        "cs_diarization_turns": [],
        "final_clips": [],
    }
    if not keep_video_path:
        keys["cs_video_path"] = None
        keys["source_video_path"] = None

    for key, default in keys.items():
        if key in st.session_state:
            st.session_state[key] = default
            cleared[key] = True

    try:
        from clip_engine.semantic_ranking import release_embedding_model

        release_embedding_model()
        cleared["embedding_model"] = True
    except Exception:
        pass

    try:
        from clip_engine.gpu_cleanup import cleanup_gpu_after_phase

        cleanup_gpu_after_phase("session_clear", whisper=True, yolo=True, embeddings=False)
        cleared["gpu_cleanup"] = True
    except Exception:
        pass

    st.session_state.cs_status = "Session memory cleared (video file on disk unchanged)."
    return cleared


def upload_size_warning(uploaded_size_bytes: int) -> str | None:
    """Return warning text if upload is very large."""
    mb = uploaded_size_bytes / (1024 * 1024)
    if mb > CLIP_STUDIO_MAX_UPLOAD_MB * 0.85:
        return (
            f"Upload is {mb:.0f} MB (near {CLIP_STUDIO_MAX_UPLOAD_MB} MB limit). "
            "Long files increase RAM use during transcribe and analyze."
        )
    if mb > 2048:
        return (
            f"Upload is {mb:.0f} MB. Prefer GPU transcription and restart the app between "
            "very long podcast sessions to avoid python.exe memory growth."
        )
    return None
