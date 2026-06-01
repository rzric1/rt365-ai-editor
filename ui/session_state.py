# -*- coding: utf-8 -*-
"""Session state initialisation and long-podcast default helpers."""
from __future__ import annotations

import streamlit as st

from config import ensure_directories
from clip_engine.effective_config import (
    SESSION_ANALYSIS_DIAGNOSTICS,
    SESSION_CLIP_EDITS,
    apply_profile_non_widget_keys,
    apply_profile_widget_defaults,
)
from clip_engine.ai_profiles import profile_from_ui_label

SESSION_DIAG_INIT = "cs_diagnostics_initialized"
SESSION_DIAG_CACHE = "cs_ai_diag_cache"


def init_session_state() -> None:
    """Initialise all session state keys with defaults (idempotent)."""
    ensure_directories()
    defaults = {
        "cs_video_path": None,
        "cs_segments": [],
        "cs_formatted": "",
        "cs_clips": [],
        "cs_session_dir": None,
        "cs_status": "Upload a video to begin.",
        "cs_gpu_acceleration": True,
        "cs_whisper_model": "base",
        "cs_media_duration": 0.0,
        "cs_force_gpu_export": False,
        "cs_allow_cpu_fallback": True,
        "cs_smart_crop": True,
        "cs_export_mode_label": "Full frame fit with blurred background",
        "cs_write_sidecars": True,
        "cs_default_caption_preset": "Clean",
        "cs_target_clips": 20,
        "cs_min_gap_seconds": 30,
        "cs_similarity_threshold": 85,
        "cs_clip_style": "Balanced",
        "cs_pipeline_stats": {},
        "cs_enable_signal_boosts": True,
        "cs_enable_advanced_captions": True,
        "cs_enable_dynamic_smart_crop": True,
        "cs_enable_preview_rendering": True,
        "cs_previews": {},
        "cs_token_saver_mode": True,
        "cs_rate_limit_safe": True,
        "cs_use_analysis_cache": True,
        "cs_max_tokens_budget": 60_000,
        "cs_openai_call_delay": 0.75,
        "cs_openai_status": "",
        "cs_upload_reused": False,
        "cs_discovery_mode": True,
        "cs_long_defaults_applied": False,
        "cs_pending_long_defaults": False,
        "cs_ai_profile_label": "SAFE (Recommended)",
        "cs_enable_gpu_prefilter": True,
        "cs_session_telemetry": {},
        "cs_diarization_turns": [],
        "cs_speaker_names": {},
        "cs_clip_strategy": "Balanced",
        "cs_platform_target": "TikTok/Reels/Shorts",
        "cs_title_style": "Curiosity",
        SESSION_CLIP_EDITS: {},
        SESSION_ANALYSIS_DIAGNOSTICS: {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "cs_ai_profile_label" not in st.session_state:
        st.session_state.cs_ai_profile_label = "SAFE (Recommended)"
    _profile_boot = (
        "cs_max_clip_seconds" not in st.session_state
        or "cs_context_before" not in st.session_state
        or "cs_max_tokens_budget" not in st.session_state
    )
    _prof = profile_from_ui_label(str(st.session_state.get("cs_ai_profile_label", "")))
    if _profile_boot:
        apply_profile_non_widget_keys(st.session_state, _prof)
        apply_profile_widget_defaults(st.session_state, _prof)
    if "cs_max_tokens_budget" not in st.session_state:
        st.session_state.cs_max_tokens_budget = _prof.max_tokens
    if "cs_token_saver_mode" not in st.session_state:
        st.session_state.cs_token_saver_mode = _prof.token_saver
    if "cs_enable_gpu_prefilter" not in st.session_state:
        st.session_state.cs_enable_gpu_prefilter = True


def flush_pending_long_defaults() -> None:
    """Apply long-podcast widget defaults once, before sidebar widgets render."""
    if not st.session_state.pop("cs_pending_long_defaults", False):
        return
    if st.session_state.get("cs_long_defaults_applied"):
        return
    dur = float(st.session_state.get("cs_media_duration") or 0)
    if dur < 30 * 60:
        return
    st.session_state.cs_discovery_mode = True
    st.session_state.cs_min_clip_seconds = 15
    st.session_state.cs_max_clip_seconds = 120
    st.session_state.cs_min_gap_seconds = 35
    st.session_state.cs_similarity_threshold = 85
    apply_profile_non_widget_keys(
        st.session_state, profile_from_ui_label("SAFE (Recommended)")
    )
    if str(st.session_state.get("cs_clip_style", "")) == "Balanced":
        st.session_state.cs_clip_style = "Micro clips"
    st.session_state.cs_long_defaults_applied = True


def apply_long_podcast_defaults() -> None:
    """Queue long-podcast defaults for next run (before widgets). Does not mutate widget keys."""
    dur = float(st.session_state.get("cs_media_duration") or 0)
    if dur < 30 * 60:
        return
    if st.session_state.get("cs_long_defaults_applied"):
        return
    st.session_state.cs_pending_long_defaults = True
