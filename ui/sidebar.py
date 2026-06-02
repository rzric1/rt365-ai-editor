# -*- coding: utf-8 -*-
"""Sidebar rendering for RT365 AI Clip Studio."""
from __future__ import annotations

import logging
import os

import streamlit as st

from config import ENV_FFMPEG_BINARY, ENV_OPENAI_API_KEY, LOGS_DIR
from clip_engine.telemetry import get_session_telemetry, render_telemetry_markdown
from clip_engine.clip_pipeline import PipelineOpenAIConfig  # noqa: F401 (used by callers)
from clip_engine.openai_resilience import get_json_telemetry
from clip_engine.effective_config import (
    SESSION_ANALYSIS_DIAGNOSTICS,
    apply_profile_non_widget_keys,
    get_cached_token_plan,
)
from clip_engine.ai_profiles import (
    PROFILE_LABELS,
    get_profile_help_text,
    profile_from_ui_label,
)
from clip_engine.clip_scoring import CLIP_STRATEGIES, PLATFORM_TARGETS, TITLE_STYLES
from clip_engine.clip_style import CLIP_STYLE_OPTIONS
from clip_engine.captions import CaptionPreset
from clip_engine.ffmpeg_gpu import (
    faster_whisper_cuda_available,
    get_gpu_acceleration_status,
    get_last_nvenc_probe_log,
    log_nvenc_probe_command_explicit,
    should_attempt_nvenc_on_export,
)
from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_version_line
from clip_engine.cuda_diagnostics import (
    CUDA_STACK_REFERENCE,
    collect_ai_acceleration_diagnostics,
    invalidate_cuda_runtime_probe_cache,
    log_ai_acceleration_startup,
)
from clip_engine.dependency_status import get_dependency_report, render_status_markdown
from clip_engine.analysis_cache import clear_all_analysis_cache
from ui_helpers import open_folder
from config import CLIP_STUDIO_OUTPUT_DIR
from clip_engine.license_check import get_license_status, TRIAL_EXPORT_LIMIT

try:
    from clip_engine.gpu_pipeline import get_rtx_pipeline_status
except Exception as _gpu_err:
    def get_rtx_pipeline_status():  # type: ignore[misc]
        return {"_error": str(_gpu_err), "embeddings_available": False}

logger = logging.getLogger("clip_studio")

CAPTION_PRESET_OPTIONS: list[CaptionPreset] = [
    "Clean", "Bold Viral", "Podcast", "Minimal",
    "Viral", "Podcast Pro", "Documentary", "Gaming", "Cinematic",
]

SESSION_DIAG_INIT = "cs_diagnostics_initialized"
SESSION_DIAG_CACHE = "cs_ai_diag_cache"


def _on_ai_profile_changed() -> None:
    apply_profile_non_widget_keys(
        st.session_state,
        profile_from_ui_label(str(st.session_state.get("cs_ai_profile_label", ""))),
    )


def _run_startup_diagnostics_once() -> None:
    if st.session_state.get(SESSION_DIAG_INIT):
        return
    ensure_ffmpeg_on_path(log=True)
    log_nvenc_probe_command_explicit()
    log_ai_acceleration_startup()
    st.session_state[SESSION_DIAG_INIT] = True
    logger.info("diagnostics initialized for session")


def _get_ai_diagnostics(*, refresh: bool = False):
    if refresh:
        invalidate_cuda_runtime_probe_cache()
        from clip_engine.ffmpeg_gpu import invalidate_nvenc_cache
        invalidate_nvenc_cache()
        st.session_state.pop(SESSION_DIAG_CACHE, None)
    cached = st.session_state.get(SESSION_DIAG_CACHE)
    if cached is not None and not refresh:
        return cached
    diag = collect_ai_acceleration_diagnostics(refresh_cuda_probe=refresh)
    st.session_state[SESSION_DIAG_CACHE] = diag
    return diag


def render_sidebar() -> None:
    """Render the full sidebar."""
    from clip_engine.export_vertical import EXPORT_MODE_LABELS

    _lic = get_license_status()
    if _lic["trial"] and _lic["enforcement_active"]:
        st.sidebar.warning(
            f"⚠️ **TRIAL MODE** — {TRIAL_EXPORT_LIMIT} exports per session.\n\n"
            "[Purchase a license](https://rt365.ai/buy) to unlock full access."
        )
    elif _lic["licensed"]:
        st.sidebar.success("✅ Licensed")

    _run_startup_diagnostics_once()
    ensure_ffmpeg_on_path(log=False)

    api_key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
    ffmpeg_path = ensure_ffmpeg_on_path()
    ffmpeg_ver = get_ffmpeg_version_line()
    gpu_status = get_gpu_acceleration_status()
    cuda_whisper = faster_whisper_cuda_available()
    ai_diag = _get_ai_diagnostics()

    with st.sidebar:
        st.header("Settings")

        from ui.stability_ui import render_stability_controls

        render_stability_controls()
        st.divider()

        st.checkbox("GPU acceleration (NVENC exports + local Whisper)", key="cs_gpu_acceleration")
        gpu_on = bool(st.session_state.get("cs_gpu_acceleration", True))
        force_gpu_export = bool(st.session_state.get("cs_force_gpu_export", False))
        will_try_nvenc = should_attempt_nvenc_on_export(prefer_gpu=gpu_on, force_gpu_mode=force_gpu_export)

        st.checkbox("Force GPU export (bypass NVENC probe gate)", key="cs_force_gpu_export")
        st.checkbox("Allow CPU fallback for exports", key="cs_allow_cpu_fallback")

        st.divider()
        st.subheader("Crop & captions")
        st.selectbox(
            "Export mode",
            list(EXPORT_MODE_LABELS.keys()),
            key="cs_export_mode_label",
            help=(
                "Full frame fit: preserves entire 16:9 frame with blurred background. Safe default.\n"
                "Smart crop: detects faces and keeps all people in frame (requires opencv-python-headless).\n"
                "Center crop: simple 9:16 center cut."
            ),
        )
        st.checkbox("Write SRT/ASS sidecar files", key="cs_write_sidecars")
        st.selectbox("Default caption preset", CAPTION_PRESET_OPTIONS, key="cs_default_caption_preset")

        st.divider()
        st.subheader("AI editing intelligence")
        st.checkbox("Enable AI signal boosts", key="cs_enable_signal_boosts",
                    help="Local heuristic scoring for emotion, pacing, hooks — no extra LLM calls.")
        st.checkbox("Enable advanced captions", key="cs_enable_advanced_captions",
                    help="Per-word karaoke highlighting when pysubs2 is installed; phrase fallback otherwise.")
        st.checkbox("Enable dynamic smart crop", key="cs_enable_dynamic_smart_crop",
                    help="Smooth camera movement in smart crop mode (YOLO or OpenCV).")
        st.checkbox("Enable preview rendering", key="cs_enable_preview_rendering",
                    help="Show 'Generate preview' size button on each clip card.")

        dep_report = get_dependency_report()
        with st.expander("Optional dependency status", expanded=False):
            st.markdown(render_status_markdown(dep_report))

        st.divider()
        st.caption("**Video export hardware**")
        if not gpu_on:
            st.info("CPU fallback active - libx264.")
        elif will_try_nvenc and gpu_status.nvenc_probe_ok:
            st.success("GPU export active - h264_nvenc probe passed.")
        elif will_try_nvenc and not gpu_status.nvenc_probe_ok:
            st.warning("GPU export will be attempted - NVENC self-test did not pass.")
        else:
            st.info("CPU fallback active - NVENC not listed.")

        st.caption(f"ffmpeg lists h264_nvenc: **{gpu_status.ffmpeg_nvenc_listed}**")
        st.caption(f"NVENC runtime probe: **{gpu_status.nvenc_probe_ok}**")
        with st.expander('Why "listed=True, probe=False"?', expanded=False):
            st.markdown(
                "- **listed** = ffmpeg advertises `h264_nvenc` in `-encoders`.\n"
                "- **probe** = a real NVENC encode at **640×360** succeeded.\n"
                "- A probe failure with *Frame Dimension less than the minimum* was a **test-size issue**.\n"
                "- Try **Force GPU export** or set `FORCE_NVENC_EXPORT=1` in `.env`."
            )
        with st.expander("Last NVENC probe log", expanded=False):
            st.code(get_last_nvenc_probe_log() or "(no probe run yet)", language=None)

        st.divider()
        st.caption("**Transcription backend**")
        if gpu_on and ai_diag.cuda_runtime_probe_ok and cuda_whisper:
            st.success("faster-whisper on CUDA")
        elif gpu_on and ai_diag.ctranslate2_cuda_devices > 0 and not ai_diag.cuda_runtime_probe_ok:
            st.info("faster-whisper CPU (int8) - CUDA skipped")
        elif gpu_on:
            st.caption("Install faster-whisper + ctranslate2 (CUDA) or use OPENAI_API_KEY.")
        else:
            st.caption("GPU mode off - OpenAI Whisper API used.")

        with st.expander("AI acceleration diagnostics", expanded=False):
            if st.button("Refresh CUDA / NVENC probes", width="stretch", key="cs_ai_diag_refresh"):
                _get_ai_diagnostics(refresh=True)
                st.rerun()
            st.markdown(ai_diag.to_detail_markdown())
            with st.expander("CUDA reference"):
                st.markdown(CUDA_STACK_REFERENCE)

        _sizes = ["tiny", "base", "small", "medium", "large-v3"]
        st.selectbox("faster-whisper model size", _sizes, key="cs_whisper_model")

        if not api_key:
            st.warning("Set `OPENAI_API_KEY` in `.env` for cloud Whisper + clip analysis.")

        st.divider()
        st.subheader("Clip duration")
        st.number_input("Minimum clip length (core, s)", min_value=5, max_value=600, step=1, key="cs_min_clip_seconds")
        st.number_input("Maximum clip length (cap, s)", min_value=10, max_value=120, step=1, key="cs_max_clip_seconds")
        st.number_input("Context before clip (s)", min_value=0, max_value=120, step=1, key="cs_context_before")
        st.number_input("Context after clip (s)", min_value=0, max_value=120, step=1, key="cs_context_after")
        st.checkbox("Allow final clip to exceed max length", value=False, key="cs_allow_exceed_max")

        st.divider()
        st.subheader("AI reliability profile")
        st.selectbox(
            "AI Reliability Profile",
            list(PROFILE_LABELS.values()),
            key="cs_ai_profile_label",
            help=get_profile_help_text(),
            on_change=_on_ai_profile_changed,
        )
        _active_profile = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", ""))
        st.caption(_active_profile.description)
        if _active_profile.warning:
            st.warning(_active_profile.warning)
        with st.expander("Active AI configuration", expanded=True):
            st.markdown(f"**Fast model:** `{_active_profile.fast_model}`")
            st.markdown(f"**Quality model:** `{_active_profile.quality_model}`")
            st.markdown(f"**JSON fallback:** `{_active_profile.json_fallback_model}`")
            st.markdown(f"**Token budget:** `{_active_profile.max_tokens:,}`")
            st.markdown(f"**Token saver:** `{_active_profile.token_saver}`")
            st.markdown(f"**Discovery mode:** `{_active_profile.discovery_mode}`")
            st.markdown(f"**GPU prefilter:** `{st.session_state.get('cs_enable_gpu_prefilter', True)}`")
            st.markdown(f"**Max GPT passes:** `{_active_profile.max_gpt_passes}`")
            st.markdown(f"**Max clip length (profile):** `{int(_active_profile.max_clip_length)}` s")
            st.markdown(
                f"**Context (profile):** `{int(_active_profile.context_before)}` s before / "
                f"`{int(_active_profile.context_after)}` s after"
            )
            st.markdown(
                f"**GPU shortlist:** `{_active_profile.target_gpu_shortlist_min}`–"
                f"`{_active_profile.target_gpu_shortlist_max}` | "
                f"**Max GPT regions:** `{_active_profile.max_active_gpt_regions}`"
            )
        st.checkbox(
            "GPU local prefilter (RTX embeddings + local candidates)",
            key="cs_enable_gpu_prefilter",
            help="Run semantic ranking on GPU before GPT refinement (recommended).",
        )

        with st.expander("RTX 4090 AI Pipeline Status", expanded=False):
            if st.button("Refresh GPU status", key="cs_rtx_refresh"):
                with st.spinner("Probing torch, embeddings, and CUDA…"):
                    try:
                        st.session_state.cs_rtx_status = get_rtx_pipeline_status()
                    except Exception as exc:
                        st.session_state.cs_rtx_status = {
                            "_error": str(exc), "python_version": "?", "torch_installed": False,
                        }
            rtx = st.session_state.get("cs_rtx_status")
            if rtx is None:
                st.caption(
                    "GPU pipeline probes are deferred until you click **Refresh GPU status** "
                    "(avoids torchcodec/sentence-transformers import failures during startup)."
                )
            else:
                if rtx.get("_error"):
                    st.warning(f"Pipeline status check failed (non-fatal). Embeddings may be unavailable: {rtx['_error']}")
                st.markdown(f"**Python:** `{rtx.get('python_version', '?')}`")
                st.markdown(f"**torch installed:** `{rtx.get('torch_installed')}`")
                if rtx.get("torch_installed"):
                    st.markdown(f"**torch version:** `{rtx.get('torch_version')}`")
                    st.markdown(f"**torch CUDA available:** `{rtx.get('torch_cuda_available')}`")
                    st.markdown(f"**torch CUDA device count:** `{rtx.get('torch_cuda_device_count')}`")
                    if rtx.get("torch_cuda_device_name"):
                        st.markdown(f"**torch CUDA device:** `{rtx.get('torch_cuda_device_name')}`")
                st.markdown(f"**sentence-transformers installed:** `{rtx.get('sentence_transformers_installed')}`")
                st.markdown(f"**Embeddings device (selected):** `{rtx.get('embeddings_device_selected', 'cpu')}`")
                st.markdown(f"**Embeddings on GPU:** `{rtx.get('embeddings_on_gpu')}`")
                st.markdown(f"**GPU (semantic):** `{rtx.get('gpu_name', 'n/a')}`")
                st.markdown(f"**Diarization on GPU:** `{rtx.get('diarization_on_gpu')}`")
                st.markdown(f"**faster-whisper CUDA:** `{rtx.get('faster_whisper_cuda')}`")
                st.markdown(f"**Local ranking enabled:** `{rtx.get('local_ranking_enabled')}`")
                if rtx.get("torch_installed") and not rtx.get("torch_cuda_available"):
                    st.warning(
                        "PyTorch is installed but CUDA is not available. Embeddings will run on CPU. "
                        "For best RTX 4090 performance, use a Python 3.11 virtual environment with "
                        "CUDA-enabled PyTorch (`scripts/setup_python311_ai_env.ps1`)."
                    )
                if rtx.get("gpu_memory"):
                    st.markdown(f"**GPU memory:** `{rtx['gpu_memory']}`")
                tel = get_json_telemetry()
                if tel.get("gpt5_success") or tel.get("json_fallback"):
                    st.markdown(
                        f"**GPT-5 JSON success rate:** `{tel.get('gpt5_success_rate_pct')}%` | "
                        f"**Fallback rate:** `{tel.get('fallback_rate_pct')}%`"
                    )

        with st.expander("GPU Candidate Explorer", expanded=False):
            explorer = (st.session_state.get("cs_pipeline_stats") or {}).get("gpu_explorer_rows") or []
            if not explorer:
                st.caption("Run **Analyze for high-retention clips** with GPU prefilter enabled to populate.")
            else:
                st.dataframe(explorer, width="stretch", hide_index=True)

        st.divider()
        st.subheader("OpenAI usage & safety")
        st.checkbox("Rate Limit Safe Mode", key="cs_rate_limit_safe",
                    help="Exponential backoff on 429 errors with retry/resume.")
        st.checkbox("Use cached analysis if available", key="cs_use_analysis_cache")
        st.slider("Delay between OpenAI calls (sec)", min_value=0.0, max_value=3.0, step=0.25, key="cs_openai_call_delay")
        if st.button("Clear analysis cache", width="stretch"):
            n = clear_all_analysis_cache()
            st.success(f"Cleared {n} cached analysis entries.")

        tel = st.session_state.get("cs_session_telemetry") or {}
        with st.expander("OpenAI Session Telemetry", expanded=False):
            try:
                st.markdown(render_telemetry_markdown(tel))
            except Exception as exc:
                logger.exception("Telemetry panel render failed")
                st.caption(f"Diagnostics display error: {exc}")
            if st.button("Refresh diagnostics from session", width="stretch", key="cs_tel_refresh"):
                st.session_state.cs_session_telemetry = get_session_telemetry().to_dict()
                st.rerun()
            st.caption(f"Rotating logs: `{LOGS_DIR}` (app.log, openai.log, gpu.log, exports.log)")

        if float(st.session_state.get("cs_media_duration") or 0) >= 30 * 60:
            st.caption("Long video detected — SAFE profile + GPU prefilter recommended.")
        st.divider()
        st.subheader("Creator controls (Opus-style+)")
        st.selectbox("Clip strategy", list(CLIP_STRATEGIES), key="cs_clip_strategy")
        st.selectbox("Platform target", list(PLATFORM_TARGETS), key="cs_platform_target")
        st.selectbox("Title style", list(TITLE_STYLES), key="cs_title_style")

        st.divider()
        st.subheader("Diversity & coverage")
        st.selectbox(
            "Clip style",
            CLIP_STYLE_OPTIONS,
            key="cs_clip_style",
            help=(
                "Micro clips: 30-75s sharp moments. "
                "Balanced: 45-100s. Long story: 90-160s narrative arcs."
            ),
        )
        st.number_input("Target number of clips", min_value=5, max_value=50, step=5, key="cs_target_clips",
                        help="How many unique clips to find across the full video.")
        st.number_input("Min gap between clips (s)", min_value=10, max_value=300, step=10, key="cs_min_gap_seconds",
                        help="Clips must be at least this many seconds apart.")
        st.slider("Duplicate similarity threshold (%)", min_value=20, max_value=95, step=5, key="cs_similarity_threshold",
                  help="Clips more similar than this % are considered duplicates and removed.")

        st.divider()
        st.caption("**FFmpeg**")
        if ffmpeg_path:
            st.code(ffmpeg_path, language=None)
            st.caption(ffmpeg_ver or "version: (unknown)")
        else:
            st.error(f"FFmpeg not found. Set **{ENV_FFMPEG_BINARY}** in `.env` or install FFmpeg.")

        if st.button("Open exports folder", width="stretch"):
            from config import ensure_directories
            ensure_directories()
            open_folder(CLIP_STUDIO_OUTPUT_DIR)
            st.toast("Opened outputs/clips")
