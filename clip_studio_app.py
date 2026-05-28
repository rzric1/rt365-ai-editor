"""
AI Clip Studio - general short-form clip generator (Streamlit).

Upload MP4/MOV -> Whisper transcript -> GPT clip picks -> export 9:16 + captions (ffmpeg).

Run: streamlit run clip_studio_app.py
Requires: ffmpeg (auto-detected or FFMPEG_BINARY in .env), OPENAI_API_KEY for cloud Whisper + clip AI
Optional: faster-whisper + CUDA for local GPU transcription when "GPU acceleration" is ON.
         opencv-python-headless for face-detection smart crop.
"""

from __future__ import annotations

import logging
import os
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
load_dotenv(_ROOT / ".env")

from config import (  # noqa: E402
    CLIP_STUDIO_MAX_UPLOAD_BYTES,
    CLIP_STUDIO_MAX_UPLOAD_MB,
    CLIP_STUDIO_OUTPUT_DIR,
    ENV_FFMPEG_BINARY,
    ENV_OPENAI_API_KEY,
    LOGS_DIR,
    PROJECT_ROOT,
    ensure_directories,
)
from clip_engine.telemetry import (  # noqa: E402
    classify_exception,
    configure_rotating_logs,
    get_session_telemetry,
    render_telemetry_markdown,
    reset_session_telemetry,
)
from clip_engine.clip_analysis import get_session_tokens  # noqa: E402
from clip_engine.clip_pipeline import run_full_clip_pipeline, PipelineOpenAIConfig  # noqa: E402
from clip_engine.openai_resilience import OpenAIRateLimitError, token_saver_pass_config
from clip_engine.effective_config import (  # noqa: E402
    get_cached_token_plan,
    plan_analysis_token_estimate,
)
from clip_engine.analysis_cache import clear_all_analysis_cache, hash_transcript  # noqa: E402
from clip_engine.ai_profiles import (  # noqa: E402
    PROFILE_LABELS,
    get_ai_profile,
    get_profile_help_text,
    profile_from_ui_label,
)
from clip_engine.effective_config import (  # noqa: E402
    ClipStudioEffectiveConfig,
    SESSION_ANALYSIS_DIAGNOSTICS,
    SESSION_CLIP_EDITS,
    SESSION_FORCE_REANALYZE,
    SESSION_RESOLVED_MODELS,
    apply_profile_non_widget_keys,
    apply_profile_widget_defaults,
    build_analysis_fingerprint,
    get_invalidation_reason,
    log_widget_rerun_noop,
    resolve_models_for_session,
    store_analysis_snapshot,
)
from clip_engine.clip_scoring import (  # noqa: E402
    CLIP_STRATEGIES,
    PLATFORM_TARGETS,
    TITLE_STYLES,
)
try:
    from clip_engine.gpu_pipeline import get_rtx_pipeline_status  # noqa: E402
except Exception as _gpu_pipeline_import_err:
    import logging as _logging
    _logging.getLogger("clip_studio").warning(
        "gpu_pipeline import failed (%s) — RTX status panel will show defaults.",
        _gpu_pipeline_import_err,
    )
    def get_rtx_pipeline_status():  # type: ignore[misc]
        return {"_error": str(_gpu_pipeline_import_err), "embeddings_available": False}
from clip_engine.openai_resilience import get_json_telemetry  # noqa: E402
from clip_engine.clip_style import CLIP_STYLE_OPTIONS, ClipStyle  # noqa: E402
from clip_engine.clip_metadata import (  # noqa: E402
    ground_clip_metadata_against_window,
    write_clip_audit_json,
)
from clip_engine.token_tracking import get_tracker  # noqa: E402
from clip_engine.export_vertical import (  # noqa: E402
    EXPORT_MODE_LABELS,
    export_clip_preview,
    export_filename_stem,
    export_vertical_clip_with_captions,
)
from clip_engine.media_probe import get_media_duration_seconds  # noqa: E402
from clip_engine.captions import CAPTION_PRESETS, CaptionPreset  # noqa: E402
from clip_engine.ffmpeg_gpu import (  # noqa: E402
    faster_whisper_cuda_available,
    get_gpu_acceleration_status,
    get_last_nvenc_probe_log,
    log_nvenc_probe_command_explicit,
    should_attempt_nvenc_on_export,
)
from clip_engine.transcription import (  # noqa: E402
    transcribe_video,
)
from clip_engine.speaker_analysis import diarize_audio_file  # noqa: E402
from clip_engine.transcription_utils import (  # noqa: E402
    extract_transcript_excerpt,
    merge_segments_into_sentences,
    segments_to_prompt_transcript,
)
from clip_engine.ffmpeg_resolve import (  # noqa: E402
    ensure_ffmpeg_on_path,
    get_ffmpeg_version_line,
)
from clip_engine.cuda_diagnostics import (  # noqa: E402
    CUDA_STACK_REFERENCE,
    collect_ai_acceleration_diagnostics,
    invalidate_cuda_runtime_probe_cache,
    log_ai_acceleration_startup,
)
from clip_engine.dependency_status import get_dependency_report, render_status_markdown  # noqa: E402
from clip_engine.upload_manifest import clean_duplicate_uploads, save_upload_once  # noqa: E402
from ui_helpers import open_folder  # noqa: E402

st.set_page_config(page_title="AI Clip Studio", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clip_studio")

CAPTION_PRESET_OPTIONS: list[CaptionPreset] = [
    "Clean", "Bold Viral", "Podcast", "Minimal",
    "Viral", "Podcast Pro", "Documentary", "Gaming", "Cinematic",
]
PREVIEW_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "previews"
SESSION_DIAG_INIT = "cs_diagnostics_initialized"
SESSION_DIAG_CACHE = "cs_ai_diag_cache"


def _run_startup_diagnostics_once() -> None:
    """Run hardware/ffmpeg diagnostics once per Streamlit session."""
    if st.session_state.get(SESSION_DIAG_INIT):
        logger.info("[DIAGNOSTICS] already initialized; skipping duplicate diagnostics")
        return
    ensure_ffmpeg_on_path(log=True)
    log_nvenc_probe_command_explicit()
    log_ai_acceleration_startup()
    st.session_state[SESSION_DIAG_INIT] = True
    logger.info("diagnostics initialized for session")


def _get_ai_diagnostics(*, refresh: bool = False):
    """Cached AI acceleration diagnostics (refreshed only on explicit user action)."""
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


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
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
    # Stale sessions: ensure profile-derived keys exist after ai_profiles updates.
    if "cs_max_tokens_budget" not in st.session_state:
        st.session_state.cs_max_tokens_budget = _prof.max_tokens
    if "cs_token_saver_mode" not in st.session_state:
        st.session_state.cs_token_saver_mode = _prof.token_saver
    if "cs_enable_gpu_prefilter" not in st.session_state:
        st.session_state.cs_enable_gpu_prefilter = True


def _on_ai_profile_changed() -> None:
    """Sync non-widget session keys from profile (widget keys are user-controlled)."""
    apply_profile_non_widget_keys(
        st.session_state,
        profile_from_ui_label(str(st.session_state.get("cs_ai_profile_label", ""))),
    )


def _flush_pending_long_defaults() -> None:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_long_podcast_defaults() -> None:
    """Queue long-podcast defaults for next run (before widgets). Does not mutate widget keys."""
    dur = float(st.session_state.get("cs_media_duration") or 0)
    if dur < 30 * 60:
        return
    if st.session_state.get("cs_long_defaults_applied"):
        return
    st.session_state.cs_pending_long_defaults = True


def _uploaded_file_size_bytes(upload) -> int:
    try:
        return int(upload.size)
    except Exception:
        return len(upload.getbuffer())


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / (1024**3):.2f} GB"
    if n >= 1024**2:
        return f"{n / (1024**2):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} bytes"


def _slug(s: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "_", s.strip())[:max_len]
    return s or "clip"


def _sync_series_export_checkbox(series_wids: list[str], source_wid: str) -> None:
    checked = bool(st.session_state.get(f"ex_{source_wid}", False))
    for w in series_wids:
        if w != source_wid:
            st.session_state[f"ex_{w}"] = checked


def _expand_export_selection(clips: list, selected: list[dict]) -> list[dict]:
    """Include every part when any part of a series is selected for export."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in selected:
        wid = str(c.get("_wid", ""))
        if wid and wid not in seen:
            out.append(c)
            seen.add(wid)
        for sib in clips:
            if (
                c.get("is_part_of_series")
                and sib.get("series_id") == c.get("series_id")
            ):
                swid = str(sib.get("_wid", ""))
                if swid and swid not in seen:
                    out.append(sib)
                    seen.add(swid)
                    st.session_state[f"ex_{swid}"] = True
    return out


def _assign_speakers_to_segments(
    segments: list[dict],
    turns: list[dict],
    name_map: dict[str, str],
) -> list[dict]:
    """Tag each transcript segment with the speaker that overlaps it most."""
    labeled = []
    for seg in segments:
        s0 = float(seg.get("start", 0))
        s1 = float(seg.get("end", s0))
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in turns:
            t0 = float(turn.get("start", 0))
            t1 = float(turn.get("end", t0))
            overlap = max(0.0, min(t1, s1) - max(t0, s0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(turn.get("speaker", ""))
        new_seg = dict(seg)
        if best_speaker:
            new_seg["speaker"] = name_map.get(best_speaker, best_speaker)
        labeled.append(new_seg)
    return labeled


def _run_clip_analysis(
    formatted: str,
    api_key: str,
    segments: list[dict],
    creator_note: str,
    min_c: float,
    max_c: float,
    ctx_b: float,
    ctx_a: float,
    allow_over: bool,
    media_dur: float,
    target_count: int = 20,
    min_gap_seconds: float = 60.0,
    similarity_threshold: float = 0.45,
    clip_style: ClipStyle = "Balanced",
    video_filename: str = "",
    status_callback=None,
) -> tuple[list[dict], dict]:
    """Run full clip pipeline. Returns (clips, pipeline_stats_dict)."""
    pipe_stats: dict = {}
    note = creator_note.strip() or None
    reset_session_telemetry()

    effective = ClipStudioEffectiveConfig.from_session(st.session_state)
    models = resolve_models_for_session(st.session_state)
    token_saver = effective.token_saver
    style_name = str(clip_style)
    est = plan_analysis_token_estimate(
        formatted,
        get_ai_profile(effective.profile_name),
        target_count=target_count,
        clip_style=style_name,
        emit_logs=True,
    )
    st.session_state.cs_token_estimate = est.to_dict()

    openai_config = PipelineOpenAIConfig(
        token_saver_mode=bool(
            st.session_state.get("cs_token_saver_mode", effective.token_saver)
        ),
        rate_limit_safe=bool(st.session_state.get("cs_rate_limit_safe", True)),
        use_cache=bool(st.session_state.get("cs_use_analysis_cache", True)),
        max_tokens_budget=int(
            st.session_state.get("cs_max_tokens_budget", effective.token_budget)
        ),
        call_delay_seconds=float(st.session_state.get("cs_openai_call_delay", 0.75)),
        status_callback=status_callback,
        model_fast=models.fast_model,
        model_final=models.quality_model,
        model_quality=models.quality_model,
        json_fallback_model=models.json_fallback_model,
        ai_profile_name=effective.profile_name,
        enable_gpu_prefilter=effective.gpu_prefilter,
    )

    clips, stats, tracker = run_full_clip_pipeline(
        formatted,
        api_key,
        segments,
        media_duration=media_dur,
        creator_note=note,
        clip_style=clip_style,
        user_min_seconds=min_c,
        user_max_seconds=max_c,
        context_before=ctx_b,
        context_after=ctx_a,
        allow_exceed_max=allow_over,
        target_count=target_count,
        min_gap_seconds=min_gap_seconds,
        similarity_threshold=similarity_threshold,
        video_filename=video_filename,
        enable_signal_boosts=bool(st.session_state.get("cs_enable_signal_boosts", True)),
        enable_speaker_signals=bool(st.session_state.get("cs_enable_signal_boosts", True)),
        openai_config=openai_config,
        discovery_mode=effective.discovery_mode,
        clip_strategy=effective.clip_strategy,
        platform_target=effective.platform_target,
        title_style=effective.title_style,
    )
    pipe_stats = stats.to_dict() if hasattr(stats, "to_dict") else {}

    video_id = video_filename or str(st.session_state.get("cs_video_path", ""))
    t_hash = hash_transcript(formatted, segments)
    fp = build_analysis_fingerprint(
        st.session_state,
        video_identity=video_id,
        transcript_hash=t_hash,
    )
    diag = {
        "cache_hit": bool(pipe_stats.get("cache_hit")),
        "cache_miss_reason": pipe_stats.get("cache_miss_reason", ""),
        "invalidation_reason": st.session_state.pop(SESSION_FORCE_REANALYZE, False) and "explicit_reanalyze" or None,
        "model_fast": models.fast_model,
        "model_quality": models.quality_model,
        "json_fallback": models.json_fallback_model,
        "json_telemetry": pipe_stats.get("json_telemetry") or get_json_telemetry(),
        "transcript_duration": media_dur,
        "raw_candidates": pipe_stats.get("raw_candidates", 0),
        "rescued_candidates": pipe_stats.get("rescued_candidates", 0),
        "final_clips": len(clips),
        "boundary_repairs": pipe_stats.get("boundary_repairs", 0),
        "title_repairs": pipe_stats.get("title_repairs", 0),
        "openai_calls_used": pipe_stats.get("openai_calls_used", 0),
        "analysis_fingerprint": fp,
    }
    store_analysis_snapshot(
        st.session_state,
        effective=effective,
        fingerprint=fp,
        video_identity=video_id,
        transcript_hash=t_hash,
        diagnostics=diag,
    )

    export_dict = tracker.to_export_dict(
        target_clips=target_count,
        final_clip_count=len(clips),
        model=models.quality_model,
    )
    export_dict["model_fast"] = models.fast_model
    export_dict["model_quality"] = models.quality_model
    export_dict["ai_profile"] = effective.profile_name
    export_dict["json_telemetry"] = getattr(stats, "json_telemetry", None) or get_json_telemetry()
    export_dict["token_estimate"] = est.to_dict()
    st.session_state.cs_token_tracker = export_dict
    st.session_state.cs_session_telemetry = get_session_telemetry().to_dict()
    pipe_stats["session_telemetry"] = st.session_state.cs_session_telemetry

    for c in clips:
        if not c.get("_wid"):
            c["_wid"] = uuid.uuid4().hex
        if "caption_preset" not in c:
            c["caption_preset"] = str(st.session_state.get("cs_default_caption_preset", "Clean"))
        tid = c.get("_wid", "")
        if tid and tid in tracker.per_clip:
            c["_token_usage"] = tracker.per_clip[tid]

    return clips, pipe_stats


# ---------------------------------------------------------------------------
# Score breakdown widget
# ---------------------------------------------------------------------------

def _render_clip_map(clips: list[dict], media_duration: float) -> None:
    """Render a visual timeline showing where clips are distributed."""
    if not clips or media_duration <= 0:
        return
    with st.expander("Clip map - timeline coverage", expanded=False):
        BAR_W = 600
        rows = []
        for i, c in enumerate(clips):
            t0 = float(c.get("start_seconds", 0))
            t1 = float(c.get("end_seconds", t0 + 60))
            x_start = int((t0 / media_duration) * BAR_W)
            x_end = max(x_start + 4, int((t1 / media_duration) * BAR_W))
            score = int(c.get("composite_score", 50))
            hook = c.get("hook_title", "").replace('"', "")
            rows.append(
                f'<div title="#{i+1} {hook} | {t0:.0f}s-{t1:.0f}s | score={score}" '
                f'style="position:absolute;left:{x_start}px;width:{x_end-x_start}px;height:18px;'
                f'background:hsl({score*1.2},80%,55%);border-radius:3px;opacity:0.85;"></div>'
            )
        region_labels = ["early-mid", "mid", "late-mid", "end"]
        region_divs = ""
        for i in range(1, 5):
            x = int((i / 5) * BAR_W)
            lbl = region_labels[i - 1]
            region_divs += (
                f'<div style="position:absolute;left:{x}px;top:0;height:100%;border-left:1px dashed #555;"></div>'
                f'<div style="position:absolute;left:{x+2}px;top:2px;font-size:9px;color:#888">{lbl}</div>'
            )
        html = (
            f'<div style="position:relative;width:{BAR_W}px;height:28px;'
            f'background:#1a1a2e;border-radius:4px;margin:8px 0;">'
            f'{region_divs}{"".join(rows)}</div>'
            f'<div style="font-size:11px;color:#888;margin-top:4px;">'
            f'0s --- {media_duration/2:.0f}s --- {media_duration:.0f}s &nbsp;|&nbsp; '
            f'{len(clips)} clips &nbsp;|&nbsp; color = score (green=high)</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


def _render_score_breakdown(c: dict) -> None:
    scores = c.get("scores", {})
    signal_scores = c.get("signal_scores", {})
    speaker_signals = c.get("speaker_signals", {})
    virality = int(c.get("virality_score", 0))
    if virality:
        st.metric("Virality", f"{virality}/100")
        if c.get("virality_explanation"):
            st.caption(c["virality_explanation"])
        breakdown = c.get("virality_breakdown") or {}
        if breakdown:
            cols = st.columns(4)
            for i, (k, v) in enumerate(breakdown.items()):
                cols[i % 4].caption(f"{k.replace('_', ' ').title()}: **{v}**")
    hook_q = int(c.get("hook_quality_score", 0))
    if hook_q:
        st.caption(f"Hook quality: **{hook_q}/100**" + (f" — {c['hook_warning']}" if c.get("hook_warning") else ""))
    if c.get("boundary_status") == "repaired" or c.get("boundary_repaired"):
        st.info("Boundary repaired to nearest complete sentence.")
    elif c.get("boundary_warning"):
        st.warning(c["boundary_warning"])
    with st.expander("Score breakdown", expanded=False):
        if scores:
            cols = st.columns(3)
            for i, (dim, val) in enumerate(scores.items()):
                label = dim.replace("_", " ").title()
                cols[i % 3].metric(label, f"{val}/100")
        composite = int(c.get("composite_score", 0))
        orig = c.get("original_composite_score")
        if orig and orig != composite:
            st.caption(f"GPT score: {orig}/100 → boosted: {composite}/100")
        st.progress(composite / 100, text=f"Composite: {composite}/100")

        if signal_scores:
            st.markdown("**AI signal scores** (local heuristics, no extra tokens)")
            sig_cols = st.columns(5)
            for i, key in enumerate(
                ("emotion_spike", "pacing", "curiosity_gap", "scroll_stopping_hook", "audience_reaction")
            ):
                val = signal_scores.get(key, 0)
                sig_cols[i].metric(key.replace("_", " ").title(), f"{val}/100")
            boost = signal_scores.get("signal_boost", 0)
            if boost:
                st.caption(f"Signal boost: +{boost} | {signal_scores.get('reason', '')}")

        if speaker_signals:
            st.markdown("**Speaker / debate signals**")
            sp_cols = st.columns(3)
            sp_cols[0].metric("Speaker energy", f"{speaker_signals.get('speaker_energy', 0)}/100")
            sp_cols[1].metric("Interruption", f"{speaker_signals.get('interruption_score', 0)}/100")
            sp_cols[2].metric("Debate", f"{speaker_signals.get('debate_score', 0)}/100")
            sp_boost = c.get("speaker_boost", 0)
            if sp_boost:
                st.caption(f"Speaker boost: +{sp_boost} | {speaker_signals.get('reason', '')}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        ensure_directories()
        configure_rotating_logs(LOGS_DIR)
        _init_state()
        _flush_pending_long_defaults()
        _run_startup_diagnostics_once()
        log_widget_rerun_noop(st.session_state)
        ensure_ffmpeg_on_path(log=False)

        st.title("AI Clip Studio")
        st.caption(
            "Topic-agnostic short-form clips for TikTok, YouTube Shorts, and Reels - "
            "transcribe, score moments, edit titles, export vertical 9:16 with burned-in captions."
        )

        api_key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
        ffmpeg_path = ensure_ffmpeg_on_path()
        ffmpeg_ver = get_ffmpeg_version_line()
        gpu_status = get_gpu_acceleration_status()
        cuda_whisper = faster_whisper_cuda_available()
        ai_diag = _get_ai_diagnostics()

        # ------------------------------------------------------------------ SIDEBAR
        with st.sidebar:
            st.header("Settings")

            # GPU / export
            st.checkbox(
                "GPU acceleration (NVENC exports + local Whisper)",
                key="cs_gpu_acceleration",
            )
            gpu_on = bool(st.session_state.get("cs_gpu_acceleration", True))
            force_gpu_export = bool(st.session_state.get("cs_force_gpu_export", False))
            will_try_nvenc = should_attempt_nvenc_on_export(prefer_gpu=gpu_on, force_gpu_mode=force_gpu_export)

            st.checkbox("Force GPU export (bypass NVENC probe gate)", key="cs_force_gpu_export")
            st.checkbox("Allow CPU fallback for exports", key="cs_allow_cpu_fallback")

            # Export mode
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
            st.selectbox(
                "Default caption preset",
                CAPTION_PRESET_OPTIONS,
                key="cs_default_caption_preset",
            )

            st.divider()
            st.subheader("AI editing intelligence")
            st.checkbox(
                "Enable AI signal boosts",
                key="cs_enable_signal_boosts",
                help="Local heuristic scoring for emotion, pacing, hooks — no extra LLM calls.",
            )
            st.checkbox(
                "Enable advanced captions",
                key="cs_enable_advanced_captions",
                help="Per-word karaoke highlighting when pysubs2 is installed; phrase fallback otherwise.",
            )
            st.checkbox(
                "Enable dynamic smart crop",
                key="cs_enable_dynamic_smart_crop",
                help="Smooth camera movement in smart crop mode (YOLO or OpenCV).",
            )
            st.checkbox(
                "Enable preview rendering",
                key="cs_enable_preview_rendering",
                help="Show 'Generate preview' size button on each clip card.",
            )

            dep_report = get_dependency_report()
            with st.expander("Optional dependency status", expanded=False):
                st.markdown(render_status_markdown(dep_report))

            # GPU status
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
                    "- **probe** = a real NVENC encode at **640×360** succeeded (minimum frame size for many drivers).\n"
                    "- A probe failure with *Frame Dimension less than the minimum* was a **test-size issue**, not a broken GPU encoder.\n"
                    "- Other failures may mean GPU busy, old driver, or remote desktop — see **Last NVENC probe log** stderr tail.\n"
                    "- Try **Force GPU export** or set `FORCE_NVENC_EXPORT=1` in `.env`."
                )
            with st.expander("Last NVENC probe log", expanded=False):
                st.code(get_last_nvenc_probe_log() or "(no probe run yet)", language=None)

            # CUDA / Whisper
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

            # Clip duration
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
                                "_error": str(exc),
                                "python_version": "?",
                                "torch_installed": False,
                            }
                rtx = st.session_state.get("cs_rtx_status")
                if rtx is None:
                    st.caption(
                        "GPU pipeline probes are deferred until you click **Refresh GPU status** "
                        "(avoids torchcodec/sentence-transformers import failures during startup)."
                    )
                else:
                    if rtx.get("_error"):
                        st.warning(
                            "Pipeline status check failed (non-fatal). "
                            f"Embeddings may be unavailable: {rtx['_error']}"
                        )
                    st.markdown(f"**Python:** `{rtx.get('python_version', '?')}`")
                    st.markdown(f"**torch installed:** `{rtx.get('torch_installed')}`")
                    if rtx.get("torch_installed"):
                        st.markdown(f"**torch version:** `{rtx.get('torch_version')}`")
                        st.markdown(f"**torch CUDA available:** `{rtx.get('torch_cuda_available')}`")
                        st.markdown(f"**torch CUDA device count:** `{rtx.get('torch_cuda_device_count')}`")
                        if rtx.get("torch_cuda_device_name"):
                            st.markdown(f"**torch CUDA device:** `{rtx.get('torch_cuda_device_name')}`")
                    st.markdown(
                        f"**sentence-transformers installed:** `{rtx.get('sentence_transformers_installed')}`"
                    )
                    st.markdown(
                        f"**Embeddings device (selected):** `{rtx.get('embeddings_device_selected', 'cpu')}`"
                    )
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
            st.checkbox(
                "Rate Limit Safe Mode",
                key="cs_rate_limit_safe",
                help="Exponential backoff on 429 errors with retry/resume.",
            )
            st.checkbox(
                "Use cached analysis if available",
                key="cs_use_analysis_cache",
            )
            st.slider(
                "Delay between OpenAI calls (sec)",
                min_value=0.0,
                max_value=3.0,
                step=0.25,
                key="cs_openai_call_delay",
            )
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

            # FFmpeg
            st.divider()
            st.caption("**FFmpeg**")
            if ffmpeg_path:
                st.code(ffmpeg_path, language=None)
                st.caption(ffmpeg_ver or "version: (unknown)")
            else:
                st.error(f"FFmpeg not found. Set **{ENV_FFMPEG_BINARY}** in `.env` or install FFmpeg.")

            if st.button("Open exports folder", width="stretch"):
                ensure_directories()
                open_folder(CLIP_STUDIO_OUTPUT_DIR)
                st.toast("Opened outputs/clips")

        # ------------------------------------------------------------------ UPLOAD
        st.subheader("1. Upload video")
        st.info(
            "For DaVinci Resolve exports, use **Local File Path** mode to avoid copies entirely."
        )
        input_mode = st.radio(
            "Video source",
            ["Browser upload", "Local file path"],
            horizontal=True,
            key="cs_video_input_mode",
        )
        if input_mode == "Local file path":
            local_path = st.text_input(
                "Path to video on disk",
                placeholder=r"C:\Videos\export.mp4",
                key="cs_local_video_path",
            )
            if st.button("Use local file", type="primary"):
                if not local_path.strip():
                    st.error("Enter a file path.")
                else:
                    lp = Path(local_path.strip()).expanduser()
                    if not lp.is_file():
                        st.error(f"File not found: `{lp}`")
                    elif lp.suffix.lower() not in (".mp4", ".mov", ".m4v"):
                        st.error("Supported formats: MP4, MOV, M4V.")
                    else:
                        resolved = lp.resolve()
                        st.session_state.cs_video_path = resolved
                        st.session_state[SESSION_FORCE_REANALYZE] = True
                        try:
                            st.session_state.cs_media_duration = get_media_duration_seconds(resolved)
                            _apply_long_podcast_defaults()
                        except Exception:
                            logger.exception("ffprobe duration failed")
                        st.session_state.cs_status = f"Using local file: `{resolved}`"
                        st.rerun()
            if st.session_state.cs_video_path and input_mode == "Local file path":
                st.success(str(st.session_state.cs_video_path))

        st.caption(
            f"**Upload limit:** up to **{CLIP_STUDIO_MAX_UPLOAD_MB} MB (~4 GB)** per file. "
            "Large files take time in browser before Save appears."
        )
        with st.expander("Upload troubleshooting"):
            st.markdown(
                f"- Server cap: `maxUploadSize = {CLIP_STUDIO_MAX_UPLOAD_MB}` (MB) in `.streamlit/config.toml`\n"
                "- Restart Streamlit after changing `config.toml`\n"
                "- Reverse proxies (nginx, Cloudflare) may impose a smaller body limit\n"
                "- Keep browser tab open during upload"
            )

        if input_mode == "Browser upload":
            up = st.file_uploader("MP4 / MOV / M4V", type=["mp4", "mov", "m4v"], key="cs_upload")
            if up is not None:
                sz = _uploaded_file_size_bytes(up)
                if sz > CLIP_STUDIO_MAX_UPLOAD_BYTES:
                    st.error(
                        f"File is **{_format_size(sz)}** - exceeds limit of **{CLIP_STUDIO_MAX_UPLOAD_MB} MB**. "
                        "Compress with `compress_video.bat` or raise `maxUploadSize` in `config.toml`."
                    )
                else:
                    st.success(f"Ready: **{up.name}** - {_format_size(sz)}. Click **Save** to continue.")
                if st.button("Save upload to project", type="primary", disabled=sz > CLIP_STUDIO_MAX_UPLOAD_BYTES):
                    bar = st.progress(0.0, text="Preparing to save...")
                    try:
                        path, reused = save_upload_once(up, progress_bar=bar)
                        st.session_state.cs_video_path = path
                        st.session_state.cs_upload_reused = reused
                        if not reused:
                            st.session_state.cs_segments = []
                            st.session_state.cs_formatted = ""
                            st.session_state.cs_clips = []
                            st.session_state.cs_session_dir = None
                            st.session_state.cs_media_duration = 0.0
                            st.session_state[SESSION_FORCE_REANALYZE] = True
                        try:
                            st.session_state.cs_media_duration = get_media_duration_seconds(path)
                            _apply_long_podcast_defaults()
                        except Exception:
                            logger.exception("ffprobe duration failed")
                        if reused:
                            st.session_state.cs_status = (
                                f"Reused: `{path.relative_to(PROJECT_ROOT)}` ({_format_size(sz)})"
                            )
                        else:
                            st.session_state.cs_status = (
                                f"Saved: `{path.relative_to(PROJECT_ROOT)}` ({_format_size(sz)})"
                            )
                        try:
                            bar.progress(1.0, text="Saved." if not reused else "Reused existing file.")
                        except TypeError:
                            bar.progress(1.0)
                        st.rerun()
                    except OSError as exc:
                        st.error(f"Could not write file (disk full or permissions?): {exc}")
                    finally:
                        bar.empty()

            if st.button("Clean duplicate uploads", width="stretch"):
                with st.spinner("Scanning uploads folder..."):
                    result = clean_duplicate_uploads()
                moved = int(result.get("moved", 0))
                saved = int(result.get("bytes_saved", 0))
                if moved:
                    st.success(
                        f"Moved **{moved}** duplicate file(s) to `uploads/_duplicates/` "
                        f"(~{_format_size(saved)})."
                    )
                else:
                    st.info("No duplicate uploads found to move.")

        if st.session_state.get("cs_upload_reused"):
            st.info("This video was already saved. Reusing existing project file.")

        if st.session_state.cs_video_path:
            try:
                rel = st.session_state.cs_video_path.relative_to(PROJECT_ROOT)
                st.success(str(rel))
            except ValueError:
                st.success(str(st.session_state.cs_video_path))

        # ------------------------------------------------------------------ TRANSCRIBE
        st.subheader("2. Transcript")
        can_transcribe = bool(
            st.session_state.cs_video_path
            and (api_key or bool(st.session_state.get("cs_gpu_acceleration", True)))
        )
        if not st.session_state.cs_video_path:
            st.info("Save a video first.")
        elif not can_transcribe:
            st.warning("Enable **GPU acceleration** (local Whisper) or add **OPENAI_API_KEY** for cloud Whisper.")
        else:
            whisper_lang = st.text_input(
                "Whisper language (optional ISO code)",
                placeholder="e.g. en - leave empty for auto",
            )
            if st.button("Transcribe", type="primary"):
                work = CLIP_STUDIO_OUTPUT_DIR / "_work"
                work.mkdir(parents=True, exist_ok=True)
                lang = whisper_lang.strip() or None
                model_sz = str(st.session_state.get("cs_whisper_model", "base"))
                prefer_gpu = bool(st.session_state.get("cs_gpu_acceleration", True))
                with st.spinner("Transcribing (may take a while for long videos)"):
                    try:
                        segs, full = transcribe_video(
                            st.session_state.cs_video_path,
                            api_key,
                            work_dir=work,
                            language=lang,
                            prefer_gpu=prefer_gpu,
                            faster_whisper_model=model_sz,
                        )
                        # Merge into sentence groups for better LLM context
                        merged = merge_segments_into_sentences(segs)
                        st.session_state.cs_segments = merged if merged else segs
                        st.session_state.cs_formatted = segments_to_prompt_transcript(
                            st.session_state.cs_segments
                        )
                        st.session_state.cs_clips = []
                        st.session_state[SESSION_FORCE_REANALYZE] = True
                        try:
                            st.session_state.cs_media_duration = get_media_duration_seconds(
                                Path(st.session_state.cs_video_path)
                            )
                        except Exception:
                            pass
                        _apply_long_podcast_defaults()
                        st.session_state.cs_status = (
                            f"Transcribed {len(segs)} raw segments -> "
                            f"{len(st.session_state.cs_segments)} sentence groups."
                        )
                        st.session_state.cs_session_telemetry = get_session_telemetry().to_dict()
                    except Exception as e:
                        logger.exception("Transcribe failed")
                        category, user_msg = classify_exception(e)
                        st.session_state.cs_session_telemetry = get_session_telemetry().to_dict()
                        if category == "StreamlitStateError":
                            st.warning(user_msg)
                            st.caption(f"Technical detail: {e}")
                        else:
                            st.error(f"**{category}:** {user_msg}")

        if st.session_state.get("cs_status"):
            st.caption(st.session_state.cs_status)

        if st.session_state.cs_segments:
            with st.expander("Transcript preview", expanded=False):
                st.text_area(
                    "Timestamped transcript",
                    st.session_state.cs_formatted[:50_000]
                    + ("..." if len(st.session_state.cs_formatted) > 50_000 else ""),
                    height=200,
                    disabled=True,
                )
            if api_key and st.session_state.cs_formatted:
                _prof = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", ""))
                _plan = get_cached_token_plan(
                    st.session_state,
                    st.session_state.cs_formatted,
                    _prof,
                    target_count=int(st.session_state.get("cs_target_clips", 20)),
                    clip_style=str(st.session_state.get("cs_clip_style", "Balanced")),
                    emit_logs=False,
                )
                _budget = _prof.max_tokens
                _pruned = _plan.after_prune
                if _pruned > _budget:
                    st.warning(
                        f"This run may use approximately **{_pruned:,}** tokens after pruning. "
                        f"Budget is **{_budget:,}**. Token Saver Mode will be enforced."
                    )
                else:
                    st.caption(
                        f"Estimated tokens (after GPU pruning): ~{_pruned:,} "
                        f"(budget {_budget:,}) | regions: {_plan.effective_regions}, passes: {_plan.effective_passes}"
                    )

        # ------------------------------------------------------------------ DIARIZATION (optional)
        _diar_turns: list[dict] = st.session_state.get("cs_diarization_turns") or []
        # Prefer the canonical work WAV; fall back to any _whisper_input.wav under outputs/
        _work_wav = CLIP_STUDIO_OUTPUT_DIR / "_work" / "_whisper_input.wav"
        if not _work_wav.is_file():
            _fallback_wavs = sorted(CLIP_STUDIO_OUTPUT_DIR.rglob("_whisper_input.wav"))
            if _fallback_wavs:
                _work_wav = _fallback_wavs[0]
        if st.session_state.cs_segments:
            st.subheader("2b. Speaker Diarization (optional)")
            if not _work_wav.is_file():
                st.caption(
                    "Diarization WAV not found. Run transcription first so a WAV is available."
                )
            if st.button(
                "Detect Speakers (faster-whisper)",
                key="cs_run_diarization",
                disabled=not _work_wav.is_file(),
            ):
                with st.spinner("Detecting speaker turns via faster-whisper gap analysis..."):
                    try:
                        _new_turns = diarize_audio_file(str(_work_wav))
                        st.session_state.cs_diarization_turns = _new_turns
                        _diar_turns = _new_turns
                        if _new_turns:
                            _n_sp = len({t["speaker"] for t in _new_turns})
                            st.success(f"Found {len(_new_turns)} speaker turns across {_n_sp} speaker(s).")
                        else:
                            st.warning(
                                "No speaker turns detected. "
                                "Ensure faster-whisper is installed (`pip install faster-whisper`)."
                            )
                    except Exception as _diar_exc:
                        logger.exception("Speaker diarization failed")
                        st.error(f"Diarization failed: {_diar_exc}")

            if _diar_turns:
                _unique_speakers = sorted({t["speaker"] for t in _diar_turns})
                st.caption(
                    f"{len(_unique_speakers)} speaker(s): {', '.join(_unique_speakers)}"
                )
                st.markdown("**Rename speakers** (leave blank = keep generic label)")
                _name_map: dict[str, str] = {}
                _sp_cols = st.columns(min(len(_unique_speakers), 4))
                for _si, _sp_id in enumerate(_unique_speakers):
                    _prev_name = (st.session_state.get("cs_speaker_names") or {}).get(_sp_id, "")
                    _entered = _sp_cols[_si % len(_sp_cols)].text_input(
                        _sp_id,
                        value=_prev_name,
                        placeholder="e.g. Host",
                        key=f"sp_name_{_sp_id}",
                    )
                    _name_map[_sp_id] = _entered.strip() if _entered.strip() else _sp_id
                st.session_state.cs_speaker_names = _name_map

                # Merge speaker labels into cs_segments and regenerate formatted transcript
                _labeled_segs = _assign_speakers_to_segments(
                    st.session_state.cs_segments, _diar_turns, _name_map
                )
                st.session_state.cs_segments = _labeled_segs
                st.session_state.cs_formatted = segments_to_prompt_transcript(_labeled_segs)
                st.caption(
                    "Speaker labels applied — re-run **Analyze** to include "
                    "`[Host]`/`[Guest]` context in the GPT prompt."
                )

        # ------------------------------------------------------------------ CLIP AI
        st.subheader("3. AI clip suggestions")

        _tel_main = st.session_state.get("cs_session_telemetry") or {}
        if _tel_main:
            with st.expander("Session diagnostics (timing, tokens, rejections)", expanded=False):
                try:
                    st.markdown(render_telemetry_markdown(_tel_main))
                except Exception as exc:
                    logger.exception("Step 3 diagnostics render failed")
                    st.caption(f"Diagnostics display error: {exc}")

        creator_note = st.text_area(
            "Optional creator note",
            placeholder="e.g. audience is beginners; brand is playful - leave blank for topic-agnostic",
            height=68,
        )

        col_analyze, col_reanalyze, col_rescore, col_more = st.columns([2, 1, 1, 1])

        def _get_analysis_params() -> dict:
            min_c = float(st.session_state.get("cs_min_clip_seconds", 25))
            max_c = float(st.session_state.get("cs_max_clip_seconds", 160))
            if min_c > max_c:
                min_c, max_c = max_c, min_c
            return dict(
                formatted=st.session_state.cs_formatted,
                api_key=api_key,
                segments=st.session_state.cs_segments,
                creator_note=creator_note,
                min_c=min_c,
                max_c=max_c,
                ctx_b=float(st.session_state.get("cs_context_before", 8)),
                ctx_a=float(st.session_state.get("cs_context_after", 12)),
                allow_over=bool(st.session_state.get("cs_allow_exceed_max", False)),
                media_dur=float(st.session_state.get("cs_media_duration") or 0.0),
                target_count=int(st.session_state.get("cs_target_clips", 20)),
                min_gap_seconds=float(st.session_state.get("cs_min_gap_seconds", 60)),
                similarity_threshold=float(st.session_state.get("cs_similarity_threshold", 45)) / 100.0,
                clip_style=str(st.session_state.get("cs_clip_style", "Balanced")),
                video_filename=str(st.session_state.cs_video_path.name if st.session_state.cs_video_path else ""),
            )

        with col_analyze:
            can_analyze = bool(st.session_state.cs_formatted and api_key)
            est_data = st.session_state.get("cs_token_estimate") or {}
            if est_data:
                budget = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", "")).max_tokens
                st.caption(
                    f"Est. tokens: **~{est_data.get('estimated_total_tokens', 0):,}** "
                    f"(budget {budget:,}) | ~{est_data.get('estimated_calls', 0)} API calls"
                )
            openai_status = st.session_state.get("cs_openai_status", "")
            if openai_status:
                st.info(openai_status)
            if st.button("Analyze for high-retention clips", type="primary", disabled=not can_analyze, width="stretch"):
                status_slot = st.empty()

                def _status_cb(msg: str) -> None:
                    st.session_state.cs_openai_status = msg
                    status_slot.info(msg)

                with st.spinner("Scoring clips (multi-pass — may take 30-120s on long podcasts)"):
                    try:
                        params = _get_analysis_params()
                        vid = params.get("video_filename") or str(st.session_state.get("cs_video_path", ""))
                        th = hash_transcript(params["formatted"], params["segments"])
                        inv = get_invalidation_reason(
                            st.session_state,
                            video_identity=vid,
                            transcript_hash=th,
                        )
                        if inv is None and st.session_state.get("cs_clips") and not st.session_state.get(SESSION_FORCE_REANALYZE):
                            from clip_engine.clip_finalizer import ensure_clips_finalized

                            segs = st.session_state.get("cs_segments") or []
                            export_max = min(
                                float(st.session_state.get("cs_max_clip_seconds", 120)),
                                120.0,
                            )
                            finalized, _did_finalize = ensure_clips_finalized(
                                list(st.session_state.cs_clips),
                                segs,
                                min_duration=float(st.session_state.get("cs_min_clip_seconds", 25)),
                                max_duration=export_max,
                            )
                            if _did_finalize:
                                st.session_state.cs_clips = finalized
                            logger.info("[SESSION] no-op analysis rerun — clips still valid")
                            st.session_state.cs_status = (
                                f"Using existing {len(st.session_state.cs_clips)} clips "
                                "(edit hooks/times freely — no new OpenAI calls)."
                            )
                            st.session_state.cs_openai_status = ""
                            st.rerun()
                        clips, pipe_stats = _run_clip_analysis(**params, status_callback=_status_cb)
                        pipe_stats = pipe_stats or {}
                        st.session_state.cs_clips = clips or []
                        st.session_state.cs_pipeline_stats = pipe_stats
                        if pipe_stats.get("cache_hit"):
                            st.session_state.cs_status = "Loaded cached analysis — no OpenAI tokens used."
                        else:
                            st.session_state.cs_status = f"Suggested {len(clips)} clips."
                        st.session_state.cs_openai_status = ""
                        logger.info(
                            "[lifecycle] Clip analysis finished — %d clips; app remains active",
                            len(clips or []),
                        )
                        # Release GPU memory after heavy analysis to prevent post-analysis crash
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logger.info("[cuda] empty_cache() called after clip analysis")
                        except Exception as exc:
                            logger.debug("CUDA cache release skipped: %s", exc)
                    except OpenAIRateLimitError as e:
                        logger.exception("Clip analysis rate limited")
                        st.error(
                            f"OpenAI rate limit at **{e.stage}** (model `{e.model}`). "
                            f"{e.mitigation} Partial progress saved — click Analyze again to resume."
                        )
                    except Exception as e:
                        logger.exception("Clip analysis failed")
                        category, user_msg = classify_exception(e)
                        st.session_state.cs_session_telemetry = get_session_telemetry().to_dict()
                        st.error(f"**{category}:** {user_msg}")
                if not api_key:
                    st.info("Add **OPENAI_API_KEY** to `.env` to run clip analysis.")

        with col_reanalyze:
            if st.session_state.cs_formatted and api_key:
                if st.button("Re-analyze", width="stretch", help="Force fresh OpenAI analysis (ignores cache)"):
                    st.session_state[SESSION_FORCE_REANALYZE] = True
                    st.rerun()

        with col_rescore:
            if st.session_state.cs_clips:
                if st.button("Re-score clips", width="stretch", disabled=not can_analyze):
                    st.session_state[SESSION_FORCE_REANALYZE] = True
                    with st.spinner("Re-scoring..."):
                        try:
                            clips, pipe_stats = _run_clip_analysis(**_get_analysis_params())
                            pipe_stats = pipe_stats or {}
                            st.session_state.cs_clips = clips or []
                            st.session_state.cs_pipeline_stats = pipe_stats
                            st.session_state.cs_status = f"Re-scored: {len(clips)} clips."
                            st.rerun()
                        except Exception as e:
                            st.error(f"Re-score failed: {e}")

        with col_more:
            if st.session_state.cs_clips:
                if st.button("Find more clips", width="stretch", disabled=not can_analyze):
                    with st.spinner("Finding more..."):
                        try:
                            extra, extra_stats = _run_clip_analysis(**_get_analysis_params())
                            extra_stats = extra_stats or {}
                            existing_ranges = {
                                (c.get("start_seconds"), c.get("end_seconds"))
                                for c in st.session_state.cs_clips
                            }
                            new_clips = [
                                c for c in extra
                                if (c.get("start_seconds"), c.get("end_seconds")) not in existing_ranges
                            ]
                            st.session_state.cs_clips.extend(new_clips)
                            st.session_state.cs_pipeline_stats = extra_stats
                            st.session_state.cs_status = f"Added {len(new_clips)} new clips ({len(st.session_state.cs_clips)} total)."
                            st.rerun()
                        except Exception as e:
                            st.error(f"Find more failed: {e}")

        # ------------------------------------------------------------------ CLIP CARDS
        clips: list = st.session_state.get("cs_clips") or []
        if clips:
            if not all(bool(c.get("finalizer_checked")) for c in clips):
                from clip_engine.clip_finalizer import ensure_clips_finalized

                segs_for_finalize = st.session_state.get("cs_segments") or []
                ui_max = min(float(st.session_state.get("cs_max_clip_seconds", 120)), 120.0)
                clips, _ui_finalized = ensure_clips_finalized(
                    clips,
                    segs_for_finalize,
                    min_duration=float(st.session_state.get("cs_min_clip_seconds", 25)),
                    max_duration=ui_max,
                )
                if _ui_finalized:
                    st.session_state.cs_clips = clips
            media_dur_for_map = float(st.session_state.get("cs_media_duration") or 0.0)
            pipe_stats = st.session_state.get("cs_pipeline_stats") or {}
            target_req = int(pipe_stats.get("target_clips", st.session_state.get("cs_target_clips", 20)))
            st.success(f"**{len(clips)} clips** selected (target: {target_req}) - check Export to include in batch.")

            if pipe_stats:
                st.caption(
                    f"Pipeline pool: **{pipe_stats.get('gpu_shortlist', 0)}** GPU shortlist | "
                    f"**{pipe_stats.get('raw_ai_candidates', pipe_stats.get('raw_candidates', '?'))}** raw AI | "
                    f"**{pipe_stats.get('valid_after_schema', '?')}** valid | "
                    f"**{pipe_stats.get('rescued_candidates', 0)}** rescued | "
                    f"**{pipe_stats.get('local_fallback_candidates', 0)}** local fallback | "
                    f"**{pipe_stats.get('raw_candidates', '?')}** in pool"
                    + (
                        f" | profile: **{pipe_stats.get('ai_profile', 'SAFE')}**"
                        if pipe_stats.get("ai_profile")
                        else ""
                    )
                    + (
                        f" | GPT passes: **{pipe_stats.get('gpt_passes_used', '?')}**"
                        if pipe_stats.get("gpt_passes_used")
                        else ""
                    )
                )
                st.caption(
                    f"Rejected: **{pipe_stats.get('rejected_invalid_time', 0)}** time | "
                    f"**{pipe_stats.get('rejected_duration', 0)}** duration | "
                    f"**{pipe_stats.get('rejected_empty_transcript', 0)}** empty text | "
                    f"**{pipe_stats.get('rejected_overlap_early', 0)}** early overlap dedupe"
                )
                st.caption(
                    f"Selection: **{pipe_stats.get('removed_overlap', 0)}** overlap | "
                    f"**{pipe_stats.get('removed_duplicates', 0)}** duplicate similarity | "
                    f"**{pipe_stats.get('final_clips', len(clips))}** final"
                    + (" | Discovery Mode ON" if pipe_stats.get("discovery_mode") else "")
                    + (
                        f" | models: `{pipe_stats.get('model_fast', 'n/a')}` / `{pipe_stats.get('model_quality', 'n/a')}`"
                    )
                    + (
                        f" | est. ~{int((st.session_state.get('cs_token_estimate') or {}).get('after_prune') or pipe_stats.get('estimated_tokens', 0)):,} tokens"
                        if (st.session_state.get("cs_token_estimate") or {}).get("after_prune")
                        or pipe_stats.get("estimated_tokens")
                        else ""
                    )
                    + (" | **cache hit**" if pipe_stats.get("cache_hit") else "")
                    + (" | **resumed**" if pipe_stats.get("resumed_from_progress") else "")
                    + (
                        f" | **{pipe_stats.get('expansion_pass_count', 0)}** expansion pass(es)"
                        if pipe_stats.get("expansion_pass_ran")
                        else ""
                    )
                    + (
                        f" | **{pipe_stats.get('rejected_ungrounded', 0)}** rejected (metadata)"
                        if pipe_stats.get("rejected_ungrounded")
                        else ""
                    )
                )
                if int(pipe_stats.get("final_clips", len(clips))) < 12:
                    st.warning(
                        "Only "
                        f"{pipe_stats.get('final_clips', len(clips))} clips found. Discovery Mode can rescue "
                        "borderline moments and add local transcript-window candidates."
                    )
                for w in pipe_stats.get("warnings", []):
                    st.warning(w)

            diag = st.session_state.get(SESSION_ANALYSIS_DIAGNOSTICS) or {}
            with st.expander("Analysis diagnostics", expanded=False):
                st.markdown(
                    f"- **Cache:** {'hit' if diag.get('cache_hit') else 'miss'} "
                    f"{('(' + str(diag.get('cache_miss_reason')) + ')') if diag.get('cache_miss_reason') else ''}\n"
                    f"- **Invalidation:** {diag.get('invalidation_reason') or 'none (widgets do not invalidate)'}\n"
                    f"- **Models:** `{diag.get('model_fast', 'n/a')}` / `{diag.get('model_quality', 'n/a')}`\n"
                    f"- **JSON fallback used:** {bool((diag.get('json_telemetry') or {}).get('json_fallback'))}\n"
                    f"- **Transcript duration:** {diag.get('transcript_duration', 0):.0f}s\n"
                    f"- **Raw candidates:** {diag.get('raw_candidates', pipe_stats.get('raw_candidates', 0))}\n"
                    f"- **Rescued:** {diag.get('rescued_candidates', pipe_stats.get('rescued_candidates', 0))}\n"
                    f"- **Final clips:** {diag.get('final_clips', len(clips))}\n"
                    f"- **Boundary repairs:** {diag.get('boundary_repairs', pipe_stats.get('boundary_repairs', 0))}\n"
                    f"- **Title repairs:** {diag.get('title_repairs', pipe_stats.get('title_repairs', 0))}\n"
                    f"- **OpenAI calls (this analysis):** {diag.get('openai_calls_used', pipe_stats.get('openai_calls_used', 0))}\n"
                    f"- **Fingerprint:** `{diag.get('analysis_fingerprint', '')}`"
                )
                dsc = pipe_stats.get("discovery_scan") or {}
                if dsc or pipe_stats.get("gpu_local_candidates") is not None:
                    with st.expander("Discovery scan diagnostics", expanded=False):
                        st.markdown(
                            f"- **Windows scanned:** {dsc.get('windows_scanned', 0)}\n"
                            f"- **Windows kept:** {dsc.get('windows_kept', 0)}\n"
                            f"- **Windows rejected:** {dsc.get('windows_rejected', 0)}\n"
                            f"- **Emotion triggers:** {dsc.get('emotion_triggers', 0)}\n"
                            f"- **Curiosity triggers:** {dsc.get('curiosity_triggers', 0)}\n"
                            f"- **Pacing triggers:** {dsc.get('pacing_triggers', 0)}\n"
                            f"- **Hook triggers:** {dsc.get('hook_triggers', 0)}\n"
                            f"- **Story phrase triggers:** {dsc.get('story_phrase_triggers', 0)}\n"
                            f"- **Transcript-only candidates:** {dsc.get('transcript_only_candidates', 0)}\n"
                            f"- **Fallback generated:** {dsc.get('fallback_generated', 0)}\n"
                            f"- **Discovery boosts:** {dsc.get('discovery_boost_activations', 0)}"
                        )
                        rej = dsc.get("rejection_reasons") or {}
                        if rej:
                            st.caption(
                                "**Rejection reasons**\n"
                                + "\n".join(f"  - {k}: {v}" for k, v in sorted(rej.items(), key=lambda x: -x[1]))
                            )

            dg = pipe_stats.get("duration_governor") or {}
            occ = pipe_stats.get("timeline_occupancy") or {}
            if dg or occ:
                with st.expander("Duration / expansion governor", expanded=False):
                    for stage, stg in dg.items():
                        if isinstance(stg, dict):
                            st.markdown(
                                f"**{stage}**: checked {stg.get('checked', 0)} | "
                                f"soft-clamped {stg.get('clamped_soft', 0)} | "
                                f"hard-clamped {stg.get('clamped_hard', 0)} | "
                                f"over soft before {stg.get('over_soft_before', 0)}"
                            )
                    fin = occ.get("final") or {}
                    if fin:
                        st.markdown(
                            f"**Timeline (final):** {fin.get('clip_count', 0)} clips | "
                            f"union {fin.get('union_seconds', 0)}s | "
                            f"overlap {fin.get('overlap_seconds', 0)}s "
                            f"(ratio {fin.get('overlap_ratio', 0)}) | "
                            f">{fin.get('over_soft_cap', 0)} over 90s | "
                            f"max {fin.get('max_duration', 0)}s"
                        )
                        durs = fin.get("durations") or []
                        if durs:
                            st.caption(f"Durations (s): {', '.join(str(d) for d in durs[:14])}")

            fr = pipe_stats.get("finalizer_report") or {}
            if fr or any(pipe_stats.get(k) for k in (
                "finalizer_expanded", "finalizer_merged", "finalizer_rejected", "finalizer_hooks_repaired"
            )):
                with st.expander("Finalizer Report", expanded=False):
                    st.markdown(
                        f"- **Checked:** {fr.get('checked', pipe_stats.get('final_clips', len(clips)))}\n"
                        f"- **Expanded:** {fr.get('expanded', pipe_stats.get('finalizer_expanded', 0))}\n"
                        f"- **Merged:** {fr.get('merged', pipe_stats.get('finalizer_merged', 0))}\n"
                        f"- **Soft warnings:** {fr.get('soft_warnings', 0)}\n"
                        f"- **Hard rejections:** {fr.get('hard_rejections', fr.get('rejected', pipe_stats.get('finalizer_rejected', 0)))}\n"
                        f"- **Hooks repaired:** {fr.get('hooks_repaired', pipe_stats.get('finalizer_hooks_repaired', 0))}\n"
                        f"- **Kept:** {fr.get('kept', len(clips))}"
                    )
                    warn_counts = [
                        ("low_hook_warning", "Hook below ideal"),
                        ("short_duration_warning", "Short 20–25s"),
                        ("dangling_ending_warning", "Dangling ending"),
                        ("host_question_warning", "Host-question opening"),
                        ("metadata_grounding_warning", "Metadata grounding"),
                    ]
                    warn_lines = [
                        f"  - {label}: {fr.get(key, 0)}"
                        for key, label in warn_counts
                        if int(fr.get(key, 0)) > 0
                    ]
                    if warn_lines:
                        st.caption("**Warning counts**\n" + "\n".join(warn_lines))
                    repairs = fr.get("hook_repairs") or []
                    if repairs:
                        for row in repairs[:8]:
                            st.caption(
                                f"Hook #{int(row.get('index', 0)) + 1}: "
                                f"\"{str(row.get('old', ''))[:50]}\" → "
                                f"\"{str(row.get('new', ''))[:50]}\" "
                                f"(score {row.get('score', '?')})"
                            )

            tok = get_session_tokens()
            if tok["total"] > 0:
                cost = tok.get("estimated_cost_usd") or (
                    (tok["prompt"] * 2.50 + tok["completion"] * 10.00) / 1_000_000
                )
                st.caption(
                    f"Session tokens: **{tok['total']:,}** "
                    f"(prompt: {tok['prompt']:,} / completion: {tok['completion']:,}) | "
                    f"Est. cost: **${cost:.4f}** (GPT-4o)"
                )
                tracker_export = st.session_state.get("cs_token_tracker") or {}
                if tracker_export.get("tokens_avoided_cache"):
                    st.caption(f"Tokens avoided by cache: **{tracker_export['tokens_avoided_cache']:,}**")
                if tracker_export.get("retry_tokens"):
                    st.caption(f"Retry tokens (est.): **{tracker_export['retry_tokens']:,}**")
                per_stage = tracker_export.get("per_stage") or {}
                if per_stage:
                    with st.expander("Token usage by stage", expanded=False):
                        for stage, usage in per_stage.items():
                            st.caption(
                                f"**{stage}**: {usage.get('total_tokens', 0):,} tokens "
                                f"({usage.get('call_count', 0)} calls)"
                            )

            _render_clip_map(clips, media_dur_for_map)

            series_export_wids: dict[str, list[str]] = {}
            for c in clips:
                if c.get("is_part_of_series") and c.get("series_id"):
                    sid = str(c["series_id"])
                    wid = str(c.get("_wid") or "")
                    if wid:
                        series_export_wids.setdefault(sid, []).append(wid)

            for i, c in enumerate(clips):
                wid = c.get("_wid") or str(i)
                fs_default = float(c.get("start_seconds", c.get("start", 0)))
                fe_default = float(c.get("end_seconds", c.get("end", 0)))
                o_s = float(c.get("original_start", fs_default))
                o_e = float(c.get("original_end", fe_default))
                score = int(c.get("composite_score", 0))
                platforms = c.get("platform_fit", [])
                platform_str = " | ".join(platforms) if platforms else "n/a"
                warnings = c.get("warnings", [])

                with st.container(border=True):
                    header_col, score_col, export_col = st.columns([4, 1, 1])
                    with header_col:
                        st.markdown(f"### #{i+1} - {c.get('hook_title', 'Untitled clip')}")
                        if c.get("is_part_of_series"):
                            part_n = int(c.get("part_number", 1))
                            part_total = int(c.get("part_total", 2))
                            try:
                                st.badge(f"Part {part_n} of {part_total}")
                            except Exception:
                                st.markdown(f"**Part {part_n} of {part_total}**")
                            st.caption(
                                "This clip is part of a series — both parts will export together"
                            )
                        st.caption(f"Platform: **{platform_str}** | Signal: `{c.get('dominant_signal', 'n/a')}` | Style: `{c.get('caption_style', 'n/a')}`")
                    with score_col:
                        virality = int(c.get("virality_score", score))
                        color = "High" if virality >= 80 else ("Mid" if virality >= 65 else "Low")
                        st.metric(f"{color} Virality", f"{virality}/100")
                        if c.get("virality_explanation"):
                            st.caption(str(c["virality_explanation"])[:72])
                        hook_q = int(c.get("hook_quality_score", 0))
                        if hook_q:
                            st.caption(f"Hook: {hook_q}/100")
                    with export_col:
                        series_wids = series_export_wids.get(str(c.get("series_id", "")), [wid])
                        st.checkbox(
                            "Export",
                            value=bool(c.get("export", True)),
                            key=f"ex_{wid}",
                            on_change=_sync_series_export_checkbox,
                            args=(series_wids, wid),
                        )

                    # Editable fields row
                    e1, e2, e3, e4 = st.columns([2, 1, 1, 1])
                    with e1:
                        edits = st.session_state.get(SESSION_CLIP_EDITS) or {}
                        default_hook = edits.get(wid, {}).get(
                            "hook_title", str(c.get("hook_title", "") or "")
                        )
                        st.text_input(
                            "Hook / title (export only — does not re-run AI)",
                            value=default_hook,
                            key=f"hook_widget_{wid}",
                        )
                    with e2:
                        st.number_input(
                            "Start (s)",
                            value=fs_default,
                            min_value=0.0,
                            step=0.5,
                            format="%.1f",
                            key=f"start_{wid}",
                        )
                    with e3:
                        st.number_input(
                            "End (s)",
                            value=fe_default,
                            min_value=0.0,
                            step=0.5,
                            format="%.1f",
                            key=f"end_{wid}",
                        )
                    with e4:
                        preset_default = str(c.get("caption_preset", st.session_state.get("cs_default_caption_preset", "Clean")))
                        preset_idx = CAPTION_PRESET_OPTIONS.index(preset_default) if preset_default in CAPTION_PRESET_OPTIONS else 0
                        st.selectbox("Caption preset", CAPTION_PRESET_OPTIONS, index=preset_idx, key=f"preset_{wid}")

                    # Computed duration from editable fields
                    t0_val = float(st.session_state.get(f"start_{wid}", fs_default))
                    t1_val = float(st.session_state.get(f"end_{wid}", fe_default))
                    dur_val = max(0.0, t1_val - t0_val)
                    boundary_label = c.get("boundary_status", "ok")
                    if c.get("boundary_repaired") or boundary_label == "repaired":
                        boundary_label = "repaired"
                    elif c.get("boundary_warning"):
                        boundary_label = "warning"
                    exp_s = float(c.get("expanded_start", fs_default))
                    exp_e = float(c.get("expanded_end", fe_default))
                    orig_dur = float(c.get("original_duration", max(0.0, o_e - o_s)))
                    exp_dur = float(c.get("expanded_duration", dur_val))
                    growth_s = float(c.get("growth_seconds", max(0.0, exp_dur - orig_dur)))
                    growth_pct = float(c.get("growth_percent", 0))
                    merge_n = int(c.get("merge_source_count", 1))
                    st.caption(
                        f"Duration: **{dur_val:.1f}s** | Original: **{orig_dur:.1f}s** → "
                        f"Expanded: **{exp_dur:.1f}s** (+{growth_s:.1f}s, {growth_pct:.0f}%) | "
                        f"Merge sources: **{merge_n}** | Boundary: **{boundary_label}**"
                    )
                    if c.get("expansion_justification"):
                        st.caption(f"Expansion: {c['expansion_justification']}")

                    grounding = int(c.get("grounding_confidence", 0))
                    if grounding > 0:
                        g_label = "Strong" if grounding >= 50 else ("Weak" if grounding >= 25 else "Poor")
                        st.caption(f"Metadata grounding: **{grounding}%** ({g_label})")
                    if grounding < 25 or any("Metadata may not match" in str(w) for w in warnings):
                        st.warning("Metadata may not match final clip window.")

                    excerpt = extract_transcript_excerpt(
                        st.session_state.get("cs_segments") or [],
                        t0_val,
                        t1_val,
                        max_chars=900,
                    ) or c.get("grounded_transcript_excerpt", "")
                    if excerpt:
                        with st.expander("Transcript used for this clip", expanded=False):
                            st.text(excerpt)

                    tok_clip = c.get("_token_usage")
                    if tok_clip and tok_clip.get("total", 0) > 0:
                        st.caption(
                            f"Clip tokens: **{tok_clip['total']:,}** "
                            f"(prompt {tok_clip.get('prompt', 0):,} / completion {tok_clip.get('completion', 0):,})"
                        )

                    # Reason / context
                    reason_col, context_col = st.columns(2)
                    with reason_col:
                        st.markdown("**Why this clip**")
                        st.write(c.get("selection_reason", c.get("reason", "n/a")))
                    with context_col:
                        st.markdown("**Why this framing**")
                        st.write(c.get("ai_context_reason", c.get("context_reason", "n/a")))

                    if c.get("expansion_note"):
                        st.caption(f"Note: {c['expansion_note'].strip()}")

                    if warnings:
                        with st.expander("Warnings", expanded=False):
                            for w in warnings:
                                st.warning(w)

                    signal_scores = c.get("signal_scores", {})
                    speaker_signals = c.get("speaker_signals", {})
                    if signal_scores or speaker_signals:
                        sig_row = st.columns(6)
                        if signal_scores:
                            sig_row[0].caption(f"Emotion **{signal_scores.get('emotion_spike', 0)}**")
                            sig_row[1].caption(f"Pacing **{signal_scores.get('pacing', 0)}**")
                            sig_row[2].caption(f"Curiosity **{signal_scores.get('curiosity_gap', 0)}**")
                            sig_row[3].caption(f"Hook **{signal_scores.get('scroll_stopping_hook', 0)}**")
                        if speaker_signals:
                            sig_row[4].caption(f"Debate **{speaker_signals.get('debate_score', 0)}**")
                        reason = signal_scores.get("reason") or speaker_signals.get("reason", "")
                        if reason:
                            sig_row[5].caption(f"Boost: {reason[:60]}")

                    _render_score_breakdown(c)

                    if st.session_state.get("cs_enable_preview_rendering", True) and st.session_state.cs_video_path:
                        preview_key = f"preview_{wid}"
                        if st.button("Generate preview", key=f"btn_{preview_key}"):
                            PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                            preview_path = PREVIEW_OUTPUT_DIR / f"{wid}_preview.mp4"
                            export_mode_label = str(st.session_state.get("cs_export_mode_label", "Full frame fit with blurred background"))
                            export_mode = EXPORT_MODE_LABELS.get(export_mode_label, "full_fit")
                            preset = str(st.session_state.get(f"preset_{wid}", c.get("caption_preset", "Clean")))
                            try:
                                with st.spinner("Rendering preview..."):
                                    export_clip_preview(
                                        Path(st.session_state.cs_video_path),
                                        preview_path,
                                        t0_val, t1_val,
                                        st.session_state.get("cs_segments") or [],
                                        caption_preset=preset,
                                        export_mode=export_mode,
                                        advanced_captions=bool(st.session_state.get("cs_enable_advanced_captions", True)),
                                        dynamic_smart_crop=bool(st.session_state.get("cs_enable_dynamic_smart_crop", True)),
                                        prefer_gpu=bool(st.session_state.get("cs_gpu_acceleration", True)),
                                        allow_cpu_fallback=bool(st.session_state.get("cs_allow_cpu_fallback", True)),
                                    )
                                st.session_state.cs_previews[wid] = str(preview_path)
                                st.success("Preview ready")
                            except Exception as e:
                                st.error(f"Preview failed: {e}")
                        saved_preview = st.session_state.get("cs_previews", {}).get(wid)
                        if saved_preview and Path(saved_preview).is_file():
                            st.video(saved_preview)

        # ------------------------------------------------------------------ EXPORT
        st.subheader("4. Export vertical (9:16) + captions")
        st.caption(
            "Each MP4 uses the **edited** start/end times from the clip cards above. "
            "SRT and ASS sidecar files are written alongside each clip when enabled."
        )

        if not st.session_state.cs_video_path:
            st.info("Save a video upload first.")
        elif not clips:
            st.info("Run clip analysis to get exportable segments.")
        else:
            if st.button("Export selected clips", type="primary"):
                try:
                    session = CLIP_STUDIO_OUTPUT_DIR / datetime.now().strftime("session_%Y%m%d_%H%M%S")
                    session.mkdir(parents=True, exist_ok=True)
                    st.session_state.cs_session_dir = session
                    video = Path(st.session_state.cs_video_path)
                    segs = st.session_state.cs_segments
                    prefer_gpu = bool(st.session_state.get("cs_gpu_acceleration", True))
                    force_gpu = bool(st.session_state.get("cs_force_gpu_export", False))
                    allow_cpu = bool(st.session_state.get("cs_allow_cpu_fallback", True))
                    export_mode_label = str(st.session_state.get("cs_export_mode_label", "Full frame fit with blurred background"))
                    export_mode = EXPORT_MODE_LABELS.get(export_mode_label, "full_fit")
                    write_sidecars = bool(st.session_state.get("cs_write_sidecars", True))

                    to_export = _expand_export_selection(
                        clips,
                        [
                            c for c in clips
                            if st.session_state.get(
                                f"ex_{c.get('_wid', '')}", c.get("export", True)
                            )
                        ],
                    )
                    if not to_export:
                        st.warning("No clips selected for export. Check at least one 'Export' checkbox.")
                    else:
                        from clip_engine.clip_finalizer import (
                            ensure_clips_finalized,
                            validate_clip_for_export,
                        )

                        export_min = float(st.session_state.get("cs_min_clip_seconds", 25))
                        export_max = min(
                            float(st.session_state.get("cs_max_clip_seconds", 120)),
                            120.0,
                        )
                        to_export, _finalized_on_export = ensure_clips_finalized(
                            to_export,
                            segs,
                            min_duration=export_min,
                            max_duration=export_max,
                        )
                        if _finalized_on_export:
                            logger.info(
                                "[CLIP FINALIZER] export guard finalized %d clip(s) before export",
                                len(to_export),
                            )

                        exported = 0
                        failed = 0
                        prog = st.progress(0.0)
                        status_area = st.empty()
                        tracker = get_tracker()
                        audit_clips: list[dict] = []

                        for idx, c in enumerate(to_export):
                            wid = c.get("_wid", str(idx))
                            pre_t0 = float(st.session_state.get(f"start_{wid}", c.get("start_seconds", c.get("start", 0))))
                            pre_t1 = float(st.session_state.get(f"end_{wid}", c.get("end_seconds", c.get("end", 0))))
                            pre_hook = str(
                                st.session_state.get(
                                    f"hook_widget_{wid}",
                                    st.session_state.get(
                                        f"hook_{wid}", c.get("hook_title", f"clip_{idx+1}")
                                    ),
                                )
                            ).strip()
                            pre_check = dict(c)
                            pre_check["start_seconds"] = pre_t0
                            pre_check["end_seconds"] = pre_t1
                            pre_check["hook_title"] = pre_hook
                            ok_export, skip_reason = validate_clip_for_export(
                                pre_check,
                                min_duration=max(1.0, export_min * 0.5),
                                max_duration=export_max + 5.0,
                            )
                            if not ok_export:
                                failed += 1
                                st.warning(f"Skipped clip {idx + 1}: {skip_reason}")
                                logger.warning(
                                    "[EXPORT] skipped clip=%s reason=%s",
                                    wid,
                                    skip_reason,
                                )
                                prog.progress((idx + 1) / len(to_export))
                                continue

                            user_hook = str(
                                st.session_state.get(
                                    f"hook_widget_{wid}",
                                    st.session_state.get(
                                        f"hook_{wid}", c.get("hook_title", f"clip_{idx+1}")
                                    ),
                                )
                            )
                            t0 = float(st.session_state.get(f"start_{wid}", c.get("start_seconds", c.get("start", 0))))
                            t1 = float(st.session_state.get(f"end_{wid}", c.get("end_seconds", c.get("end", 0))))
                            preset = str(st.session_state.get(f"preset_{wid}", c.get("caption_preset", "Clean")))

                            export_clip = dict(c)
                            export_clip["start_seconds"] = t0
                            export_clip["end_seconds"] = t1
                            export_clip["hook_title"] = user_hook
                            export_clip["export_title"] = user_hook
                            export_title = user_hook
                            corrected_title = user_hook.strip()

                            if api_key and bool(st.session_state.get("cs_reground_on_export", False)):
                                from clip_engine.effective_config import resolve_models_from_effective_config
                                from clip_engine.effective_config import get_durable_effective_config
                                _durable = get_durable_effective_config(st.session_state)
                                _models = resolve_models_from_effective_config(_durable)
                                export_clip = ground_clip_metadata_against_window(
                                    export_clip,
                                    segs,
                                    api_key,
                                    tracker=tracker,
                                    force_regenerate=True,
                                    resolved_models=_models,
                                )
                                corrected_title = str(export_clip.get("hook_title", user_hook)).strip() or user_hook
                                export_clip["hook_title"] = corrected_title
                                export_clip["grounded_hook_title"] = corrected_title
                                export_clip["export_title"] = corrected_title
                                export_title = corrected_title
                                tid = export_clip.get("_wid", wid)
                                if tid and tid in tracker.per_clip:
                                    export_clip["_token_usage"] = tracker.per_clip[tid]

                            base = export_filename_stem(
                                export_clip, idx + 1, _slug(export_title)
                            )
                            out = session / f"{base}_9x16.mp4"

                            write_clip_audit_json(export_clip, session / f"clip_{idx+1:02d}_audit.json", index=idx + 1)
                            audit_clips.append(export_clip)

                            status_area.info(f"Exporting {idx+1}/{len(to_export)}: {export_title}...")
                            try:
                                result = export_vertical_clip_with_captions(
                                    video, out, t0, t1, segs,
                                    prefer_gpu=prefer_gpu,
                                    force_gpu_export=force_gpu,
                                    allow_cpu_fallback=allow_cpu,
                                    caption_preset=preset,
                                    export_mode=export_mode,
                                    write_sidecars=write_sidecars,
                                    advanced_captions=bool(st.session_state.get("cs_enable_advanced_captions", True)),
                                    dynamic_smart_crop=bool(st.session_state.get("cs_enable_dynamic_smart_crop", True)),
                                )
                                exported += 1
                                if api_key and corrected_title:
                                    st.session_state[f"hook_{wid}"] = corrected_title
                                    st.session_state.pop(f"grounded_hook_{wid}", None)
                                    for clip in st.session_state.get("cs_clips") or []:
                                        if str(clip.get("_wid", "")) == str(wid):
                                            clip["hook_title"] = corrected_title
                                            clip["grounded_hook_title"] = corrected_title
                                            break
                                logger.info(
                                    "[EXPORT] completed path=%s encoder=%s mode=%s",
                                    out,
                                    result.get("encoder_used"),
                                    result.get("export_mode"),
                                )
                            except Exception as e:
                                failed += 1
                                st.warning(f"Skipped **{base}**: {e}")
                                logger.exception("Export failed for %s", base)

                            prog.progress((idx + 1) / len(to_export))

                        status_area.empty()
                        prog.empty()

                        tracker.write_json(
                            session / "token_usage.json",
                            target_clips=int(st.session_state.get("cs_target_clips", 20)),
                            final_clip_count=exported,
                        )

                        logger.info(
                            "[lifecycle] Export batch finished — exported=%d failed=%d; app remains active",
                            exported,
                            failed,
                        )
                        if exported:
                            st.success(f"Exported **{exported}** clip(s) to `{session.relative_to(PROJECT_ROOT)}`")
                        if failed:
                            st.error(f"{failed} clip(s) failed - check warnings above.")

                except Exception as e:
                    logger.exception("Export batch failed")
                    category, user_msg = classify_exception(e)
                    st.session_state.cs_session_telemetry = get_session_telemetry().to_dict()
                    st.error(f"**{category}:** {user_msg}")

            out_dir = st.session_state.get("cs_session_dir")
            if out_dir and Path(out_dir).is_dir():
                st.markdown("**Latest export folder**")
                st.code(str(Path(out_dir).relative_to(PROJECT_ROOT)), language=None)
                if st.button("Open latest export in Explorer"):
                    open_folder(Path(out_dir))
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