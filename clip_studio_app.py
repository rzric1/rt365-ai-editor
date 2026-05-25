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
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

from config import (  # noqa: E402
    CLIP_STUDIO_MAX_UPLOAD_BYTES,
    CLIP_STUDIO_MAX_UPLOAD_MB,
    CLIP_STUDIO_OUTPUT_DIR,
    ENV_FFMPEG_BINARY,
    ENV_OPENAI_API_KEY,
    PROJECT_ROOT,
    ensure_directories,
)
from clip_engine.clip_analysis import get_session_tokens  # noqa: E402
from clip_engine.clip_pipeline import run_full_clip_pipeline, PipelineOpenAIConfig  # noqa: E402
from clip_engine.openai_resilience import OpenAIRateLimitError, estimate_pipeline_tokens, token_saver_pass_config  # noqa: E402
from clip_engine.analysis_cache import clear_all_analysis_cache  # noqa: E402
from config import get_openai_model, get_openai_model_fast, get_openai_model_json_fallback  # noqa: E402
from clip_engine.clip_style import CLIP_STYLE_OPTIONS, ClipStyle  # noqa: E402
from clip_engine.clip_metadata import (  # noqa: E402
    ground_clip_metadata_against_window,
    write_clip_audit_json,
)
from clip_engine.token_tracking import get_tracker  # noqa: E402
from clip_engine.export_vertical import export_vertical_clip_with_captions  # noqa: E402
from clip_engine.export_vertical import export_clip_preview  # noqa: E402
from clip_engine.media_probe import get_media_duration_seconds  # noqa: E402
from clip_engine.captions import CAPTION_PRESETS, CaptionPreset  # noqa: E402
from clip_engine.export_vertical import EXPORT_MODE_LABELS  # noqa: E402
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clip_studio")
ensure_ffmpeg_on_path(log=True)
log_nvenc_probe_command_explicit()
log_ai_acceleration_startup()

CAPTION_PRESET_OPTIONS: list[CaptionPreset] = [
    "Clean", "Bold Viral", "Podcast", "Minimal",
    "Viral", "Podcast Pro", "Documentary", "Gaming", "Cinematic",
]
PREVIEW_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "previews"


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
        "cs_min_gap_seconds": 60,
        "cs_similarity_threshold": 45,
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    note = creator_note.strip() or None

    token_saver = bool(st.session_state.get("cs_token_saver_mode", True))
    style_name = str(clip_style)
    n_passes, max_rounds, _ = token_saver_pass_config(style_name)
    if not token_saver:
        n_passes = 3
        max_rounds = 2
    est = estimate_pipeline_tokens(
        formatted,
        target_count=target_count,
        n_passes=n_passes,
        max_pass_rounds=max_rounds,
        token_saver_mode=token_saver,
    )
    st.session_state.cs_token_estimate = est.to_dict()

    openai_config = PipelineOpenAIConfig(
        token_saver_mode=token_saver,
        rate_limit_safe=bool(st.session_state.get("cs_rate_limit_safe", True)),
        use_cache=bool(st.session_state.get("cs_use_analysis_cache", True)),
        max_tokens_budget=int(st.session_state.get("cs_max_tokens_budget", 60_000)),
        call_delay_seconds=float(st.session_state.get("cs_openai_call_delay", 0.75)),
        status_callback=status_callback,
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
    )

    export_dict = tracker.to_export_dict(
        target_clips=target_count,
        final_clip_count=len(clips),
        model=get_openai_model(),
    )
    export_dict["model_fast"] = get_openai_model_fast()
    export_dict["model_quality"] = get_openai_model()
    export_dict["token_estimate"] = est.to_dict()
    st.session_state.cs_token_tracker = export_dict

    for c in clips:
        if not c.get("_wid"):
            c["_wid"] = uuid.uuid4().hex
        if "caption_preset" not in c:
            c["caption_preset"] = str(st.session_state.get("cs_default_caption_preset", "Clean"))
        tid = c.get("_wid", "")
        if tid and tid in tracker.per_clip:
            c["_token_usage"] = tracker.per_clip[tid]

    return clips, stats.to_dict()


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
    _init_state()
    ensure_ffmpeg_on_path(log=False)

    st.set_page_config(page_title="AI Clip Studio", layout="wide", initial_sidebar_state="expanded")
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
    ai_diag = collect_ai_acceleration_diagnostics()

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
                "- **listed** = ffmpeg advertises the encoder.\n"
                "- **probe** = a tiny real NVENC encode was attempted. Failure usually means GPU busy, old driver, or remote desktop session.\n"
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
            if st.button("Refresh CUDA / NVENC probes", use_container_width=True, key="cs_ai_diag_refresh"):
                invalidate_cuda_runtime_probe_cache()
                from clip_engine.ffmpeg_gpu import invalidate_nvenc_cache
                invalidate_nvenc_cache()
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
        st.number_input("Minimum clip length (core, s)", min_value=5, max_value=600, value=25, step=1, key="cs_min_clip_seconds")
        st.number_input("Maximum clip length (cap, s)", min_value=10, max_value=600, value=160, step=1, key="cs_max_clip_seconds")
        st.number_input("Context before clip (s)", min_value=0, max_value=120, value=8, step=1, key="cs_context_before")
        st.number_input("Context after clip (s)", min_value=0, max_value=120, value=12, step=1, key="cs_context_after")
        st.checkbox("Allow final clip to exceed max length", value=False, key="cs_allow_exceed_max")

        st.divider()
        st.subheader("OpenAI usage & safety")
        st.checkbox(
            "Token Saver Mode",
            key="cs_token_saver_mode",
            help="Fewer GPT passes, smaller candidate pool, skip strong grounding regen.",
        )
        st.checkbox(
            "Rate Limit Safe Mode",
            key="cs_rate_limit_safe",
            help="Exponential backoff on 429 errors with retry/resume.",
        )
        st.checkbox(
            "Use cached analysis if available",
            key="cs_use_analysis_cache",
        )
        st.number_input(
            "Max tokens per analysis run",
            min_value=10_000,
            max_value=500_000,
            step=5_000,
            key="cs_max_tokens_budget",
        )
        st.slider(
            "Delay between OpenAI calls (sec)",
            min_value=0.0,
            max_value=3.0,
            step=0.25,
            key="cs_openai_call_delay",
        )
        st.caption(f"Fast model: `{get_openai_model_fast()}`")
        st.caption(f"Quality model: `{get_openai_model()}`")
        st.caption(
            f"JSON fallback (if gpt-5* returns empty/invalid JSON): `{get_openai_model_json_fallback()}`"
        )
        if st.button("Clear analysis cache", use_container_width=True):
            n = clear_all_analysis_cache()
            st.success(f"Cleared {n} cached analysis entries.")

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
        st.slider("Duplicate similarity threshold (%)", min_value=20, max_value=80, step=5, key="cs_similarity_threshold",
            help="Clips more similar than this % are considered duplicates and removed.")

        # FFmpeg
        st.divider()
        st.caption("**FFmpeg**")
        if ffmpeg_path:
            st.code(ffmpeg_path, language=None)
            st.caption(ffmpeg_ver or "version: (unknown)")
        else:
            st.error(f"FFmpeg not found. Set **{ENV_FFMPEG_BINARY}** in `.env` or install FFmpeg.")

        if st.button("Open exports folder", use_container_width=True):
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
                    try:
                        st.session_state.cs_media_duration = get_media_duration_seconds(resolved)
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
                    try:
                        st.session_state.cs_media_duration = get_media_duration_seconds(path)
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

        if st.button("Clean duplicate uploads", use_container_width=True):
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
                    st.session_state.cs_status = (
                        f"Transcribed {len(segs)} raw segments -> "
                        f"{len(st.session_state.cs_segments)} sentence groups."
                    )
                except Exception as e:
                    logger.exception("Transcribe failed")
                    st.error(f"Transcription failed: {e}")

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
            _style = str(st.session_state.get("cs_clip_style", "Balanced"))
            _n, _r, _ = token_saver_pass_config(_style)
            if not st.session_state.get("cs_token_saver_mode", True):
                _n, _r = 3, 2
            _pre = estimate_pipeline_tokens(
                st.session_state.cs_formatted,
                target_count=int(st.session_state.get("cs_target_clips", 20)),
                n_passes=_n,
                max_pass_rounds=_r,
                token_saver_mode=bool(st.session_state.get("cs_token_saver_mode", True)),
            )
            _budget = int(st.session_state.get("cs_max_tokens_budget", 60_000))
            if _pre.estimated_total_tokens > _budget:
                st.warning(
                    f"This run may use approximately **{_pre.estimated_total_tokens:,}** tokens. "
                    f"Budget is **{_budget:,}**. Token Saver Mode will be enforced."
                )
            else:
                st.caption(
                    f"Estimated analysis tokens: ~{_pre.estimated_total_tokens:,} "
                    f"(budget {_budget:,}) | ~{_pre.estimated_calls} API calls"
                )

    # ------------------------------------------------------------------ CLIP AI
    st.subheader("3. AI clip suggestions")

    creator_note = st.text_area(
        "Optional creator note",
        placeholder="e.g. audience is beginners; brand is playful - leave blank for topic-agnostic",
        height=68,
    )

    col_analyze, col_rescore, col_more = st.columns([2, 1, 1])

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
            budget = int(st.session_state.get("cs_max_tokens_budget", 60_000))
            st.caption(
                f"Est. tokens: **~{est_data.get('estimated_total_tokens', 0):,}** "
                f"(budget {budget:,}) | ~{est_data.get('estimated_calls', 0)} API calls"
            )
        openai_status = st.session_state.get("cs_openai_status", "")
        if openai_status:
            st.info(openai_status)
        if st.button("Analyze for high-retention clips", type="primary", disabled=not can_analyze, use_container_width=True):
            status_slot = st.empty()

            def _status_cb(msg: str) -> None:
                st.session_state.cs_openai_status = msg
                status_slot.info(msg)

            with st.spinner("Scoring clips (multi-pass — may take 30-120s on long podcasts)"):
                try:
                    params = _get_analysis_params()
                    if st.session_state.cs_formatted:
                        style = str(st.session_state.get("cs_clip_style", "Balanced"))
                        n_passes, max_rounds, _ = token_saver_pass_config(style)
                        if not st.session_state.get("cs_token_saver_mode", True):
                            n_passes, max_rounds = 3, 2
                        pre_est = estimate_pipeline_tokens(
                            st.session_state.cs_formatted,
                            target_count=int(st.session_state.get("cs_target_clips", 20)),
                            n_passes=n_passes,
                            max_pass_rounds=max_rounds,
                            token_saver_mode=bool(st.session_state.get("cs_token_saver_mode", True)),
                        )
                        st.session_state.cs_token_estimate = pre_est.to_dict()
                    clips, pipe_stats = _run_clip_analysis(**params, status_callback=_status_cb)
                    st.session_state.cs_clips = clips
                    st.session_state.cs_pipeline_stats = pipe_stats
                    if pipe_stats.get("cache_hit"):
                        st.session_state.cs_status = "Loaded cached analysis — no OpenAI tokens used."
                    else:
                        st.session_state.cs_status = f"Suggested {len(clips)} clips."
                    st.session_state.cs_openai_status = ""
                except OpenAIRateLimitError as e:
                    logger.exception("Clip analysis rate limited")
                    st.error(
                        f"OpenAI rate limit at **{e.stage}** (model `{e.model}`). "
                        f"{e.mitigation} Partial progress saved — click Analyze again to resume."
                    )
                except Exception as e:
                    logger.exception("Clip analysis failed")
                    st.error(f"Clip analysis failed: {e}")
            if not api_key:
                st.info("Add **OPENAI_API_KEY** to `.env` to run clip analysis.")

    with col_rescore:
        if st.session_state.cs_clips:
            if st.button("Re-score clips", use_container_width=True, disabled=not can_analyze):
                with st.spinner("Re-scoring..."):
                    try:
                        clips, pipe_stats = _run_clip_analysis(**_get_analysis_params())
                        st.session_state.cs_clips = clips
                        st.session_state.cs_pipeline_stats = pipe_stats
                        st.session_state.cs_status = f"Re-scored: {len(clips)} clips."
                        st.rerun()
                    except Exception as e:
                        st.error(f"Re-score failed: {e}")

    with col_more:
        if st.session_state.cs_clips:
            if st.button("Find more clips", use_container_width=True, disabled=not can_analyze):
                with st.spinner("Finding more..."):
                    try:
                        extra, extra_stats = _run_clip_analysis(**_get_analysis_params())
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
        media_dur_for_map = float(st.session_state.get("cs_media_duration") or 0.0)
        pipe_stats = st.session_state.get("cs_pipeline_stats") or {}
        target_req = int(pipe_stats.get("target_clips", st.session_state.get("cs_target_clips", 20)))
        st.success(f"**{len(clips)} clips** selected (target: {target_req}) - check Export to include in batch.")

        if pipe_stats:
            st.caption(
                f"Pipeline: **{pipe_stats.get('raw_candidates', '?')}** raw candidates | "
                f"**{pipe_stats.get('removed_overlap', 0)}** removed (overlap) | "
                f"**{pipe_stats.get('removed_duplicates', 0)}** removed (duplicates) | "
                f"**{pipe_stats.get('final_clips', len(clips))}** final"
                + (
                    f" | models: `{pipe_stats.get('model_fast', 'n/a')}` / `{pipe_stats.get('model_quality', 'n/a')}`"
                )
                + (
                    f" | est. ~{pipe_stats.get('estimated_tokens', 0):,} tokens"
                    if pipe_stats.get("estimated_tokens")
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
            for w in pipe_stats.get("warnings", []):
                st.warning(w)

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
                    st.caption(f"Platform: **{platform_str}** | Signal: `{c.get('dominant_signal', 'n/a')}` | Style: `{c.get('caption_style', 'n/a')}`")
                with score_col:
                    color = "High" if score >= 80 else ("Mid" if score >= 65 else "Low")
                    st.metric(f"{color} Score", f"{score}/100")
                with export_col:
                    st.checkbox("Export", value=bool(c.get("export", True)), key=f"ex_{wid}")

                # Editable fields row
                e1, e2, e3, e4 = st.columns([2, 1, 1, 1])
                with e1:
                    st.text_input("Hook / title", value=str(c.get("hook_title", "") or ""), key=f"hook_{wid}")
                    grounded_hook = st.session_state.get(f"grounded_hook_{wid}")
                    user_hook = st.session_state.get(f"hook_{wid}", c.get("hook_title", ""))
                    if grounded_hook and str(grounded_hook).strip() != str(user_hook).strip():
                        st.caption(
                            f"Last export used grounded title: **{grounded_hook}** "
                            f"(widget value unchanged — edit manually if desired)"
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
                st.caption(f"Duration: **{dur_val:.1f}s** | AI core: {o_s:.1f}s - {o_e:.1f}s ({max(0.0, o_e-o_s):.1f}s)")

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

                to_export = [
                    c for c in clips
                    if st.session_state.get(f"ex_{c.get('_wid', '')}", c.get("export", True))
                ]
                if not to_export:
                    st.warning("No clips selected for export. Check at least one 'Export' checkbox.")
                else:
                    exported = 0
                    failed = 0
                    prog = st.progress(0.0)
                    status_area = st.empty()
                    tracker = get_tracker()
                    audit_clips: list[dict] = []

                    for idx, c in enumerate(to_export):
                        wid = c.get("_wid", str(idx))
                        user_hook = str(
                            st.session_state.get(f"hook_{wid}", c.get("hook_title", f"clip_{idx+1}"))
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

                        if api_key:
                            export_clip = ground_clip_metadata_against_window(
                                export_clip,
                                segs,
                                api_key,
                                tracker=tracker,
                                force_regenerate=True,
                            )
                            corrected_title = str(export_clip.get("hook_title", user_hook)).strip() or user_hook
                            export_clip["hook_title"] = corrected_title
                            export_clip["grounded_hook_title"] = corrected_title
                            export_clip["export_title"] = corrected_title
                            export_title = corrected_title
                            if corrected_title != user_hook.strip():
                                st.session_state[f"grounded_hook_{wid}"] = corrected_title
                            tid = export_clip.get("_wid", wid)
                            if tid and tid in tracker.per_clip:
                                export_clip["_token_usage"] = tracker.per_clip[tid]

                        base = f"{idx+1:02d}_{_slug(export_title)}"
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
                            logger.info(
                                "Exported %s - mode=%s encoder=%s res=%s",
                                out.name,
                                result.get("export_mode"),
                                result.get("encoder_used"),
                                result.get("resolution"),
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

                    if exported:
                        st.success(f"Exported **{exported}** clip(s) to `{session.relative_to(PROJECT_ROOT)}`")
                    if failed:
                        st.error(f"{failed} clip(s) failed - check warnings above.")

            except Exception as e:
                logger.exception("Export batch failed")
                st.error(f"Export failed: {e}")

    out_dir = st.session_state.get("cs_session_dir")
    if out_dir and Path(out_dir).is_dir():
        st.markdown("**Latest export folder**")
        st.code(str(Path(out_dir).relative_to(PROJECT_ROOT)), language=None)
        if st.button("Open latest export in Explorer"):
            open_folder(Path(out_dir))


if __name__ == "__main__":
    main()
