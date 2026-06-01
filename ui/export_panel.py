# -*- coding: utf-8 -*-
"""Export panel for RT365 AI Clip Studio."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import (
    CLIP_STUDIO_OUTPUT_DIR,
    ENV_OPENAI_API_KEY,
    PROJECT_ROOT,
)
from clip_engine.telemetry import classify_exception, get_session_telemetry
from clip_engine.token_tracking import get_tracker
from clip_engine.export_vertical import (
    EXPORT_MODE_LABELS,
    export_filename_stem,
    export_vertical_clip_with_captions,
)
from clip_engine.clip_metadata import ground_clip_metadata_against_window, write_clip_audit_json
from clip_engine.captions import CaptionPreset
from ui_helpers import open_folder

logger = logging.getLogger("clip_studio")

CAPTION_PRESET_OPTIONS: list[CaptionPreset] = [
    "Clean", "Bold Viral", "Podcast", "Minimal",
    "Viral", "Podcast Pro", "Documentary", "Gaming", "Cinematic",
]


def _slug(s: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "_", s.strip())[:max_len]
    return s or "clip"


def _expand_export_selection(clips: list, selected: list[dict]) -> list[dict]:
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


def render_export_panel() -> None:
    """Render the Export vertical (9:16) + captions section."""
    api_key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
    clips: list = st.session_state.get("cs_clips") or []

    st.subheader("4. Export vertical (9:16) + captions")
    st.caption(
        "Each MP4 uses the **edited** start/end times from the clip cards above. "
        "SRT and ASS sidecar files are written alongside each clip when enabled."
    )

    if not st.session_state.cs_video_path:
        st.info("Save a video upload first.")
        return
    if not clips:
        st.info("Run clip analysis to get exportable segments.")
        return

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
                    if st.session_state.get(f"ex_{c.get('_wid', '')}", c.get("export", True))
                ],
            )
            if not to_export:
                st.warning("No clips selected for export. Check at least one 'Export' checkbox.")
            else:
                from clip_engine.clip_finalizer import ensure_clips_finalized, validate_clip_for_export

                export_min = float(st.session_state.get("cs_min_clip_seconds", 25))
                export_max = min(float(st.session_state.get("cs_max_clip_seconds", 120)), 120.0)
                to_export, _finalized_on_export = ensure_clips_finalized(
                    to_export, segs, min_duration=export_min, max_duration=export_max,
                )
                if _finalized_on_export:
                    logger.info("[CLIP FINALIZER] export guard finalized %d clip(s) before export", len(to_export))

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
                            st.session_state.get(f"hook_{wid}", c.get("hook_title", f"clip_{idx+1}")),
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
                        logger.warning("[EXPORT] skipped clip=%s reason=%s", wid, skip_reason)
                        prog.progress((idx + 1) / len(to_export))
                        continue

                    user_hook = str(
                        st.session_state.get(
                            f"hook_widget_{wid}",
                            st.session_state.get(f"hook_{wid}", c.get("hook_title", f"clip_{idx+1}")),
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
                        from clip_engine.effective_config import (
                            resolve_models_from_effective_config,
                            get_durable_effective_config,
                        )
                        _durable = get_durable_effective_config(st.session_state)
                        _models = resolve_models_from_effective_config(_durable)
                        export_clip = ground_clip_metadata_against_window(
                            export_clip, segs, api_key, tracker=tracker,
                            force_regenerate=True, resolved_models=_models,
                        )
                        corrected_title = str(export_clip.get("hook_title", user_hook)).strip() or user_hook
                        export_clip["hook_title"] = corrected_title
                        export_clip["grounded_hook_title"] = corrected_title
                        export_clip["export_title"] = corrected_title
                        export_title = corrected_title
                        tid = export_clip.get("_wid", wid)
                        if tid and tid in tracker.per_clip:
                            export_clip["_token_usage"] = tracker.per_clip[tid]

                    base = export_filename_stem(export_clip, idx + 1, _slug(export_title))
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
                            out, result.get("encoder_used"), result.get("export_mode"),
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
                    exported, failed,
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
