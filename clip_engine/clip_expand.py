"""
clip_engine/clip_expand.py
Clip expansion/finalization with pause-aware boundary adjustment.
Expands AI core windows by context seconds, snaps to sentence/pause boundaries,
enforces min/max duration limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from clip_engine.clip_duration_governor import (
    HARD_CAP_SECONDS,
    SOFT_CAP_SECONDS,
    clamp_clip_to_duration_policy,
    ensure_expansion_baseline,
    refresh_expansion_diagnostics,
    scaled_context_padding,
)

logger = logging.getLogger("clip_engine.clip_expand")


@dataclass
class ClipExpansionSettings:
    context_before: float = 8.0
    context_after: float = 12.0
    min_clip_seconds: float = 25.0
    max_clip_seconds: float = 90.0
    hard_max_seconds: float | None = None
    allow_exceed_max: bool = False
    pause_snap_tolerance: float = 2.0   # seconds to look for a pause boundary


def finalize_clips_after_ai(
    clips: list[dict],
    media_duration: float,
    segments: list[dict],
    settings: ClipExpansionSettings | None = None,
) -> list[dict]:
    """
    Expand each clip's AI-core window by context seconds, snap to boundaries,
    clamp to media duration and max length.
    """
    if settings is None:
        settings = ClipExpansionSettings()

    from clip_engine.transcription_utils import detect_pauses, find_nearest_pause_boundary
    pauses = detect_pauses(segments) if segments else []

    effective_max = min(
        settings.max_clip_seconds,
        settings.hard_max_seconds or HARD_CAP_SECONDS,
        SOFT_CAP_SECONDS,
    )
    if settings.allow_exceed_max:
        effective_max = min(
            settings.max_clip_seconds,
            settings.hard_max_seconds or HARD_CAP_SECONDS,
        )

    finalized: list[dict] = []
    for raw in clips:
        c = ensure_expansion_baseline(dict(raw))
        core_start = float(c["original_start"])
        core_end = float(c["original_end"])
        core_dur = max(0.01, core_end - core_start)

        ctx_b, ctx_a = scaled_context_padding(
            core_dur, settings.context_before, settings.context_after,
        )

        exp_start = max(0.0, core_start - ctx_b)
        exp_end = (
            min(media_duration, core_end + ctx_a)
            if media_duration > 0
            else core_end + ctx_a
        )

        snapped_start = find_nearest_pause_boundary(
            exp_start, pauses, direction="before", tolerance=settings.pause_snap_tolerance,
        )
        if snapped_start is not None and snapped_start < core_start:
            exp_start = snapped_start
            c.setdefault("expansion_note", "")
            c["expansion_note"] = (c.get("expansion_note") or "") + f" Start snapped to pause at {snapped_start:.1f}s."

        snapped_end = find_nearest_pause_boundary(
            exp_end, pauses, direction="after", tolerance=settings.pause_snap_tolerance,
        )
        if snapped_end is not None and snapped_end > core_end:
            exp_end = snapped_end
            c["expansion_note"] = (c.get("expansion_note") or "") + f" End snapped to pause at {snapped_end:.1f}s."

        duration = exp_end - exp_start
        if duration > effective_max and not settings.allow_exceed_max:
            exp_end = exp_start + effective_max
            c["expansion_note"] = (c.get("expansion_note") or "") + f" Trimmed to {effective_max:.0f}s max."

        duration = exp_end - exp_start
        if duration < settings.min_clip_seconds:
            needed = settings.min_clip_seconds - duration
            exp_end = min(media_duration or exp_end + needed, exp_end + needed)
            if exp_end - exp_start > effective_max and not settings.allow_exceed_max:
                exp_end = exp_start + effective_max
            c["expansion_note"] = (c.get("expansion_note") or "") + " Extended toward min length."

        c["start_seconds"] = round(exp_start, 3)
        c["end_seconds"] = round(exp_end, 3)
        c, _ = clamp_clip_to_duration_policy(c, media_duration, pre_virality=True)
        c = refresh_expansion_diagnostics(c)

        logger.debug(
            "Clip expanded: core=%.1f-%.1f -> export=%.1f-%.1f (%.1fs, growth=%.1fs)",
            core_start,
            core_end,
            c["expanded_start"],
            c["expanded_end"],
            c["duration"],
            c.get("growth_seconds", 0),
        )
        finalized.append(c)

    return finalized
