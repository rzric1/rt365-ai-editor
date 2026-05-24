"""
clip_engine/clip_expand.py
Clip expansion/finalization with pause-aware boundary adjustment.
Expands AI core windows by context seconds, snaps to sentence/pause boundaries,
enforces min/max duration limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("clip_engine.clip_expand")


@dataclass
class ClipExpansionSettings:
    context_before: float = 8.0
    context_after: float = 12.0
    min_clip_seconds: float = 25.0
    max_clip_seconds: float = 160.0
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

    # Build pause list for boundary snapping
    from clip_engine.transcription_utils import detect_pauses, find_nearest_pause_boundary
    pauses = detect_pauses(segments) if segments else []

    finalized: list[dict] = []
    for c in clips:
        c = dict(c)
        core_start = float(c.get("start_seconds", c.get("start", 0)))
        core_end = float(c.get("end_seconds", c.get("end", 0)))

        # Store original AI core
        c["original_start"] = core_start
        c["original_end"] = core_end

        # Expand by context
        exp_start = max(0.0, core_start - settings.context_before)
        exp_end = min(media_duration, core_end + settings.context_after) if media_duration > 0 else core_end + settings.context_after

        # Try to snap start to a nearby pause boundary (cleaner cut point)
        snapped_start = find_nearest_pause_boundary(
            exp_start, pauses, direction="before", tolerance=settings.pause_snap_tolerance
        )
        if snapped_start is not None and snapped_start < core_start:
            exp_start = snapped_start
            c.setdefault("expansion_note", "")
            c["expansion_note"] = (c.get("expansion_note") or "") + f" Start snapped to pause at {snapped_start:.1f}s."

        snapped_end = find_nearest_pause_boundary(
            exp_end, pauses, direction="after", tolerance=settings.pause_snap_tolerance
        )
        if snapped_end is not None and snapped_end > core_end:
            exp_end = snapped_end
            c["expansion_note"] = (c.get("expansion_note") or "") + f" End snapped to pause at {snapped_end:.1f}s."

        # Enforce max length (hard cap always wins unless allow_exceed_max)
        duration = exp_end - exp_start
        effective_max = settings.max_clip_seconds
        if settings.hard_max_seconds is not None:
            effective_max = min(effective_max, settings.hard_max_seconds)
        if duration > effective_max and not settings.allow_exceed_max:
            exp_end = exp_start + effective_max
            c["expansion_note"] = (c.get("expansion_note") or "") + " Trimmed to max length."
        elif settings.hard_max_seconds is not None and duration > settings.hard_max_seconds:
            exp_end = exp_start + settings.hard_max_seconds
            c["expansion_note"] = (c.get("expansion_note") or "") + " Hard-capped to style max."

        # Enforce min length
        duration = exp_end - exp_start
        if duration < settings.min_clip_seconds:
            # Extend end
            needed = settings.min_clip_seconds - duration
            exp_end = min(media_duration or exp_end + needed, exp_end + needed)
            c["expansion_note"] = (c.get("expansion_note") or "") + " Extended to meet min length."

        c["start_seconds"] = round(exp_start, 3)
        c["end_seconds"] = round(exp_end, 3)
        c["duration"] = round(exp_end - exp_start, 3)

        logger.debug(
            "Clip expanded: core=%.1f-%.1f -> export=%.1f-%.1f (%.1fs)",
            core_start, core_end, exp_start, exp_end, c["duration"],
        )
        finalized.append(c)

    return finalized
