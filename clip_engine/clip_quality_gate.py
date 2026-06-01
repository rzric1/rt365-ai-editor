# -*- coding: utf-8 -*-
"""
clip_engine/clip_quality_gate.py
Validate and repair clips before they reach the UI; never crash the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from clip_engine.clip_boundaries import ends_with_dangling_word, hook_title_is_incomplete
from clip_engine.clip_scoring import assess_hook_quality, repair_hook_title_local
from clip_engine.transcription_utils import extract_transcript_window

logger = logging.getLogger("clip_engine.clip_quality_gate")


@dataclass
class QualityGateStats:
    input_count: int = 0
    passed: int = 0
    repaired: int = 0
    warnings: int = 0
    dropped: int = 0
    issues: list[str] = field(default_factory=list)


def _validate_times(clip: dict, media_duration: float) -> str | None:
    try:
        t0 = float(clip.get("start_seconds", clip.get("start", -1)))
        t1 = float(clip.get("end_seconds", clip.get("end", -1)))
    except (TypeError, ValueError):
        return "invalid_times"
    if t0 < 0 or t1 <= t0:
        return "invalid_times"
    if media_duration > 0 and t0 >= media_duration:
        return "invalid_times"
    return None


def _repair_clip_once(
    clip: dict,
    segments: list[dict],
    *,
    max_duration: float,
    min_duration: float,
    media_duration: float,
) -> dict:
    c = dict(clip)
    issue = _validate_times(c, media_duration)
    if issue:
        return c

    t0 = float(c.get("start_seconds", 0))
    t1 = float(c.get("end_seconds", t0))
    dur = t1 - t0

    if dur > max_duration + 0.5:
        c["end_seconds"] = round(t0 + max_duration, 3)
        c.setdefault("warnings", []).append(
            f"Duration trimmed to {max_duration:.0f}s cap."
        )
        c["quality_repaired"] = True

    title = str(c.get("hook_title", "")).strip()
    if not title:
        window = extract_transcript_window(segments, t0, t1)
        c["hook_title"] = repair_hook_title_local("", window)
        c.setdefault("warnings", []).append("Generated title from transcript.")
        c["quality_repaired"] = True
    elif hook_title_is_incomplete(title) or ends_with_dangling_word(title):
        window = extract_transcript_window(segments, t0, t1)
        repaired_title = repair_hook_title_local(title, window)
        if repaired_title != title:
            c["hook_title_before_repair"] = title
            c["hook_title"] = repaired_title
            c["hook_title_repaired"] = True
            c.setdefault("warnings", []).append("Repaired incomplete hook title.")
            c["quality_repaired"] = True

    hook_score, hook_warn = assess_hook_quality(str(c.get("hook_title", "")))
    c["hook_quality_score"] = hook_score
    if hook_warn:
        c["hook_warning"] = hook_warn

    if not c.get("virality_score"):
        c["virality_score"] = int(c.get("composite_score", 50))
    if not c.get("virality_breakdown"):
        c["virality_breakdown"] = {"composite_fallback": int(c.get("composite_score", 50))}

    window = extract_transcript_window(segments, t0, float(c.get("end_seconds", t1)))
    if not window.strip():
        c.setdefault("warnings", []).append("Empty transcript in clip window.")
        c["quality_warning"] = "empty_transcript"
    elif ends_with_dangling_word(window):
        c.setdefault("warnings", []).append("Transcript may end mid-thought.")

    t0 = float(c.get("start_seconds", 0))
    t1 = float(c.get("end_seconds", t0))
    if t1 - t0 < min_duration:
        c["quality_warning"] = "below_min_duration"

    return c


def run_quality_gate(
    clips: list[dict],
    segments: list[dict],
    *,
    media_duration: float,
    max_duration: float,
    min_duration: float,
) -> tuple[list[dict], QualityGateStats]:
    """
    Validate clips; repair once when possible; keep usable clips with warnings.
    """
    stats = QualityGateStats(input_count=len(clips))
    out: list[dict] = []

    for clip in clips:
        try:
            issue = _validate_times(clip, media_duration)
            if issue == "invalid_times":
                stats.dropped += 1
                stats.issues.append(f"dropped invalid times: {clip.get('hook_title', '')[:40]}")
                continue

            repaired = _repair_clip_once(
                clip,
                segments,
                max_duration=max_duration,
                min_duration=min_duration,
                media_duration=media_duration,
            )
            if repaired.get("quality_repaired"):
                stats.repaired += 1

            dur = float(repaired.get("end_seconds", 0)) - float(repaired.get("start_seconds", 0))
            if dur > max_duration + 2.0:
                stats.dropped += 1
                stats.issues.append("dropped: exceeds hard max duration")
                continue

            window = extract_transcript_window(
                segments,
                float(repaired.get("start_seconds", 0)),
                float(repaired.get("end_seconds", 0)),
            )
            if not window.strip() and not repaired.get("grounded_transcript_excerpt"):
                stats.dropped += 1
                stats.issues.append("dropped: empty transcript")
                continue

            if repaired.get("warnings") or repaired.get("quality_warning"):
                stats.warnings += 1

            out.append(repaired)
            stats.passed += 1
        except Exception as exc:
            logger.warning("Quality gate error for clip: %s", exc)
            stats.warnings += 1
            nc = dict(clip)
            nc.setdefault("warnings", []).append(f"Quality gate error: {exc}")
            out.append(nc)
            stats.passed += 1

    return out, stats


__all__ = ["QualityGateStats", "run_quality_gate"]
