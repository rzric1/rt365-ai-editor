# -*- coding: utf-8 -*-
"""
clip_engine/clip_split_parts.py
Split long narrative clips into Part 1 / Part 2 for export.
"""

from __future__ import annotations

import logging
import re
import uuid

from clip_engine.transcription_utils import merge_segments_into_sentences

logger = logging.getLogger("clip_engine.clip_split_parts")

MIN_PART_SECONDS = 20.0
# Target part length band (used when evaluating viable split windows).
TARGET_PART_MIN = 45.0
TARGET_PART_MAX = 60.0

NARRATIVE_CONTINUATION_RE = re.compile(
    r"\b(continues|part of|story|journey|explains|then|after that)\b",
    re.IGNORECASE,
)
_PART_SUFFIX_RE = re.compile(r"\s*\(Part\s+\d+\)\s*$", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


def flag_split_recommended(clip: dict, max_clip_length: float) -> None:
    """
    Mark clip for two-part split when duration exceeds max and score >= 75
    (or narrative continuation language in reasoning).
    Clips with score < 75 are not flagged (truncated elsewhere).
    """
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    dur = t1 - t0
    if dur <= max_clip_length:
        return

    score = int(clip.get("composite_score", 0) or 0)
    if score < 75:
        return

    text = " ".join(
        str(clip.get(field, ""))
        for field in ("selection_reason", "ai_context_reason", "hook_title", "context_reason")
    )
    reason = "high_score_long"
    if NARRATIVE_CONTINUATION_RE.search(text):
        reason = "narrative_continuation"

    clip["split_recommended"] = True
    clip["split_reason"] = reason


def _base_title(title: str) -> str:
    return _PART_SUFFIX_RE.sub("", str(title or "").strip()).strip() or "Clip"


def _clip_export_range(clip: dict) -> tuple[float, float]:
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    return t0, t1


def _clip_core_range(clip: dict) -> tuple[float, float]:
    t0, t1 = _clip_export_range(clip)
    core_start = float(clip.get("original_start", t0))
    core_end = float(clip.get("original_end", t1))
    if core_end <= core_start:
        core_start, core_end = t0, t1
    return core_start, core_end


def _segments_in_range(segments: list[dict], t0: float, t1: float) -> list[dict]:
    return [
        s
        for s in segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]


def _sentence_boundary_times(segments: list[dict], t0: float, t1: float) -> list[float]:
    window = _segments_in_range(segments, t0, t1)
    if not window:
        return []
    sentences = merge_segments_into_sentences(window)
    times: list[float] = []
    for sent in sentences:
        times.append(float(sent.get("start", 0)))
        if sent.get("has_sentence_end"):
            times.append(float(sent.get("end", 0)))
    return sorted({round(t, 3) for t in times if t0 < t < t1})


def _split_time_from_formatted_transcript(
    formatted_transcript: str,
    target: float,
    t0: float,
    t1: float,
) -> float | None:
    """Parse [HH:MM:SS] lines and pick boundary nearest target within range."""
    if not formatted_transcript.strip():
        return None

    def _parse_ts(line: str) -> float | None:
        m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]", line.strip())
        if not m:
            return None
        h, mi, s = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return float(h * 3600 + mi * 60 + s)

    boundaries: list[float] = []
    for line in formatted_transcript.splitlines():
        ts = _parse_ts(line)
        if ts is None:
            continue
        if t0 < ts < t1:
            boundaries.append(ts)
        if SENTENCE_END_RE.search(line):
            boundaries.append(ts)

    if not boundaries:
        return None
    return min(boundaries, key=lambda t: abs(t - target))


def find_series_split_time(
    clip: dict,
    segments: list[dict] | None,
    formatted_transcript: str,
) -> float:
    """Nearest sentence boundary to AI core midpoint, else export midpoint."""
    export_start, export_end = _clip_export_range(clip)
    core_start, core_end = _clip_core_range(clip)
    core_mid = (core_start + core_end) / 2.0

    split = None
    if segments:
        boundaries = _sentence_boundary_times(segments, export_start, export_end)
        if boundaries:
            split = min(boundaries, key=lambda t: abs(t - core_mid))

    if split is None and formatted_transcript:
        split = _split_time_from_formatted_transcript(
            formatted_transcript, core_mid, export_start, export_end
        )

    if split is None:
        split = (export_start + export_end) / 2.0

    split = max(export_start + MIN_PART_SECONDS, min(export_end - MIN_PART_SECONDS, split))
    return round(split, 3)


def split_clip_into_parts(
    clip: dict,
    formatted_transcript: str,
    *,
    segments: list[dict] | None = None,
) -> list[dict]:
    """
    Split one long clip into two export parts at a sentence boundary near
    the AI core midpoint. Returns [clip] unchanged if split is not viable.
    """
    export_start, export_end = _clip_export_range(clip)
    duration = export_end - export_start
    if clip.get("is_part_of_series") or not clip.get("split_recommended"):
        return [clip]
    if duration < MIN_PART_SECONDS * 2:
        return [clip]

    split = find_series_split_time(clip, segments, formatted_transcript)
    part1_end = split
    part2_start = split

    dur1 = part1_end - export_start
    dur2 = export_end - part2_start
    if dur1 < MIN_PART_SECONDS or dur2 < MIN_PART_SECONDS:
        note = (
            f"Series split skipped: parts would be {dur1:.0f}s / {dur2:.0f}s "
            f"(minimum {MIN_PART_SECONDS:.0f}s each)."
        )
        out = dict(clip)
        out.setdefault("warnings", [])
        if note not in out["warnings"]:
            out["warnings"].append(note)
        out["split_recommended"] = False
        return [out]

    series_id = str(clip.get("clip_id") or clip.get("_wid") or uuid.uuid4().hex)
    base = _base_title(str(clip.get("hook_title", "Clip")))

    def _make_part(part_num: int, p_start: float, p_end: float) -> dict:
        part = dict(clip)
        part["start_seconds"] = round(p_start, 3)
        part["end_seconds"] = round(p_end, 3)
        part["original_start"] = round(p_start, 3)
        part["original_end"] = round(p_end, 3)
        part["hook_title"] = f"{base} (Part {part_num})"
        part["is_part_of_series"] = True
        part["series_id"] = series_id
        part["part_number"] = part_num
        part["part_total"] = 2
        part["clip_id"] = series_id
        part["split_recommended"] = False
        part["split_from_parent"] = True
        part.pop("_wid", None)
        part.setdefault("warnings", [])
        note = f"Split from parent window {export_start:.0f}s–{export_end:.0f}s."
        if note not in part["warnings"]:
            part["warnings"].append(note)
        return part

    part1 = _make_part(1, export_start, part1_end)
    part2 = _make_part(2, part2_start, export_end)

    logger.info(
        '[CLIP SPLIT] clip="%s" duration=%.0fs → Part 1: %.0fs-%.0fs | Part 2: %.0fs-%.0fs',
        base[:60],
        duration,
        export_start,
        part1_end,
        part2_start,
        export_end,
    )
    return [part1, part2]


def apply_recommended_series_splits(
    clips: list[dict],
    formatted_transcript: str,
    *,
    segments: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Replace split_recommended clips with two-part series; return (clips, split_count)."""
    out: list[dict] = []
    split_count = 0
    for clip in clips:
        if clip.get("split_recommended") and not clip.get("split_from_parent"):
            parts = split_clip_into_parts(
                clip, formatted_transcript, segments=segments
            )
            if len(parts) == 2:
                out.extend(parts)
                split_count += 1
            else:
                out.append(parts[0] if parts else clip)
        else:
            out.append(clip)
    return out, split_count


def apply_series_hook_filter(clips: list[dict], kept: list[dict]) -> list[dict]:
    """Keep or drop all parts of a series together after hook filtering."""
    kept_ids = {id(c) for c in kept}
    by_series: dict[str, list[dict]] = {}
    for c in clips:
        if c.get("is_part_of_series") and c.get("series_id"):
            by_series.setdefault(str(c["series_id"]), []).append(c)

    adjusted = list(kept)
    adjusted_ids = {id(c) for c in adjusted}
    for sid, members in by_series.items():
        member_ids = {id(m) for m in members}
        any_kept = bool(member_ids & kept_ids)
        all_kept = member_ids <= kept_ids
        if any_kept and not all_kept:
            for m in members:
                if id(m) not in adjusted_ids:
                    adjusted.append(m)
                    adjusted_ids.add(id(m))
        elif not any_kept and len(member_ids & kept_ids) == 0:
            continue
    return adjusted
