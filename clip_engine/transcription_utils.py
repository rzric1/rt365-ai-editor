"""
clip_engine/transcription_utils.py
Improved transcript segmentation utilities:
- Merge tiny segments into sentence groups
- Detect pauses
- Snap clip boundaries to sentence starts/ends
- Word-level timing preservation helpers
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import NamedTuple

logger = logging.getLogger("clip_engine.transcription_utils")

# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------

SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


def merge_segments_into_sentences(
    segments: list[dict],
    *,
    max_gap_seconds: float = 1.2,
    min_sentence_words: int = 4,
    max_sentence_seconds: float = 15.0,
) -> list[dict]:
    """
    Merge short whisper segments into natural sentence groups.

    Rules:
    - Merge if gap between segments < max_gap_seconds AND merged text < max_sentence_seconds
    - Force-split on sentence-ending punctuation (. ! ?)
    - Preserve word_timestamps if present in original segments.

    Returns list of merged segment dicts with keys:
      start, end, text, word_count, has_sentence_end, source_segments
    """
    if not segments:
        return []

    merged: list[dict] = []
    current: dict | None = None

    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue

        if current is None:
            current = _new_merged(seg_start, seg_end, seg_text, seg)
            continue

        gap = seg_start - current["end"]
        merged_dur = seg_end - current["start"]
        would_end_sentence = bool(SENTENCE_END_RE.search(current["text"]))

        if would_end_sentence or gap >= max_gap_seconds or merged_dur > max_sentence_seconds:
            merged.append(current)
            current = _new_merged(seg_start, seg_end, seg_text, seg)
        else:
            current["end"] = seg_end
            current["text"] = current["text"].rstrip() + " " + seg_text
            current["word_count"] += len(seg_text.split())
            current["source_segments"].append(seg)

    if current:
        merged.append(current)

    # Remove very short merged segments (filler words)
    result = [m for m in merged if m["word_count"] >= min_sentence_words]
    logger.debug("merge_segments: %d raw -> %d merged sentence groups", len(segments), len(result))
    return result


def _new_merged(start: float, end: float, text: str, source_seg: dict) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "word_count": len(text.split()),
        "has_sentence_end": bool(SENTENCE_END_RE.search(text)),
        "source_segments": [source_seg],
    }


# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------

@dataclass
class PauseInfo:
    before_seconds: float   # timestamp of silence start
    after_seconds: float    # timestamp of silence end
    duration: float


def detect_pauses(
    segments: list[dict],
    min_pause_seconds: float = 0.8,
) -> list[PauseInfo]:
    """Return list of pauses (gaps between segments) longer than min_pause_seconds."""
    pauses: list[PauseInfo] = []
    for i in range(1, len(segments)):
        gap_start = float(segments[i - 1].get("end", 0))
        gap_end = float(segments[i].get("start", 0))
        dur = gap_end - gap_start
        if dur >= min_pause_seconds:
            pauses.append(PauseInfo(gap_start, gap_end, dur))
    return pauses


def find_nearest_pause_boundary(
    t: float,
    pauses: list[PauseInfo],
    direction: str = "nearest",   # "before" | "after" | "nearest"
    tolerance: float = 3.0,
) -> float | None:
    """Find a pause boundary near time t. Returns the boundary time or None."""
    candidates: list[tuple[float, float]] = []  # (distance, time)
    for p in pauses:
        if direction in ("before", "nearest") and abs(p.before_seconds - t) <= tolerance:
            candidates.append((abs(p.before_seconds - t), p.before_seconds))
        if direction in ("after", "nearest") and abs(p.after_seconds - t) <= tolerance:
            candidates.append((abs(p.after_seconds - t), p.after_seconds))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def segments_to_prompt_transcript(segments: list[dict]) -> str:
    """
    Convert segments to a timestamped text block for LLM prompts.
    Format: [HH:MM:SS] text
    If a segment has a "speaker" key: [HH:MM:SS] [Speaker]: text
    """
    lines: list[str] = []
    for seg in segments:
        t = float(seg.get("start", 0))
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = str(seg.get("speaker", "")).strip()
        if speaker:
            lines.append(f"[{h:02d}:{m:02d}:{s:02d}] [{speaker}]: {text}")
        else:
            lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {text}")
    return "\n".join(lines)


def segments_to_srt(segments: list[dict], offset: float = 0.0) -> str:
    """Convert segments to SRT subtitle format, optionally with a time offset."""
    lines: list[str] = []
    idx = 1
    for seg in segments:
        t0 = float(seg.get("start", 0)) - offset
        t1 = float(seg.get("end", 0)) - offset
        if t1 <= 0 or t0 < 0:
            continue
        t0 = max(0.0, t0)
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_srt_ts(t0)} --> {_srt_ts(t1)}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def _srt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_transcript_window(segments: list[dict], start: float, end: float) -> str:
    """Plain text spoken between start and end (inclusive overlap)."""
    parts: list[str] = []
    for seg in segments:
        s0 = float(seg.get("start", 0))
        s1 = float(seg.get("end", 0))
        if s1 <= start or s0 >= end:
            continue
        text = str(seg.get("text", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def extract_transcript_excerpt(
    segments: list[dict],
    start: float,
    end: float,
    *,
    max_chars: int = 900,
    max_lines: int = 10,
) -> str:
    """Timestamped excerpt for UI display (5-10 lines, ~600-1000 chars)."""
    lines: list[str] = []
    for seg in segments:
        s0 = float(seg.get("start", 0))
        s1 = float(seg.get("end", 0))
        if s1 <= start or s0 >= end:
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        h = int(s0 // 3600)
        m = int((s0 % 3600) // 60)
        s = int(s0 % 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {text}")
        if len(lines) >= max_lines:
            break
    excerpt = "\n".join(lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3] + "..."
    return excerpt


# ---------------------------------------------------------------------------
# Clip boundary snap (sentence-level)
# ---------------------------------------------------------------------------

def snap_clip_to_sentences(
    start: float,
    end: float,
    segments: list[dict],
    snap_tolerance: float = 2.5,
) -> tuple[float, float]:
    """
    Snap start/end times to the nearest sentence boundary.
    Returns (snapped_start, snapped_end).
    """
    if not segments:
        return start, end

    seg_starts = [float(s.get("start", 0)) for s in segments]
    seg_ends = [float(s.get("end", 0)) for s in segments]

    def nearest(candidates: list[float], target: float) -> float:
        best = min(candidates, key=lambda x: abs(x - target))
        return best if abs(best - target) <= snap_tolerance else target

    new_start = nearest(seg_starts, start)
    new_end = nearest(seg_ends, end)
    if new_end > new_start:
        return new_start, new_end
    return start, end
