# -*- coding: utf-8 -*-
"""
Load podcast / reaction transcripts for AI analysis.

Supported formats:
  - SubRip (.srt)
  - Bracket timecode (.srt or .txt): blocks starting with
      [HH:MM:SS:FF - HH:MM:SS:FF]
    The FF field is converted using TRANSCRIPT_BRACKET_FPS (default 24).
  - JSON (.json) with a simple "segments" list (see README)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from config import get_transcript_bracket_fps

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """One timed line of speech (or caption)."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class TranscriptDocument:
    """Full transcript plus segments — used for chunking and prompts."""

    source_path: Path
    segments: List[TranscriptSegment] = field(default_factory=list)

    def plain_text_with_timestamps(self) -> str:
        """
        Flatten segments into a single string with [HH:MM:SS] prefixes.

        The model uses these timestamps to emit marker times in seconds.
        """
        lines: List[str] = []
        for seg in self.segments:
            ts = _format_hhmmss(seg.start_seconds)
            lines.append(f"[{ts}] {seg.text.strip()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SubRip (.srt)
# ---------------------------------------------------------------------------

_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _srt_timestamp_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt_raw(raw: str) -> List[TranscriptSegment]:
    """Parse SRT content into segments (no file I/O)."""
    blocks = re.split(r"\n\s*\n", raw.strip())
    segments: List[TranscriptSegment] = []
    idx = 0
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        time_line_idx = 0
        if _SRT_TIME_RE.search(lines[0]) is None and len(lines) >= 2:
            time_line_idx = 1
        m = _SRT_TIME_RE.search(lines[time_line_idx])
        if not m:
            continue
        start = _srt_timestamp_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _srt_timestamp_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        text_lines = lines[time_line_idx + 1 :]
        text = " ".join(text_lines)
        idx += 1
        segments.append(
            TranscriptSegment(index=idx, start_seconds=start, end_seconds=end, text=text)
        )
    return segments


def load_srt(path: Path) -> TranscriptDocument:
    """Parse a basic SRT file into segments."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return TranscriptDocument(source_path=path, segments=_parse_srt_raw(raw))


# ---------------------------------------------------------------------------
# Bracket timecode: [HH:MM:SS:FF - HH:MM:SS:FF]
# ---------------------------------------------------------------------------

# Optional text after the closing ] on the same line is included in the segment body.
_BRACKET_HEADER_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$"
)


def _hhmmssff_to_seconds(h: str, m: str, s: str, f: str, fps: float) -> float:
    """Convert HH, MM, SS, and frame index FF to seconds using the given FPS."""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(f) / fps


def _looks_like_bracket_timestamp_transcript(raw: str) -> bool:
    """
    True if the first non-empty line looks like a Resolve-style bracket range.

    Example: [00:00:49:14 - 00:01:10:04]
    """
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return bool(_BRACKET_HEADER_RE.match(stripped))
    return False


def parse_bracket_timestamp_transcript(
    raw: str,
    *,
    fps: float,
    source_path: Path,
) -> TranscriptDocument:
    """
    Parse transcript text into segments. Each block:

        [HH:MM:SS:FF - HH:MM:SS:FF]
        Optional dialogue lines...

    Start/end seconds use the FF field as frames at `fps` (e.g. 49 + 14/24 ≈ 49.58 s).
    """
    lines = raw.splitlines()
    segments: List[TranscriptSegment] = []
    idx = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        m = _BRACKET_HEADER_RE.match(stripped)
        if not m:
            i += 1
            continue

        sh, sm, ss, sf = m.group(1), m.group(2), m.group(3), m.group(4)
        eh, em, es, ef = m.group(5), m.group(6), m.group(7), m.group(8)
        same_line_rest = (m.group(9) or "").strip()

        start_sec = _hhmmssff_to_seconds(sh, sm, ss, sf, fps)
        end_sec = _hhmmssff_to_seconds(eh, em, es, ef, fps)

        i += 1
        text_parts: List[str] = []
        if same_line_rest:
            text_parts.append(same_line_rest)

        while i < len(lines):
            nxt = lines[i]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                i += 1
                continue
            if _BRACKET_HEADER_RE.match(nxt_stripped):
                break
            text_parts.append(nxt_stripped)
            i += 1

        text = " ".join(text_parts).strip()
        idx += 1
        segments.append(
            TranscriptSegment(
                index=idx,
                start_seconds=start_sec,
                end_seconds=end_sec,
                text=text,
            )
        )

    return TranscriptDocument(source_path=source_path, segments=segments)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _coerce_segment(obj: Any, fallback_index: int) -> TranscriptSegment:
    if not isinstance(obj, dict):
        raise ValueError(f"Segment must be an object, got: {type(obj).__name__}")

    start = obj.get("start_seconds")
    if start is None:
        start = obj.get("start")
    end = obj.get("end_seconds")
    if end is None:
        end = obj.get("end")
    text = obj.get("text", "")

    if start is None or end is None:
        raise ValueError('Each segment needs start_seconds/start and end_seconds/end (JSON).')

    return TranscriptSegment(
        index=int(obj.get("index", fallback_index)),
        start_seconds=float(start),
        end_seconds=float(end),
        text=str(text),
    )


def load_json(path: Path) -> TranscriptDocument:
    """
    Parse JSON transcript.

    Expected shape:
      { "segments": [ { "start_seconds": 0.0, "end_seconds": 3.5, "text": "..." }, ... ] }

    Aliases `start` / `end` are also accepted for convenience.
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict) and "segments" in data:
        raw_segments = data["segments"]
    elif isinstance(data, list):
        raw_segments = data
    else:
        raise ValueError('JSON transcript must be a list of segments or { "segments": [...] }.')

    if not isinstance(raw_segments, list):
        raise ValueError('"segments" must be a JSON array.')

    segments = [
        _coerce_segment(item, fallback_index=i + 1) for i, item in enumerate(raw_segments)
    ]
    return TranscriptDocument(source_path=path, segments=segments)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _normalize_transcript_raw(raw: str) -> str:
    """Strip BOM so heuristics see the first real character."""
    return raw.lstrip("\ufeff")


def load_transcript(path: Path) -> TranscriptDocument:
    """
    Dispatch on file extension and content.

    .srt  — Standard SRT, or bracket [HH:MM:SS:FF - ...] blocks (Resolve exports).
    .txt  — Treated as bracket timecode transcript.
    .json — Segment JSON.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_json(path)

    if suffix == ".txt":
        raw = _normalize_transcript_raw(path.read_text(encoding="utf-8", errors="replace"))
        fps = get_transcript_bracket_fps()
        doc = parse_bracket_timestamp_transcript(raw, fps=fps, source_path=path)
        logger.info(
            "Bracket timestamp transcript (.txt): loaded %s segment(s) at %.4f fps (FF field).",
            len(doc.segments),
            fps,
        )
        return doc

    if suffix == ".srt":
        raw = _normalize_transcript_raw(path.read_text(encoding="utf-8", errors="replace"))
        fps = get_transcript_bracket_fps()

        if _looks_like_bracket_timestamp_transcript(raw):
            doc = parse_bracket_timestamp_transcript(raw, fps=fps, source_path=path)
            logger.info(
                "Bracket timestamp transcript (.srt): loaded %s segment(s) at %.4f fps (FF field).",
                len(doc.segments),
                fps,
            )
            return doc

        srt_segments = _parse_srt_raw(raw)
        if srt_segments:
            return TranscriptDocument(source_path=path, segments=srt_segments)

        doc = parse_bracket_timestamp_transcript(raw, fps=fps, source_path=path)
        if doc.segments:
            logger.info(
                "Bracket timestamp transcript (.srt fallback after empty SRT parse): "
                "loaded %s segment(s) at %.4f fps (FF field).",
                len(doc.segments),
                fps,
            )
            return doc

        return TranscriptDocument(source_path=path, segments=[])

    raise ValueError(f"Unsupported transcript type: {suffix} (use .srt, .txt, or .json)")


def _format_hhmmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _run_bracket_parser_self_test() -> None:
    """Quick sanity check: [00:00:49:14 - 00:01:10:04] → ~49.58 s start at 24 fps."""
    sample = (
        "[00:00:49:14 - 00:01:10:04]\n"
        "Justin, how are you doing?\n"
    )
    fps = 24.0
    doc = parse_bracket_timestamp_transcript(sample, fps=fps, source_path=Path("<self-test>"))
    assert len(doc.segments) == 1, f"expected 1 segment, got {len(doc.segments)}"
    seg = doc.segments[0]
    expected_start = 49.0 + 14.0 / 24.0
    assert abs(seg.start_seconds - expected_start) < 0.001, seg.start_seconds
    expected_end = 70.0 + 4.0 / 24.0
    assert abs(seg.end_seconds - expected_end) < 0.001, seg.end_seconds
    assert "Justin" in seg.text
    print(
        "transcript_loader bracket self-test OK:",
        f"start={seg.start_seconds:.4f}s (expect ~49.58), end={seg.end_seconds:.4f}s, text={seg.text!r}",
    )


if __name__ == "__main__":
    _run_bracket_parser_self_test()
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        fps = get_transcript_bracket_fps()
        raw = _normalize_transcript_raw(p.read_text(encoding="utf-8", errors="replace"))
        doc = parse_bracket_timestamp_transcript(raw, fps=fps, source_path=p)
        print(f"File: {p} | fps={fps} | segments={len(doc.segments)}")
        for s in doc.segments[:20]:
            print(
                f"  #{s.index} {s.start_seconds:.4f}s – {s.end_seconds:.4f}s | {s.text[:80]!r}"
                + ("…" if len(s.text) > 80 else "")
            )
        if len(doc.segments) > 20:
            print(f"  ... and {len(doc.segments) - 20} more")
