"""
clip_engine/clip_boundaries.py
Complete-thought boundary detection and repair for clip windows.
"""

from __future__ import annotations

import logging
import re

from clip_engine.transcription_utils import extract_transcript_window

logger = logging.getLogger("clip_engine.clip_boundaries")

DANGLING_END_WORDS = frozenset({
    "and", "but", "so", "because", "the", "a", "an", "to", "of", "with",
    "that", "which", "when", "while", "or", "if", "as", "at", "for", "in",
    "on", "by", "from", "into", "about", "than", "then", "also", "just",
    "even", "still", "yet", "nor", "though", "although", "unless", "until",
})

TITLE_TRAILING_CONNECTORS = frozenset({
    "and", "but", "so", "into", "with", "for", "of",
})

ARTICLE_WORDS = frozenset({"a", "an", "the"})

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")
TITLE_TOKEN_RE = re.compile(r"[A-Za-z']+")
ARTICLE_NOUN_END_RE = re.compile(
    r"\b(?:a|an|the)\s+[A-Za-z][A-Za-z'-]*\s*$",
    re.IGNORECASE,
)

# Common truncated medical stems/fragments frequently seen in clipped titles.
MID_MEDICAL_TERM_END_RE = re.compile(
    r"\b(?:cardio|neuro|gastro|immuno|dermato|osteo|pulmo|nephro|onco|endo|hemo|patho)\s*$",
    re.IGNORECASE,
)

PARTIAL_TITLE_PHRASE_END_RE = re.compile(
    r"(?:\bi'm sorry\b|\bi am sorry\b|\band shots\b|\ba pet\b)\s*$",
    re.IGNORECASE,
)

PARTIAL_HOOK_FRAGMENT_RE = re.compile(
    r"(?:"
    r"that would have been|would have been|ever since I was|she was|he was|"
    r"so they|and I|and he|and she|so every day|pretty heartbreaking\.?\s*so|"
    r"don't know about|do not know about|know about|"
    r"threatening suicide again and I|shattered in her brain\.?\s*so"
    r")\s*$",
    re.IGNORECASE,
)

INCOMPLETE_ABOUT_TITLE_RE = re.compile(
    r"\b(?:about|regarding|concerning)\s*$",
    re.IGNORECASE,
)


def ends_with_dangling_word(text: str) -> bool:
    """True if text ends on a weak connector/article (mid-thought)."""
    words = TITLE_TOKEN_RE.findall((text or "").strip())
    if not words:
        return False
    return words[-1].lower() in DANGLING_END_WORDS


def hook_title_is_incomplete(title: str) -> bool:
    """
    Detect incomplete hook-title endings, including required edge patterns:
    - trailing connectors (and, but, so, into, with, for, of)
    - article + noun ending ("a PET", "an MRI", "the process")
    - mid-medical-term fragments ("cardio", "neuro", ...)
    - short partial phrases ("I'm sorry", "and shots", "a PET")
    """
    t = (title or "").strip()
    if not t:
        return True
    if t[-1] in ",;:-":
        return True

    words = TITLE_TOKEN_RE.findall(t)
    if not words:
        return True

    last = words[-1].lower()
    if last in DANGLING_END_WORDS or last in TITLE_TRAILING_CONNECTORS:
        return True

    if ARTICLE_NOUN_END_RE.search(t):
        return True

    if MID_MEDICAL_TERM_END_RE.search(t):
        return True

    if PARTIAL_TITLE_PHRASE_END_RE.search(t):
        return True

    if PARTIAL_HOOK_FRAGMENT_RE.search(t):
        return True

    if INCOMPLETE_ABOUT_TITLE_RE.search(t) and len(words) >= 4:
        return True

    if len(words) >= 2 and words[-2].lower() in ARTICLE_WORDS:
        return True

    if not SENTENCE_END_RE.search(t) and len(words) <= 3:
        return True

    return False


def starts_mid_sentence(text: str) -> bool:
    """Heuristic: clip window does not begin at a sentence start."""
    t = (text or "").strip()
    if not t:
        return False
    if t[0].islower():
        return True
    first = t.split(None, 1)[0] if t.split() else ""
    starters = {"and", "but", "so", "because", "then", "also", "or", "yet", "still"}
    return first.lower().rstrip(",") in starters


def _segment_sentences(segments: list[dict], t0: float, t1: float) -> list[tuple[float, float, str]]:
    """Build sentence-like spans from transcript segments inside [t0, t1]."""
    window = [
        s for s in segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]
    if not window:
        return []

    spans: list[tuple[float, float, str]] = []
    buf_start = float(window[0].get("start", t0))
    buf_end = float(window[0].get("end", t0))
    buf_text = str(window[0].get("text", "")).strip()

    for seg in window[1:]:
        seg_start = float(seg.get("start", buf_end))
        seg_end = float(seg.get("end", seg_start))
        seg_text = str(seg.get("text", "")).strip()
        gap = seg_start - buf_end
        combined = f"{buf_text} {seg_text}".strip()
        ends_sentence = bool(SENTENCE_END_RE.search(buf_text))

        if gap > 1.2 or ends_sentence:
            if buf_text:
                spans.append((buf_start, buf_end, buf_text))
            buf_start = seg_start
            buf_text = seg_text
        else:
            buf_text = combined
        buf_end = seg_end

    if buf_text:
        spans.append((buf_start, buf_end, buf_text))
    return spans


def _last_complete_sentence_end(
    spans: list[tuple[float, float, str]],
    *,
    max_end: float,
) -> float | None:
    for _, end, text in reversed(spans):
        if end > max_end + 0.05:
            continue
        if SENTENCE_END_RE.search(text.strip()) and not ends_with_dangling_word(text):
            return end
    return None


def _first_sentence_start(
    spans: list[tuple[float, float, str]],
    *,
    min_start: float,
) -> float | None:
    for start, _, text in spans:
        if start < min_start - 0.05:
            continue
        t = text.strip()
        if not t:
            continue
        if not starts_mid_sentence(t):
            return start
        if SENTENCE_END_RE.search(t) or len(t.split()) >= 4:
            return start
    return spans[0][0] if spans else None


def snap_clip_to_sentence_boundaries(
    clip: dict,
    segments: list[dict],
    *,
    max_duration: float,
    min_duration: float = 5.0,
) -> tuple[dict, bool, str | None]:
    """
    Adjust clip start/end to sentence boundaries when transcript data allows.
    Returns (clip, repaired, warning).
    """
    c = dict(clip)
    t0 = float(c.get("start_seconds", c.get("start", 0)))
    t1 = float(c.get("end_seconds", c.get("end", t0)))
    dur = t1 - t0
    if dur <= 0 or not segments:
        return c, False, None

    window_text = extract_transcript_window(segments, t0, t1)
    spans = _segment_sentences(segments, t0, t1)
    if not spans:
        if ends_with_dangling_word(window_text) or starts_mid_sentence(window_text):
            c["boundary_warning"] = "Could not snap to sentence boundaries (sparse transcript)."
            c.setdefault("warnings", []).append(c["boundary_warning"])
        return c, False, c.get("boundary_warning")

    repaired = False
    warning: str | None = None
    new_t0, new_t1 = t0, t1

    first_start = _first_sentence_start(spans, min_start=t0)
    if first_start is not None and first_start > t0 + 0.25 and first_start < t1 - min_duration:
        new_t0 = first_start
        repaired = True

    max_end = new_t0 + max_duration
    last_end = _last_complete_sentence_end(spans, max_end=min(new_t1, max_end))
    if last_end is not None and last_end > new_t0 + min_duration:
        if last_end < new_t1 - 0.2:
            new_t1 = last_end
            repaired = True
    elif new_t1 - new_t0 > max_duration and last_end is not None:
        new_t1 = min(last_end, new_t0 + max_duration)
        repaired = True

    if new_t1 - new_t0 > max_duration:
        new_t1 = new_t0 + max_duration
        repaired = True

    if new_t1 - new_t0 < min_duration:
        return c, False, "Boundary repair skipped: would violate minimum duration."

    final_text = extract_transcript_window(segments, new_t0, new_t1)
    if ends_with_dangling_word(final_text):
        for _, end, text in reversed(spans):
            if end <= new_t0 + min_duration:
                continue
            if end > new_t0 + max_duration:
                continue
            if SENTENCE_END_RE.search(text.strip()) and not ends_with_dangling_word(text):
                candidate_end = min(end, new_t0 + max_duration)
                if candidate_end - new_t0 >= min_duration:
                    new_t1 = candidate_end
                    repaired = True
                    break
        final_text = extract_transcript_window(segments, new_t0, new_t1)

    if ends_with_dangling_word(final_text) or starts_mid_sentence(final_text):
        warning = "Clip boundaries may cut mid-thought; transcript alignment incomplete."
        c["boundary_warning"] = warning
        c.setdefault("warnings", []).append(warning)
        c["boundary_status"] = "warning"
    elif repaired:
        c["boundary_status"] = "repaired"
        c.pop("boundary_warning", None)
    else:
        c["boundary_status"] = "ok"
        c.pop("boundary_warning", None)

    if repaired and (abs(new_t0 - t0) > 0.1 or abs(new_t1 - t1) > 0.1):
        c["start_seconds"] = round(new_t0, 3)
        c["end_seconds"] = round(new_t1, 3)
        c["boundary_repaired"] = True
        logger.info(
            "Boundary repair %.1f-%.1f -> %.1f-%.1f",
            t0, t1, new_t0, new_t1,
        )
        return c, True, warning

    return c, False, warning


def apply_boundary_repairs(
    clips: list[dict],
    segments: list[dict],
    *,
    max_duration: float,
    min_duration: float = 5.0,
) -> tuple[list[dict], int]:
    """Repair all clips; never raises."""
    out: list[dict] = []
    repairs = 0
    for clip in clips:
        try:
            fixed, did_repair, _ = snap_clip_to_sentence_boundaries(
                clip,
                segments,
                max_duration=max_duration,
                min_duration=min_duration,
            )
            if did_repair:
                repairs += 1
            out.append(fixed)
        except Exception as exc:
            logger.warning("Boundary repair failed for clip: %s", exc)
            nc = dict(clip)
            nc.setdefault("warnings", []).append(f"Boundary repair error: {exc}")
            nc["boundary_status"] = "error"
            out.append(nc)
    return out, repairs


__all__ = [
    "DANGLING_END_WORDS",
    "apply_boundary_repairs",
    "ends_with_dangling_word",
    "hook_title_is_incomplete",
    "snap_clip_to_sentence_boundaries",
    "starts_mid_sentence",
]
