# -*- coding: utf-8 -*-
"""
clip_engine/speaker_signals.py
Speaker energy, interruption, and debate detection.
Transcript-based fallback; pyannote.audio diarization reserved for future use.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("clip_engine.speaker_signals")

DEBATE_KEYWORDS = {
    "disagree", "wrong", "actually", "no that's", "but i think", "on the other hand",
    "however", "debate", "argument", "pushback", "counter", "rebuttal", "not true",
    "that's false", "i disagree", "let me push back", "devil's advocate",
}

QUESTION_TENSION = re.compile(
    r"\?\s*(but|however|well|actually|no)\b",
    re.IGNORECASE,
)

INTERRUPTION_MARKERS = re.compile(
    r"(\-\-|\[crosstalk\]|\[overlap\]|\[interrupt\]|—)",
    re.IGNORECASE,
)


def _window_segments(
    candidate_clip: dict,
    transcript_segments: list[dict],
) -> list[dict]:
    t0 = float(candidate_clip.get("start_seconds", candidate_clip.get("start", 0)))
    t1 = float(candidate_clip.get("end_seconds", candidate_clip.get("end", t0 + 60)))
    return [
        s for s in transcript_segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]


def _extract_speaker(seg: dict) -> str | None:
    for key in ("speaker", "speaker_id", "speaker_label"):
        val = seg.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _detect_speaker_alternation(segments: list[dict]) -> tuple[int, int]:
    """Return (alternations, unique_speakers)."""
    speakers: list[str] = []
    for seg in segments:
        sp = _extract_speaker(seg)
        if sp:
            speakers.append(sp)
    if len(speakers) < 2:
        return 0, len(set(speakers))

    alternations = sum(
        1 for i in range(1, len(speakers)) if speakers[i] != speakers[i - 1]
    )
    return alternations, len(set(speakers))


def _detect_short_exchanges(segments: list[dict]) -> int:
    """Count rapid short back-and-forth segments (proxy for interruption)."""
    if len(segments) < 3:
        return 0
    count = 0
    for i in range(len(segments) - 1):
        dur_a = float(segments[i].get("end", 0)) - float(segments[i].get("start", 0))
        dur_b = float(segments[i + 1].get("end", 0)) - float(segments[i + 1].get("start", 0))
        gap = float(segments[i + 1].get("start", 0)) - float(segments[i].get("end", 0))
        if dur_a <= 3.0 and dur_b <= 3.0 and gap <= 0.5:
            count += 1
    return count


def analyze_speaker_signals(
    candidate_clip: dict,
    transcript_segments: list[dict],
) -> dict[str, Any]:
    """
    Analyze speaker energy, interruptions, and debate tension.
    Uses speaker labels when present; falls back to segment timing heuristics.
    """
    window = _window_segments(candidate_clip, transcript_segments)
    text = " ".join(str(s.get("text", "")) for s in window).lower()

    if not window:
        return {
            "speaker_energy": 0,
            "interruption_score": 0,
            "debate_score": 0,
            "reason": "No transcript in clip window",
        }

    alternations, unique_speakers = _detect_speaker_alternation(window)
    short_exchanges = _detect_short_exchanges(window)
    interruption_markers = len(INTERRUPTION_MARKERS.findall(text))
    debate_hits = sum(1 for kw in DEBATE_KEYWORDS if kw in text)
    question_tension = len(QUESTION_TENSION.findall(text))

    # Speaker energy: rapid alternation + multi-speaker
    speaker_energy = 25.0
    if unique_speakers >= 2:
        speaker_energy += 20
    if alternations >= 4:
        speaker_energy += min(40, alternations * 5)
    elif alternations >= 2:
        speaker_energy += 15
    speaker_energy = min(100.0, speaker_energy)

    # Interruption score
    interruption_score = 20.0 + short_exchanges * 12 + interruption_markers * 15
    if alternations >= 3 and unique_speakers >= 2:
        interruption_score += 15
    interruption_score = min(100.0, interruption_score)

    # Debate score
    debate_score = 25.0 + debate_hits * 15 + question_tension * 10
    if "?" in text and any(w in text for w in ("but", "however", "actually", "wrong")):
        debate_score += 12
    debate_score = min(100.0, debate_score)

    reasons: list[str] = []
    if unique_speakers >= 2:
        reasons.append(f"{unique_speakers} speakers")
    if alternations >= 3:
        reasons.append(f"{alternations} alternations")
    if short_exchanges >= 2:
        reasons.append("rapid exchanges")
    if debate_hits:
        reasons.append("debate keywords")
    reason = "; ".join(reasons) if reasons else "Single-speaker / low tension"

    return {
        "speaker_energy": int(round(speaker_energy)),
        "interruption_score": int(round(interruption_score)),
        "debate_score": int(round(debate_score)),
        "reason": reason,
    }


def compute_speaker_boost(speaker_signals: dict[str, Any]) -> float:
    """Return 0-8 point additive boost from speaker/debate signals."""
    avg = (
        speaker_signals.get("speaker_energy", 0)
        + speaker_signals.get("interruption_score", 0)
        + speaker_signals.get("debate_score", 0)
    ) / 3.0
    if avg >= 70:
        return 8.0
    if avg >= 55:
        return 5.0
    if avg >= 40:
        return 2.0
    return 0.0


def apply_speaker_signals_to_clips(
    clips: list[dict],
    transcript_segments: list[dict],
    *,
    enabled: bool = True,
) -> list[dict]:
    """Apply speaker signal scores and optional ranking boost."""
    if not enabled or not clips:
        return clips

    for clip in clips:
        signals = analyze_speaker_signals(clip, transcript_segments)
        boost = compute_speaker_boost(signals)
        clip["speaker_signals"] = signals
        clip["speaker_boost"] = round(boost, 1)
        if boost > 0:
            current = float(clip.get("composite_score", clip.get("original_composite_score", 50)))
            clip["composite_score"] = int(min(100, round(current + boost)))

    logger.info("Applied speaker signals to %d clips", len(clips))
    return clips
