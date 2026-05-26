"""
clip_engine/local_candidate_discovery.py
Fast local pre-candidate extraction (no OpenAI).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from clip_engine.clip_signals import (
    AUDIENCE_REACTIONS,
    CURIOSITY_HOOKS,
    EMOTION_WORDS,
    FUNNY_INDICATORS,
    analyze_curiosity_gap,
    analyze_emotion_spikes,
    analyze_pacing,
    analyze_scroll_stopping_hook,
    compute_signal_boosts,
)
from clip_engine.clip_style import ClipStyle, get_clip_style_profile
from clip_engine.transcription_utils import detect_pauses, merge_segments_into_sentences

logger = logging.getLogger("clip_engine.local_candidate_discovery")

QUESTION_RE = re.compile(r"\?")
HOOK_PHRASES = (
    "here's the thing",
    "the truth is",
    "nobody talks about",
    "what they don't tell you",
    "wait until",
    "you won't believe",
    "the craziest",
    "hot take",
)


def _window_bounds(style: ClipStyle | str, user_min: float, user_max: float) -> tuple[float, float, float]:
    profile = get_clip_style_profile(style, user_min_seconds=user_min, user_max_seconds=user_max)
    if style == "Micro clips":
        return max(user_min, 30.0), min(user_max, 75.0), 22.0
    if style == "Long story clips":
        return max(user_min, 90.0), min(user_max, 150.0), 40.0
    return max(user_min, 45.0), min(user_max, 100.0), 30.0


def _score_sentence_energy(text: str, segments: list[dict]) -> float:
    lower = text.lower()
    score = 40.0
    words = set(re.findall(r"\b[a-z]{3,}\b", lower))
    if words & EMOTION_WORDS:
        score += 18
    if words & FUNNY_INDICATORS:
        score += 12
    if any(p in lower for p in CURIOSITY_HOOKS):
        score += 14
    if any(p in lower for p in HOOK_PHRASES):
        score += 12
    if QUESTION_RE.search(text):
        score += 8
    if any(r in lower for r in AUDIENCE_REACTIONS):
        score += 10
    emo = analyze_emotion_spikes(segments)
    pace = analyze_pacing(segments)
    cur = analyze_curiosity_gap(text)
    hook = analyze_scroll_stopping_hook(text)
    score += (
        emo.get("emotion_spike", 0) * 0.12
        + pace.get("pacing", 0) * 0.08
        + cur.get("curiosity_gap", 0) * 0.12
        + hook.get("scroll_stopping_hook", 0) * 0.15
    )
    return min(100.0, score)


def _assign_region(mid: float, media_duration: float) -> str:
    if media_duration <= 0:
        return "middle"
    names = ["beginning", "early_middle", "middle", "late_middle", "ending"]
    idx = min(len(names) - 1, int(mid / media_duration * len(names)))
    return names[idx]


def discover_local_candidates(
    segments: list[dict],
    media_duration: float,
    *,
    clip_style: ClipStyle | str = "Balanced",
    user_min_seconds: float = 25.0,
    user_max_seconds: float = 160.0,
    max_candidates: int = 180,
) -> list[dict]:
    """
    Generate 50-200 lightweight candidate windows from transcript heuristics.
    """
    if not segments or media_duration <= 0:
        return []

    win_min, win_max, step = _window_bounds(clip_style, user_min_seconds, user_max_seconds)
    sentences = merge_segments_into_sentences(segments)
    if len(sentences) < 2:
        return []

    pauses = detect_pauses(segments, min_pause_seconds=0.75)
    pause_times = {round(p.after_seconds, 2) for p in pauses}

    raw_windows: list[dict] = []
    i = 0
    while i < len(sentences):
        start = float(sentences[i].get("start", 0))
        j = i
        texts: list[str] = []
        window_segs: list[dict] = []
        while j < len(sentences):
            window_segs.append(sentences[j])
            texts.append(str(sentences[j].get("text", "")).strip())
            end = float(sentences[j].get("end", start))
            dur = end - start
            if dur >= win_max:
                break
            if dur >= win_min and (
                j == len(sentences) - 1
                or round(end, 1) in pause_times
                or bool(re.search(r"[.!?]\s*$", texts[-1]))
            ):
                break
            j += 1
        if j == i:
            j += 1
        end = float(sentences[min(j, len(sentences) - 1)].get("end", start))
        dur = end - start
        if dur >= win_min * 0.85 and dur <= win_max * 1.1:
            text = " ".join(texts).strip()
            if len(text.split()) >= 8:
                energy = _score_sentence_energy(text, window_segs)
                mid = (start + end) / 2
                raw_windows.append(
                    {
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                        "hook_title": " ".join(text.split()[:8])[:60],
                        "composite_score": int(round(energy)),
                        "local_rank_score": round(energy, 1),
                        "selection_reason": "Local heuristic pre-candidate.",
                        "ai_context_reason": "Discovered via punctuation, pacing, and signal heuristics.",
                        "dominant_signal": "educational",
                        "source": "local_prefilter",
                        "confidence": 0.6,
                        "_region": _assign_region(mid, media_duration),
                        "_pass": "local_prefilter",
                        "warnings": [],
                    }
                )
        i = max(i + 1, j)

    raw_windows.sort(key=lambda x: int(x.get("composite_score", 0)), reverse=True)

    kept: list[dict] = []
    for c in raw_windows:
        t0 = float(c["start_seconds"])
        t1 = float(c["end_seconds"])
        if any(abs(t0 - float(k["start_seconds"])) < step * 0.5 for k in kept):
            continue
        kept.append(c)
        if len(kept) >= max_candidates:
            break

    for c in kept[: min(len(kept), 80)]:
        boosts = compute_signal_boosts(c, segments)
        c["composite_score"] = min(
            85,
            int(c.get("composite_score", 50)) + int(boosts.get("signal_boost", 0)),
        )
        c["local_signals"] = {
            k: boosts.get(k)
            for k in (
                "emotion_spike",
                "pacing",
                "curiosity_gap",
                "scroll_stopping_hook",
                "audience_reaction",
            )
        }

    logger.info("Local pre-candidates: %d windows (max=%d)", len(kept), max_candidates)
    return kept
