"""
clip_engine/clip_scoring.py
Transparent virality scoring and hook title quality assessment.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from clip_engine.clip_boundaries import (
    ends_with_dangling_word,
    starts_mid_sentence,
)
from clip_engine.transcription_utils import extract_transcript_window

logger = logging.getLogger("clip_engine.clip_scoring")

CLIP_STRATEGIES = (
    "Balanced",
    "Viral Moments",
    "Educational Nuggets",
    "Debate/Controversy",
    "Emotional Story",
    "Podcast Highlights",
)

PLATFORM_TARGETS = (
    "TikTok/Reels/Shorts",
    "YouTube Shorts",
    "Instagram Reels",
    "LinkedIn",
    "Podcast teaser",
)

TITLE_STYLES = (
    "Curiosity",
    "Bold statement",
    "Educational",
    "Emotional",
    "Debate",
    "Clean/professional",
)

_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "Viral Moments": {"hook_strength": 1.2, "curiosity": 1.15, "pacing": 1.1},
    "Educational Nuggets": {"clarity": 1.2, "complete_thought": 1.15, "hook_strength": 0.95},
    "Debate/Controversy": {"debate": 1.3, "emotion": 1.1, "hook_strength": 1.05},
    "Emotional Story": {"emotion": 1.25, "complete_thought": 1.1, "hook_strength": 1.0},
    "Podcast Highlights": {"standalone_context": 1.15, "clarity": 1.1, "pacing": 1.05},
    "Balanced": {},
}


def assess_hook_quality(title: str) -> tuple[int, str | None]:
    """
    Score hook title 0-100. Returns (hook_quality_score, hook_warning).
    """
    t = (title or "").strip()
    if not t:
        return 0, "Empty hook title"
    words = t.split()
    if len(words) < 3:
        return 35, "Hook title is very short"
    score = 72
    if ends_with_dangling_word(t):
        return max(20, score - 45), "Hook ends on a dangling word (incomplete thought)"
    if t[-1] in ",;:":
        return max(25, score - 40), "Hook ends mid-sentence"
    if not re.search(r"[.!?]$", t) and len(words) > 6:
        score -= 5
    if t.islower():
        score -= 8
    if len(words) > 12:
        score -= 10
        warn = "Hook may be too long for Shorts"
    else:
        warn = None
    if re.search(r"\b(how|why|what|secret|truth|nobody)\b", t, re.I):
        score += 8
    if re.search(r"\b(never|always|everyone|no one|shocking|insane)\b", t, re.I):
        score += 6
    return min(100, max(0, score)), warn


def repair_hook_title_local(title: str, window_text: str = "") -> str:
    """Rewrite a bad hook locally without OpenAI."""
    t = (title or "").strip()
    if not t:
        if window_text:
            return repair_hook_title_local(
                " ".join(window_text.split()[:8]).strip(".,;:!?"),
                "",
            )
        return "Untitled Clip Moment"

    words = t.split()
    while words and words[-1].lower().rstrip(".,;:") in {
        "and", "but", "so", "because", "the", "a", "an", "to", "of", "with",
        "that", "which", "when", "while", "or", "if", "about", "for",
    }:
        words.pop()

    if not words and window_text:
        sents = re.split(r"(?<=[.!?])\s+", window_text.strip())
        for sent in sents:
            w = sent.split()
            if len(w) >= 4:
                words = w[:10]
                break

    if not words:
        return "Key Moment From This Episode"

    repaired = " ".join(words).strip(".,;:!? ")
    if repaired:
        repaired = repaired[0].upper() + repaired[1:]
    return repaired[:80] or "Key Moment From This Episode"


def _dim_score(base: float, hits: int, cap: int = 3) -> int:
    ratio = min(1.0, hits / max(cap, 1))
    return int(min(20, max(0, base * ratio)))


def compute_virality_score(
    clip: dict,
    segments: list[dict],
    *,
    clip_strategy: str = "Balanced",
    platform_target: str = "TikTok/Reels/Shorts",
    title_style: str = "Curiosity",
) -> tuple[int, dict[str, int], str]:
    """
    Compute virality_score 0-100, breakdown dict, and short explanation.
    """
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    text = extract_transcript_window(segments, t0, t1).lower()
    title = str(clip.get("hook_title", ""))
    hook_q, _ = assess_hook_quality(title)

    scores = clip.get("scores") or {}
    signal = clip.get("signal_scores") or {}

    hook_strength = _dim_score(
        18,
        int(scores.get("hook_strength", 0) / 25)
        + int(signal.get("scroll_stopping_hook", 0) / 30)
        + (1 if hook_q >= 70 else 0),
        cap=4,
    )
    emotion = _dim_score(
        15,
        int(signal.get("emotion_spike", 0) / 35)
        + int(scores.get("emotional_impact", 0) / 30),
        cap=4,
    )
    curiosity = _dim_score(
        14,
        int(signal.get("curiosity_gap", 0) / 35)
        + int(scores.get("curiosity", 0) / 30),
        cap=4,
    )
    debate = _dim_score(
        10,
        int((clip.get("speaker_signals") or {}).get("debate_score", 0) / 40),
        cap=3,
    )
    pacing = _dim_score(12, int(signal.get("pacing", 0) / 35), cap=3)
    clarity = _dim_score(15, int(scores.get("clarity", 0) / 30), cap=4)
    complete_thought = 14
    standalone = 12
    quoteability = 10
    platform_fit = 10

    penalties = 0
    boosts = 0

    if starts_mid_sentence(text):
        penalties += 12
        complete_thought -= 8
    if ends_with_dangling_word(text):
        penalties += 12
        complete_thought -= 8
    if clip.get("boundary_warning") or clip.get("boundary_status") == "warning":
        penalties += 6
    if hook_q < 50:
        penalties += 8
        hook_strength -= 4
    if ends_with_dangling_word(title):
        penalties += 10

    if re.search(r"\b(lesson|learned|realize|turned out|here's what)\b", text):
        boosts += 6
        clarity += 2
    if re.search(r"\b(never|secret|shocking|controversy|wrong about)\b", text):
        boosts += 5
        curiosity += 2
    if re.search(r"\b(cry|cried|trauma|love|hate|afraid|devastated)\b", text):
        boosts += 4
        emotion += 2
    if re.search(r"[.!?][\"')\]]*\s*$", text.strip()):
        boosts += 3
        complete_thought += 2
    else:
        complete_thought -= 4

    if len(text.split()) < 40:
        standalone -= 4
        penalties += 4

    breakdown = {
        "hook_strength": max(0, min(20, hook_strength)),
        "emotion": max(0, min(15, emotion)),
        "curiosity": max(0, min(14, curiosity)),
        "debate": max(0, min(10, debate)),
        "pacing": max(0, min(12, pacing)),
        "clarity": max(0, min(15, clarity)),
        "complete_thought": max(0, min(16, complete_thought)),
        "standalone_context": max(0, min(12, standalone)),
        "quoteability": max(0, min(10, quoteability)),
        "platform_fit": max(0, min(10, platform_fit)),
    }

    weights = _STRATEGY_WEIGHTS.get(clip_strategy, {})
    total = 0.0
    for key, val in breakdown.items():
        w = weights.get(key, 1.0)
        total += val * w

    if platform_target == "LinkedIn":
        breakdown["platform_fit"] = min(10, breakdown["platform_fit"] + 2)
        total += 2
    elif platform_target == "Podcast teaser":
        breakdown["standalone_context"] = min(12, breakdown["standalone_context"] + 2)
        total += 2

    if title_style == "Debate":
        breakdown["debate"] = min(10, breakdown["debate"] + 2)
    elif title_style == "Educational":
        breakdown["clarity"] = min(15, breakdown["clarity"] + 2)

    total = total - penalties + boosts
    virality = int(max(0, min(100, round(total))))

    parts = []
    if hook_q >= 75:
        parts.append("strong hook")
    if breakdown["emotion"] >= 10:
        parts.append("emotional energy")
    if breakdown["curiosity"] >= 9:
        parts.append("curiosity gap")
    if breakdown["debate"] >= 7:
        parts.append("debate potential")
    if complete_thought >= 12 and not clip.get("boundary_warning"):
        parts.append("complete thought")
    if penalties >= 10:
        parts.append("boundary/title penalties applied")
    explanation = ", ".join(parts) if parts else "balanced moment across clarity and pacing"

    return virality, breakdown, explanation


def apply_virality_to_clip(
    clip: dict,
    segments: list[dict],
    *,
    clip_strategy: str = "Balanced",
    platform_target: str = "TikTok/Reels/Shorts",
    title_style: str = "Curiosity",
) -> dict:
    """Attach virality_score, breakdown, hook quality fields to clip."""
    c = dict(clip)
    title = str(c.get("hook_title", ""))
    hook_score, hook_warn = assess_hook_quality(title)
    if hook_score < 55 or ends_with_dangling_word(title):
        repaired = repair_hook_title_local(title, extract_transcript_window(
            segments,
            float(c.get("start_seconds", 0)),
            float(c.get("end_seconds", 0)),
        ))
        if repaired != title:
            c["hook_title_before_repair"] = title
            c["hook_title"] = repaired
            c["hook_title_repaired"] = True
            title = repaired
            hook_score, hook_warn = assess_hook_quality(title)

    c["hook_quality_score"] = hook_score
    if hook_warn:
        c["hook_warning"] = hook_warn
        c.setdefault("warnings", []).append(f"Hook: {hook_warn}")

    virality, breakdown, explanation = compute_virality_score(
        c,
        segments,
        clip_strategy=clip_strategy,
        platform_target=platform_target,
        title_style=title_style,
    )
    c["virality_score"] = virality
    c["virality_breakdown"] = breakdown
    c["virality_explanation"] = explanation
    return c


def apply_virality_to_clips(
    clips: list[dict],
    segments: list[dict],
    *,
    clip_strategy: str = "Balanced",
    platform_target: str = "TikTok/Reels/Shorts",
    title_style: str = "Curiosity",
) -> tuple[list[dict], int]:
    """Apply virality scoring to all clips; returns (clips, title_repairs)."""
    out: list[dict] = []
    repairs = 0
    for clip in clips:
        try:
            fixed = apply_virality_to_clip(
                clip,
                segments,
                clip_strategy=clip_strategy,
                platform_target=platform_target,
                title_style=title_style,
            )
            if fixed.get("hook_title_repaired"):
                repairs += 1
            out.append(fixed)
        except Exception as exc:
            logger.warning("Virality scoring failed: %s", exc)
            nc = dict(clip)
            nc.setdefault("virality_score", int(nc.get("composite_score", 50)))
            out.append(nc)
    return out, repairs


__all__ = [
    "CLIP_STRATEGIES",
    "PLATFORM_TARGETS",
    "TITLE_STYLES",
    "apply_virality_to_clip",
    "apply_virality_to_clips",
    "assess_hook_quality",
    "compute_virality_score",
    "repair_hook_title_local",
]
