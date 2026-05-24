"""
clip_engine/clip_signals.py
Local heuristic signal scoring for clip ranking boosts.
No LLM calls — pure transcript/text analysis.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("clip_engine.clip_signals")

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

EMOTION_WORDS = {
    "love", "hate", "fear", "angry", "crying", "cried", "devastated", "heartbroken",
    "amazing", "incredible", "unbelievable", "shocked", "terrified", "furious",
    "grateful", "blessed", "miserable", "hopeless", "desperate", "overwhelmed",
    "excited", "thrilled", "horrified", "disgusted", "betrayed", "abandoned",
    "trauma", "traumatic", "abuse", "violence", "death", "died", "suicide",
    "cancer", "addiction", "struggle", "survived", "resilience", "resilient",
    "loss", "grief", "mourning", "painful", "suffering", "broken", "healing",
}

DRAMATIC_TURNS = {
    "but then", "suddenly", "everything changed", "turned out", "plot twist",
    "never expected", "out of nowhere", "that's when", "little did i know",
    "changed my life", "changed everything", "never the same", "until one day",
}

FUNNY_INDICATORS = {
    "hilarious", "funny", "laugh", "laughed", "laughing", "lol", "haha",
    "joke", "comedy", "ridiculous", "absurd", "crazy story", "you won't believe",
}

LAUGHTER_MARKERS = re.compile(
    r"\b(lol|lmao|haha|hehe|\[laugh|\(laugh|\*laugh)\b",
    re.IGNORECASE,
)

CURIOSITY_HOOKS = {
    "secret", "truth", "nobody talks about", "what they don't tell you",
    "here's why", "the reason", "you need to know", "most people don't",
    "hidden", "revealed", "exposed", "shocking", "surprising",
}

WAIT_WHAT_OPENINGS = re.compile(
    r"^(wait|hold on|stop|what\??|no way|are you serious|you're kidding)\b",
    re.IGNORECASE,
)

STRONG_OPENINGS = re.compile(
    r"^(so|look|listen|here's the thing|the craziest|the wildest|"
    r"i never|nobody|everyone|the day i|when i)\b",
    re.IGNORECASE,
)

AUDIENCE_REACTIONS = {
    "wow", "whoa", "oh my god", "omg", "no way", "seriously", "really",
    "that's insane", "that's crazy", "unbelievable", "mind blown",
    "give it up", "round of applause", "standing ovation",
}

FAST_PACING_SHORT_SEGMENTS = 2.5  # seconds
FAST_PACING_MIN_ALTERNATIONS = 4


def _clip_window_text(
    candidate_clip: dict,
    transcript_segments: list[dict],
) -> tuple[str, list[dict]]:
    """Extract text and segments overlapping the clip window."""
    t0 = float(candidate_clip.get("start_seconds", candidate_clip.get("start", 0)))
    t1 = float(candidate_clip.get("end_seconds", candidate_clip.get("end", t0 + 60)))
    window_segs = [
        s for s in transcript_segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]
    text = " ".join(str(s.get("text", "")).strip() for s in window_segs).strip()
    if not text:
        text = str(candidate_clip.get("hook_title", "")) + " " + str(
            candidate_clip.get("selection_reason", "")
        )
    return text, window_segs


def _score_from_hits(hits: int, max_hits: int, base: float = 30.0) -> float:
    if hits <= 0:
        return base * 0.3
    ratio = min(1.0, hits / max(max_hits, 1))
    return min(100.0, base + ratio * (100.0 - base))


def analyze_emotion_spikes(transcript_segments: list[dict]) -> dict[str, Any]:
    """Detect emotional words, dramatic turns, trauma/resilience moments."""
    text = " ".join(str(s.get("text", "")) for s in transcript_segments).lower()
    if not text.strip():
        return {"emotion_spike": 0, "reason": "No transcript text"}

    words = set(re.findall(r"\b[a-z]{3,}\b", text))
    emotion_hits = len(words & EMOTION_WORDS)
    dramatic_hits = sum(1 for phrase in DRAMATIC_TURNS if phrase in text)
    funny_hits = len(words & FUNNY_INDICATORS) + (1 if LAUGHTER_MARKERS.search(text) else 0)

    score = _score_from_hits(emotion_hits + dramatic_hits * 2, 8, base=25)
    if funny_hits:
        score = min(100, score + funny_hits * 8)

    reasons: list[str] = []
    if emotion_hits >= 2:
        reasons.append(f"{emotion_hits} emotional keywords")
    if dramatic_hits:
        reasons.append("dramatic turn")
    if funny_hits:
        reasons.append("humor/laughter")
    reason = "; ".join(reasons) if reasons else "Low emotional intensity"

    return {"emotion_spike": int(round(score)), "reason": reason}


def analyze_pacing(transcript_segments: list[dict]) -> dict[str, Any]:
    """Detect fast back-and-forth and short powerful segments."""
    if len(transcript_segments) < 2:
        return {"pacing": 35, "reason": "Single segment — moderate pacing"}

    short_count = 0
    total = len(transcript_segments)
    for seg in transcript_segments:
        dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
        word_count = len(str(seg.get("text", "")).split())
        if dur <= FAST_PACING_SHORT_SEGMENTS and word_count >= 3:
            short_count += 1
        elif word_count <= 8 and dur <= 4.0:
            short_count += 1  # short powerful quote

    short_ratio = short_count / total
    avg_dur = sum(
        float(s.get("end", 0)) - float(s.get("start", 0)) for s in transcript_segments
    ) / total

    score = 40.0
    if short_ratio >= 0.5:
        score += 35
    elif short_ratio >= 0.3:
        score += 20
    if avg_dur <= 3.0:
        score += 15
    elif avg_dur <= 5.0:
        score += 8

    score = min(100.0, score)
    reason = (
        f"Fast pacing ({short_count}/{total} short segments)"
        if short_ratio >= 0.3
        else f"Moderate pacing (avg {avg_dur:.1f}s/segment)"
    )
    return {"pacing": int(round(score)), "reason": reason}


def analyze_audience_reactions(transcript_segments: list[dict]) -> dict[str, Any]:
    """Detect laughter indicators and audience reaction phrases."""
    text = " ".join(str(s.get("text", "")) for s in transcript_segments).lower()
    if not text.strip():
        return {"audience_reaction": 0, "reason": "No transcript text"}

    reaction_hits = sum(1 for phrase in AUDIENCE_REACTIONS if phrase in text)
    laugh_hits = len(LAUGHTER_MARKERS.findall(text))
    funny_words = len(set(re.findall(r"\b[a-z]{3,}\b", text)) & FUNNY_INDICATORS)

    total_hits = reaction_hits + laugh_hits + funny_words
    score = _score_from_hits(total_hits, 5, base=20)

    reasons: list[str] = []
    if laugh_hits:
        reasons.append("laughter markers")
    if reaction_hits:
        reasons.append(f"{reaction_hits} reaction phrases")
    if funny_words and not reasons:
        reasons.append("humor indicators")
    reason = "; ".join(reasons) if reasons else "No audience reaction signals"

    return {"audience_reaction": int(round(score)), "reason": reason}


def analyze_curiosity_gap(text: str) -> dict[str, Any]:
    """Score curiosity hooks and open loops in text."""
    if not text.strip():
        return {"curiosity_gap": 0, "reason": "Empty text"}

    lower = text.lower()
    hook_hits = sum(1 for phrase in CURIOSITY_HOOKS if phrase in lower)
    question_marks = lower.count("?")
    ellipsis = lower.count("...") + lower.count("…")
    cliffhanger = bool(re.search(r"\b(but|however|yet|until)\b.{0,40}$", lower.strip()))

    score = 25.0 + hook_hits * 12 + min(question_marks, 3) * 8 + ellipsis * 5
    if cliffhanger:
        score += 15
    score = min(100.0, score)

    reasons: list[str] = []
    if hook_hits:
        reasons.append(f"{hook_hits} curiosity hooks")
    if question_marks:
        reasons.append("open questions")
    if cliffhanger:
        reasons.append("cliffhanger ending")
    reason = "; ".join(reasons) if reasons else "Low curiosity gap"

    return {"curiosity_gap": int(round(score)), "reason": reason}


def analyze_scroll_stopping_hook(text: str) -> dict[str, Any]:
    """Score first-sentence hook strength for scroll-stopping opens."""
    if not text.strip():
        return {"scroll_stopping_hook": 0, "reason": "Empty text"}

    first_sentence = re.split(r"[.!?]\s+", text.strip())[0].strip()
    first_words = first_sentence.split()[:12]
    opening = " ".join(first_words).lower()

    score = 30.0
    reasons: list[str] = []

    if WAIT_WHAT_OPENINGS.search(opening):
        score += 30
        reasons.append('"wait/what" opening')
    if STRONG_OPENINGS.search(opening):
        score += 20
        reasons.append("strong opener")
    if len(first_words) <= 8 and len(first_sentence) >= 15:
        score += 15
        reasons.append("punchy short hook")
    if any(w in opening for w in ("never", "secret", "crazy", "worst", "best", "first time")):
        score += 12
        reasons.append("high-impact words")
    if opening.startswith(("i ", "we ", "my ", "when ")):
        score += 8
        reasons.append("personal story open")

    score = min(100.0, score)
    reason = "; ".join(reasons) if reasons else "Weak opening hook"
    return {"scroll_stopping_hook": int(round(score)), "reason": reason}


def compute_signal_boosts(
    candidate_clip: dict,
    transcript_segments: list[dict],
) -> dict[str, Any]:
    """
    Compute all signal scores and a composite boost for ranking.
    Returns structured scores plus signal_boost (0-15 point additive boost).
    """
    text, window_segs = _clip_window_text(candidate_clip, transcript_segments)

    emotion = analyze_emotion_spikes(window_segs)
    pacing = analyze_pacing(window_segs)
    audience = analyze_audience_reactions(window_segs)
    curiosity = analyze_curiosity_gap(text)
    hook = analyze_scroll_stopping_hook(text)

    # Weighted signal average (0-100)
    weights = {
        "emotion_spike": 0.25,
        "pacing": 0.15,
        "audience_reaction": 0.15,
        "curiosity_gap": 0.20,
        "scroll_stopping_hook": 0.25,
    }
    scores = {
        "emotion_spike": emotion["emotion_spike"],
        "pacing": pacing["pacing"],
        "audience_reaction": audience["audience_reaction"],
        "curiosity_gap": curiosity["curiosity_gap"],
        "scroll_stopping_hook": hook["scroll_stopping_hook"],
    }
    weighted_avg = sum(scores[k] * weights[k] for k in weights)

    # Additive boost: 0-15 points on composite_score when signals are strong
    if weighted_avg >= 75:
        signal_boost = 15.0
    elif weighted_avg >= 60:
        signal_boost = 10.0
    elif weighted_avg >= 45:
        signal_boost = 5.0
    else:
        signal_boost = 0.0

    # Build reason from top signals
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [f"{k.replace('_', ' ')}={v}" for k, v in ranked[:2] if v >= 50]
    sub_reasons = [emotion.get("reason", ""), hook.get("reason", "")]
    reason_parts = top + [r for r in sub_reasons if r and r not in ("Low emotional intensity", "Weak opening hook")]
    reason = "; ".join(dict.fromkeys(reason_parts))[:200] if reason_parts else "Baseline signals"

    return {
        **scores,
        "signal_weighted_avg": int(round(weighted_avg)),
        "signal_boost": round(signal_boost, 1),
        "reason": reason,
    }


def apply_signal_boosts_to_clips(
    clips: list[dict],
    transcript_segments: list[dict],
    *,
    enabled: bool = True,
) -> list[dict]:
    """Apply signal scores to each clip. Preserves original composite_score."""
    if not enabled or not clips:
        return clips

    for clip in clips:
        signals = compute_signal_boosts(clip, transcript_segments)
        clip["signal_scores"] = signals
        original = float(clip.get("composite_score", 50))
        boost = float(signals.get("signal_boost", 0))
        clip["original_composite_score"] = int(original)
        clip["boosted_composite_score"] = int(min(100, round(original + boost)))
        if boost > 0:
            clip["composite_score"] = clip["boosted_composite_score"]

    logger.info("Applied signal boosts to %d clips", len(clips))
    return clips
