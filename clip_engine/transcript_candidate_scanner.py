# -*- coding: utf-8 -*-
"""
clip_engine/transcript_candidate_scanner.py

High-recall local candidate discovery from transcript windows.
Used when GPU prefilter is empty or the candidate pool is starved.
No OpenAI calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from clip_engine.clip_signals import (
    AUDIENCE_REACTIONS,
    CURIOSITY_HOOKS,
    EMOTION_WORDS,
    analyze_curiosity_gap,
    analyze_emotion_spikes,
    analyze_pacing,
    analyze_scroll_stopping_hook,
    compute_signal_boosts,
)
from clip_engine.transcription_utils import merge_segments_into_sentences

logger = logging.getLogger("clip_engine.transcript_candidate_scanner")

STORY_TURN_PHRASES = (
    "then he shot",
    "then she shot",
    "i remember",
    "i'll never forget",
    "i was terrified",
    "we almost died",
    "she killed herself",
    "he killed himself",
    "i never told anyone",
    "that changed everything",
    "the doctor said",
    "i thought i was going to die",
    "my mom",
    "my dad",
    "my mother",
    "my father",
    "childhood",
    "alcoholic",
    "addiction",
    "abuse",
    "trauma",
    "suicide",
    "divorce",
    "cancer",
    "miscarriage",
    "miscarried",
    "overdose",
    "overdosed",
    "homeless",
    "prison",
    "jail",
    "arrested",
    "cheated",
    "affair",
    "betrayed",
    "abandoned",
    "orphan",
    "foster",
    "adopted",
    "rape",
    "assault",
    "molest",
    "beaten",
    "abusive",
    "drunk",
    "sober",
    "rehab",
    "relapse",
    "panic attack",
    "breakdown",
    "mental health",
    "depression",
    "anxiety",
)

CONFLICT_PHRASES = (
    "fight",
    "argued",
    "screaming",
    "yelling",
    "threatened",
    "threatening",
    "police",
    "court",
    "lawsuit",
    "divorce",
    "custody",
    "restraining order",
)

TRAUMA_PHRASES = (
    "trauma",
    "ptsd",
    "triggered",
    "flashback",
    "nightmare",
    "nightmares",
    "grief",
    "mourning",
    "funeral",
    "hospital",
    "icu",
    "coma",
    "life support",
    "widow",
    "orphan",
)

ROLLING_DURATIONS_DISCOVERY = (20.0, 35.0, 50.0, 75.0)
ROLLING_DURATIONS_NORMAL = (25.0, 40.0, 55.0, 75.0)
ROLLING_STEP_SECONDS = 12.0
MIN_WORDS_DISCOVERY = 5
MIN_WORDS_NORMAL = 6
MIN_SCORE_DISCOVERY = 28
MIN_SCORE_NORMAL = 35

TOKEN_RE = re.compile(r"\b[a-z]{3,}\b")


@dataclass
class DiscoveryScanStats:
    windows_scanned: int = 0
    windows_kept: int = 0
    windows_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    emotion_triggers: int = 0
    curiosity_triggers: int = 0
    pacing_triggers: int = 0
    hook_triggers: int = 0
    story_phrase_triggers: int = 0
    conflict_triggers: int = 0
    trauma_triggers: int = 0
    keyword_hits: int = 0
    fallback_generated: int = 0
    transcript_only_candidates: int = 0
    discovery_boost_activations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows_scanned": self.windows_scanned,
            "windows_kept": self.windows_kept,
            "windows_rejected": self.windows_rejected,
            "rejection_reasons": dict(self.rejection_reasons),
            "emotion_triggers": self.emotion_triggers,
            "curiosity_triggers": self.curiosity_triggers,
            "pacing_triggers": self.pacing_triggers,
            "hook_triggers": self.hook_triggers,
            "story_phrase_triggers": self.story_phrase_triggers,
            "conflict_triggers": self.conflict_triggers,
            "trauma_triggers": self.trauma_triggers,
            "keyword_hits": self.keyword_hits,
            "fallback_generated": self.fallback_generated,
            "transcript_only_candidates": self.transcript_only_candidates,
            "discovery_boost_activations": self.discovery_boost_activations,
        }


def _record_reject(stats: DiscoveryScanStats, reason: str) -> None:
    stats.windows_rejected += 1
    stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1


def _assign_region(mid: float, media_duration: float) -> str:
    if media_duration <= 0:
        return "middle"
    names = ["beginning", "early_middle", "middle", "late_middle", "ending"]
    idx = min(len(names) - 1, int(mid / media_duration * len(names)))
    return names[idx]


def _phrase_hits(lower: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for p in phrases if p in lower)


def _score_window(
    text: str,
    window_segs: list[dict],
    *,
    stats: DiscoveryScanStats,
    discovery_mode: bool,
) -> float:
    lower = text.lower()
    score = 32.0 if discovery_mode else 38.0

    story_hits = _phrase_hits(lower, STORY_TURN_PHRASES)
    conflict_hits = _phrase_hits(lower, CONFLICT_PHRASES)
    trauma_hits = _phrase_hits(lower, TRAUMA_PHRASES)
    if story_hits:
        score += 12 + story_hits * 4
        stats.story_phrase_triggers += 1
        stats.keyword_hits += story_hits
    if conflict_hits:
        score += 8 + conflict_hits * 3
        stats.conflict_triggers += 1
        stats.keyword_hits += conflict_hits
    if trauma_hits:
        score += 10 + trauma_hits * 3
        stats.trauma_triggers += 1
        stats.keyword_hits += trauma_hits

    words = set(TOKEN_RE.findall(lower))
    emo_word_hits = len(words & EMOTION_WORDS)
    if emo_word_hits:
        score += 14
        stats.emotion_triggers += 1
        stats.keyword_hits += emo_word_hits
    cur_phrase_hits = sum(1 for p in CURIOSITY_HOOKS if p in lower)
    if cur_phrase_hits:
        score += 10
        stats.curiosity_triggers += 1
        stats.keyword_hits += cur_phrase_hits
    if any(p in lower for p in AUDIENCE_REACTIONS):
        score += 6

    emo = analyze_emotion_spikes(window_segs)
    pace = analyze_pacing(window_segs)
    cur = analyze_curiosity_gap(text)
    hook = analyze_scroll_stopping_hook(text)
    if emo.get("emotion_spike", 0) >= 40:
        stats.emotion_triggers += 1
    if pace.get("pacing", 0) >= 45:
        stats.pacing_triggers += 1
    if cur.get("curiosity_gap", 0) >= 40:
        stats.curiosity_triggers += 1
    if hook.get("scroll_stopping_hook", 0) >= 40:
        stats.hook_triggers += 1

    score += emo.get("emotion_spike", 0) * 0.14
    score += pace.get("pacing", 0) * 0.10
    score += cur.get("curiosity_gap", 0) * 0.12
    score += hook.get("scroll_stopping_hook", 0) * 0.14

    if discovery_mode:
        score += 6
        stats.discovery_boost_activations += 1

    return min(100.0, score)


def _infer_title(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    if len(words) < 4:
        return "Key Moment From The Conversation"
    title = " ".join(words[:9]).strip(".,;:!? ")
    return title[0].upper() + title[1:] if title else "Key Moment From The Conversation"


def _overlaps_existing(
    t0: float,
    t1: float,
    existing: list[tuple[float, float]],
    gap: float,
) -> bool:
    for e0, e1 in existing:
        if t0 < e1 + gap and e0 < t1 + gap:
            return True
    return False


def scan_transcript_candidates(
    segments: list[dict],
    media_duration: float,
    *,
    discovery_mode: bool = True,
    user_min_seconds: float = 15.0,
    user_max_seconds: float = 120.0,
    max_candidates: int = 80,
    existing: list[dict] | None = None,
    min_gap_seconds: float = 18.0,
) -> tuple[list[dict], DiscoveryScanStats]:
    """
    Rolling overlapping transcript windows at multiple durations.
    Prioritizes recall — ranking happens later in the pipeline.
    """
    stats = DiscoveryScanStats()
    if not segments or media_duration <= 0:
        return [], stats

    min_words = MIN_WORDS_DISCOVERY if discovery_mode else MIN_WORDS_NORMAL
    min_score = MIN_SCORE_DISCOVERY if discovery_mode else MIN_SCORE_NORMAL
    durations = list(ROLLING_DURATIONS_DISCOVERY if discovery_mode else ROLLING_DURATIONS_NORMAL)
    durations = [
        d for d in durations
        if d >= max(12.0, user_min_seconds * 0.85) and d <= min(user_max_seconds * 1.05, media_duration)
    ]
    if not durations:
        durations = [max(20.0, user_min_seconds), min(50.0, user_max_seconds)]

    sentences = merge_segments_into_sentences(
        segments,
        min_sentence_words=3 if discovery_mode else 4,
    )
    if len(sentences) < 1:
        return [], stats

    existing_ranges: list[tuple[float, float]] = []
    for c in existing or []:
        try:
            existing_ranges.append(
                (
                    float(c.get("start_seconds", c.get("start", 0))),
                    float(c.get("end_seconds", c.get("end", 0))),
                )
            )
        except (TypeError, ValueError):
            continue

    candidates: list[dict] = []
    gap = max(12.0, min_gap_seconds * 0.55) if discovery_mode else min_gap_seconds

    t = 0.0
    while t < media_duration - min(durations):
        for target_dur in durations:
            stats.windows_scanned += 1
            t1 = min(t + target_dur, media_duration)
            if t1 - t < user_min_seconds * 0.75:
                _record_reject(stats, "too_short")
                continue

            window_sents = [
                s
                for s in sentences
                if float(s.get("end", 0)) > t and float(s.get("start", 0)) < t1
            ]
            if not window_sents:
                _record_reject(stats, "no_sentences_in_window")
                continue

            s0 = float(window_sents[0].get("start", t))
            s1 = float(window_sents[-1].get("end", t1))
            dur = s1 - s0
            if dur < user_min_seconds * 0.7:
                _record_reject(stats, "duration_below_min")
                continue
            if dur > user_max_seconds * 1.08:
                _record_reject(stats, "duration_above_max")
                continue

            if _overlaps_existing(s0, s1, existing_ranges, gap):
                _record_reject(stats, "overlap_existing")
                continue
            if _overlaps_existing(
                s0,
                s1,
                [(float(c["start_seconds"]), float(c["end_seconds"])) for c in candidates],
                gap * 0.45,
            ):
                _record_reject(stats, "overlap_candidate")
                continue

            text = " ".join(str(s.get("text", "")).strip() for s in window_sents).strip()
            if len(text.split()) < min_words:
                _record_reject(stats, "too_few_words")
                continue

            energy = _score_window(text, window_sents, stats=stats, discovery_mode=discovery_mode)
            if energy < min_score:
                _record_reject(stats, "low_score")
                continue

            mid = (s0 + s1) / 2
            clip: dict[str, Any] = {
                "start_seconds": round(s0, 3),
                "end_seconds": round(s1, 3),
                "hook_title": _infer_title(text)[:72],
                "composite_score": int(round(energy)),
                "local_rank_score": round(energy, 1),
                "selection_reason": "Transcript rolling-window discovery.",
                "ai_context_reason": "High-recall local scan (emotion/story/pacing heuristics).",
                "dominant_signal": "emotional",
                "source": "transcript_scanner",
                "confidence": 0.58 if discovery_mode else 0.52,
                "_region": _assign_region(mid, media_duration),
                "_pass": "transcript_scan",
                "warnings": [],
            }
            boosts = compute_signal_boosts(clip, segments)
            clip["composite_score"] = min(
                88,
                int(clip["composite_score"]) + int(boosts.get("signal_boost", 0)),
            )
            clip["local_signals"] = {
                k: boosts.get(k)
                for k in (
                    "emotion_spike",
                    "pacing",
                    "curiosity_gap",
                    "scroll_stopping_hook",
                    "audience_reaction",
                )
            }
            candidates.append(clip)
            existing_ranges.append((s0, s1))
            stats.windows_kept += 1
            stats.transcript_only_candidates += 1

        t += ROLLING_STEP_SECONDS

    candidates.sort(key=lambda x: int(x.get("composite_score", 0)), reverse=True)
    out = candidates[:max_candidates]
    stats.fallback_generated = len(out)
    logger.info(
        "[DISCOVERY SCAN] scanned=%d kept=%d rejected=%d emotion=%d curiosity=%d story=%d",
        stats.windows_scanned,
        len(out),
        stats.windows_rejected,
        stats.emotion_triggers,
        stats.curiosity_triggers,
        stats.story_phrase_triggers,
    )
    return out, stats


__all__ = [
    "DiscoveryScanStats",
    "scan_transcript_candidates",
    "STORY_TURN_PHRASES",
]
