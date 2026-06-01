# -*- coding: utf-8 -*-
"""
clip_engine/local_candidate_discovery.py
Fast local pre-candidate extraction (no OpenAI).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clip_engine.discovery_forensics import DiscoveryForensics

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


def _window_bounds(
    style: ClipStyle | str,
    user_min: float,
    user_max: float,
    *,
    discovery_mode: bool = False,
) -> tuple[float, float, float]:
    profile = get_clip_style_profile(style, user_min_seconds=user_min, user_max_seconds=user_max)
    if discovery_mode:
        if style == "Micro clips":
            return max(15.0, user_min, 18.0), min(user_max, 90.0), 14.0
        if style == "Long story clips":
            return max(user_min, 60.0), min(user_max, 150.0), 28.0
        return max(18.0, user_min, 20.0), min(user_max, 110.0), 16.0
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
    discovery_mode: bool = False,
    forensics: DiscoveryForensics | None = None,
) -> list[dict]:
    """
    Generate 50-200 lightweight candidate windows from transcript heuristics.
    """
    if not segments or media_duration <= 0:
        if forensics:
            forensics.record_stage(
                "local_candidate_discovery",
                input_count=len(segments or []),
                output_count=0,
                rejection_reasons={"no_segments_or_duration": 1},
                note="missing transcript segments or media_duration",
            )
        return []

    win_min, win_max, step = _window_bounds(
        clip_style,
        user_min_seconds,
        user_max_seconds,
        discovery_mode=discovery_mode,
    )
    min_words = 5 if discovery_mode else 8
    sentences = merge_segments_into_sentences(
        segments,
        min_sentence_words=3 if discovery_mode else 4,
    )
    if len(sentences) < 1:
        from clip_engine.transcript_candidate_scanner import scan_transcript_candidates

        scanned, scan_stats = scan_transcript_candidates(
            segments,
            media_duration,
            discovery_mode=discovery_mode,
            user_min_seconds=user_min_seconds,
            user_max_seconds=user_max_seconds,
            max_candidates=max_candidates,
        )
        if forensics:
            forensics.merge_scan_stats(scan_stats.to_dict())
            forensics.record_stage(
                "local_sentence_merge_empty_scanner",
                input_count=len(segments),
                output_count=len(scanned),
                rejection_reasons=scan_stats.rejection_reasons,
                note="no merged sentences — used rolling scanner",
            )
        return scanned

    pauses = detect_pauses(segments, min_pause_seconds=0.75)
    pause_times = {round(p.after_seconds, 2) for p in pauses}

    reject_reasons: dict[str, int] = {}
    windows_scanned = 0
    hit_totals = {"emotion_hits": 0, "curiosity_hits": 0, "story_turn_hits": 0, "trauma_hits": 0, "keyword_hits": 0}

    def _rej(reason: str) -> None:
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    raw_windows: list[dict] = []
    i = 0
    while i < len(sentences):
        windows_scanned += 1
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
        min_dur = win_min * (0.75 if discovery_mode else 0.85)
        if dur < min_dur:
            _rej("duration_below_min")
        elif dur > win_max * 1.12:
            _rej("duration_above_max")
        elif dur >= min_dur and dur <= win_max * 1.12:
            text = " ".join(texts).strip()
            if len(text.split()) < min_words:
                _rej("too_few_words")
            else:
                from clip_engine.discovery_forensics import count_lexicon_hits

                hits = count_lexicon_hits(text)
                for k, v in hits.items():
                    hit_totals[k] = hit_totals.get(k, 0) + v
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
            _rej("dedupe_start_overlap")
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

    if len(kept) < (15 if discovery_mode else 8):
        from clip_engine.transcript_candidate_scanner import scan_transcript_candidates

        extra, scan_stats = scan_transcript_candidates(
            segments,
            media_duration,
            discovery_mode=discovery_mode,
            user_min_seconds=user_min_seconds,
            user_max_seconds=user_max_seconds,
            max_candidates=max(60, max_candidates - len(kept)),
            existing=kept,
            min_gap_seconds=14.0 if discovery_mode else 22.0,
        )
        if forensics:
            forensics.merge_scan_stats(scan_stats.to_dict())
            forensics.record_stage(
                "local_discovery_scanner_supplement",
                input_count=len(kept),
                output_count=len(kept) + len(extra),
                rejected_count=scan_stats.windows_rejected,
                rejection_reasons=scan_stats.rejection_reasons,
            )
        if extra:
            logger.info(
                "[LOCAL DISCOVERY] transcript scanner added %d windows (had %d)",
                len(extra),
                len(kept),
            )
            kept.extend(extra)
            kept.sort(key=lambda x: int(x.get("composite_score", 0)), reverse=True)
            kept = kept[:max_candidates]

    if forensics:
        forensics.windows_scanned += windows_scanned
        forensics.windows_rejected += sum(reject_reasons.values())
        forensics.emotion_hits += hit_totals.get("emotion_hits", 0)
        forensics.curiosity_hits += hit_totals.get("curiosity_hits", 0)
        forensics.story_turn_hits += hit_totals.get("story_turn_hits", 0)
        forensics.trauma_hits += hit_totals.get("trauma_hits", 0)
        forensics.keyword_hits += hit_totals.get("keyword_hits", 0)
        forensics.record_stage(
            "local_candidate_discovery",
            input_count=windows_scanned,
            output_count=len(kept),
            rejected_count=sum(reject_reasons.values()),
            rejection_reasons=reject_reasons,
            note=f"raw_windows={len(raw_windows)} sentences={len(sentences)}",
        )

    logger.info(
        "Local pre-candidates: %d windows (max=%d, discovery_mode=%s, scanned=%d rejected=%d)",
        len(kept),
        max_candidates,
        discovery_mode,
        windows_scanned,
        sum(reject_reasons.values()),
    )
    return kept
