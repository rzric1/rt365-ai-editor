# -*- coding: utf-8 -*-
"""
clip_engine/clip_discovery.py
Discovery mode: relaxed validation, candidate rescue, and local transcript-window fallbacks.
No extra OpenAI calls for local fallback generation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clip_engine.discovery_forensics import DiscoveryForensics

from clip_engine.clip_signals import compute_signal_boosts
from clip_engine.clip_split_parts import flag_split_recommended
from clip_engine.clip_style import ClipStyle, ClipStyleProfile
from clip_engine.transcription_utils import extract_transcript_window, merge_segments_into_sentences

logger = logging.getLogger("clip_engine.clip_discovery")

HARD_MIN_DURATION_SECONDS = 12.0


def empty_pool_stats() -> dict[str, int]:
    return {
        "raw_ai_candidates": 0,
        "valid_after_schema": 0,
        "rescued_candidates": 0,
        "local_fallback_candidates": 0,
        "rejected_invalid_time": 0,
        "rejected_duration": 0,
        "rejected_empty_transcript": 0,
        "rejected_overlap_early": 0,
    }


def _clip_range(c: dict) -> tuple[float, float]:
    t0 = float(c.get("start_seconds", c.get("start", 0)))
    t1 = float(c.get("end_seconds", c.get("end", t0)))
    return t0, t1


def _infer_title_from_text(text: str, max_words: int = 8) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        words = sent.split()
        if len(words) >= 4:
            title = " ".join(words[:max_words]).strip(".,!?")
            if title:
                return title[:60]
    words = text.split()
    return " ".join(words[:max_words]) if words else "Clip moment"


def validate_clip_candidate(
    c: dict,
    *,
    min_clip_seconds: float,
    max_clip_seconds: float,
    min_score: int,
    discovery_mode: bool,
    segments: list[dict] | None = None,
    media_duration: float = 0.0,
) -> tuple[list[str], list[str]]:
    """Return (hard_issues, soft_issues). Hard issues mean reject; soft can be rescued/warned."""
    hard: list[str] = []
    soft: list[str] = []
    t0, t1 = _clip_range(c)
    dur = t1 - t0

    if t1 <= t0:
        hard.append("end <= start")
    if t0 < 0:
        hard.append("negative start")
    if media_duration > 0 and t0 > media_duration + 1.0:
        hard.append("start beyond media duration")

    hard_floor = max(HARD_MIN_DURATION_SECONDS, min_clip_seconds * (0.8 if discovery_mode else 1.0))
    if dur < hard_floor:
        hard.append(f"Too short: {dur:.1f}s")
    max_allowed = max_clip_seconds * (1.08 if discovery_mode else 1.0)
    if dur > max_allowed and not c.get("split_recommended"):
        hard.append(f"Too long: {dur:.1f}s")

    if segments:
        window = extract_transcript_window(segments, t0, t1)
        if len(window.split()) < 3:
            hard.append("Empty transcript window")

    score = int(c.get("composite_score", 0) or 0)
    score_floor = min_score if not discovery_mode else max(35, min_score - 18)
    if score < score_floor:
        soft.append(f"Low composite_score {score} < {score_floor}")
    elif score < min_score:
        soft.append(f"Below preferred score {score} < {min_score}")

    if not str(c.get("hook_title", "")).strip():
        soft.append("Missing hook_title")
    if not str(c.get("selection_reason", "")).strip():
        soft.append("Missing selection_reason")
    if not str(c.get("dominant_signal", "")).strip():
        soft.append("Missing dominant_signal")

    return hard, soft


def rescue_clip_candidate(
    c: dict,
    *,
    min_clip_seconds: float,
    max_clip_seconds: float,
    segments: list[dict] | None,
    media_duration: float,
) -> dict:
    """Repair borderline candidates: clamp times, defaults, inferred title."""
    out = dict(c)
    t0, t1 = _clip_range(out)
    if media_duration > 0:
        t0 = max(0.0, min(t0, media_duration - 1.0))
        t1 = max(t0 + 1.0, min(t1, media_duration))
    if t1 <= t0:
        t1 = t0 + max(min_clip_seconds, HARD_MIN_DURATION_SECONDS)

    dur = t1 - t0
    if dur < min_clip_seconds:
        t1 = min(t0 + min_clip_seconds, media_duration if media_duration > 0 else t0 + min_clip_seconds)
    if dur > max_clip_seconds and not out.get("split_recommended"):
        t1 = t0 + max_clip_seconds

    out["start_seconds"] = round(t0, 3)
    out["end_seconds"] = round(t1, 3)

    window = ""
    if segments:
        window = extract_transcript_window(segments, t0, t1)
    if not str(out.get("hook_title", "")).strip():
        out["hook_title"] = _infer_title_from_text(window or str(out.get("selection_reason", "")))
    if not str(out.get("selection_reason", "")).strip():
        out["selection_reason"] = f"Key moment: {out.get('hook_title', 'Clip')[:80]}"
    if not str(out.get("ai_context_reason", "")).strip():
        out["ai_context_reason"] = "Standalone window with clear spoken content."
    if not str(out.get("dominant_signal", "")).strip():
        out["dominant_signal"] = "educational"
    if not out.get("platform_fit"):
        out["platform_fit"] = ["TikTok", "YouTube Shorts"]
    if not out.get("caption_style"):
        out["caption_style"] = "Bold Viral"
    if int(out.get("composite_score", 0) or 0) < 48:
        out["composite_score"] = max(48, int(out.get("composite_score", 0) or 0))

    out.setdefault("warnings", [])
    if "rescued_candidate" not in out["warnings"]:
        out["warnings"].append("rescued_candidate")
    return out


def process_raw_clips(
    clips_raw: list[dict],
    *,
    min_clip_seconds: float,
    max_clip_seconds: float,
    min_score: int,
    discovery_mode: bool,
    segments: list[dict] | None,
    media_duration: float,
    region_label: str = "",
    pass_name: str = "",
) -> tuple[list[dict], dict[str, int]]:
    """Validate/rescue raw AI clips. Returns (accepted, region_stats)."""
    stats = {
        "raw": len(clips_raw),
        "valid": 0,
        "rescued": 0,
        "rejected_invalid_time": 0,
        "rejected_duration": 0,
        "rejected_empty_transcript": 0,
    }
    accepted: list[dict] = []

    for raw in clips_raw:
        c = dict(raw)
        c["_region"] = region_label or c.get("_region", "")
        c["_pass"] = pass_name or c.get("_pass", "")

        flag_split_recommended(c, max_clip_seconds)

        rescued = rescue_clip_candidate(
            c,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
            segments=segments,
            media_duration=media_duration,
        )
        was_rescued = rescued != c or "rescued_candidate" in rescued.get("warnings", [])
        hard, soft = validate_clip_candidate(
            rescued,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
            min_score=min_score,
            discovery_mode=discovery_mode,
            segments=segments,
            media_duration=media_duration,
        )

        if hard:
            from clip_engine.telemetry import log_clip_reject

            clip_id = f"{region_label}_{rescued.get('hook_title', '')[:24]}"
            for issue in hard:
                if "end <=" in issue or "negative" in issue or "beyond media" in issue:
                    stats["rejected_invalid_time"] += 1
                    log_clip_reject(
                        "invalid_time",
                        issue=issue,
                        candidate_clip=clip_id,
                        region=region_label,
                    )
                elif "Too short" in issue:
                    stats["rejected_duration"] += 1
                    dur = float(rescued.get("end_seconds", 0)) - float(
                        rescued.get("start_seconds", 0)
                    )
                    log_clip_reject(
                        "duration_too_short",
                        duration=round(dur, 1),
                        minimum=min_clip_seconds,
                        candidate_clip=clip_id,
                    )
                elif "Too long" in issue:
                    stats["rejected_duration"] += 1
                    dur = float(rescued.get("end_seconds", 0)) - float(
                        rescued.get("start_seconds", 0)
                    )
                    log_clip_reject(
                        "duration_too_long",
                        duration=round(dur, 1),
                        maximum=max_clip_seconds,
                        candidate_clip=clip_id,
                    )
                elif "Empty transcript" in issue:
                    stats["rejected_empty_transcript"] += 1
                    log_clip_reject(
                        "empty_transcript",
                        candidate_clip=clip_id,
                        region=region_label,
                    )
            logger.debug("Hard reject region=%s: %s (%s)", region_label, hard, rescued.get("hook_title"))
            continue

        if soft and not discovery_mode:
            # Non-discovery: still accept rescued clips with soft issues as warnings
            rescued.setdefault("warnings", []).extend(soft)

        if was_rescued or soft:
            stats["rescued"] += 1
            rescued.setdefault("warnings", []).extend(soft)

        accepted.append(rescued)
        stats["valid"] += 1

    return accepted, stats


def _ranges_overlap(a: tuple[float, float], b: tuple[float, float], gap: float) -> bool:
    return a[0] < b[1] + gap and b[0] < a[1] + gap


def _window_duration_bounds(style: ClipStyle) -> tuple[float, float, float]:
    """Return (ideal_min, ideal_max, step_seconds) for local windows."""
    if style == "Micro clips":
        return 30.0, 75.0, 28.0
    if style == "Long story clips":
        return 90.0, 150.0, 45.0
    return 45.0, 100.0, 35.0


def generate_local_fallback_candidates(
    segments: list[dict],
    media_duration: float,
    *,
    clip_style: ClipStyle | str,
    profile: ClipStyleProfile,
    target_count: int,
    existing: list[dict],
    min_gap_seconds: float = 35.0,
    user_min_seconds: float = 15.0,
    user_max_seconds: float = 90.0,
    discovery_mode: bool = False,
    forensics: DiscoveryForensics | None = None,
) -> list[dict]:
    """
    Build additional candidates from transcript sentence windows using local signal scoring.
    """
    if not segments or media_duration <= 0:
        if forensics:
            forensics.record_stage(
                "local_fallback",
                input_count=len(segments or []),
                output_count=0,
                rejection_reasons={"no_segments_or_duration": 1},
            )
        return []

    from clip_engine.transcript_candidate_scanner import scan_transcript_candidates

    scanned, scan_stats = scan_transcript_candidates(
        segments,
        media_duration,
        discovery_mode=discovery_mode,
        user_min_seconds=user_min_seconds,
        user_max_seconds=min(user_max_seconds, profile.ai_max_clip_seconds),
        max_candidates=max(target_count * 3, 40),
        existing=existing,
        min_gap_seconds=max(14.0, min_gap_seconds * 0.6) if discovery_mode else min_gap_seconds,
    )
    if scanned:
        if forensics:
            forensics.merge_scan_stats(scan_stats.to_dict())
            forensics.fallback_candidates_generated = len(scanned)
            forensics.record_stage(
                "local_fallback_scanner",
                input_count=len(existing),
                output_count=len(scanned),
                rejected_count=scan_stats.windows_rejected,
                rejection_reasons=scan_stats.rejection_reasons,
            )
        logger.info(
            "Local fallback (transcript scanner): %d candidates (discovery_mode=%s)",
            len(scanned),
            discovery_mode,
        )
        return scanned

    style = clip_style if clip_style in ("Balanced", "Micro clips", "Long story clips") else "Balanced"
    win_min, win_max, step = _window_duration_bounds(style)  # type: ignore[arg-type]
    if discovery_mode:
        win_min = max(15.0, user_min_seconds, 18.0)
        win_max = min(user_max_seconds, win_max, profile.ai_max_clip_seconds)
        step = min(step, 16.0)
    else:
        win_min = max(user_min_seconds, win_min)
        win_max = min(user_max_seconds, win_max, profile.ai_max_clip_seconds)

    sentences = merge_segments_into_sentences(
        segments,
        min_sentence_words=3 if discovery_mode else 4,
    )
    if not sentences:
        if forensics:
            forensics.record_stage(
                "local_fallback_sentence_windows",
                input_count=len(segments),
                output_count=0,
                rejection_reasons={"no_sentences": 1},
            )
        return []

    existing_ranges = [_clip_range(c) for c in existing]
    candidates: list[dict] = []
    fb_rejects: dict[str, int] = {}
    windows_scanned = 0

    t = float(sentences[0].get("start", 0))
    while t < media_duration - win_min:
        windows_scanned += 1
        end_target = min(t + win_max, media_duration)
        end_min = t + win_min

        window_sents = [
            s for s in sentences
            if float(s.get("end", 0)) > t and float(s.get("start", 0)) < end_target
        ]
        if not window_sents:
            fb_rejects["no_sentences_in_window"] = fb_rejects.get("no_sentences_in_window", 0) + 1
            t += step
            continue

        s0 = float(window_sents[0].get("start", t))
        s1 = float(window_sents[-1].get("end", end_target))
        dur = s1 - s0
        if dur < win_min:
            fb_rejects["duration_below_min"] = fb_rejects.get("duration_below_min", 0) + 1
            t += step * 0.5
            continue
        if dur > win_max * 1.05:
            fb_rejects["duration_above_max"] = fb_rejects.get("duration_above_max", 0) + 1
            t += step
            continue

        r = (s0, s1)
        if any(_ranges_overlap(r, er, min_gap_seconds) for er in existing_ranges):
            fb_rejects["overlap_existing"] = fb_rejects.get("overlap_existing", 0) + 1
            t += step
            continue
        if any(_ranges_overlap(r, _clip_range(c), min_gap_seconds * 0.5) for c in candidates):
            fb_rejects["overlap_candidate"] = fb_rejects.get("overlap_candidate", 0) + 1
            t += step
            continue

        text = " ".join(str(s.get("text", "")).strip() for s in window_sents).strip()
        if len(text.split()) < (5 if discovery_mode else 8):
            fb_rejects["too_few_words"] = fb_rejects.get("too_few_words", 0) + 1
            t += step
            continue

        clip: dict[str, Any] = {
            "start_seconds": round(s0, 3),
            "end_seconds": round(s1, 3),
            "hook_title": _infer_title_from_text(text),
            "composite_score": 55,
            "selection_reason": "Locally scored transcript window.",
            "ai_context_reason": "Fallback window selected by local signal heuristics.",
            "dominant_signal": "educational",
            "caption_style": "Bold Viral",
            "platform_fit": ["TikTok", "YouTube Shorts"],
            "warnings": ["locally generated fallback candidate"],
            "source": "local_transcript_window",
            "confidence": 0.55,
            "_region": "local_fallback",
            "_pass": "local",
        }

        signals = compute_signal_boosts(clip, segments)
        local_score = int(signals.get("signal_weighted_avg", 50))
        clip["composite_score"] = max(48, min(78, local_score))
        clip["scores"] = {
            "hook_strength": signals.get("scroll_stopping_hook", 50),
            "emotional_intensity": signals.get("emotion_spike", 50),
            "retention_potential": signals.get("curiosity_gap", 50),
            "standalone_clarity": signals.get("pacing", 50),
        }
        clip["local_signal_reason"] = signals.get("reason", "")

        candidates.append(clip)
        existing_ranges.append(r)
        t += step

    candidates.sort(key=lambda x: int(x.get("composite_score", 0)), reverse=True)
    need = max(target_count * 2 - len(existing), target_count - len(existing), 0)
    need = min(need, max(target_count, 15))
    selected = candidates[:need]
    if forensics:
        forensics.windows_scanned += windows_scanned
        forensics.windows_rejected += sum(fb_rejects.values())
        forensics.fallback_candidates_generated = len(selected)
        forensics.record_stage(
            "local_fallback_sentence_windows",
            input_count=windows_scanned,
            output_count=len(selected),
            rejected_count=sum(fb_rejects.values()),
            rejection_reasons=fb_rejects,
        )
    logger.info(
        "Local fallback generated %d candidates (scanned=%d rejected=%d)",
        len(selected),
        windows_scanned,
        sum(fb_rejects.values()),
    )
    return selected


def merge_pool_stats(target: dict[str, int], addition: dict[str, int]) -> None:
    for k, v in addition.items():
        if k in target:
            target[k] += int(v)
        else:
            target[k] = int(v)
