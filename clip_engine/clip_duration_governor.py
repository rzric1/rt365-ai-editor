# -*- coding: utf-8 -*-
"""
clip_engine/clip_duration_governor.py
Central duration policy: target 30–90s, soft cap 90s, hard cap 120s.
High-virality clips (score > 90) may use the full hard cap; others stay at soft cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("clip_engine.clip_duration_governor")

TARGET_MIN_SECONDS = 30.0
TARGET_MAX_SECONDS = 90.0
SOFT_CAP_SECONDS = 90.0
HARD_CAP_SECONDS = 120.0
VIRALITY_HARD_CAP_EXCEPTION = 90
MAX_GROWTH_PERCENT = 100.0


@dataclass
class DurationGovernorStats:
    checked: int = 0
    clamped_soft: int = 0
    clamped_hard: int = 0
    clamped_growth: int = 0
    over_soft_before: int = 0
    over_hard_before: int = 0
    over_growth_limit_before: int = 0
    justified_over_soft: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "clamped_soft": self.clamped_soft,
            "clamped_hard": self.clamped_hard,
            "clamped_growth": self.clamped_growth,
            "over_soft_before": self.over_soft_before,
            "over_hard_before": self.over_hard_before,
            "over_growth_limit_before": self.over_growth_limit_before,
            "justified_over_soft": self.justified_over_soft,
            "notes": self.notes[:20],
        }


def clip_virality_score(clip: dict) -> int:
    return int(clip.get("virality_score", clip.get("composite_score", 0)) or 0)


def effective_max_duration(clip: dict) -> float:
    """Max export duration for this clip (soft 90 or hard 120 if viral)."""
    if clip_virality_score(clip) > VIRALITY_HARD_CAP_EXCEPTION:
        return HARD_CAP_SECONDS
    return SOFT_CAP_SECONDS


def _core_bounds(clip: dict) -> tuple[float, float]:
    """Immutable AI-core window (pinned before any expansion)."""
    o0 = float(clip.get("ai_core_start", clip.get("original_start", 0)))
    o1 = float(clip.get("ai_core_end", clip.get("original_end", o0)))
    return o0, o1


def pin_ai_core_window(clip: dict) -> dict:
    """
    Freeze the AI-core timestamps once, before context/boundary/finalizer expansion.
    Never overwrites ai_core_* if already set.
    """
    c = dict(clip)
    t0 = float(c.get("start_seconds", c.get("start", 0)))
    t1 = float(c.get("end_seconds", c.get("end", t0)))
    if "ai_core_start" not in c:
        c["ai_core_start"] = round(t0, 3)
        c["ai_core_end"] = round(t1, 3)
    c["original_start"] = float(c["ai_core_start"])
    c["original_end"] = float(c["ai_core_end"])
    if "merge_source_count" not in c:
        c["merge_source_count"] = 1
    return c


def ensure_expansion_baseline(clip: dict) -> dict:
    """Preserve AI-core window; sync original_* from pinned core when present."""
    c = dict(clip)
    if "ai_core_start" in c:
        c["original_start"] = float(c["ai_core_start"])
        c["original_end"] = float(c["ai_core_end"])
    else:
        t0 = float(c.get("start_seconds", c.get("start", 0)))
        t1 = float(c.get("end_seconds", c.get("end", t0)))
        if "original_start" not in c:
            c["original_start"] = round(t0, 3)
        if "original_end" not in c:
            c["original_end"] = round(t1, 3)
    if "merge_source_count" not in c:
        c["merge_source_count"] = 1
    return c


def max_allowed_span_for_core(
    clip: dict,
    core_duration: float,
    duration_cap: float,
    *,
    pre_virality: bool = False,
) -> float:
    """Max export span from core: duration cap and 100% growth rule."""
    if core_duration <= 0.01:
        return duration_cap
    growth_cap = core_duration * (1.0 + MAX_GROWTH_PERCENT / 100.0)
    if pre_virality or not growth_over_100_allowed(clip):
        return min(duration_cap, growth_cap)
    return duration_cap


def growth_over_100_allowed(clip: dict) -> bool:
    """Allow >100% growth only with high virality and explicit justification flag."""
    if clip_virality_score(clip) <= VIRALITY_HARD_CAP_EXCEPTION:
        return False
    return bool(clip.get("expansion_over_100_approved"))


def build_expansion_reason(clip: dict, actions: list[str] | None = None) -> str:
    """Structured expansion_reason for diagnostics and logs."""
    parts: list[str] = []
    if actions:
        parts.append("actions=" + ",".join(actions))
    for key in (
        "expansion_note",
        "finalizer_action",
        "finalizer_reason",
        "boundary_status",
    ):
        val = clip.get(key)
        if val:
            parts.append(f"{key}={val}")
    dur = float(clip.get("expanded_duration", clip.get("duration", 0)))
    core = float(clip.get("original_duration", 0))
    growth_pct = float(clip.get("growth_percent", 0))
    if dur > SOFT_CAP_SECONDS + 0.5:
        parts.append(
            f"over_soft_cap({dur:.0f}s core={core:.0f}s growth={growth_pct:.0f}%)"
        )
    if growth_pct > MAX_GROWTH_PERCENT + 0.5 and not growth_over_100_allowed(clip):
        parts.append(f"growth_exceeds_{MAX_GROWTH_PERCENT:.0f}pct_unapproved")
    if clip_virality_score(clip) > VIRALITY_HARD_CAP_EXCEPTION:
        parts.append(f"virality_{clip_virality_score(clip)}_hard_cap_ok")
    return "; ".join(parts) if parts else "within_policy"


def refresh_expansion_diagnostics(clip: dict) -> dict:
    """Compute expanded_* and growth_* from pinned AI core vs current window."""
    c = ensure_expansion_baseline(clip)
    o0, o1 = _core_bounds(c)
    c["original_start"] = round(o0, 3)
    c["original_end"] = round(o1, 3)
    e0 = float(c.get("start_seconds", c.get("start", o0)))
    e1 = float(c.get("end_seconds", c.get("end", o1)))
    c["expanded_start"] = round(e0, 3)
    c["expanded_end"] = round(e1, 3)
    core = max(0.01, o1 - o0)
    export_dur = max(0.0, e1 - e0)
    growth = max(0.0, export_dur - core)
    c["original_duration"] = round(core, 3)
    c["expanded_duration"] = round(export_dur, 3)
    c["growth_seconds"] = round(growth, 2)
    c["growth_percent"] = round(100.0 * growth / core, 1)
    c["duration"] = round(export_dur, 3)
    c["merge_source_count"] = int(c.get("merge_source_count", 1) or 1)
    reasons = []
    if export_dur > SOFT_CAP_SECONDS + 0.5:
        reasons.append("over_90s")
    if c["growth_percent"] > MAX_GROWTH_PERCENT + 0.5:
        reasons.append(f"growth>{MAX_GROWTH_PERCENT:.0f}%")
    if reasons or c.get("expansion_note") or c.get("finalizer_action"):
        c["expansion_reason"] = build_expansion_reason(c)
        c["expansion_justification"] = build_expansion_justification(c)
    else:
        c.pop("expansion_reason", None)
        c.pop("expansion_justification", None)
    return c


def build_expansion_justification(clip: dict) -> str:
    """Human-readable reason when export window exceeds soft cap or growth policy."""
    dur = float(clip.get("expanded_duration", clip.get("duration", 0)))
    core = float(clip.get("original_duration", 0))
    virality = clip_virality_score(clip)
    growth = float(clip.get("growth_seconds", 0))
    growth_pct = float(clip.get("growth_percent", 0))
    parts: list[str] = [
        f"Export {dur:.0f}s vs AI core {core:.0f}s "
        f"(+{growth:.0f}s, {growth_pct:.0f}% growth).",
    ]
    if virality > VIRALITY_HARD_CAP_EXCEPTION:
        parts.append(f"Virality {virality}/100 allows up to {HARD_CAP_SECONDS:.0f}s hard cap.")
    else:
        parts.append(f"Virality {virality}/100 — target ≤{SOFT_CAP_SECONDS:.0f}s.")
    if growth_pct > MAX_GROWTH_PERCENT + 0.5:
        if growth_over_100_allowed(clip):
            parts.append(f">{MAX_GROWTH_PERCENT:.0f}% growth approved (high virality).")
        else:
            parts.append(
                f"Exceeded {MAX_GROWTH_PERCENT:.0f}% growth limit without approval — clamped."
            )
    reason = clip.get("expansion_reason")
    if reason:
        parts.append(str(reason))
    return " ".join(parts)


def log_over_soft_justifications(clips: list[dict], *, stage: str) -> int:
    """Log justification for every clip still above soft cap after a pipeline stage."""
    logged = 0
    for idx, clip in enumerate(clips):
        c = refresh_expansion_diagnostics(clip)
        dur = float(c.get("expanded_duration", 0))
        growth_pct = float(c.get("growth_percent", 0))
        if dur <= SOFT_CAP_SECONDS + 0.5 and growth_pct <= MAX_GROWTH_PERCENT + 0.5:
            continue
        justification = c.get("expansion_justification") or build_expansion_justification(c)
        logger.info(
            "[DURATION %s] clip=%d dur=%.1fs original=%.1fs growth=%.1fs (%.0f%%) "
            "merge_sources=%d | %s",
            stage,
            idx,
            dur,
            float(c.get("original_duration", 0)),
            float(c.get("growth_seconds", 0)),
            growth_pct,
            int(c.get("merge_source_count", 1)),
            justification,
        )
        logged += 1
    return logged


def compute_timeline_occupancy(
    clips: list[dict],
    media_duration: float,
) -> dict:
    """
    Measure how much timeline selected clips cover (sum of spans vs union vs pairwise overlap).
    """
    if not clips:
        return {
            "clip_count": 0,
            "sum_span_seconds": 0.0,
            "union_seconds": 0.0,
            "overlap_seconds": 0.0,
            "overlap_ratio": 0.0,
            "over_soft_cap": 0,
            "over_hard_cap": 0,
            "over_growth_100pct": 0,
            "max_duration": 0.0,
            "mean_duration": 0.0,
            "durations": [],
        }

    spans: list[tuple[float, float, float]] = []
    over_growth = 0
    for c in clips:
        c = refresh_expansion_diagnostics(c)
        t0 = float(c.get("start_seconds", c.get("start", 0)))
        t1 = float(c.get("end_seconds", c.get("end", t0)))
        dur = max(0.0, t1 - t0)
        spans.append((t0, t1, dur))
        if float(c.get("growth_percent", 0)) > MAX_GROWTH_PERCENT + 0.5:
            over_growth += 1

    spans.sort(key=lambda x: x[0])
    sum_span = sum(s[2] for s in spans)
    durations = [round(s[2], 1) for s in spans]

    union = 0.0
    cur_end = -1.0
    for t0, t1, _ in spans:
        if t0 > cur_end:
            union += t1 - t0
            cur_end = t1
        elif t1 > cur_end:
            union += t1 - cur_end
            cur_end = t1

    overlap_pairs = 0
    overlap_seconds = 0.0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a0, a1, _ = spans[i]
            b0, b1, _ = spans[j]
            if b0 >= a1:
                break
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            if inter > 0.5:
                overlap_pairs += 1
                overlap_seconds += inter

    return {
        "clip_count": len(clips),
        "sum_span_seconds": round(sum_span, 1),
        "union_seconds": round(union, 1),
        "overlap_seconds": round(overlap_seconds, 1),
        "overlap_ratio": round(overlap_seconds / max(union, 1.0), 3),
        "overlap_pairs": overlap_pairs,
        "over_soft_cap": sum(1 for d in durations if d > SOFT_CAP_SECONDS),
        "over_hard_cap": sum(1 for d in durations if d > HARD_CAP_SECONDS),
        "over_growth_100pct": over_growth,
        "max_duration": max(durations) if durations else 0.0,
        "mean_duration": round(sum_span / len(durations), 1) if durations else 0.0,
        "durations": sorted(durations, reverse=True)[:20],
        "media_duration": round(media_duration, 1) if media_duration > 0 else None,
        "occupancy_pct": round(100.0 * union / media_duration, 1) if media_duration > 0 else None,
    }


def merge_allowed_max_duration(a: dict, b: dict, max_duration: float) -> float:
    """Max combined span allowed when merging two clips (soft 90 unless high virality)."""
    best_v = max(clip_virality_score(a), clip_virality_score(b))
    if best_v > VIRALITY_HARD_CAP_EXCEPTION:
        return min(max_duration, HARD_CAP_SECONDS)
    return min(max_duration, SOFT_CAP_SECONDS)


def scaled_context_padding(
    core_duration: float,
    context_before: float,
    context_after: float,
) -> tuple[float, float]:
    """Reduce context padding when the AI core is already long (limits overlap)."""
    if core_duration >= 78.0:
        return min(context_before, 2.0), min(context_after, 4.0)
    if core_duration >= 65.0:
        return min(context_before, 3.0), min(context_after, 6.0)
    if core_duration >= 52.0:
        return min(context_before, 4.0), min(context_after, 8.0)
    if core_duration >= 40.0:
        return min(context_before, 5.0), min(context_after, 10.0)
    return context_before, context_after


def _shrink_window_around_core(
    t0: float,
    t1: float,
    core0: float,
    core1: float,
    max_span: float,
    media_duration: float,
) -> tuple[float, float]:
    """Shrink export window toward AI core while respecting max_span."""
    if t1 - t0 <= max_span + 0.01:
        return t0, t1
    core_mid = (core0 + core1) / 2.0
    half = max_span / 2.0
    new_t0 = max(0.0, core_mid - half)
    new_t1 = new_t0 + max_span
    if core0 < new_t0:
        shift = core0 - new_t0
        new_t0 += shift
        new_t1 += shift
    if core1 > new_t1:
        shift = core1 - new_t1
        new_t0 += shift
        new_t1 += shift
    if media_duration > 0:
        new_t1 = min(new_t1, media_duration)
        new_t0 = max(0.0, new_t1 - max_span)
    return new_t0, new_t1


def clamp_clip_to_duration_policy(
    clip: dict,
    media_duration: float,
    *,
    pre_virality: bool = False,
) -> tuple[dict, list[str]]:
    """
    Enforce soft/hard caps and 100% growth limit on start_seconds/end_seconds.
    Shrinks around pinned AI core instead of only trimming the end.
    """
    c = refresh_expansion_diagnostics(ensure_expansion_baseline(clip))
    actions: list[str] = []
    core0, core1 = _core_bounds(c)
    core_dur = max(0.01, core1 - core0)
    t0 = float(c["expanded_start"])
    t1 = float(c["expanded_end"])
    dur = t1 - t0

    if pre_virality:
        cap = SOFT_CAP_SECONDS
    else:
        cap = effective_max_duration(c)

    max_span = max_allowed_span_for_core(c, core_dur, cap, pre_virality=pre_virality)

    if dur > HARD_CAP_SECONDS + 0.25 and clip_virality_score(c) <= VIRALITY_HARD_CAP_EXCEPTION:
        actions.append(f"hard_clamp_{HARD_CAP_SECONDS:.0f}s")
        max_span = min(max_span, HARD_CAP_SECONDS)
    elif dur > HARD_CAP_SECONDS + 0.25:
        actions.append(f"hard_clamp_{HARD_CAP_SECONDS:.0f}s")
        max_span = min(max_span, HARD_CAP_SECONDS)

    growth_pct = 100.0 * max(0.0, dur - core_dur) / core_dur
    if growth_pct > MAX_GROWTH_PERCENT + 0.5 and not growth_over_100_allowed(c):
        actions.append(f"growth_clamp_{MAX_GROWTH_PERCENT:.0f}pct")
        max_span = min(max_span, core_dur * (1.0 + MAX_GROWTH_PERCENT / 100.0))

    if dur > max_span + 0.25:
        if max_span <= SOFT_CAP_SECONDS:
            actions.append(f"span_clamp_{max_span:.0f}s")
        t0, t1 = _shrink_window_around_core(
            t0, t1, core0, core1, max_span, media_duration,
        )
        dur = t1 - t0

    if dur > cap + 0.25:
        label = "soft" if cap <= SOFT_CAP_SECONDS else "viral_hard"
        actions.append(f"{label}_clamp_{cap:.0f}s")
        t0, t1 = _shrink_window_around_core(t0, t1, core0, core1, cap, media_duration)
        dur = t1 - t0

    if media_duration > 0:
        t1 = min(t1, media_duration)
        t0 = max(0.0, min(t0, t1 - 1.0))

    if abs(t0 - c["expanded_start"]) > 0.01 or abs(t1 - c["expanded_end"]) > 0.01:
        c["start_seconds"] = round(t0, 3)
        c["end_seconds"] = round(t1, 3)
        c.setdefault("warnings", [])
        c["warnings"].append(
            f"Duration governor: {', '.join(actions)}."
        )

    c = refresh_expansion_diagnostics(c)
    c["expansion_reason"] = build_expansion_reason(c, actions)
    if actions and float(c.get("expanded_duration", 0)) > SOFT_CAP_SECONDS + 0.5:
        c["expansion_justification"] = build_expansion_justification(c)
        logger.info(
            "[DURATION GOVERNOR] clamped core=%.1fs export %.1fs→%.1fs (%.0f%% growth) | %s",
            core_dur,
            dur,
            float(c["expanded_duration"]),
            float(c["growth_percent"]),
            c["expansion_justification"],
        )
    return c, actions


def apply_duration_policy_batch(
    clips: list[dict],
    media_duration: float,
    *,
    pre_virality: bool = False,
) -> tuple[list[dict], DurationGovernorStats]:
    stats = DurationGovernorStats()
    out: list[dict] = []
    for clip in clips:
        stats.checked += 1
        c = refresh_expansion_diagnostics(ensure_expansion_baseline(clip))
        dur_before = float(c.get("expanded_duration", c.get("duration", 0)))
        growth_before = float(c.get("growth_percent", 0))
        if dur_before > SOFT_CAP_SECONDS:
            stats.over_soft_before += 1
        if dur_before > HARD_CAP_SECONDS:
            stats.over_hard_before += 1
        if growth_before > MAX_GROWTH_PERCENT + 0.5:
            stats.over_growth_limit_before += 1

        fixed, actions = clamp_clip_to_duration_policy(
            c, media_duration, pre_virality=pre_virality,
        )
        if actions:
            if any("soft" in a or "span_clamp" in a for a in actions):
                stats.clamped_soft += 1
            if any("hard_clamp" in a for a in actions):
                stats.clamped_hard += 1
            if any("growth_clamp" in a for a in actions):
                stats.clamped_growth += 1
            stats.notes.extend(actions[:3])
        elif float(fixed.get("expanded_duration", 0)) > SOFT_CAP_SECONDS:
            stats.justified_over_soft += 1
        out.append(fixed)
    log_over_soft_justifications(out, stage="policy_batch")
    return out, stats


__all__ = [
    "HARD_CAP_SECONDS",
    "SOFT_CAP_SECONDS",
    "TARGET_MAX_SECONDS",
    "TARGET_MIN_SECONDS",
    "VIRALITY_HARD_CAP_EXCEPTION",
    "MAX_GROWTH_PERCENT",
    "DurationGovernorStats",
    "apply_duration_policy_batch",
    "build_expansion_justification",
    "build_expansion_reason",
    "clamp_clip_to_duration_policy",
    "clip_virality_score",
    "effective_max_duration",
    "ensure_expansion_baseline",
    "pin_ai_core_window",
    "refresh_expansion_diagnostics",
    "scaled_context_padding",
    "compute_timeline_occupancy",
    "log_over_soft_justifications",
    "merge_allowed_max_duration",
    "max_allowed_span_for_core",
]
