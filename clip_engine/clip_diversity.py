"""
clip_engine/clip_diversity.py
Timeline diversity, overlap prevention, and semantic deduplication for clip selection.

Pipeline:
  1. bucket_transcript()     — split transcript into timeline regions
  2. deduplicate_clips()     — remove overlapping / similar clips
  3. enforce_diversity()     — ensure clips spread across all regions
  4. rank_final_clips()      — score by uniqueness + quality, return top N
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger("clip_engine.clip_diversity")

MIN_HOOK_SCORE = 35
MIN_CLIPS_FLOOR = 5
NEARBY_CONTENT_START_GAP = 60.0


@dataclass
class DiversityPipelineStats:
    input_count: int = 0
    removed_overlap: int = 0
    removed_duplicates: int = 0
    output_count: int = 0


# ---------------------------------------------------------------------------
# Timeline regions
# ---------------------------------------------------------------------------

REGION_NAMES = ["beginning", "early_middle", "middle", "late_middle", "ending"]


class TimelineRegion(NamedTuple):
    name: str
    start: float   # seconds
    end: float     # seconds


def build_timeline_regions(media_duration: float, n_regions: int = 5) -> list[TimelineRegion]:
    """Divide media duration into N equal regions."""
    if media_duration <= 0:
        media_duration = 3600.0  # fallback 1hr
    chunk = media_duration / n_regions
    names = REGION_NAMES if n_regions == 5 else [f"region_{i+1}" for i in range(n_regions)]
    return [
        TimelineRegion(name=names[i], start=i * chunk, end=(i + 1) * chunk)
        for i in range(n_regions)
    ]


def assign_region(clip: dict, regions: list[TimelineRegion]) -> str:
    """Return the region name the clip's midpoint falls in."""
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0 + 60)))
    mid = (t0 + t1) / 2
    for r in regions:
        if r.start <= mid < r.end:
            return r.name
    return regions[-1].name


def bucket_transcript(
    segments: list[dict],
    media_duration: float,
    n_regions: int = 5,
) -> dict[str, list[dict]]:
    """
    Split transcript segments into timeline buckets.
    Returns {region_name: [segments]} dict.
    """
    regions = build_timeline_regions(media_duration, n_regions)
    buckets: dict[str, list[dict]] = {r.name: [] for r in regions}
    for seg in segments:
        t = float(seg.get("start", 0))
        for r in regions:
            if r.start <= t < r.end:
                buckets[r.name].append(seg)
                break
    return buckets


# ---------------------------------------------------------------------------
# Overlap prevention
# ---------------------------------------------------------------------------

def clips_overlap(a: dict, b: dict, min_gap_seconds: float = 15.0) -> bool:
    """
    Return True if clips a and b overlap OR are within min_gap_seconds of each other.
    """
    a0 = float(a.get("start_seconds", a.get("start", 0)))
    a1 = float(a.get("end_seconds", a.get("end", a0)))
    b0 = float(b.get("start_seconds", b.get("start", 0)))
    b1 = float(b.get("end_seconds", b.get("end", b0)))

    # True overlap
    if a0 < b1 and b0 < a1:
        return True
    # Too close
    gap = max(a0, b0) - min(a1, b1)
    return gap < min_gap_seconds


def remove_overlapping_clips(
    clips: list[dict],
    min_gap_seconds: float = 15.0,
) -> tuple[list[dict], int]:
    """
    Given a list of clips sorted by score (desc), remove any clip that overlaps
    or is too close to a higher-scored clip. Returns (kept, removed_count).
    """
    kept: list[dict] = []
    removed = 0
    for clip in clips:
        conflict = any(clips_overlap(clip, k, min_gap_seconds) for k in kept)
        if not conflict:
            kept.append(clip)
        else:
            removed += 1
            from clip_engine.telemetry import log_clip_reject

            conflict_clip = next(
                (k for k in kept if clips_overlap(clip, k, min_gap_seconds)), None
            )
            log_clip_reject(
                "timeline_overlap",
                candidate_clip=(
                    f"{clip.get('hook_title', '?')} @"
                    f"{float(clip.get('start_seconds', clip.get('start', 0))):.0f}s"
                ),
                existing_clip=(
                    f"{(conflict_clip or {}).get('hook_title', '?')} @"
                    f"{float((conflict_clip or {}).get('start_seconds', (conflict_clip or {}).get('start', 0))):.0f}s"
                ),
                min_gap_seconds=min_gap_seconds,
            )
            logger.debug(
                "Overlap removed: %.1f-%.1f '%s'",
                float(clip.get("start_seconds", 0)),
                float(clip.get("end_seconds", 0)),
                clip.get("hook_title", ""),
            )
    return kept, removed


# ---------------------------------------------------------------------------
# Simple text similarity (no external deps)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> set[str]:
    """Lowercase word set, stripped of common stop words."""
    STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
            "is","was","are","were","be","been","i","you","he","she","we","they",
            "it","this","that","so","just","like","about","what","how","when","why"}
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP}


def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity between word sets of two strings. Range 0-1."""
    wa = _normalize(a)
    wb = _normalize(b)
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    union = wa | wb
    return len(intersection) / len(union)


def _clip_text(clip: dict) -> str:
    """Combine hook + reason + context into one text blob for comparison."""
    parts = [
        str(clip.get("hook_title", "")),
        str(clip.get("selection_reason", "")),
        str(clip.get("ai_context_reason", "")),
        str(clip.get("dominant_signal", "")),
        str(clip.get("grounded_transcript_excerpt", "")),
    ]
    return " ".join(p for p in parts if p)


def _clip_start_seconds(clip: dict) -> float:
    return float(clip.get("start_seconds", clip.get("start", 0)))


def _clip_hook_score(clip: dict) -> int:
    if clip.get("hook_score") is not None:
        return int(clip.get("hook_score", 0))
    sig = clip.get("signal_scores") or clip.get("local_signals") or {}
    if sig.get("scroll_stopping_hook") is not None:
        return int(sig.get("scroll_stopping_hook", 0))
    scores = clip.get("scores") or {}
    return int(scores.get("hook_strength", 0))


def filter_minimum_hook_score(clips: list[dict]) -> tuple[list[dict], int]:
    """
    Drop clips with hook_score below MIN_HOOK_SCORE unless that would leave
    fewer than MIN_CLIPS_FLOOR clips (then keep the highest hook scores).
    """
    if not clips:
        return clips, 0

    strong = [c for c in clips if _clip_hook_score(c) >= MIN_HOOK_SCORE]
    if len(strong) >= MIN_CLIPS_FLOOR:
        kept_ids = {id(c) for c in strong}
        removed = 0
        for c in clips:
            if id(c) not in kept_ids:
                removed += 1
                from clip_engine.telemetry import log_clip_reject

                log_clip_reject(
                    "weak_hook",
                    hook_score=_clip_hook_score(c),
                    threshold=MIN_HOOK_SCORE,
                    clip=c.get("hook_title", ""),
                )
        return strong, removed

    ranked = sorted(clips, key=_clip_hook_score, reverse=True)
    kept = ranked[: min(MIN_CLIPS_FLOOR, len(ranked))]
    kept_ids = {id(c) for c in kept}
    removed = 0
    for c in clips:
        if id(c) not in kept_ids:
            removed += 1
            if _clip_hook_score(c) < MIN_HOOK_SCORE:
                from clip_engine.telemetry import log_clip_reject

                log_clip_reject(
                    "weak_hook",
                    hook_score=_clip_hook_score(c),
                    threshold=MIN_HOOK_SCORE,
                    clip=c.get("hook_title", ""),
                )
    from clip_engine.clip_split_parts import apply_series_hook_filter

    kept = apply_series_hook_filter(clips, kept)
    return kept, removed


def remove_content_overlap_nearby(
    clips: list[dict],
    *,
    similarity_threshold: float = 0.45,
    max_start_gap_seconds: float = NEARBY_CONTENT_START_GAP,
) -> tuple[list[dict], int]:
    """
    Remove lower-scored clips that are semantically similar to a kept clip and
    whose start times are within max_start_gap_seconds (start-to-start).
    """
    if len(clips) <= 1:
        return clips, 0

    from clip_engine.semantic_ranking import (
        embeddings_available,
        generate_embeddings,
        semantic_similarity,
    )

    if not embeddings_available():
        return clips, 0

    texts = [_clip_text(c) for c in clips]
    try:
        emb = generate_embeddings(texts)
    except Exception as exc:
        logger.warning("Content overlap nearby check skipped: %s", exc)
        return clips, 0

    sorted_idx = sorted(
        range(len(clips)),
        key=lambda i: int(clips[i].get("composite_score", 0)),
        reverse=True,
    )
    kept: list[dict] = []
    kept_idx: list[int] = []
    removed = 0

    for i in sorted_idx:
        clip = clips[i]
        vec = emb[i]
        conflict = False
        for ki in kept_idx:
            start_gap = abs(_clip_start_seconds(clip) - _clip_start_seconds(clips[ki]))
            if start_gap >= max_start_gap_seconds:
                continue
            sim = semantic_similarity(vec, emb[ki])
            if sim >= similarity_threshold:
                from clip_engine.telemetry import log_clip_reject

                log_clip_reject(
                    "content_overlap_nearby",
                    start_a=int(_clip_start_seconds(clips[ki])),
                    start_b=int(_clip_start_seconds(clip)),
                    similarity=round(sim, 2),
                )
                logger.debug(
                    "Content overlap nearby (%.2f): %.0fs vs %.0fs '%s' ~ '%s'",
                    sim,
                    _clip_start_seconds(clips[ki]),
                    _clip_start_seconds(clip),
                    clips[ki].get("hook_title", ""),
                    clip.get("hook_title", ""),
                )
                removed += 1
                conflict = True
                break
        if not conflict:
            kept.append(clip)
            kept_idx.append(i)

    return kept, removed


def remove_semantic_duplicates(
    clips: list[dict],
    similarity_threshold: float = 0.80,
) -> tuple[list[dict], int]:
    """
    Remove clips that are semantically too similar to a higher-ranked clip.
    Returns (kept, removed_count).

    Floor rule: clips whose start times differ by more than 60 s are NEVER
    deduped regardless of text similarity (different moments in the video).
    """
    START_GAP_FLOOR = 60.0  # seconds
    kept: list[dict] = []
    removed = 0
    for clip in clips:
        clip_text = _clip_text(clip)
        is_dup = False
        for k in kept:
            start_gap = abs(_clip_start_seconds(clip) - _clip_start_seconds(k))
            if start_gap > START_GAP_FLOOR:
                continue  # different moments — never dedup
            sim = text_similarity(clip_text, _clip_text(k))
            if sim >= similarity_threshold:
                from clip_engine.telemetry import log_clip_reject

                log_clip_reject(
                    "duplicate_similarity",
                    similarity=round(sim, 2),
                    existing_clip=(
                        f"{k.get('hook_title', '?')} @"
                        f"{float(k.get('start_seconds', k.get('start', 0))):.0f}s"
                    ),
                    candidate_clip=(
                        f"{clip.get('hook_title', '?')} @"
                        f"{float(clip.get('start_seconds', clip.get('start', 0))):.0f}s"
                    ),
                )
                logger.debug(
                    "Semantic dup (%.2f): '%s' ~ '%s'",
                    sim,
                    clip.get("hook_title", ""),
                    k.get("hook_title", ""),
                )
                is_dup = True
                removed += 1
                break
        if not is_dup:
            kept.append(clip)
    return kept, removed


# ---------------------------------------------------------------------------
# Timeline diversity enforcement
# ---------------------------------------------------------------------------

def enforce_timeline_diversity(
    clips: list[dict],
    regions: list[TimelineRegion],
    target_count: int = 20,
    min_per_region: int = 1,
) -> list[dict]:
    """
    Ensure clips are spread across timeline regions.
    Strategy:
      1. Guarantee min_per_region clips from each region (if available).
      2. Fill remaining slots with highest-scoring clips.
    """
    # Assign regions
    for c in clips:
        c["_region"] = assign_region(c, regions)

    # Group by region
    by_region: dict[str, list[dict]] = {r.name: [] for r in regions}
    for c in clips:
        by_region[c["_region"]].append(c)

    selected: list[dict] = []

    # Phase 1: guarantee min_per_region from each region
    for r in regions:
        region_clips = sorted(
            by_region[r.name],
            key=lambda x: int(x.get("composite_score", 0)),
            reverse=True,
        )
        added = 0
        for c in region_clips:
            if added >= min_per_region:
                break
            if not any(clips_overlap(c, s, min_gap_seconds=15.0) for s in selected):
                selected.append(c)
                added += 1
        if added == 0:
            logger.info("No clips available for region: %s", r.name)

    already_selected_ids = {id(c) for c in selected}

    # Phase 2: fill remaining slots with best remaining clips
    remaining = [
        c for c in clips
        if id(c) not in already_selected_ids
    ]
    remaining.sort(key=lambda x: int(x.get("composite_score", 0)), reverse=True)

    for c in remaining:
        if len(selected) >= target_count:
            break
        if not any(clips_overlap(c, s, min_gap_seconds=15.0) for s in selected):
            selected.append(c)

    # Final sort by timeline position
    selected.sort(key=lambda x: float(x.get("start_seconds", x.get("start", 0))))

    logger.info(
        "Timeline diversity: %d clips across %d regions (target=%d)",
        len(selected), len(regions), target_count,
    )
    return selected


# ---------------------------------------------------------------------------
# Uniqueness scoring
# ---------------------------------------------------------------------------

def score_uniqueness(clips: list[dict]) -> list[dict]:
    """
    Add a 'uniqueness_score' (0-100) to each clip based on how different
    its content is from all other clips in the set.
    """
    texts = [_clip_text(c) for c in clips]
    for i, clip in enumerate(clips):
        if len(clips) <= 1:
            clip["uniqueness_score"] = 100
            continue
        sims = [
            text_similarity(texts[i], texts[j])
            for j in range(len(clips)) if j != i
        ]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        clip["uniqueness_score"] = int(round((1.0 - avg_sim) * 100))
    return clips


def underrepresented_regions(
    clips: list[dict],
    media_duration: float,
    n_regions: int = 5,
) -> list[str]:
    """Return timeline region names with fewer clips than average coverage."""
    regions = build_timeline_regions(media_duration, n_regions)
    counts: dict[str, int] = {r.name: 0 for r in regions}
    for c in clips:
        counts[assign_region(c, regions)] += 1
    avg = len(clips) / max(1, n_regions)
    return [name for name, cnt in counts.items() if cnt < max(1, avg * 0.5)]


def compute_final_score(clip: dict) -> float:
    """
    Weighted final score combining composite_score + uniqueness_score.
    """
    quality = float(clip.get("composite_score", 50))
    uniqueness = float(clip.get("uniqueness_score", 50))
    return quality * 0.65 + uniqueness * 0.35


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_diversity_pipeline(
    clips: list[dict],
    media_duration: float,
    target_count: int = 20,
    min_gap_seconds: float = 15.0,
    similarity_threshold: float = 0.80,
    n_regions: int = 5,
    min_per_region: int = 1,
    return_stats: bool = False,
    *,
    relax_if_under_target: bool = False,
) -> list[dict] | tuple[list[dict], DiversityPipelineStats]:
    """
    Full diversity pipeline:
      1. Sort by quality score
      2. Remove overlapping clips
      3. Remove semantic duplicates
      4. Score uniqueness
      5. Enforce timeline diversity
      6. Return top target_count clips sorted by timeline position

    When relax_if_under_target is True and output is below target_count,
    re-runs selection with progressively smaller min_gap_seconds.

    The effective gap is the larger of min_gap_seconds and a relative formula
    based on media duration: max(10, media_duration * 0.005).  This prevents
    60-second no-go zones on short clips and keeps them sane on very long ones.
    """
    stats = DiversityPipelineStats(input_count=len(clips))

    # Relative gap: 0.5% of media duration, clamped to [10s, min_gap_seconds].
    # Never exceeds the caller-requested min_gap_seconds so the pipeline
    # argument still acts as a ceiling — just not an artificially large floor.
    relative_gap = max(10.0, media_duration * 0.005) if media_duration > 0 else min_gap_seconds
    effective_gap = min(min_gap_seconds, relative_gap)

    logger.info(
        "Diversity pipeline: %d candidates -> target=%d, gap=%.0fs (relative=%.0fs), sim_thresh=%.2f",
        len(clips), target_count, effective_gap, relative_gap, similarity_threshold,
    )

    if not clips:
        stats.output_count = 0
        return (clips, stats) if return_stats else clips

    # Use the effective (smaller) gap; relaxation steps go down from there.
    gap_steps = [effective_gap]
    if relax_if_under_target:
        gap_steps.extend([
            max(10.0, effective_gap * 0.6),
            max(5.0, effective_gap * 0.3),
        ])

    best_result: list[dict] = []
    best_stats = stats

    for gap in gap_steps:
        working = sorted(clips, key=lambda x: int(x.get("composite_score", 0)), reverse=True)
        working, removed_overlap = remove_overlapping_clips(working, min_gap_seconds=gap)
        working, _ = remove_content_overlap_nearby(
            working, similarity_threshold=similarity_threshold
        )
        working, removed_dup = remove_semantic_duplicates(
            working, similarity_threshold=similarity_threshold
        )
        working = score_uniqueness(working)
        regions = build_timeline_regions(media_duration, n_regions)
        working = enforce_timeline_diversity(
            working, regions,
            target_count=target_count,
            min_per_region=min_per_region,
        )

        if len(working) > len(best_result):
            best_result = working
            best_stats = DiversityPipelineStats(
                input_count=len(clips),
                removed_overlap=removed_overlap,
                removed_duplicates=removed_dup,
                output_count=len(working),
            )

        if len(working) >= target_count or not relax_if_under_target:
            stats = best_stats
            logger.info(
                "After diversity (gap=%.0fs): %d clips (overlap removed=%d, dup removed=%d)",
                gap, len(best_result), best_stats.removed_overlap, best_stats.removed_duplicates,
            )
            return (best_result, stats) if return_stats else best_result

    stats = best_stats
    logger.info("After diversity enforcement: %d clips", len(best_result))
    return (best_result, stats) if return_stats else best_result
