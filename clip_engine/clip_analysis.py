"""
clip_engine/clip_analysis.py
Improved clip intelligence:
- Multi-region transcript chunking (forces full timeline coverage)
- Large candidate pool (60 candidates -> filtered to target)
- Strict JSON schema with retry logic
- Validation pass + sentence-boundary snapping
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clip_engine.token_tracking import TokenTracker, get_tracker, reset_tracker
from clip_engine.clip_discovery import empty_pool_stats, process_raw_clips
from clip_engine.openai_resilience import (
    JSON_STRICT_RULES,
    call_openai_chat_json,
    estimate_tokens_rough,
    get_call_context,
    truncate_text_safe,
)
from config import get_openai_model_fast

logger = logging.getLogger("clip_engine.clip_analysis")

PASS_DEFINITIONS: tuple[tuple[str, str], ...] = (
    (
        "primary",
        "Find the best story, emotional, educational, and high-retention clips.",
    ),
    (
        "gems",
        "Find hidden gems: short quotes, funny moments, hot takes, and surprising one-liners.",
    ),
    (
        "micro",
        "Find strong standalone MICRO-clips: one powerful idea, sharp hook, clean ending, 30-75 seconds preferred.",
    ),
)

EXPANSION_PASS = (
    "expansion",
    "Find additional unique clips in underrepresented moments. Avoid topics already covered in excluded clips.",
)


def get_session_tokens() -> dict:
    return get_tracker().to_session_dict()


def reset_session_tokens() -> None:
    reset_tracker()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class ClipPromptSettings:
    min_clip_seconds: float = 25.0
    max_clip_seconds: float = 160.0
    ideal_min_seconds: float = 45.0
    ideal_max_seconds: float = 120.0
    max_clips: int = 8           # per region chunk (total = max_clips * n_chunks)
    min_score: int = 55
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.5
    n_transcript_chunks: int = 5  # divide transcript into this many regions


SCORE_DIMENSIONS = [
    "hook_strength",
    "emotional_intensity",
    "controversy_debate",
    "educational_value",
    "story_arc",
    "quote_worthiness",
    "standalone_clarity",
    "retention_potential",
    "ending_strength",
]


# ---------------------------------------------------------------------------
# Transcript chunking
# ---------------------------------------------------------------------------

def _chunk_transcript(transcript: str, n_chunks: int) -> list[tuple[str, str]]:
    """
    Split transcript into n_chunks roughly equal parts (line-based fallback).
    Returns list of (region_label, text) tuples.
    """
    lines = [l for l in transcript.splitlines() if l.strip()]
    if not lines:
        return [("full", transcript)]

    chunk_size = max(1, math.ceil(len(lines) / n_chunks))
    region_names = ["beginning", "early_middle", "middle", "late_middle", "ending"]
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(lines))
        region = region_names[i] if i < len(region_names) else f"region_{i+1}"
        chunk_text = "\n".join(lines[start:end])
        if chunk_text.strip():
            chunks.append((region, chunk_text))
    return chunks


def _chunk_transcript_by_time(
    segments: list[dict],
    media_duration: float,
    n_chunks: int,
) -> list[tuple[str, str]]:
    """Split transcript into equal TIME regions using segment timestamps."""
    from clip_engine.transcription_utils import segments_to_prompt_transcript

    if not segments or media_duration <= 0:
        return [("full", segments_to_prompt_transcript(segments))]

    region_names = ["beginning", "early_middle", "middle", "late_middle", "ending"]
    chunk_dur = media_duration / n_chunks
    chunks: list[tuple[str, str]] = []

    for i in range(n_chunks):
        t_start = i * chunk_dur
        t_end = (i + 1) * chunk_dur if i < n_chunks - 1 else media_duration + 0.001
        region = region_names[i] if i < len(region_names) else f"region_{i+1}"
        region_segs = [
            s for s in segments
            if float(s.get("end", 0)) > t_start and float(s.get("start", 0)) < t_end
        ]
        if not region_segs:
            continue
        text = segments_to_prompt_transcript(region_segs)
        if text.strip():
            chunks.append((region, text))

    return chunks or [("full", segments_to_prompt_transcript(segments))]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(
    settings: ClipPromptSettings,
    region_label: str,
    optional_note: str | None,
    pass_focus: str = "",
    excluded_summary: str = "",
) -> str:
    dim_list = "\n".join(f"  - {d}" for d in SCORE_DIMENSIONS)
    note_block = f"\nCreator note: {optional_note}\n" if optional_note else ""
    focus_block = f"\nPASS FOCUS: {pass_focus}\n" if pass_focus else ""
    exclude_block = f"\nALREADY SELECTED (avoid similar moments):\n{excluded_summary}\n" if excluded_summary else ""
    return f"""You are an expert short-form video editor for TikTok, YouTube Shorts, and Instagram Reels.
You are analyzing the {region_label.upper()} section of a longer video.
{note_block}{focus_block}{exclude_block}
Find up to {settings.max_clips} of the BEST clip moments in THIS section.

RULES:
- Each clip MUST have a clear hook at the start, a clear point, and a clean ending.
- Do NOT pick clips that start or end mid-sentence.
- STRONGLY prefer clips between {settings.ideal_min_seconds}s and {settings.ideal_max_seconds}s.
- Never suggest clips shorter than {settings.min_clip_seconds}s or longer than {settings.max_clip_seconds}s.
- One powerful idea per clip. Avoid broad multi-topic windows.
- Use the EXACT timestamps from the transcript - do not guess or interpolate.
- hook_title MUST describe content actually spoken in that exact time range.
- Score each clip across these dimensions (0-100 each):
{dim_list}
- Compute composite_score = weighted average (weight hook_strength and ending_strength 1.5x).
- Prefer clips with composite_score >= {settings.min_score}, but include strong borderline moments when unsure.
- Return as many distinct strong moments as you can find in this section (up to {settings.max_clips}).
- For caption_style choose one of: Clean | Bold Viral | Podcast | Minimal
- For platform_fit list applicable: TikTok | YouTube Shorts | Instagram Reels | LinkedIn

OUTPUT: Respond with ONLY valid JSON. No markdown. No code fences. No explanations.
{JSON_STRICT_RULES}
Schema:
{{
  "clips": [
    {{
      "rank": 1,
      "start_seconds": 12.4,
      "end_seconds": 67.1,
      "hook_title": "Short punchy title under 8 words",
      "composite_score": 82,
      "scores": {{
        "hook_strength": 85,
        "emotional_intensity": 70,
        "controversy_debate": 60,
        "educational_value": 90,
        "story_arc": 80,
        "quote_worthiness": 75,
        "standalone_clarity": 88,
        "retention_potential": 83,
        "ending_strength": 79
      }},
      "dominant_signal": "educational",
      "selection_reason": "One sentence max.",
      "ai_context_reason": "One sentence on why this window works.",
      "caption_style": "Bold Viral",
      "platform_fit": ["TikTok", "YouTube Shorts"],
      "warnings": []
    }}
  ]
}}"""


def _build_user_prompt(region_label: str, chunk_text: str, max_chars: int | None = None) -> str:
    ctx = get_call_context()
    limit = max_chars if max_chars is not None else ctx.max_chunk_chars
    chunk_text, _ = truncate_text_safe(chunk_text, limit, label=f"region_{region_label}")
    return f"TRANSCRIPT SECTION ({region_label}):\n{chunk_text}"


ANALYSIS_JSON_SCHEMA_HINT = (
    '{"clips": [{"rank": 1, "start_seconds": 0, "end_seconds": 0, "hook_title": "", '
    '"composite_score": 0, "scores": {}, "dominant_signal": "", "selection_reason": "", '
    '"ai_context_reason": "", "caption_style": "", "platform_fit": [], "warnings": []}]}'
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sentence-boundary snapping
# ---------------------------------------------------------------------------

def _snap_to_boundaries(clips: list[dict], segments: list[dict], tol: float = 2.5) -> list[dict]:
    if not segments:
        return clips
    seg_starts = [float(s.get("start", 0)) for s in segments]
    seg_ends = [float(s.get("end", 0)) for s in segments]

    def nearest(candidates: list[float], target: float) -> float:
        best = min(candidates, key=lambda x: abs(x - target))
        return best if abs(best - target) <= tol else target

    snapped = []
    for c in clips:
        c = dict(c)
        ns = nearest(seg_starts, float(c["start_seconds"]))
        ne = nearest(seg_ends, float(c["end_seconds"]))
        if ne > ns:
            if ns != c["start_seconds"] or ne != c["end_seconds"]:
                c.setdefault("warnings", [])
                c["warnings"].append(f"Snapped: {c['start_seconds']:.1f}->{ns:.1f}s / {c['end_seconds']:.1f}->{ne:.1f}s")
            c["start_seconds"] = ns
            c["end_seconds"] = ne
        snapped.append(c)
    return snapped


# ---------------------------------------------------------------------------
# Single-region API call
# ---------------------------------------------------------------------------

def _analyze_region(
    client,
    region_label: str,
    chunk_text: str,
    settings: ClipPromptSettings,
    optional_note: str | None,
    *,
    pass_name: str = "primary",
    pass_focus: str = "",
    excluded_summary: str = "",
    tracker: TokenTracker | None = None,
    model: str | None = None,
    progress: Any | None = None,
    cache_key: str = "",
    discovery_mode: bool = False,
    segments: list[dict] | None = None,
    media_duration: float = 0.0,
    region_stats: dict[str, int] | None = None,
) -> list[dict]:
    """Call OpenAI for one transcript region. Returns validated/rescued clips."""
    from clip_engine.analysis_cache import AnalysisProgress, save_progress

    tracker = tracker or get_tracker()
    model = model or get_openai_model_fast()
    stage = f"clip_analysis_{pass_name}_{region_label}"

    if progress and isinstance(progress, AnalysisProgress):
        if progress.is_done(pass_name, region_label):
            logger.info("Skipping completed step: %s / %s", pass_name, region_label)
            return []

    system_prompt = _build_system_prompt(
        settings, region_label, optional_note, pass_focus, excluded_summary
    )
    user_prompt = _build_user_prompt(region_label, chunk_text)
    prompt_estimate = estimate_tokens_rough(system_prompt + user_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        logger.info(
            "Region '%s' pass '%s' model=%s est_tokens=%d",
            region_label, pass_name, model, prompt_estimate,
        )
        data = call_openai_chat_json(
            client,
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=3500,
            response_format={"type": "json_object"},
            stage=stage,
            schema_hint=ANALYSIS_JSON_SCHEMA_HINT,
            tracker=tracker,
            prompt_estimate=prompt_estimate,
        )
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        clips_raw = data.get("clips", [])
        if not isinstance(clips_raw, list):
            raise ValueError("'clips' not a list")
    except (json.JSONDecodeError, ValueError, RuntimeError) as e:
        logger.error(
            "Region '%s' pass '%s' failed (parse/repair exhausted): %s",
            region_label, pass_name, e,
        )
        return []

    min_score = settings.min_score - 12 if discovery_mode else settings.min_score
    validated, rs = process_raw_clips(
        clips_raw,
        min_clip_seconds=settings.min_clip_seconds,
        max_clip_seconds=settings.max_clip_seconds,
        min_score=min_score,
        discovery_mode=discovery_mode,
        segments=segments,
        media_duration=media_duration,
        region_label=region_label,
        pass_name=pass_name,
    )
    if region_stats is not None:
        region_stats["raw_ai_candidates"] = region_stats.get("raw_ai_candidates", 0) + rs["raw"]
        region_stats["valid_after_schema"] = region_stats.get("valid_after_schema", 0) + rs["valid"]
        region_stats["rescued_candidates"] = region_stats.get("rescued_candidates", 0) + rs["rescued"]
        region_stats["rejected_invalid_time"] = region_stats.get("rejected_invalid_time", 0) + rs[
            "rejected_invalid_time"
        ]
        region_stats["rejected_duration"] = region_stats.get("rejected_duration", 0) + rs["rejected_duration"]
        region_stats["rejected_empty_transcript"] = region_stats.get("rejected_empty_transcript", 0) + rs[
            "rejected_empty_transcript"
        ]

    if progress and isinstance(progress, AnalysisProgress) and cache_key:
        progress.mark_done(pass_name, region_label)
        progress.partial_candidates.extend(validated)
        progress.partial_candidates = dedupe_candidates_by_time(
            progress.partial_candidates, min_gap_seconds=10.0,
        )
        save_progress(progress)

    logger.info(
        "Region '%s' pass '%s': %d raw -> %d valid clips",
        region_label, pass_name, len(clips_raw), len(validated),
    )
    return validated


# ---------------------------------------------------------------------------
# Candidate deduplication
# ---------------------------------------------------------------------------

def _clip_time_range(c: dict) -> tuple[float, float]:
    t0 = float(c.get("start_seconds", c.get("start", 0)))
    t1 = float(c.get("end_seconds", c.get("end", t0)))
    return t0, t1


def _ranges_overlap(a: tuple[float, float], b: tuple[float, float], gap: float = 0.0) -> bool:
    return a[0] < b[1] + gap and b[0] < a[1] + gap


def dedupe_candidates_by_time(candidates: list[dict], min_gap_seconds: float = 15.0) -> list[dict]:
    """Remove overlapping candidates, keeping highest composite_score."""
    sorted_c = sorted(candidates, key=lambda x: int(x.get("composite_score", 0)), reverse=True)
    kept: list[dict] = []
    for c in sorted_c:
        r = _clip_time_range(c)
        if any(_ranges_overlap(r, _clip_time_range(k), min_gap_seconds) for k in kept):
            continue
        kept.append(c)
    return kept


def _summarize_excluded_clips(clips: list[dict], max_items: int = 12) -> str:
    lines = []
    for c in clips[:max_items]:
        t0, t1 = _clip_time_range(c)
        lines.append(f"- {t0:.0f}s-{t1:.0f}s: {c.get('hook_title', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-pass candidate collection
# ---------------------------------------------------------------------------

def _duration_sort_key(c: dict, ideal_min: float, ideal_max: float) -> tuple:
    """Prefer clips in ideal duration band, then by score."""
    t0, t1 = _clip_time_range(c)
    dur = t1 - t0
    in_band = ideal_min <= dur <= ideal_max
    over_ideal = max(0.0, dur - ideal_max)
    return (0 if in_band else 1, over_ideal, -int(c.get("composite_score", 0)))


def collect_candidates_multipass(
    transcript: str,
    api_key: str,
    optional_content_note: str | None = None,
    prompt_settings: ClipPromptSettings | None = None,
    segments: list[dict] | None = None,
    media_duration: float = 0.0,
    pool_target: int = 60,
    tracker: TokenTracker | None = None,
    passes: tuple[str, ...] | None = None,
    exclude_clips: list[dict] | None = None,
    *,
    region_filter: tuple[str, ...] | None = None,
    max_pass_rounds: int = 1,
    progress: Any | None = None,
    cache_key: str = "",
    model: str | None = None,
    discovery_mode: bool = False,
    pool_stats: dict[str, int] | None = None,
) -> list[dict]:
    """Run analysis passes until candidate pool reaches pool_target (or rounds exhausted)."""
    import openai

    settings = prompt_settings or ClipPromptSettings()
    tracker = tracker or get_tracker()
    client = openai.OpenAI(api_key=api_key)
    model = model or get_openai_model_fast()
    ctx = get_call_context()

    if segments and media_duration > 0:
        chunks = _chunk_transcript_by_time(segments, media_duration, settings.n_transcript_chunks)
    else:
        chunks = _chunk_transcript(transcript, settings.n_transcript_chunks)

    if region_filter:
        chunks = [(r, t) for r, t in chunks if r in region_filter] or chunks

    pass_map = {name: focus for name, focus in PASS_DEFINITIONS}
    pass_map[EXPANSION_PASS[0]] = EXPANSION_PASS[1]

    if passes:
        pass_list = [(p, pass_map.get(p, "")) for p in passes if p in pass_map]
    else:
        if discovery_mode:
            pass_list = list(PASS_DEFINITIONS)
        elif ctx.token_saver_mode:
            pass_list = list(PASS_DEFINITIONS[:2])  # primary + gems
        else:
            pass_list = list(PASS_DEFINITIONS)

    excluded_summary = _summarize_excluded_clips(exclude_clips or [])
    all_candidates: list[dict] = []
    stats = pool_stats if pool_stats is not None else empty_pool_stats()

    if progress and getattr(progress, "partial_candidates", None):
        all_candidates = list(progress.partial_candidates)

    for _round in range(max(1, max_pass_rounds)):
        if len(all_candidates) >= pool_target and passes:
            break
        for pass_name, pass_focus in pass_list:
            for region_label, chunk_text in chunks:
                before_dedupe = len(all_candidates)
                region_clips = _analyze_region(
                    client,
                    region_label,
                    chunk_text,
                    settings,
                    optional_content_note,
                    pass_name=pass_name,
                    pass_focus=pass_focus,
                    excluded_summary=excluded_summary if pass_name == "expansion" else "",
                    tracker=tracker,
                    model=model,
                    progress=progress,
                    cache_key=cache_key,
                    discovery_mode=discovery_mode,
                    segments=segments,
                    media_duration=media_duration,
                    region_stats=stats,
                )
                all_candidates.extend(region_clips)
                gap = 8.0 if discovery_mode else 10.0
                all_candidates = dedupe_candidates_by_time(all_candidates, min_gap_seconds=gap)
                stats["rejected_overlap_early"] = stats.get("rejected_overlap_early", 0) + max(
                    0, before_dedupe + len(region_clips) - len(all_candidates)
                )
            if not passes and len(all_candidates) >= pool_target * (1 if ctx.token_saver_mode else 2):
                break
        if len(all_candidates) >= pool_target:
            break

    if segments and all_candidates:
        all_candidates = _snap_to_boundaries(all_candidates, segments)

    all_candidates.sort(
        key=lambda x: _duration_sort_key(
            x, settings.ideal_min_seconds, settings.ideal_max_seconds
        ),
    )
    logger.info("Multipass candidates collected: %d (target pool=%d)", len(all_candidates), pool_target)
    return all_candidates


# ---------------------------------------------------------------------------
# Main entry point (backward compatible)
# ---------------------------------------------------------------------------

def suggest_clips_from_transcript(
    transcript: str,
    api_key: str,
    optional_content_note: str | None = None,
    prompt_settings: ClipPromptSettings | None = None,
    segments: list[dict] | None = None,
    target_count: int = 20,
    media_duration: float = 0.0,
) -> list[dict]:
    """
    Analyze transcript in multiple regions, collect large candidate pool,
    validate, snap to boundaries, and return all candidates for diversity pipeline.
    """
    reset_tracker()
    settings = prompt_settings or ClipPromptSettings()
    pool_target = max(target_count * 3, 45)

    all_candidates = collect_candidates_multipass(
        transcript,
        api_key,
        optional_content_note=optional_content_note,
        prompt_settings=settings,
        segments=segments,
        media_duration=media_duration,
        pool_target=pool_target,
    )

    if not all_candidates:
        logger.warning("No clips found across all regions.")
        return []

    return all_candidates
