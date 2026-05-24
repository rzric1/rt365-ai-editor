"""
clip_engine/clip_pipeline.py
Orchestrates multi-pass candidate generation, diversity, expansion, split, and grounding.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("clip_engine.clip_pipeline")


@dataclass
class PipelineOpenAIConfig:
    """OpenAI usage controls for run_full_clip_pipeline."""

    token_saver_mode: bool = True
    rate_limit_safe: bool = True
    use_cache: bool = True
    max_tokens_budget: int = 60_000
    call_delay_seconds: float = 0.75
    status_callback: Callable[[str], None] | None = None
    model_fast: str | None = None
    model_final: str | None = None

    # Aliases for alternate naming (clip_studio / docs)
    @property
    def rate_limit_safe_mode(self) -> bool:
        return self.rate_limit_safe

    @property
    def use_cached_analysis(self) -> bool:
        return self.use_cache

    @property
    def max_tokens_per_analysis(self) -> int:
        return self.max_tokens_budget

    @property
    def delay_between_calls(self) -> float:
        return self.call_delay_seconds


@dataclass
class PipelineStats:
    target_clips: int = 20
    raw_candidates: int = 0
    removed_overlap: int = 0
    removed_duplicates: int = 0
    after_diversity: int = 0
    after_split: int = 0
    final_clips: int = 0
    expansion_pass_ran: bool = False
    expansion_pass_count: int = 0
    rejected_ungrounded: int = 0
    warnings: list[str] = field(default_factory=list)
    cache_hit: bool = False
    resumed_from_progress: bool = False
    estimated_tokens: int = 0
    model_fast: str = ""
    model_quality: str = ""

    def to_dict(self) -> dict:
        return {
            "target_clips": self.target_clips,
            "raw_candidates": self.raw_candidates,
            "removed_overlap": self.removed_overlap,
            "removed_duplicates": self.removed_duplicates,
            "after_diversity": self.after_diversity,
            "after_split": self.after_split,
            "final_clips": self.final_clips,
            "expansion_pass_ran": self.expansion_pass_ran,
            "expansion_pass_count": self.expansion_pass_count,
            "rejected_ungrounded": self.rejected_ungrounded,
            "warnings": self.warnings,
            "cache_hit": self.cache_hit,
            "resumed_from_progress": self.resumed_from_progress,
            "estimated_tokens": self.estimated_tokens,
            "model_fast": self.model_fast,
            "model_quality": self.model_quality,
        }


from clip_engine.analysis_cache import (
    AnalysisProgress,
    build_cache_key,
    clear_progress,
    load_cached_analysis,
    load_progress,
    save_cached_analysis,
)
from clip_engine.clip_analysis import (
    ClipPromptSettings,
    collect_candidates_multipass,
    dedupe_candidates_by_time,
)
from clip_engine.clip_diversity import run_diversity_pipeline, underrepresented_regions
from clip_engine.clip_expand import ClipExpansionSettings, finalize_clips_after_ai
from clip_engine.clip_metadata import ground_all_clips_metadata
from clip_engine.clip_split import split_long_clips
from clip_engine.clip_style import ClipStyle, get_clip_style_profile
from clip_engine.clip_signals import apply_signal_boosts_to_clips
from clip_engine.speaker_signals import apply_speaker_signals_to_clips
from clip_engine.openai_resilience import (
    OpenAICallContext,
    OpenAIRateLimitError,
    estimate_pipeline_tokens,
    set_call_context,
    token_saver_pass_config,
)
from clip_engine.token_tracking import TokenTracker, get_tracker, reset_tracker
from config import get_openai_model, get_openai_model_fast


def _assign_clip_ids(clips: list[dict]) -> list[dict]:
    for c in clips:
        if not c.get("_wid"):
            c["_wid"] = uuid.uuid4().hex
    return clips


def _run_expansion_passes(
    *,
    formatted: str,
    api_key: str,
    segments: list[dict],
    media_duration: float,
    creator_note: str | None,
    prompt_settings: ClipPromptSettings,
    pool_target: int,
    tracker: TokenTracker,
    candidates: list[dict],
    selected: list[dict],
    target_count: int,
    min_gap_seconds: float,
    similarity_threshold: float,
    profile_min_score: int,
    max_rounds: int = 3,
    token_saver_mode: bool = True,
) -> tuple[list[dict], list[dict], int]:
    """Run up to max_rounds expansion passes when selected count is below target."""
    if token_saver_mode:
        max_rounds = min(max_rounds, 1)
    rounds = 0
    merged_candidates = list(candidates)

    while len(selected) < target_count and rounds < max_rounds:
        rounds += 1
        score_floor = max(42, profile_min_score - 6 - (rounds - 1) * 3)
        relaxed_settings = ClipPromptSettings(
            min_clip_seconds=prompt_settings.min_clip_seconds,
            max_clip_seconds=prompt_settings.max_clip_seconds,
            ideal_min_seconds=prompt_settings.ideal_min_seconds,
            ideal_max_seconds=prompt_settings.ideal_max_seconds,
            max_clips=min(prompt_settings.max_clips + 3, 20),
            min_score=score_floor,
            n_transcript_chunks=5,
        )
        weak_regions = underrepresented_regions(selected, media_duration)
        extra = collect_candidates_multipass(
            formatted,
            api_key,
            optional_content_note=creator_note,
            prompt_settings=relaxed_settings,
            segments=segments,
            media_duration=media_duration,
            pool_target=pool_target,
            tracker=tracker,
            passes=("expansion",),
            exclude_clips=selected,
            region_filter=tuple(weak_regions) if weak_regions else None,
            max_pass_rounds=1 if token_saver_mode else 2,
        )
        if not extra:
            break
        merged_candidates = dedupe_candidates_by_time(
            merged_candidates + extra, min_gap_seconds=12.0
        )
        selected, _ = run_diversity_pipeline(
            merged_candidates,
            media_duration=media_duration,
            target_count=target_count,
            min_gap_seconds=min_gap_seconds,
            similarity_threshold=similarity_threshold,
            n_regions=5,
            min_per_region=1,
            relax_if_under_target=True,
            return_stats=True,
        )

    return merged_candidates, selected, rounds


def run_full_clip_pipeline(
    formatted: str,
    api_key: str,
    segments: list[dict],
    *,
    media_duration: float,
    creator_note: str | None = None,
    clip_style: ClipStyle | str = "Balanced",
    user_min_seconds: float = 25.0,
    user_max_seconds: float = 160.0,
    context_before: float | None = None,
    context_after: float | None = None,
    allow_exceed_max: bool = False,
    target_count: int = 20,
    min_gap_seconds: float = 60.0,
    similarity_threshold: float = 0.45,
    video_filename: str = "",
    enable_signal_boosts: bool = True,
    enable_speaker_signals: bool = True,
    openai_config: PipelineOpenAIConfig | None = None,
) -> tuple[list[dict], PipelineStats, TokenTracker]:
    """
    Full clip pipeline:
      1. Multi-pass candidate generation (pool >= 3x target)
      2. Diversity selection on AI core windows
      3. Expansion passes if under target
      4. Split long clips
      5. Context expansion with style-aware caps
      6. Post-expand split if needed
      7. Metadata grounding on final windows
    """
    tracker = reset_tracker(video_filename=video_filename)
    stats = PipelineStats(target_clips=target_count)
    oai = openai_config or PipelineOpenAIConfig()
    model_fast = get_openai_model_fast()
    model_quality = get_openai_model()
    stats.model_fast = model_fast
    stats.model_quality = model_quality

    style_name = clip_style if clip_style in ("Balanced", "Micro clips", "Long story clips") else "Balanced"
    n_passes, max_pass_rounds, _ = token_saver_pass_config(style_name)
    if not oai.token_saver_mode:
        n_passes = 3
        max_pass_rounds = 2

    token_estimate = estimate_pipeline_tokens(
        formatted,
        target_count=target_count,
        n_passes=n_passes,
        max_pass_rounds=max_pass_rounds,
        token_saver_mode=oai.token_saver_mode,
    )
    stats.estimated_tokens = token_estimate.estimated_total_tokens

    if token_estimate.estimated_total_tokens > oai.max_tokens_budget:
        oai.token_saver_mode = True
        n_passes, max_pass_rounds, _ = token_saver_pass_config(style_name)
        stats.warnings.append(
            f"Estimated ~{token_estimate.estimated_total_tokens:,} tokens exceeds budget "
            f"({oai.max_tokens_budget:,}). Token Saver Mode enforced."
        )
        if token_estimate.estimated_total_tokens > oai.max_tokens_budget * 1.5:
            stats.warnings.append(
                "Consider lowering target clip count or max clip length before analyzing."
            )

    cache_key_obj = build_cache_key(
        video_filename=video_filename,
        formatted=formatted,
        segments=segments,
        target_clips=target_count,
        clip_style=style_name,
        min_clip_seconds=user_min_seconds,
        max_clip_seconds=user_max_seconds,
        min_gap_seconds=min_gap_seconds,
        similarity_threshold=similarity_threshold,
        token_saver_mode=oai.token_saver_mode,
        model_fast=model_fast,
        model_quality=model_quality,
        context_before=context_before if context_before is not None else 8.0,
        context_after=context_after if context_after is not None else 12.0,
    )
    cache_key = cache_key_obj.digest()

    if oai.use_cache:
        cached = load_cached_analysis(cache_key)
        if cached:
            tracker.record_cache_hit(cached.get("token_usage", {}).get("total_tokens", stats.estimated_tokens))
            stats.cache_hit = True
            stats.final_clips = len(cached.get("clips", []))
            stats.warnings.append("Loaded cached analysis — no OpenAI tokens used.")
            return cached["clips"], stats, tracker

    ctx = OpenAICallContext(
        token_saver_mode=oai.token_saver_mode,
        rate_limit_safe=oai.rate_limit_safe,
        call_delay_seconds=oai.call_delay_seconds,
        max_chunk_chars=8_000 if oai.token_saver_mode else 10_000,
        status_callback=oai.status_callback,
        tracker=tracker,
    )
    set_call_context(ctx)

    progress = load_progress(cache_key)
    if progress and progress.completed_steps:
        stats.resumed_from_progress = True
        stats.warnings.append(
            f"Resuming analysis from pass '{progress.last_pass}' / region '{progress.last_region}'."
        )
    elif progress is None:
        progress = AnalysisProgress(cache_key=cache_key)

    profile = get_clip_style_profile(
        style_name,
        user_min_seconds=user_min_seconds,
        user_max_seconds=user_max_seconds,
    )

    ctx_b = context_before if context_before is not None else profile.context_before
    ctx_a = context_after if context_after is not None else profile.context_after

    pool_multiplier = 1.5 if oai.token_saver_mode else 3.0
    pool_target = max(int(target_count * pool_multiplier), 30 if oai.token_saver_mode else 45)
    max_clips_region = profile.max_clips_per_region
    if oai.token_saver_mode:
        max_clips_region = max(6, int(max_clips_region * 0.65))

    prompt_settings = ClipPromptSettings(
        min_clip_seconds=max(user_min_seconds, profile.ideal_min_seconds * 0.85),
        max_clip_seconds=profile.ai_max_clip_seconds,
        ideal_min_seconds=profile.ideal_min_seconds,
        ideal_max_seconds=profile.ideal_max_seconds,
        max_clips=max_clips_region,
        min_score=profile.min_score,
        n_transcript_chunks=5,
    )

    try:
        candidates = collect_candidates_multipass(
            formatted,
            api_key,
            optional_content_note=creator_note,
            prompt_settings=prompt_settings,
            segments=segments,
            media_duration=media_duration,
            pool_target=pool_target,
            tracker=tracker,
            max_pass_rounds=max_pass_rounds,
            progress=progress,
            cache_key=cache_key,
            model=model_fast,
        )
        candidates = dedupe_candidates_by_time(candidates, min_gap_seconds=12.0)
        stats.raw_candidates = len(candidates)
        logger.info("Candidate pool after multipass: %d (target pool=%d)", len(candidates), pool_target)

        # If pool still too small, run another collection (skip in token saver unless very low)
        if len(candidates) < pool_target and not oai.token_saver_mode:
            boost_settings = ClipPromptSettings(
                min_clip_seconds=prompt_settings.min_clip_seconds,
                max_clip_seconds=prompt_settings.max_clip_seconds,
                ideal_min_seconds=prompt_settings.ideal_min_seconds,
                ideal_max_seconds=prompt_settings.ideal_max_seconds,
                max_clips=min(prompt_settings.max_clips + 4, 22),
                min_score=max(42, profile.min_score - 5),
                n_transcript_chunks=5,
            )
            extra_pool = collect_candidates_multipass(
                formatted,
                api_key,
                optional_content_note=creator_note,
                prompt_settings=boost_settings,
                segments=segments,
                media_duration=media_duration,
                pool_target=pool_target,
                tracker=tracker,
                max_pass_rounds=max_pass_rounds,
                progress=progress,
                cache_key=cache_key,
                model=model_fast,
            )
            candidates = dedupe_candidates_by_time(candidates + extra_pool, min_gap_seconds=10.0)
            stats.raw_candidates = len(candidates)

        # Diversity on AI core windows BEFORE expansion
        selected, div_stats = run_diversity_pipeline(
            candidates,
            media_duration=media_duration,
            target_count=target_count,
            min_gap_seconds=min_gap_seconds,
            similarity_threshold=similarity_threshold,
            n_regions=5,
            min_per_region=1,
            relax_if_under_target=True,
            return_stats=True,
        )
        stats.removed_overlap = div_stats.removed_overlap
        stats.removed_duplicates = div_stats.removed_duplicates
        stats.after_diversity = len(selected)

        if len(selected) < target_count and candidates:
            stats.expansion_pass_ran = True
            candidates, selected, exp_rounds = _run_expansion_passes(
                formatted=formatted,
                api_key=api_key,
                segments=segments,
                media_duration=media_duration,
                creator_note=creator_note,
                prompt_settings=prompt_settings,
                pool_target=pool_target,
                tracker=tracker,
                candidates=candidates,
                selected=selected,
                target_count=target_count,
                min_gap_seconds=min_gap_seconds,
                similarity_threshold=similarity_threshold,
                profile_min_score=profile.min_score,
                max_rounds=3,
                token_saver_mode=oai.token_saver_mode,
            )
            stats.expansion_pass_count = exp_rounds
            stats.after_diversity = len(selected)

        selected = split_long_clips(
            selected,
            segments,
            api_key,
            max_duration=profile.split_threshold_seconds,
            sub_clip_max=profile.sub_clip_max_seconds,
            min_duration=max(user_min_seconds, profile.ideal_min_seconds * 0.8),
            tracker=tracker,
        )
        stats.after_split = len(selected)

        if len(selected) > target_count:
            selected, _ = run_diversity_pipeline(
                selected,
                media_duration=media_duration,
                target_count=target_count,
                min_gap_seconds=min_gap_seconds,
                similarity_threshold=similarity_threshold,
                n_regions=5,
                min_per_region=1,
                return_stats=True,
            )

        exp = ClipExpansionSettings(
            context_before=ctx_b,
            context_after=ctx_a,
            min_clip_seconds=user_min_seconds,
            max_clip_seconds=profile.expansion_max_seconds,
            hard_max_seconds=profile.hard_max_export_seconds,
            allow_exceed_max=allow_exceed_max,
        )
        selected = finalize_clips_after_ai(selected, media_duration, segments, exp)

        selected = split_long_clips(
            selected,
            segments,
            api_key,
            max_duration=profile.split_threshold_seconds,
            sub_clip_max=profile.sub_clip_max_seconds,
            min_duration=user_min_seconds,
            tracker=tracker,
        )

        if len(selected) > target_count:
            selected, _ = run_diversity_pipeline(
                selected,
                media_duration=media_duration,
                target_count=target_count,
                min_gap_seconds=max(30.0, min_gap_seconds * 0.75),
                similarity_threshold=similarity_threshold,
                n_regions=5,
                min_per_region=0,
                return_stats=True,
            )

        selected = _assign_clip_ids(selected)

        selected = ground_all_clips_metadata(
            selected,
            segments,
            api_key,
            tracker=tracker,
            force_regenerate=True,
            skip_strong_grounding=oai.token_saver_mode,
        )

        grounded: list[dict] = []
        for c in selected:
            conf = int(c.get("grounding_confidence", 0))
            excerpt = str(c.get("grounded_transcript_excerpt", "")).strip()
            if conf < 15 or (len(excerpt.split()) < 6 and not c.get("metadata_grounded")):
                stats.rejected_ungrounded += 1
                logger.info(
                    "Rejected ungrounded clip: %s (confidence=%d)",
                    c.get("hook_title", ""), conf,
                )
                continue
            grounded.append(c)
        selected = grounded

        selected = apply_signal_boosts_to_clips(
            selected, segments, enabled=enable_signal_boosts,
        )
        selected = apply_speaker_signals_to_clips(
            selected, segments, enabled=enable_speaker_signals,
        )

        stats.final_clips = len(selected)

        if stats.final_clips < 15:
            stats.warnings.append(
                f"Only {stats.final_clips} clips found. Try Micro clips mode or lower minimum score."
            )
        elif stats.final_clips < stats.target_clips:
            stats.warnings.append(
                f"Found {stats.final_clips} of {stats.target_clips} requested clips."
            )
        if stats.rejected_ungrounded:
            stats.warnings.append(
                f"{stats.rejected_ungrounded} clip(s) removed because metadata did not match transcript."
            )

        if oai.use_cache and selected:
            clear_progress(cache_key)
            save_cached_analysis(
                cache_key,
                clips=selected,
                stats=stats.to_dict(),
                token_usage=tracker.to_export_dict(
                    target_clips=target_count,
                    final_clip_count=len(selected),
                    model=model_quality,
                ),
            )

        return selected, stats, tracker

    except OpenAIRateLimitError as exc:
        stats.warnings.append(
            f"OpenAI rate limit at stage '{exc.stage}' (model {exc.model}). "
            f"{exc.mitigation} Partial progress saved — retry to resume."
        )
        raise
    finally:
        set_call_context(None)


__all__ = [
    "PipelineOpenAIConfig",
    "PipelineStats",
    "run_full_clip_pipeline",
]
