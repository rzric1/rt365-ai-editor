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
    model_quality: str | None = None
    json_fallback_model: str | None = None
    ai_profile_name: str = "SAFE"
    enable_gpu_prefilter: bool = True

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
    raw_ai_candidates: int = 0
    valid_after_schema: int = 0
    rescued_candidates: int = 0
    local_fallback_candidates: int = 0
    rejected_invalid_time: int = 0
    rejected_duration: int = 0
    rejected_empty_transcript: int = 0
    rejected_overlap_early: int = 0
    removed_overlap: int = 0
    removed_duplicates: int = 0
    removed_weak_hook: int = 0
    series_splits: int = 0
    after_diversity: int = 0
    after_split: int = 0
    final_clips: int = 0
    expansion_pass_ran: bool = False
    expansion_pass_count: int = 0
    rejected_ungrounded: int = 0
    discovery_mode: bool = False
    gpu_local_candidates: int = 0
    gpu_shortlist: int = 0
    semantic_dedupe_removed: int = 0
    gpt_passes_used: int = 0
    ai_profile: str = "SAFE"
    gpu_explorer_rows: list = field(default_factory=list)
    json_telemetry: dict = field(default_factory=dict)
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
            "raw_ai_candidates": self.raw_ai_candidates,
            "valid_after_schema": self.valid_after_schema,
            "rescued_candidates": self.rescued_candidates,
            "local_fallback_candidates": self.local_fallback_candidates,
            "rejected_invalid_time": self.rejected_invalid_time,
            "rejected_duration": self.rejected_duration,
            "rejected_empty_transcript": self.rejected_empty_transcript,
            "rejected_overlap_early": self.rejected_overlap_early,
            "removed_overlap": self.removed_overlap,
            "removed_duplicates": self.removed_duplicates,
            "removed_weak_hook": self.removed_weak_hook,
            "series_splits": self.series_splits,
            "after_diversity": self.after_diversity,
            "after_split": self.after_split,
            "final_clips": self.final_clips,
            "expansion_pass_ran": self.expansion_pass_ran,
            "expansion_pass_count": self.expansion_pass_count,
            "rejected_ungrounded": self.rejected_ungrounded,
            "discovery_mode": self.discovery_mode,
            "gpu_local_candidates": self.gpu_local_candidates,
            "gpu_shortlist": self.gpu_shortlist,
            "semantic_dedupe_removed": self.semantic_dedupe_removed,
            "gpt_passes_used": self.gpt_passes_used,
            "ai_profile": self.ai_profile,
            "json_telemetry": self.json_telemetry,
            "warnings": self.warnings,
            "cache_hit": self.cache_hit,
            "resumed_from_progress": self.resumed_from_progress,
            "estimated_tokens": self.estimated_tokens,
            "model_fast": self.model_fast,
            "model_quality": self.model_quality,
            "gpu_explorer_rows": self.gpu_explorer_rows,
            "session_telemetry": get_session_telemetry().to_dict(),
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
    dedupe_candidates_exact_start,
)
from clip_engine.effective_config import (
    ResolvedModels,
    plan_analysis_token_estimate,
)
from clip_engine.clip_diversity import (
    filter_minimum_hook_score,
    run_diversity_pipeline,
    underrepresented_regions,
)
from clip_engine.clip_split_parts import (
    apply_recommended_series_splits,
    flag_split_recommended,
)
from clip_engine.clip_expand import ClipExpansionSettings, finalize_clips_after_ai
from clip_engine.clip_metadata import ground_all_clips_metadata
from clip_engine.clip_split import split_long_clips
from clip_engine.clip_style import ClipStyle, get_clip_style_profile
from clip_engine.ai_profiles import get_ai_profile
from clip_engine.clip_discovery import empty_pool_stats, generate_local_fallback_candidates
from clip_engine.gpu_pipeline import run_gpu_prefilter_pipeline
from clip_engine.clip_signals import apply_signal_boosts_to_clips
from clip_engine.speaker_signals import apply_speaker_signals_to_clips
from clip_engine.openai_resilience import (
    OpenAICallContext,
    OpenAIRateLimitError,
    estimate_pipeline_tokens,
    get_json_telemetry,
    reset_json_telemetry,
    set_call_context,
    token_saver_pass_config,
)
from clip_engine.token_tracking import TokenTracker, get_tracker, reset_tracker
from clip_engine.effective_config import resolve_models_from_profile
from clip_engine.telemetry import (
    get_session_telemetry,
    log_clip_reject,
    log_gpu_memory,
    log_pipeline_timing_summary,
    log_rejection_summary,
    log_session_tokens_summary,
    pipeline_phase,
    reset_session_telemetry,
)


def _assign_clip_ids(clips: list[dict]) -> list[dict]:
    for c in clips:
        if not c.get("_wid"):
            c["_wid"] = uuid.uuid4().hex
        if not c.get("clip_id"):
            c["clip_id"] = c["_wid"]
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
    discovery_mode: bool = False,
) -> tuple[list[dict], list[dict], int]:
    """Run up to max_rounds expansion passes when selected count is below target."""
    if token_saver_mode and not discovery_mode:
        max_rounds = min(max_rounds, 1)
    elif discovery_mode:
        max_rounds = min(max_rounds, 2)
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
        merged_candidates = dedupe_candidates_exact_start(merged_candidates + extra)
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
    discovery_mode: bool = False,
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
    reset_session_telemetry()
    if media_duration is None or media_duration <= 0:
        media_duration = 300.0
    stats = PipelineStats(target_clips=target_count, discovery_mode=discovery_mode)
    oai = openai_config or PipelineOpenAIConfig()
    pool_stats = empty_pool_stats()
    ai_prof = get_ai_profile(oai.ai_profile_name or "SAFE")
    stats.ai_profile = ai_prof.name

    if oai.model_fast and oai.model_quality:
        resolved = ResolvedModels(
            fast_model=oai.model_fast,
            quality_model=oai.model_final or oai.model_quality,
            json_fallback_model=oai.json_fallback_model or ai_prof.json_fallback_model,
            profile_name=ai_prof.name,
        )
    else:
        resolved = resolve_models_from_profile(ai_prof)
    model_fast = resolved.fast_model
    model_quality = resolved.quality_model
    json_fallback = resolved.json_fallback_model
    stats.model_fast = model_fast
    stats.model_quality = model_quality

    reset_json_telemetry()

    style_name = clip_style if clip_style in ("Balanced", "Micro clips", "Long story clips") else "Balanced"
    n_passes, max_pass_rounds, _ = token_saver_pass_config(style_name)
    if not oai.token_saver_mode:
        n_passes = 3
        max_pass_rounds = 2

    _PASS_NAMES = ("primary", "gems", "micro")
    _max_gpt_passes = max(1, min(ai_prof.max_gpt_passes, 3))
    # Always run only the first pass upfront; additional passes are conditional
    # on the final clip count (see conditional second-pass block below).
    passes_override: tuple[str, ...] | None = tuple(_PASS_NAMES[:1])
    stats.gpt_passes_used = 1

    _trim_context_for_budget = False  # set True if token saver kicks in over budget

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
        _trim_context_for_budget = True
        n_passes, max_pass_rounds, _ = token_saver_pass_config(style_name)
        stats.warnings.append(
            f"Estimated ~{token_estimate.estimated_total_tokens:,} tokens exceeds budget "
            f"({oai.max_tokens_budget:,}). Token Saver Mode enforced; context trimmed to 2s/3s."
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
        json_fallback_model=json_fallback,
        model_fast=model_fast,
        model_quality=model_quality,
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
    if _trim_context_for_budget:
        ctx_b = min(ctx_b, 2.0)
        ctx_a = min(ctx_a, 3.0)
        logger.info("[PIPELINE] Context trimmed to %.0fs/%.0fs due to token budget", ctx_b, ctx_a)

    if discovery_mode:
        pool_multiplier = 2.5 if oai.token_saver_mode else 3.0
        pool_target = max(int(target_count * pool_multiplier), 50 if oai.token_saver_mode else 60)
    else:
        pool_multiplier = 1.5 if oai.token_saver_mode else 3.0
        pool_target = max(int(target_count * pool_multiplier), 30 if oai.token_saver_mode else 45)
    max_clips_region = profile.max_clips_per_region
    if oai.token_saver_mode and not discovery_mode:
        max_clips_region = max(6, int(max_clips_region * 0.65))
    elif discovery_mode:
        max_clips_region = min(22, max_clips_region + 4)

    score_floor = max(38, profile.min_score - (15 if discovery_mode else 0))
    prompt_settings = ClipPromptSettings(
        min_clip_seconds=max(user_min_seconds, profile.ideal_min_seconds * (0.75 if discovery_mode else 0.85)),
        max_clip_seconds=profile.ai_max_clip_seconds,
        ideal_min_seconds=profile.ideal_min_seconds,
        ideal_max_seconds=profile.ideal_max_seconds,
        max_clips=max_clips_region,
        min_score=score_floor,
        n_transcript_chunks=5,
    )

    min_acceptable = max(12, int(target_count * 0.6))

    gpu_shortlist: list[dict] = []
    region_filter_gpu: tuple[str, ...] | None = None
    gpu_explorer_rows: list[dict] = []
    if oai.enable_gpu_prefilter and segments and media_duration > 0:
        try:
            log_gpu_memory("before_embeddings")
            eff_max_clip = min(user_max_seconds, ai_prof.max_clip_length)
            with pipeline_phase("semantic_prefilter"):
                gpu_shortlist, gpu_stats = run_gpu_prefilter_pipeline(
                segments,
                media_duration,
                clip_style=style_name,
                user_min_seconds=user_min_seconds,
                user_max_seconds=eff_max_clip,
                target_count=target_count,
                pool_target=pool_target,
                ai_profile=ai_prof,
                )
            log_gpu_memory("after_embeddings")
            stats.gpu_local_candidates = int(gpu_stats.get("local_prefilter_count", 0))
            stats.gpu_shortlist = int(gpu_stats.get("shortlist_count", 0))
            stats.semantic_dedupe_removed = int(gpu_stats.get("semantic_dedupe_removed", 0))
            gpu_explorer_rows = list(gpu_stats.get("explorer_rows") or [])
            regions = gpu_stats.get("active_regions") or []
            if regions:
                region_filter_gpu = tuple(regions[: ai_prof.max_active_gpt_regions])
            est_refine = int(gpu_stats.get("estimated_refinement_tokens", 0))
            if est_refine > oai.max_tokens_budget:
                oai.token_saver_mode = True
                passes_override = tuple(
                    _PASS_NAMES[: max(1, min(ai_prof.max_gpt_passes, 1))]
                )
                stats.gpt_passes_used = len(passes_override)
                stats.warnings.append(
                    f"GPU prefilter estimated ~{est_refine:,} refinement tokens exceeds "
                    f"budget ({oai.max_tokens_budget:,}); limiting to 1 GPT pass."
                )
            stats.warnings.append(
                f"GPU prefilter: {stats.gpu_shortlist} shortlist "
                f"(raw={stats.gpu_local_candidates}, est refine tokens≈{est_refine:,}, "
                f"embeddings GPU: {gpu_stats.get('embeddings_on_gpu', False)})."
            )
        except Exception as exc:
            logger.warning("GPU prefilter skipped: %s", exc)
            stats.warnings.append(f"GPU prefilter unavailable: {exc}")
    stats.gpu_explorer_rows = gpu_explorer_rows

    try:
        candidates: list[dict] = list(gpu_shortlist)
        with pipeline_phase("openai_refinement"):
            ai_candidates = collect_candidates_multipass(
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
                discovery_mode=discovery_mode,
                pool_stats=pool_stats,
                passes_override=passes_override,
                region_filter_override=region_filter_gpu,
            )
        pool_stats["local_prefilter_candidates"] = stats.gpu_shortlist
        candidates.extend(ai_candidates)
        # Early pool dedupe: start times within 5s only (not user min_gap_seconds).
        before_early = len(candidates)
        candidates = dedupe_candidates_exact_start(candidates)
        stats.rejected_overlap_early = int(pool_stats.get("rejected_overlap_early", 0)) + (
            before_early - len(candidates)
        )
        stats.json_telemetry = get_json_telemetry()
        stats.raw_candidates = len(candidates)
        stats.raw_ai_candidates = pool_stats.get("raw_ai_candidates", 0)
        stats.valid_after_schema = pool_stats.get("valid_after_schema", 0)
        stats.rescued_candidates = pool_stats.get("rescued_candidates", 0)
        stats.rejected_invalid_time = pool_stats.get("rejected_invalid_time", 0)
        stats.rejected_duration = pool_stats.get("rejected_duration", 0)
        stats.rejected_empty_transcript = pool_stats.get("rejected_empty_transcript", 0)
        logger.info("Candidate pool after multipass: %d (target pool=%d)", len(candidates), pool_target)

        # Boost collection when pool is still small (always in discovery mode; full mode otherwise)
        if len(candidates) < pool_target and (discovery_mode or not oai.token_saver_mode):
            boost_settings = ClipPromptSettings(
                min_clip_seconds=prompt_settings.min_clip_seconds,
                max_clip_seconds=prompt_settings.max_clip_seconds,
                ideal_min_seconds=prompt_settings.ideal_min_seconds,
                ideal_max_seconds=prompt_settings.ideal_max_seconds,
                max_clips=min(prompt_settings.max_clips + 4, 22),
                min_score=max(38, score_floor - 5),
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
                discovery_mode=discovery_mode,
                pool_stats=pool_stats,
                passes_override=passes_override,
                region_filter_override=region_filter_gpu,
            )
            before_boost = len(candidates)
            candidates = dedupe_candidates_exact_start(candidates + extra_pool)
            stats.rejected_overlap_early += before_boost + len(extra_pool) - len(candidates)
            stats.json_telemetry = get_json_telemetry()
            stats.raw_candidates = len(candidates)
            stats.raw_ai_candidates = pool_stats.get("raw_ai_candidates", stats.raw_ai_candidates)
            stats.valid_after_schema = pool_stats.get("valid_after_schema", stats.valid_after_schema)
            stats.rescued_candidates = pool_stats.get("rescued_candidates", stats.rescued_candidates)

        min_acceptable = max(12, int(target_count * 0.6))
        if len(candidates) < min_acceptable and segments:
            fallback = generate_local_fallback_candidates(
                segments,
                media_duration,
                clip_style=style_name,
                profile=profile,
                target_count=target_count,
                existing=candidates,
                min_gap_seconds=max(25.0, min_gap_seconds * 0.85),
                user_min_seconds=user_min_seconds,
                user_max_seconds=user_max_seconds,
            )
            if fallback:
                before_fb = len(candidates)
                candidates = dedupe_candidates_exact_start(candidates + fallback)
                stats.rejected_overlap_early += before_fb + len(fallback) - len(candidates)
                stats.local_fallback_candidates = len(fallback)
                stats.raw_candidates = len(candidates)
                stats.warnings.append(
                    f"Added {len(fallback)} local transcript-window fallback candidate(s) (no extra GPT calls)."
                )

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

        # Conditional second GPT pass: only run if clip count is below 50% of target
        # and the AI profile allows more than one pass.
        if _max_gpt_passes >= 2 and len(selected) < target_count * 0.5:
            logger.info(
                "[PIPELINE] Only %d/%d clips after pass 1 (<50%%) — running conditional gems pass",
                len(selected), target_count,
            )
            extra_passes = tuple(_PASS_NAMES[1:_max_gpt_passes])
            if extra_passes:
                try:
                    gems_candidates = collect_candidates_multipass(
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
                        discovery_mode=discovery_mode,
                        pool_stats=pool_stats,
                        passes_override=extra_passes,
                        region_filter_override=region_filter_gpu,
                    )
                    if gems_candidates:
                        before_gems = len(candidates)
                        candidates = dedupe_candidates_exact_start(candidates + gems_candidates)
                        stats.rejected_overlap_early += before_gems + len(gems_candidates) - len(candidates)
                        stats.raw_candidates = len(candidates)
                        stats.gpt_passes_used += 1
                        selected, div_gems = run_diversity_pipeline(
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
                        stats.removed_overlap += div_gems.removed_overlap
                        stats.removed_duplicates += div_gems.removed_duplicates
                        stats.after_diversity = len(selected)
                        logger.info("[PIPELINE] After gems pass: %d clips", len(selected))
                except Exception as _gems_exc:
                    logger.warning("Conditional gems pass failed: %s", _gems_exc)
                    stats.warnings.append(f"Conditional gems pass skipped: {_gems_exc}")

        if len(selected) < target_count and candidates and (discovery_mode or not oai.token_saver_mode or len(selected) < 12):
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
                discovery_mode=discovery_mode,
            )
            stats.expansion_pass_count = exp_rounds
            stats.after_diversity = len(selected)

        if len(selected) < min_acceptable and segments:
            fallback_sel = generate_local_fallback_candidates(
                segments,
                media_duration,
                clip_style=style_name,
                profile=profile,
                target_count=target_count,
                existing=candidates,
                min_gap_seconds=max(25.0, min_gap_seconds * 0.75),
                user_min_seconds=user_min_seconds,
                user_max_seconds=user_max_seconds,
            )
            if fallback_sel:
                before_m = len(candidates)
                merged = dedupe_candidates_exact_start(candidates + fallback_sel)
                stats.rejected_overlap_early += before_m + len(fallback_sel) - len(merged)
                candidates = merged
                stats.local_fallback_candidates += len(fallback_sel)
                selected, div2 = run_diversity_pipeline(
                    merged,
                    media_duration=media_duration,
                    target_count=target_count,
                    min_gap_seconds=min_gap_seconds,
                    similarity_threshold=similarity_threshold,
                    n_regions=5,
                    min_per_region=1 if discovery_mode else 1,
                    relax_if_under_target=True,
                    return_stats=True,
                )
                stats.removed_overlap += div2.removed_overlap
                stats.removed_duplicates += div2.removed_duplicates
                stats.after_diversity = len(selected)

        with pipeline_phase("clip_splitting"):
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

        with pipeline_phase("metadata_grounding"):
            selected = ground_all_clips_metadata(
                selected,
                segments,
                api_key,
                tracker=tracker,
                force_regenerate=True,
                skip_strong_grounding=oai.token_saver_mode,
                resolved_models=resolved,
            )
        log_gpu_memory("after_openai_phase")

        grounded: list[dict] = []
        ungrounded_floor = 8 if discovery_mode else 15
        for c in selected:
            conf = int(c.get("grounding_confidence", 0))
            excerpt = str(c.get("grounded_transcript_excerpt", "")).strip()
            is_local = c.get("source") == "local_transcript_window"
            if is_local and discovery_mode:
                grounded.append(c)
                continue
            if conf < ungrounded_floor or (
                len(excerpt.split()) < 6 and not c.get("metadata_grounded")
            ):
                if discovery_mode and conf >= 5 and len(excerpt.split()) >= 4:
                    c.setdefault("warnings", []).append("Low grounding confidence (kept in Discovery Mode).")
                    grounded.append(c)
                    continue
                stats.rejected_ungrounded += 1
                log_clip_reject(
                    "metadata_weak",
                    grounding_confidence=conf,
                    candidate_clip=c.get("hook_title", ""),
                    threshold=ungrounded_floor,
                )
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

        max_part_len = min(user_max_seconds, ai_prof.max_clip_length)
        for c in selected:
            flag_split_recommended(c, max_part_len)
        selected, stats.series_splits = apply_recommended_series_splits(
            selected,
            formatted,
            segments=segments,
        )
        selected = _assign_clip_ids(selected)
        selected, stats.removed_weak_hook = filter_minimum_hook_score(selected)

        stats.final_clips = len(selected)

        if stats.final_clips < 12:
            stats.warnings.append(
                f"Only {stats.final_clips} clips found (target 12+). "
                "Enable Discovery Mode to rescue borderline moments and add local transcript-window candidates."
            )
        elif stats.final_clips < 15:
            stats.warnings.append(
                f"Only {stats.final_clips} clips found (preferred 15–20 for long podcasts)."
            )
        elif stats.final_clips < stats.target_clips:
            stats.warnings.append(
                f"Found {stats.final_clips} of {stats.target_clips} requested clips."
            )
        if discovery_mode and stats.final_clips < stats.target_clips:
            stats.warnings.append(
                "Discovery Mode is ON — quantity-first ranking with duplicate protection."
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

        token_plan = plan_analysis_token_estimate(
            formatted,
            ai_prof,
            target_count=target_count,
            clip_style=style_name,
        )
        stats.estimated_tokens = token_plan.after_prune

        return selected, stats, tracker

    except OpenAIRateLimitError as exc:
        stats.warnings.append(
            f"OpenAI rate limit at stage '{exc.stage}' (model {exc.model}). "
            f"{exc.mitigation} Partial progress saved — retry to resume."
        )
        raise
    finally:
        log_session_tokens_summary()
        log_rejection_summary()
        log_pipeline_timing_summary()
        set_call_context(None)


__all__ = [
    "PipelineOpenAIConfig",
    "PipelineStats",
    "run_full_clip_pipeline",
]
