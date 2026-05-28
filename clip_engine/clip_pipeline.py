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
    cache_miss_reason: str = ""
    resumed_from_progress: bool = False
    estimated_tokens: int = 0
    model_fast: str = ""
    model_quality: str = ""
    boundary_repairs: int = 0
    title_repairs: int = 0
    quality_gate_repaired: int = 0
    openai_calls_used: int = 0
    clip_strategy: str = "Balanced"
    platform_target: str = "TikTok/Reels/Shorts"
    title_style: str = "Curiosity"
    finalizer_expanded: int = 0
    finalizer_merged: int = 0
    finalizer_rejected: int = 0
    finalizer_hooks_repaired: int = 0
    finalizer_report: dict = field(default_factory=dict)
    discovery_scan: dict = field(default_factory=dict)
    discovery_forensics: dict = field(default_factory=dict)
    duration_governor: dict = field(default_factory=dict)
    timeline_occupancy: dict = field(default_factory=dict)

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
            "cache_miss_reason": self.cache_miss_reason,
            "resumed_from_progress": self.resumed_from_progress,
            "boundary_repairs": self.boundary_repairs,
            "title_repairs": self.title_repairs,
            "quality_gate_repaired": self.quality_gate_repaired,
            "openai_calls_used": self.openai_calls_used,
            "clip_strategy": self.clip_strategy,
            "platform_target": self.platform_target,
            "title_style": self.title_style,
            "finalizer_expanded": self.finalizer_expanded,
            "finalizer_merged": self.finalizer_merged,
            "finalizer_rejected": self.finalizer_rejected,
            "finalizer_hooks_repaired": self.finalizer_hooks_repaired,
            "finalizer_report": self.finalizer_report,
            "discovery_scan": self.discovery_scan,
            "discovery_forensics": self.discovery_forensics,
            "duration_governor": self.duration_governor,
            "timeline_occupancy": self.timeline_occupancy,
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
from clip_engine.clip_boundaries import apply_boundary_repairs
from clip_engine.clip_scoring import apply_virality_to_clips, ensure_all_clip_hooks
from clip_engine.clip_finalizer import finalize_clips_with_report
from clip_engine.clip_duration_governor import (
    HARD_CAP_SECONDS,
    SOFT_CAP_SECONDS,
    apply_duration_policy_batch,
    compute_timeline_occupancy,
    log_over_soft_justifications,
    refresh_expansion_diagnostics,
)
from clip_engine.clip_quality_gate import run_quality_gate
from clip_engine.clip_style import ClipStyle, get_clip_style_profile
from clip_engine.ai_profiles import get_ai_profile
from clip_engine.clip_discovery import empty_pool_stats, generate_local_fallback_candidates
from clip_engine.discovery_forensics import DiscoveryForensics
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


def _apply_clip_quality_finalizer(
    clips: list[dict],
    segments: list[dict],
    stats: PipelineStats,
    *,
    user_min_seconds: float,
    hard_max: float,
) -> list[dict]:
    """Clip Quality Finalizer — last pass before UI/cache/export."""
    if not clips:
        return clips
    before = len(clips)
    with pipeline_phase("clip_quality_finalizer"):
        finalized, report = finalize_clips_with_report(
            clips,
            transcript_segments=segments,
            min_duration=user_min_seconds,
            max_duration=hard_max,
            merge_gap_seconds=20.0,
            logger=logger,
        )
    stats.finalizer_expanded = report.expanded
    stats.finalizer_merged = report.merged
    stats.finalizer_rejected = report.rejected
    stats.finalizer_hooks_repaired = report.hooks_repaired
    stats.finalizer_report = report.to_dict()
    stats.final_clips = len(finalized)
    if before > len(finalized):
        stats.warnings.append(
            f"Clip finalizer removed {before - len(finalized)} weak or fragmented clip(s)."
        )
    return finalized


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
    clip_strategy: str = "Balanced",
    platform_target: str = "TikTok/Reels/Shorts",
    title_style: str = "Curiosity",
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
    stats = PipelineStats(
        target_clips=target_count,
        discovery_mode=discovery_mode,
        clip_strategy=clip_strategy,
        platform_target=platform_target,
        title_style=title_style,
    )
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
        max_clip_seconds=min(user_max_seconds, ai_prof.max_clip_length, 120.0),
        min_gap_seconds=min_gap_seconds,
        similarity_threshold=similarity_threshold,
        token_saver_mode=oai.token_saver_mode,
        model_fast=model_fast,
        model_quality=model_quality,
        context_before=context_before if context_before is not None else 8.0,
        context_after=context_after if context_after is not None else 12.0,
        discovery_mode=discovery_mode,
        ai_profile_name=ai_prof.name,
        clip_strategy=clip_strategy,
        platform_target=platform_target,
        title_style=title_style,
    )
    cache_key = cache_key_obj.digest()

    if oai.use_cache:
        cached = load_cached_analysis(cache_key)
        if cached:
            tracker.record_cache_hit(cached.get("token_usage", {}).get("total_tokens", stats.estimated_tokens))
            stats.cache_hit = True
            cached_clips = list(cached.get("clips", []))
            if cached_clips and segments:
                cached_clips, cached_hook_fixes = ensure_all_clip_hooks(cached_clips, segments)
                stats.title_repairs = int(cached.get("stats", {}).get("title_repairs", 0)) + cached_hook_fixes
            hard_max_cached = min(user_max_seconds, ai_prof.max_clip_length, HARD_CAP_SECONDS)
            soft_cached = min(SOFT_CAP_SECONDS, hard_max_cached)
            cached_clips = _apply_clip_quality_finalizer(
                cached_clips,
                segments,
                stats,
                user_min_seconds=user_min_seconds,
                hard_max=soft_cached,
            )
            if cached_clips and media_duration > 0:
                cached_clips, gov_cached = apply_duration_policy_batch(
                    cached_clips, media_duration, pre_virality=False,
                )
                stats.duration_governor["cache_rehydrate"] = gov_cached.to_dict()
            stats.final_clips = len(cached_clips)
            cached_stats = cached.get("stats") or {}
            stats.boundary_repairs = int(cached_stats.get("boundary_repairs", 0))
            if not stats.title_repairs:
                stats.title_repairs = int(cached_stats.get("title_repairs", 0))
            stats.openai_calls_used = 0
            stats.warnings.append("Loaded cached analysis — no OpenAI tokens used.")
            logger.info("[CACHE] hit — skipping OpenAI pipeline (cache reuse)")
            if oai.use_cache and cached_clips:
                save_cached_analysis(
                    cache_key,
                    clips=cached_clips,
                    stats=stats.to_dict(),
                    token_usage=cached.get("token_usage", {}),
                    analysis_fingerprint=cache_key,
                )
            return cached_clips, stats, tracker
        stats.cache_miss_reason = "no_cached_entry_or_version"
        logger.info("[CACHE] miss — running full pipeline")

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
        pool_multiplier = 3.0 if oai.token_saver_mode else 3.5
        pool_target = max(int(target_count * pool_multiplier), 55 if oai.token_saver_mode else 70)
        if media_duration >= 40 * 60:
            pool_target = max(pool_target, 60)
            pool_multiplier = max(pool_multiplier, 3.2)
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

    forensics = DiscoveryForensics()
    forensics.record_stage(
        "pipeline_inputs",
        input_count=len(segments or []),
        output_count=len(segments or []) if segments and media_duration > 0 else 0,
        note=(
            f"media_duration={media_duration:.0f}s discovery_mode={discovery_mode} "
            f"gpu_prefilter={oai.enable_gpu_prefilter} pool_target={pool_target}"
        ),
    )
    if not segments or media_duration <= 0:
        forensics.notes.append("Starvation: no transcript segments or zero media duration.")
        stats.discovery_forensics = forensics.to_dict()
        stats.warnings.append(
            f"Discovery forensic: first zero at '{forensics.first_zero_stage}' — no transcript."
        )

    gpu_shortlist: list[dict] = []
    region_filter_gpu: tuple[str, ...] | None = None
    gpu_explorer_rows: list[dict] = []
    gpu_stats: dict = {}
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
                discovery_mode=discovery_mode,
                forensics=forensics,
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
            stats.discovery_scan = dict(gpu_stats.get("discovery_scan") or stats.discovery_scan)
            forensics.merge_scan_stats(stats.discovery_scan)
        except Exception as exc:
            logger.warning("GPU prefilter skipped: %s", exc)
            stats.warnings.append(f"GPU prefilter unavailable: {exc}")
            stats.gpu_explorer_rows = gpu_explorer_rows
            stats.discovery_scan = dict(gpu_stats.get("discovery_scan") or {})
            forensics.record_stage(
                "gpu_prefilter_exception",
                input_count=len(segments),
                output_count=0,
                rejection_reasons={"exception": 1},
                note=str(exc)[:200],
            )
    else:
        forensics.record_stage(
            "gpu_prefilter_skipped",
            input_count=len(segments or []),
            output_count=0,
            rejection_reasons={
                "disabled": int(not oai.enable_gpu_prefilter),
                "no_segments": int(not bool(segments)),
                "no_duration": int(media_duration <= 0),
            },
            note="GPU prefilter not run",
        )

    try:
        candidates: list[dict] = list(gpu_shortlist)
        forensics.record_stage(
            "gpu_shortlist_to_pool",
            input_count=stats.gpu_shortlist,
            output_count=len(gpu_shortlist),
        )
        with pipeline_phase("openai_refinement"):
            before_openai = len(candidates)
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
        forensics.record_stage(
            "openai_multipass",
            input_count=before_openai,
            output_count=len(ai_candidates),
            rejected_count=max(
                0,
                pool_stats.get("rejected_invalid_time", 0)
                + pool_stats.get("rejected_duration", 0)
                + pool_stats.get("rejected_empty_transcript", 0),
            ),
            rejection_reasons={
                "invalid_time": pool_stats.get("rejected_invalid_time", 0),
                "duration": pool_stats.get("rejected_duration", 0),
                "empty_transcript": pool_stats.get("rejected_empty_transcript", 0),
            },
            note=f"raw_ai={pool_stats.get('raw_ai_candidates', 0)} valid={pool_stats.get('valid_after_schema', 0)}",
        )
        pool_stats["local_prefilter_candidates"] = stats.gpu_shortlist
        candidates.extend(ai_candidates)
        forensics.record_stage(
            "pool_after_openai",
            input_count=before_openai,
            output_count=len(candidates),
        )
        # Early pool dedupe: start times within 5s only (not user min_gap_seconds).
        before_early = len(candidates)
        candidates = dedupe_candidates_exact_start(candidates)
        overlap_removed = before_early - len(candidates)
        stats.rejected_overlap_early = int(pool_stats.get("rejected_overlap_early", 0)) + (
            overlap_removed
        )
        forensics.record_stage(
            "early_start_dedupe",
            input_count=before_early,
            output_count=len(candidates),
            rejected_count=overlap_removed,
            rejection_reasons={"duplicate_start_within_5s": overlap_removed},
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

        fallback_trigger = max(10, int(target_count * 0.5))
        if discovery_mode:
            fallback_trigger = max(10, int(target_count * 0.75))
        needs_fallback = len(candidates) < fallback_trigger
        if segments and (needs_fallback or (discovery_mode and stats.gpu_shortlist < 5)):
            before_fb_pool = len(candidates)
            fallback = generate_local_fallback_candidates(
                segments,
                media_duration,
                clip_style=style_name,
                profile=profile,
                target_count=target_count,
                existing=candidates,
                min_gap_seconds=max(18.0, min_gap_seconds * 0.7) if discovery_mode else max(25.0, min_gap_seconds * 0.85),
                user_min_seconds=user_min_seconds,
                user_max_seconds=user_max_seconds,
                discovery_mode=discovery_mode,
                forensics=forensics,
            )
            forensics.record_stage(
                "local_fallback_triggered",
                input_count=before_fb_pool,
                output_count=before_fb_pool + len(fallback),
                note=f"needs_fallback={needs_fallback} generated={len(fallback)}",
            )
            if fallback:
                before_fb = len(candidates)
                candidates = dedupe_candidates_exact_start(candidates + fallback)
                stats.rejected_overlap_early += before_fb + len(fallback) - len(candidates)
                stats.local_fallback_candidates = len(fallback)
                stats.raw_candidates = len(candidates)
                if not stats.discovery_scan:
                    stats.discovery_scan = forensics.to_dict()
                stats.warnings.append(
                    f"Added {len(fallback)} local transcript-window fallback candidate(s) (no extra GPT calls)."
                )
        elif needs_fallback or (discovery_mode and stats.gpu_shortlist < 5):
            forensics.record_stage(
                "local_fallback_empty",
                input_count=len(candidates),
                output_count=len(candidates),
                note="fallback trigger fired but generate_local_fallback returned 0",
            )

        forensics.record_stage(
            "pre_diversity_pool",
            input_count=stats.raw_candidates,
            output_count=len(candidates),
        )
        stats.discovery_forensics = forensics.to_dict()
        if forensics.first_zero_stage:
            stats.warnings.append(
                f"Discovery forensic: first zero-output stage='{forensics.first_zero_stage}' "
                f"(gpu_shortlist={stats.gpu_shortlist}, raw={stats.raw_candidates}, "
                f"fallback={stats.local_fallback_candidates})."
            )
        logger.info(
            "[DISCOVERY FORENSIC] first_zero=%s gpu_gen=%d gpu_rej=%d windows_scanned=%d "
            "fallback_gen=%d",
            forensics.first_zero_stage or "none",
            forensics.gpu_candidates_generated,
            forensics.gpu_candidates_rejected,
            forensics.windows_scanned,
            forensics.fallback_candidates_generated,
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
                min_gap_seconds=max(18.0, min_gap_seconds * 0.7) if discovery_mode else max(25.0, min_gap_seconds * 0.75),
                user_min_seconds=user_min_seconds,
                user_max_seconds=user_max_seconds,
                discovery_mode=discovery_mode,
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
        selected, gov_pre = apply_duration_policy_batch(
            selected, media_duration, pre_virality=True,
        )
        stats.duration_governor["after_context_expand"] = gov_pre.to_dict()
        stats.timeline_occupancy["after_context_expand"] = compute_timeline_occupancy(
            selected, media_duration,
        )

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

        hard_max = min(user_max_seconds, ai_prof.max_clip_length, HARD_CAP_SECONDS)
        soft_max = min(SOFT_CAP_SECONDS, hard_max, profile.expansion_max_seconds)
        with pipeline_phase("boundary_repair"):
            selected, stats.boundary_repairs = apply_boundary_repairs(
                selected,
                segments,
                max_duration=soft_max,
                min_duration=user_min_seconds,
            )
        selected, gov_boundary = apply_duration_policy_batch(
            selected, media_duration, pre_virality=True,
        )
        stats.duration_governor["after_boundary_repair"] = gov_boundary.to_dict()

        with pipeline_phase("virality_scoring"):
            selected, stats.title_repairs = apply_virality_to_clips(
                selected,
                segments,
                clip_strategy=clip_strategy,
                platform_target=platform_target,
                title_style=title_style,
            )

        selected, gov_viral = apply_duration_policy_batch(
            selected, media_duration, pre_virality=False,
        )
        stats.duration_governor["after_virality"] = gov_viral.to_dict()

        with pipeline_phase("quality_gate"):
            selected, qg_stats = run_quality_gate(
                selected,
                segments,
                media_duration=media_duration,
                max_duration=soft_max,
                min_duration=user_min_seconds,
            )
            stats.quality_gate_repaired = qg_stats.repaired
            if qg_stats.dropped:
                stats.warnings.append(
                    f"Quality gate dropped {qg_stats.dropped} unusable clip(s)."
                )

        max_part_len = hard_max
        for c in selected:
            flag_split_recommended(c, max_part_len)
        selected, stats.series_splits = apply_recommended_series_splits(
            selected,
            formatted,
            segments=segments,
        )
        selected = _assign_clip_ids(selected)
        selected, stats.removed_weak_hook = filter_minimum_hook_score(selected)

        with pipeline_phase("final_hook_repair"):
            selected, final_hook_repairs = ensure_all_clip_hooks(selected, segments)
            stats.title_repairs += final_hook_repairs

        selected = _apply_clip_quality_finalizer(
            selected,
            segments,
            stats,
            user_min_seconds=user_min_seconds,
            hard_max=soft_max,
        )

        selected, gov_final = apply_duration_policy_batch(
            selected, media_duration, pre_virality=False,
        )
        stats.duration_governor["after_finalizer"] = gov_final.to_dict()
        stats.timeline_occupancy["final"] = compute_timeline_occupancy(
            selected, media_duration,
        )
        selected = [refresh_expansion_diagnostics(c) for c in selected]
        log_over_soft_justifications(selected, stage="final_export_window")
        if gov_final.over_soft_before or gov_final.clamped_soft or gov_final.clamped_hard:
            stats.warnings.append(
                f"Duration governor: {gov_final.clamped_soft} soft-clamp(s), "
                f"{gov_final.clamped_hard} hard-clamp(s) "
                f"({gov_final.over_soft_before} clip(s) exceeded {SOFT_CAP_SECONDS:.0f}s before final pass)."
            )

        stats.final_clips = len(selected)
        export_usage = tracker.to_export_dict(
            target_clips=target_count,
            final_clip_count=len(selected),
            model=model_quality,
        )
        stats.openai_calls_used = int(export_usage.get("total_calls", 0))

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
                token_usage=export_usage,
                analysis_fingerprint=cache_key,
            )

        stats.estimated_tokens = int(
            stats.estimated_tokens or token_estimate.estimated_total_tokens
        )
        logger.info(
            "[PIPELINE] completed clips=%d cache_hit=%s finalizer_rejected=%d",
            stats.final_clips,
            stats.cache_hit,
            stats.finalizer_rejected,
        )

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
