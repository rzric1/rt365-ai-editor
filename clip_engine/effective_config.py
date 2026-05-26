"""
clip_engine/effective_config.py

Merge AI profile defaults with Streamlit widget state without mutating widget keys after render.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from clip_engine.ai_profiles import AIProfile, PROFILE_MAX_QUALITY, get_ai_profile, profile_from_ui_label
from clip_engine.openai_resilience import (
    PipelineTokenEstimate,
    estimate_pipeline_tokens,
    token_saver_pass_config,
)

logger = logging.getLogger("clip_engine.effective_config")

_PROFILE_SAFE = get_ai_profile("SAFE")


@dataclass(frozen=True)
class ResolvedModels:
    """Models resolved from profile / effective config / pipeline context."""

    fast_model: str
    quality_model: str
    json_fallback_model: str
    profile_name: str


def _safe_default_models() -> ResolvedModels:
    logger.error(
        "[AI PROFILE ERROR] Missing effective config; defaulting to SAFE gpt-4o-mini"
    )
    p = _PROFILE_SAFE
    return ResolvedModels(
        fast_model=p.fast_model,
        quality_model=p.quality_model,
        json_fallback_model=p.json_fallback_model,
        profile_name=p.name,
    )


def resolve_models_from_effective_config(
    config: ClipStudioEffectiveConfig | None,
) -> ResolvedModels:
    """Return fast/quality/fallback models from effective config."""
    if config is None:
        return _safe_default_models()
    return ResolvedModels(
        fast_model=config.fast_model,
        quality_model=config.quality_model,
        json_fallback_model=config.fallback_model,
        profile_name=config.profile_name,
    )


def resolve_models_from_profile(profile: AIProfile | str | None) -> ResolvedModels:
    """Return models from an AI profile (unknown names fall back to SAFE)."""
    if profile is None:
        return _safe_default_models()
    p = profile if isinstance(profile, AIProfile) else get_ai_profile(str(profile))
    return ResolvedModels(
        fast_model=p.fast_model,
        quality_model=p.quality_model,
        json_fallback_model=p.json_fallback_model,
        profile_name=p.name,
    )


def resolve_models_from_call_context() -> ResolvedModels:
    """Read models from OpenAICallContext set by the clip pipeline."""
    from clip_engine.openai_resilience import get_call_context

    ctx = get_call_context()
    fast = (ctx.model_fast or "").strip()
    quality = (ctx.model_quality or "").strip()
    fallback = (ctx.json_fallback_model or "").strip()
    if fast and quality:
        return ResolvedModels(
            fast_model=fast,
            quality_model=quality,
            json_fallback_model=fallback or quality,
            profile_name="pipeline",
        )
    return _safe_default_models()


def init_session_default(session_state: Any, key: str, value: Any) -> None:
    """Set session key only if absent (safe before widget creation)."""
    if key not in session_state:
        session_state[key] = value


@dataclass(frozen=True)
class ClipStudioEffectiveConfig:
    """Runtime config passed to the clip pipeline (profile + UI overrides)."""

    profile_name: str
    fast_model: str
    quality_model: str
    fallback_model: str
    token_budget: int
    token_saver: bool
    discovery_mode: bool
    gpu_prefilter: bool
    max_gpt_passes: int
    max_active_gpt_regions: int
    shortlist_min: int
    shortlist_max: int
    min_clip_length: float
    max_clip_length: float
    context_before: float
    context_after: float
    min_gap_seconds: float
    duplicate_similarity: float
    target_clips: int

    @classmethod
    def from_session(cls, session_state: Any) -> ClipStudioEffectiveConfig:
        label = str(session_state.get("cs_ai_profile_label", "SAFE (Recommended)"))
        profile = profile_from_ui_label(label)
        widget_gpu = bool(session_state.get("cs_enable_gpu_prefilter", True))
        return cls(
            profile_name=profile.name,
            fast_model=profile.fast_model,
            quality_model=profile.quality_model,
            fallback_model=profile.json_fallback_model,
            token_budget=profile.max_tokens,
            token_saver=profile.token_saver,
            discovery_mode=bool(
                session_state.get("cs_discovery_mode", profile.discovery_mode)
            ),
            gpu_prefilter=profile.prefer_gpu_prefilter and widget_gpu,
            max_gpt_passes=profile.max_gpt_passes,
            max_active_gpt_regions=profile.max_active_gpt_regions,
            shortlist_min=profile.target_gpu_shortlist_min,
            shortlist_max=profile.target_gpu_shortlist_max,
            min_clip_length=float(session_state.get("cs_min_clip_seconds", 25)),
            max_clip_length=min(
                float(session_state.get("cs_max_clip_seconds", profile.max_clip_length)),
                profile.max_clip_length,
            ),
            context_before=float(
                session_state.get("cs_context_before", profile.context_before)
            ),
            context_after=float(
                session_state.get("cs_context_after", profile.context_after)
            ),
            min_gap_seconds=float(session_state.get("cs_min_gap_seconds", 60)),
            duplicate_similarity=float(session_state.get("cs_similarity_threshold", 45))
            / 100.0,
            target_clips=int(session_state.get("cs_target_clips", 20)),
        )


@dataclass
class TokenPlanResult:
    estimate: PipelineTokenEstimate
    before_prune: int
    after_prune: int
    budget: int
    pruned_reason: str | None
    effective_regions: int
    effective_passes: int
    effective_grounding_targets: int
    max_chunk_chars: int
    include_grounding: bool
    include_split: bool

    def to_dict(self) -> dict:
        d = self.estimate.to_dict()
        d.update(
            {
                "before_prune": self.before_prune,
                "after_prune": self.after_prune,
                "budget": self.budget,
                "pruned_reason": self.pruned_reason,
                "effective_regions": self.effective_regions,
                "effective_passes": self.effective_passes,
            }
        )
        return d


def plan_analysis_token_estimate(
    formatted_transcript: str,
    profile: AIProfile,
    *,
    target_count: int = 20,
    clip_style: str = "Balanced",
) -> TokenPlanResult:
    """
    Estimate GPT usage with profile limits; progressively prune until within budget.
    """
    budget = profile.max_tokens
    style_passes, style_rounds, _ = token_saver_pass_config(clip_style)
    base_passes = min(style_passes, profile.max_gpt_passes)
    base_rounds = 1 if profile.token_saver else style_rounds
    max_regions = profile.max_active_gpt_regions
    grounding_cap = min(target_count, profile.target_gpu_shortlist_max)

    initial = estimate_pipeline_tokens(
        formatted_transcript,
        target_count=target_count,
        n_regions=max_regions,
        n_passes=base_passes,
        max_pass_rounds=base_rounds,
        token_saver_mode=profile.token_saver,
    )
    before_prune = initial.estimated_total_tokens

    if before_prune <= budget or profile.name == PROFILE_MAX_QUALITY:
        logger.info(
            "[TOKEN PLAN] before_prune=%d after_prune=%d budget=%d",
            before_prune,
            before_prune,
            budget,
        )
        return TokenPlanResult(
            estimate=initial,
            before_prune=before_prune,
            after_prune=before_prune,
            budget=budget,
            pruned_reason=None,
            effective_regions=max_regions,
            effective_passes=base_passes,
            effective_grounding_targets=target_count,
            max_chunk_chars=8_000,
            include_grounding=True,
            include_split=True,
        )

    chunk_sizes = [8_000, 6_000, 4_500, 3_000] if profile.token_saver else [10_000, 8_000]
    region_range = range(max_regions, 0, -1)
    grounding_targets = [grounding_cap, max(5, grounding_cap // 2), 5]
    split_options = [True, False] if profile.token_saver else [True]
    grounding_options = [True, False] if profile.token_saver else [True]

    best: PipelineTokenEstimate | None = None
    best_meta: dict[str, Any] = {}

    for max_chunk in chunk_sizes:
        for regions in region_range:
            for g_target in grounding_targets:
                for include_split in split_options:
                    for include_grounding in grounding_options:
                        est = estimate_pipeline_tokens(
                            formatted_transcript,
                            target_count=g_target if include_grounding else 0,
                            n_regions=regions,
                            n_passes=1,
                            max_pass_rounds=1,
                            max_chunk_chars=max_chunk,
                            include_grounding=include_grounding,
                            include_split=include_split,
                            token_saver_mode=True,
                        )
                        if est.estimated_total_tokens <= budget:
                            reason = (
                                f"regions={regions},passes=1,grounding={g_target if include_grounding else 0},"
                                f"split={include_split},chunk={max_chunk}"
                            )
                            logger.info(
                                "[TOKEN PLAN] before_prune=%d after_prune=%d budget=%d",
                                before_prune,
                                est.estimated_total_tokens,
                                budget,
                            )
                            logger.info("[TOKEN PLAN] pruned_reason=%s", reason)
                            return TokenPlanResult(
                                estimate=est,
                                before_prune=before_prune,
                                after_prune=est.estimated_total_tokens,
                                budget=budget,
                                pruned_reason=reason,
                                effective_regions=regions,
                                effective_passes=1,
                                effective_grounding_targets=g_target,
                                max_chunk_chars=max_chunk,
                                include_grounding=include_grounding,
                                include_split=include_split,
                            )
                        if best is None or est.estimated_total_tokens < best.estimated_total_tokens:
                            best = est
                            best_meta = {
                                "regions": regions,
                                "g_target": g_target,
                                "include_split": include_split,
                                "include_grounding": include_grounding,
                                "max_chunk": max_chunk,
                            }

    assert best is not None
    reason = (
        f"minimal_plan_still_over_budget:{best_meta};"
        f"after={best.estimated_total_tokens}"
    )
    logger.warning(
        "[TOKEN PLAN] before_prune=%d after_prune=%d budget=%d",
        before_prune,
        best.estimated_total_tokens,
        budget,
    )
    logger.warning("[TOKEN PLAN] pruned_reason=%s", reason)
    return TokenPlanResult(
        estimate=best,
        before_prune=before_prune,
        after_prune=best.estimated_total_tokens,
        budget=budget,
        pruned_reason=reason,
        effective_regions=int(best_meta.get("regions", 1)),
        effective_passes=1,
        effective_grounding_targets=int(best_meta.get("g_target", 5)),
        max_chunk_chars=int(best_meta.get("max_chunk", 3_000)),
        include_grounding=bool(best_meta.get("include_grounding", False)),
        include_split=bool(best_meta.get("include_split", False)),
    )


def apply_profile_non_widget_keys(session_state: Any, profile: AIProfile) -> None:
    """Sync profile to session keys that are not bound to Streamlit widgets."""
    session_state.cs_ai_profile = profile.name
    session_state.cs_token_saver_mode = profile.token_saver
    session_state.cs_discovery_mode = profile.discovery_mode
    session_state.cs_max_tokens_budget = profile.max_tokens


def apply_profile_widget_defaults(session_state: Any, profile: AIProfile) -> None:
    """
    Initialize widget-bound keys only when missing (call before widgets render).
    """
    init_session_default(session_state, "cs_enable_gpu_prefilter", profile.prefer_gpu_prefilter)
    init_session_default(
        session_state, "cs_max_clip_seconds", int(profile.max_clip_length)
    )
    init_session_default(
        session_state, "cs_context_before", int(profile.context_before)
    )
    init_session_default(
        session_state, "cs_context_after", int(profile.context_after)
    )


__all__ = [
    "ClipStudioEffectiveConfig",
    "ResolvedModels",
    "TokenPlanResult",
    "apply_profile_non_widget_keys",
    "apply_profile_widget_defaults",
    "init_session_default",
    "plan_analysis_token_estimate",
    "resolve_models_from_call_context",
    "resolve_models_from_effective_config",
    "resolve_models_from_profile",
]
