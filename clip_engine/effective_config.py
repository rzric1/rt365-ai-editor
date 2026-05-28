"""
clip_engine/effective_config.py

Merge AI profile defaults with Streamlit widget state without mutating widget keys after render.
Durable analysis snapshots and cache fingerprints live here (not in widget keys).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from typing import Any

from clip_engine.ai_profiles import AIProfile, PROFILE_MAX_QUALITY, get_ai_profile, profile_from_ui_label
from clip_engine.openai_resilience import (
    PipelineTokenEstimate,
    estimate_pipeline_tokens,
    token_saver_pass_config,
)

logger = logging.getLogger("clip_engine.effective_config")

_PROFILE_SAFE = get_ai_profile("SAFE")

# Session keys (durable — not Streamlit widget keys)
SESSION_ANALYSIS_FINGERPRINT = "cs_analysis_fingerprint"
SESSION_EFFECTIVE_CONFIG = "cs_durable_effective_config"
SESSION_ANALYSIS_DIAGNOSTICS = "cs_analysis_diagnostics"
SESSION_CLIP_EDITS = "cs_clip_ui_edits"
SESSION_FORCE_REANALYZE = "cs_force_reanalyze"
SESSION_RESOLVED_MODELS = "cs_durable_resolved_models"
SESSION_TOKEN_PLAN_CACHE_KEY = "cs_token_plan_cache_key"
SESSION_TOKEN_PLAN_CACHE = "cs_token_plan_cache"

_TOKEN_PLAN_LOGGED_KEYS: set[str] = set()

# Creator control defaults
DEFAULT_CLIP_STRATEGY = "Balanced"
DEFAULT_PLATFORM_TARGET = "TikTok/Reels/Shorts"
DEFAULT_TITLE_STYLE = "Curiosity"


@dataclass(frozen=True)
class ResolvedModels:
    """Models resolved from profile / effective config / pipeline context."""

    fast_model: str
    quality_model: str
    json_fallback_model: str
    profile_name: str


def _safe_default_models(*, reason: str = "") -> ResolvedModels:
    if reason:
        logger.debug("[AI PROFILE] %s — using SAFE gpt-4o-mini defaults", reason)
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
        return _safe_default_models(reason="missing effective config")
    return ResolvedModels(
        fast_model=config.fast_model,
        quality_model=config.quality_model,
        json_fallback_model=config.fallback_model,
        profile_name=config.profile_name,
    )


def resolve_models_from_profile(profile: AIProfile | str | None) -> ResolvedModels:
    """Return models from an AI profile (unknown names fall back to SAFE)."""
    if profile is None:
        return _safe_default_models(reason="missing profile")
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
    return resolve_models_from_profile(_PROFILE_SAFE)


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
    clip_strategy: str = DEFAULT_CLIP_STRATEGY
    platform_target: str = DEFAULT_PLATFORM_TARGET
    title_style: str = DEFAULT_TITLE_STYLE
    clip_style: str = "Balanced"

    @classmethod
    def from_session(cls, session_state: Any) -> ClipStudioEffectiveConfig:
        label = str(session_state.get("cs_ai_profile_label", "SAFE (Recommended)"))
        profile = profile_from_ui_label(label)
        widget_gpu = bool(session_state.get("cs_enable_gpu_prefilter", True))
        raw_max = float(session_state.get("cs_max_clip_seconds", profile.max_clip_length))
        capped_max = min(raw_max, profile.max_clip_length, 120.0)
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
            max_clip_length=capped_max,
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
            clip_strategy=str(
                session_state.get("cs_clip_strategy", DEFAULT_CLIP_STRATEGY)
            ),
            platform_target=str(
                session_state.get("cs_platform_target", DEFAULT_PLATFORM_TARGET)
            ),
            title_style=str(session_state.get("cs_title_style", DEFAULT_TITLE_STYLE)),
            clip_style=str(session_state.get("cs_clip_style", "Balanced")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipStudioEffectiveConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: data[k] for k in names if k in data})


def build_analysis_fingerprint(
    session_state: Any,
    *,
    video_identity: str,
    transcript_hash: str,
) -> str:
    """
    Stable fingerprint for analysis cache invalidation.
    Excludes widget edit state (hook titles, export checkboxes, trim fields).
    """
    effective = ClipStudioEffectiveConfig.from_session(session_state)
    payload = {
        "video": video_identity,
        "transcript": transcript_hash,
        "profile": effective.profile_name,
        "discovery": effective.discovery_mode,
        "token_saver": effective.token_saver,
        "gpu_prefilter": effective.gpu_prefilter,
        "min_clip": effective.min_clip_length,
        "max_clip": effective.max_clip_length,
        "ctx_before": effective.context_before,
        "ctx_after": effective.context_after,
        "min_gap": effective.min_gap_seconds,
        "similarity": round(effective.duplicate_similarity, 3),
        "target": effective.target_clips,
        "clip_style": effective.clip_style,
        "clip_strategy": effective.clip_strategy,
        "platform_target": effective.platform_target,
        "title_style": effective.title_style,
        "fast_model": effective.fast_model,
        "quality_model": effective.quality_model,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_invalidation_reason(
    session_state: Any,
    *,
    video_identity: str,
    transcript_hash: str,
) -> str | None:
    """
    Return why analysis should re-run, or None if cached session analysis is still valid.
    """
    if session_state.get(SESSION_FORCE_REANALYZE):
        return "explicit_reanalyze"
    if not session_state.get("cs_formatted"):
        return "missing_transcript"
    if not video_identity:
        return "missing_video"
    prev_video = str(session_state.get("cs_analysis_video_identity", ""))
    if prev_video and prev_video != video_identity:
        return "video_changed"
    prev_hash = str(session_state.get("cs_analysis_transcript_hash", ""))
    if prev_hash and prev_hash != transcript_hash:
        return "transcript_changed"
    fp = build_analysis_fingerprint(session_state, video_identity=video_identity, transcript_hash=transcript_hash)
    prev_fp = str(session_state.get(SESSION_ANALYSIS_FINGERPRINT, ""))
    if prev_fp and prev_fp != fp:
        return "ai_settings_changed"
    if not session_state.get("cs_clips") and not prev_fp:
        return None
    if prev_fp == fp and session_state.get("cs_clips"):
        return None
    return None


def store_analysis_snapshot(
    session_state: Any,
    *,
    effective: ClipStudioEffectiveConfig,
    fingerprint: str,
    video_identity: str,
    transcript_hash: str,
    diagnostics: dict[str, Any],
) -> None:
    """Persist durable analysis state (survives widget reruns)."""
    session_state[SESSION_EFFECTIVE_CONFIG] = effective.to_dict()
    session_state[SESSION_ANALYSIS_FINGERPRINT] = fingerprint
    session_state["cs_analysis_video_identity"] = video_identity
    session_state["cs_analysis_transcript_hash"] = transcript_hash
    session_state[SESSION_ANALYSIS_DIAGNOSTICS] = diagnostics
    session_state[SESSION_RESOLVED_MODELS] = {
        "fast_model": effective.fast_model,
        "quality_model": effective.quality_model,
        "json_fallback_model": effective.fallback_model,
        "profile_name": effective.profile_name,
    }
    session_state[SESSION_FORCE_REANALYZE] = False
    logger.info(
        "[ANALYSIS] config restored from session state fingerprint=%s profile=%s",
        fingerprint,
        effective.profile_name,
    )


def get_durable_effective_config(session_state: Any) -> ClipStudioEffectiveConfig | None:
    """Return last analysis effective config, or None."""
    raw = session_state.get(SESSION_EFFECTIVE_CONFIG)
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return ClipStudioEffectiveConfig.from_dict(raw)
    except (TypeError, KeyError) as exc:
        logger.warning("[ANALYSIS] Invalid durable config: %s", exc)
        return None


def resolve_models_for_session(session_state: Any) -> ResolvedModels:
    """Prefer durable analysis config; fall back to live session."""
    raw_models = session_state.get(SESSION_RESOLVED_MODELS)
    if isinstance(raw_models, dict) and raw_models.get("fast_model") and raw_models.get("quality_model"):
        logger.debug("[ANALYSIS] config restored from session state (resolved models)")
        return ResolvedModels(
            fast_model=str(raw_models["fast_model"]),
            quality_model=str(raw_models["quality_model"]),
            json_fallback_model=str(
                raw_models.get("json_fallback_model") or raw_models["quality_model"]
            ),
            profile_name=str(raw_models.get("profile_name", "SAFE")),
        )
    durable = get_durable_effective_config(session_state)
    if durable is not None:
        logger.debug("[ANALYSIS] config restored from durable effective config")
        return resolve_models_from_effective_config(durable)
    return resolve_models_from_effective_config(
        ClipStudioEffectiveConfig.from_session(session_state)
    )


def log_widget_rerun_noop(session_state: Any) -> None:
    """Log when a Streamlit rerun does not invalidate analysis."""
    if session_state.get("cs_clips") and session_state.get(SESSION_ANALYSIS_FINGERPRINT):
        logger.info(
            "[ANALYSIS] no-op widget rerun (fingerprint=%s, clips=%d)",
            session_state.get(SESSION_ANALYSIS_FINGERPRINT),
            len(session_state.get("cs_clips") or []),
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


def _token_plan_cache_key(
    formatted_transcript: str,
    profile: AIProfile,
    *,
    target_count: int,
    clip_style: str,
) -> str:
    payload = {
        "profile": profile.name,
        "target": target_count,
        "style": clip_style,
        "len": len(formatted_transcript),
        "head": formatted_transcript[:400],
        "tail": formatted_transcript[-400:] if formatted_transcript else "",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def plan_analysis_token_estimate(
    formatted_transcript: str,
    profile: AIProfile,
    *,
    target_count: int = 20,
    clip_style: str = "Balanced",
    emit_logs: bool = True,
) -> TokenPlanResult:
    """
    Estimate GPT usage with profile limits; progressively prune until within budget.
    """
    budget = profile.max_tokens
    cache_key = _token_plan_cache_key(
        formatted_transcript,
        profile,
        target_count=target_count,
        clip_style=clip_style,
    )
    should_log = emit_logs
    if emit_logs:
        if cache_key in _TOKEN_PLAN_LOGGED_KEYS:
            should_log = False
            logger.debug("skipping duplicate diagnostics — token plan already logged for %s", cache_key)
        else:
            _TOKEN_PLAN_LOGGED_KEYS.add(cache_key)

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
        if should_log:
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
                            if should_log:
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
    if should_log:
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
    default_max = min(int(profile.max_clip_length), 120)
    init_session_default(session_state, "cs_max_clip_seconds", default_max)
    init_session_default(
        session_state, "cs_context_before", int(profile.context_before)
    )
    init_session_default(
        session_state, "cs_context_after", int(profile.context_after)
    )
    init_session_default(session_state, "cs_clip_strategy", DEFAULT_CLIP_STRATEGY)
    init_session_default(session_state, "cs_platform_target", DEFAULT_PLATFORM_TARGET)
    init_session_default(session_state, "cs_title_style", DEFAULT_TITLE_STYLE    )


def get_cached_token_plan(
    session_state: Any,
    formatted_transcript: str,
    profile: AIProfile,
    *,
    target_count: int = 20,
    clip_style: str = "Balanced",
    emit_logs: bool = False,
) -> TokenPlanResult:
    """
    Return token plan for UI display without recomputing/logging on every rerun.
    """
    cache_key = _token_plan_cache_key(
        formatted_transcript,
        profile,
        target_count=target_count,
        clip_style=clip_style,
    )
    cached_key = str(session_state.get(SESSION_TOKEN_PLAN_CACHE_KEY, ""))
    cached_plan = session_state.get(SESSION_TOKEN_PLAN_CACHE)
    if cached_key == cache_key and isinstance(cached_plan, dict):
        logger.debug("[TOKEN PLAN] cache reuse for display key=%s", cache_key)
        est = cached_plan.get("estimate") or {}
        return TokenPlanResult(
            estimate=PipelineTokenEstimate(
                estimated_total_tokens=int(
                    est.get("estimated_total_tokens", cached_plan.get("after_prune", 0))
                ),
                estimated_calls=int(est.get("estimated_calls", 0)),
                estimated_prompt_tokens=int(est.get("estimated_prompt_tokens", 0)),
                estimated_completion_tokens=int(est.get("estimated_completion_tokens", 0)),
                breakdown=dict(est.get("breakdown") or {}),
            ),
            before_prune=int(cached_plan.get("before_prune", 0)),
            after_prune=int(cached_plan.get("after_prune", 0)),
            budget=int(cached_plan.get("budget", profile.max_tokens)),
            pruned_reason=cached_plan.get("pruned_reason"),
            effective_regions=int(cached_plan.get("effective_regions", profile.max_active_gpt_regions)),
            effective_passes=int(cached_plan.get("effective_passes", 1)),
            effective_grounding_targets=int(cached_plan.get("effective_grounding_targets", target_count)),
            max_chunk_chars=int(cached_plan.get("max_chunk_chars", 8_000)),
            include_grounding=bool(cached_plan.get("include_grounding", True)),
            include_split=bool(cached_plan.get("include_split", True)),
        )

    plan = plan_analysis_token_estimate(
        formatted_transcript,
        profile,
        target_count=target_count,
        clip_style=clip_style,
        emit_logs=emit_logs,
    )
    session_state[SESSION_TOKEN_PLAN_CACHE_KEY] = cache_key
    session_state[SESSION_TOKEN_PLAN_CACHE] = plan.to_dict()
    return plan


__all__ = [
    "ClipStudioEffectiveConfig",
    "DEFAULT_CLIP_STRATEGY",
    "DEFAULT_PLATFORM_TARGET",
    "DEFAULT_TITLE_STYLE",
    "ResolvedModels",
    "SESSION_ANALYSIS_DIAGNOSTICS",
    "SESSION_ANALYSIS_FINGERPRINT",
    "SESSION_CLIP_EDITS",
    "SESSION_EFFECTIVE_CONFIG",
    "SESSION_FORCE_REANALYZE",
    "TokenPlanResult",
    "apply_profile_non_widget_keys",
    "apply_profile_widget_defaults",
    "build_analysis_fingerprint",
    "get_durable_effective_config",
    "get_invalidation_reason",
    "init_session_default",
    "log_widget_rerun_noop",
    "get_cached_token_plan",
    "plan_analysis_token_estimate",
    "resolve_models_for_session",
    "resolve_models_from_call_context",
    "resolve_models_from_effective_config",
    "resolve_models_from_profile",
    "store_analysis_snapshot",
]
