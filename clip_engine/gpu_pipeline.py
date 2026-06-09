# -*- coding: utf-8 -*-
"""

clip_engine/gpu_pipeline.py

GPU-assisted local intelligence before OpenAI refinement.

"""



from __future__ import annotations

import logging
import os
import sys
import traceback

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clip_engine.discovery_forensics import DiscoveryForensics



from clip_engine.ai_profiles import AIProfile, get_ai_profile

from clip_engine.clip_style import ClipStyle

from clip_engine.local_candidate_discovery import discover_local_candidates

from clip_engine.semantic_ranking import (

    candidate_window_text,

    cluster_segments,

    dedupe_by_embedding_similarity,

    embeddings_available,

    generate_embeddings,

    rank_candidate_segments,

    semantic_pipeline_status,

)

from clip_engine.speaker_analysis import (

    boost_candidates_from_diarization,

    boost_candidates_from_transcript_speakers,

    speaker_pipeline_status,

)



logger = logging.getLogger("clip_engine.gpu_pipeline")





def _estimate_refinement_tokens(shortlist_count: int, *, max_regions: int, n_passes: int = 1) -> int:

    """Rough token estimate for GPT refinement passes over shortlist regions."""

    regions = max(1, min(max_regions, shortlist_count))

    per_region = 4_500

    return int(regions * per_region * max(1, n_passes))





def _build_explorer_rows(

    candidates: list[dict],

    segments: list[dict],

    *,

    shortlist_max: int,

    kept_ids: set[int] | None = None,

) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for idx, c in enumerate(candidates[: max(shortlist_max * 2, 30)]):

        snippet = candidate_window_text(c, segments)

        if len(snippet) > 220:

            snippet = snippet[:217] + "..."

        sig = c.get("local_signals") or {}

        in_shortlist = kept_ids is None or idx in kept_ids

        rows.append(

            {

                "start_seconds": c.get("start_seconds"),

                "end_seconds": c.get("end_seconds"),

                "total_local_score": c.get("composite_score"),

                "semantic_score": c.get("_semantic_score"),

                "hook_score": sig.get("scroll_stopping_hook"),

                "emotion_score": sig.get("emotion_spike"),

                "cluster_id": c.get("_cluster_id"),

                "kept": in_shortlist,

                "reject_reason": None if in_shortlist else c.get("_reject_reason", "pruned"),

                "transcript_snippet": snippet,

                "region": c.get("_region"),

            }

        )

    return rows


def _shortlist_with_timeline_spread(
    candidates: list[dict],
    *,
    media_duration: float,
    shortlist_max: int,
) -> list[dict]:
    """
    Enforce full-timeline coverage: divide the media into `shortlist_max` equal
    time buckets and pick the top-scoring candidate from each bucket.

    Assumes `candidates` are already sorted best-first.
    """
    if not candidates or shortlist_max <= 0 or media_duration <= 0:
        return candidates[: max(0, shortlist_max)]

    bucket_to_best: dict[int, dict] = {}
    for c in candidates:
        t0 = float(c.get("start_seconds", 0) or 0.0)
        b = int((t0 / media_duration) * shortlist_max)
        if b < 0:
            b = 0
        if b >= shortlist_max:
            b = shortlist_max - 1
        if b not in bucket_to_best:
            bucket_to_best[b] = c
            if len(bucket_to_best) >= shortlist_max:
                break

    out = list(bucket_to_best.values())
    out.sort(key=lambda x: float(x.get("start_seconds", 0) or 0.0))
    return out



def run_gpu_prefilter_pipeline(

    segments: list[dict],

    media_duration: float,

    *,

    clip_style: ClipStyle | str = "Balanced",

    user_min_seconds: float = 25.0,

    user_max_seconds: float = 160.0,

    target_count: int = 20,

    pool_target: int = 50,

    similarity_threshold: float = 0.88,

    ai_profile: AIProfile | str | None = None,

    discovery_mode: bool = False,
    forensics: DiscoveryForensics | None = None,
    diarization_turns: list[dict] | None = None,

) -> tuple[list[dict], dict[str, Any]]:

    """

    Local pre-candidates → speaker boost → embedding dedupe → semantic rank.

    No OpenAI calls.

    """

    prof = (
        ai_profile
        if isinstance(ai_profile, AIProfile)
        else get_ai_profile(str(ai_profile or "SAFE"))
    )

    shortlist_min = prof.target_gpu_shortlist_min
    shortlist_max = prof.target_gpu_shortlist_max
    shortlist_target = max(shortlist_min, min(shortlist_max, max(target_count, shortlist_min)))

    stats: dict[str, Any] = {
        "local_prefilter_count": 0,
        "semantic_dedupe_removed": 0,
        "shortlist_count": 0,
        "embeddings_on_gpu": False,
        "local_ranking_enabled": True,
        "estimated_refinement_tokens": 0,
        "ai_profile": prof.name,
        "raw_candidates": 0,
        "semantic_prefilter_fallback": False,
        "semantic_prefilter_error": "",
        "discovery_scan": {},
    }

    # Hard safety valve for unstable environments.
    force_cpu_embeddings = os.environ.get("FORCE_CPU_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")

    try:
        local = discover_local_candidates(
            segments,
            media_duration,
            clip_style=clip_style,
            user_min_seconds=user_min_seconds,
            user_max_seconds=min(user_max_seconds, prof.max_clip_length),
            max_candidates=max(100, pool_target * 4) if discovery_mode else max(80, pool_target * 3),
            discovery_mode=discovery_mode,
            forensics=forensics,
        )

        if len(local) < 10:
            from clip_engine.transcript_candidate_scanner import scan_transcript_candidates

            rescue, scan_stats = scan_transcript_candidates(
                segments,
                media_duration,
                discovery_mode=True,
                user_min_seconds=user_min_seconds,
                user_max_seconds=min(user_max_seconds, prof.max_clip_length),
                max_candidates=max(60, shortlist_max * 2),
                existing=local,
                min_gap_seconds=14.0,
            )
            stats["discovery_scan"] = scan_stats.to_dict()
            if forensics:
                forensics.merge_scan_stats(scan_stats.to_dict())
                forensics.record_stage(
                    "gpu_transcript_scan_rescue",
                    input_count=len(local),
                    output_count=len(local) + len(rescue),
                    rejected_count=scan_stats.windows_rejected,
                    rejection_reasons=scan_stats.rejection_reasons,
                )
            if rescue:
                logger.warning(
                    "[GPU PREFILTER] local discovery starved (%d) — transcript scan added %d",
                    len(local),
                    len(rescue),
                )
                local.extend(rescue)
                local.sort(key=lambda x: int(x.get("composite_score", 0)), reverse=True)

        raw_count = len(local)
        stats["local_prefilter_count"] = raw_count
        stats["raw_candidates"] = raw_count
        if forensics:
            forensics.gpu_candidates_generated = raw_count
        logger.info("[GPU PREFILTER] raw_candidates=%d", raw_count)

        local = boost_candidates_from_transcript_speakers(local, segments)
        if diarization_turns:
            local = boost_candidates_from_diarization(local, diarization_turns)
        sem_status = semantic_pipeline_status()
        stats["embeddings_on_gpu"] = sem_status.get("device") == "cuda"

        dedupe_removed = 0
        if force_cpu_embeddings:
            logger.warning("[GPU PREFILTER] FORCE_CPU_EMBEDDINGS=1 — skipping GPU semantic embeddings")
        elif embeddings_available() and len(local) > 3:
            texts = [candidate_window_text(c, segments) for c in local]
            local, dedupe_removed = dedupe_by_embedding_similarity(
                local, texts, similarity_threshold=similarity_threshold
            )
            stats["semantic_dedupe_removed"] = dedupe_removed
            if forensics and dedupe_removed:
                forensics.record_gpu_rejection("semantic_dedupe", dedupe_removed)
            logger.info("[GPU PREFILTER] semantic_dedupe_removed=%d", dedupe_removed)

            try:
                emb = generate_embeddings(texts[: len(local)])
                clusters = cluster_segments(emb, similarity_threshold=0.82)
                for cid, group in enumerate(clusters):
                    for gi in group:
                        if gi < len(local):
                            local[gi]["_cluster_id"] = cid
            except Exception as exc:
                logger.debug("Cluster ids skipped: %s", exc)

            texts = [candidate_window_text(c, segments) for c in local]
            local = rank_candidate_segments(local, texts, top_k=shortlist_target)
            for _, c in enumerate(local):
                c["_semantic_score"] = round(float(c.get("local_rank_score", 0)) / 100.0, 3)

        before_spread = len(local)
        local = _shortlist_with_timeline_spread(
            local, media_duration=media_duration, shortlist_max=shortlist_max
        )
        if forensics and before_spread > len(local):
            forensics.record_gpu_rejection(
                "timeline_spread_prune", before_spread - len(local)
            )

        kept_ids = set(range(len(local)))
        active_regions = sorted({str(c.get("_region", "")) for c in local if c.get("_region")})
        if prof.max_active_gpt_regions and len(active_regions) > prof.max_active_gpt_regions:
            region_scores: dict[str, float] = {}
            for c in local:
                r = str(c.get("_region", ""))
                if not r:
                    continue
                region_scores[r] = region_scores.get(r, 0.0) + float(c.get("composite_score", 0))

            active_regions = [
                r
                for r, _ in sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[
                    : prof.max_active_gpt_regions
                ]
            ]

            allowed = set(active_regions)
            pruned: list[dict] = []
            region_pruned = 0
            for c in local:
                r = str(c.get("_region", ""))
                if r and r not in allowed:
                    c["_reject_reason"] = "region_cap"
                    region_pruned += 1
                    if forensics:
                        forensics.record_gpu_rejection("region_cap", 1)
                    continue
                pruned.append(c)
            local = pruned
            kept_ids = set(range(len(local)))

        est_tokens = _estimate_refinement_tokens(
            len(local),
            max_regions=prof.max_active_gpt_regions,
            n_passes=prof.max_gpt_passes,
        )
        if est_tokens > prof.max_tokens:
            while len(local) > shortlist_min and est_tokens > prof.max_tokens:
                if forensics:
                    forensics.record_gpu_rejection("token_budget_prune", 1)
                local.pop()
                est_tokens = _estimate_refinement_tokens(
                    len(local),
                    max_regions=prof.max_active_gpt_regions,
                    n_passes=prof.max_gpt_passes,
                )
            logger.info(
                "[GPU PREFILTER] token budget prune: shortlist=%d est=%d budget=%d",
                len(local),
                est_tokens,
                prof.max_tokens,
            )

        stats["shortlist_count"] = len(local)
        if forensics:
            forensics.record_stage(
                "gpu_prefilter_shortlist",
                input_count=raw_count,
                output_count=len(local),
                rejected_count=max(0, raw_count - len(local)),
                rejection_reasons=dict(forensics.gpu_rejection_reasons),
            )
        stats["estimated_refinement_tokens"] = est_tokens
        stats["active_regions"] = active_regions
        stats["semantic"] = sem_status
        stats["speaker"] = speaker_pipeline_status()
        stats["explorer_rows"] = _build_explorer_rows(
            local, segments, shortlist_max=shortlist_max, kept_ids=kept_ids
        )

        logger.info("[GPU PREFILTER] shortlist=%d", stats["shortlist_count"])
        logger.info("[GPU PREFILTER] estimated_refinement_tokens=%d", est_tokens)
        logger.info(
            "[AI PROFILE] %s budget=%d max_gpt_passes=%d max_active_regions=%d",
            prof.name,
            prof.max_tokens,
            prof.max_gpt_passes,
            prof.max_active_gpt_regions,
        )
        logger.info(
            "GPU prefilter: %d local → %d shortlist (dedupe removed=%d, regions=%s)",
            stats["local_prefilter_count"],
            stats["shortlist_count"],
            stats["semantic_dedupe_removed"],
            active_regions,
        )
        return local, stats
    except BaseException as fatal_exc:
        # Last-line guard: NEVER allow prefilter to kill Streamlit.
        stats["semantic_prefilter_fallback"] = True
        stats["local_ranking_enabled"] = False
        stats["semantic_prefilter_error"] = f"{type(fatal_exc).__name__}: {fatal_exc}"
        stats["estimated_refinement_tokens"] = 0
        stats["active_regions"] = []
        stats["semantic"] = {"_error": stats["semantic_prefilter_error"]}
        stats["speaker"] = {}
        logger.error(
            "[GPU PREFILTER FATAL] embedding path failed: %s\n%s",
            fatal_exc,
            traceback.format_exc(),
        )
        try:
            from clip_engine.transcript_candidate_scanner import scan_transcript_candidates

            rescue, scan_stats = scan_transcript_candidates(
                segments,
                media_duration,
                discovery_mode=True,
                user_min_seconds=user_min_seconds,
                user_max_seconds=min(user_max_seconds, prof.max_clip_length),
                max_candidates=max(40, shortlist_max),
            )
            stats["discovery_scan"] = scan_stats.to_dict()
            if forensics:
                forensics.merge_scan_stats(scan_stats.to_dict())
                forensics.gpu_candidates_generated = len(rescue)
                forensics.record_stage(
                    "gpu_prefilter_fatal_rescue",
                    input_count=0,
                    output_count=len(rescue[:shortlist_max]),
                    rejection_reasons=scan_stats.rejection_reasons,
                    note=stats["semantic_prefilter_error"],
                )
            stats["raw_candidates"] = len(rescue)
            stats["local_prefilter_count"] = len(rescue)
            stats["shortlist_count"] = len(rescue[:shortlist_max])
            stats["explorer_rows"] = _build_explorer_rows(
                rescue[:shortlist_max],
                segments,
                shortlist_max=shortlist_max,
            )
            logger.warning(
                "[GPU PREFILTER FATAL] recovered %d transcript-only candidates",
                stats["shortlist_count"],
            )
            return rescue[:shortlist_max], stats
        except Exception as rescue_exc:
            logger.error("[GPU PREFILTER] transcript rescue failed: %s", rescue_exc)
            stats["shortlist_count"] = 0
            stats["explorer_rows"] = []
            if forensics:
                forensics.record_stage(
                    "gpu_prefilter_fatal_failed",
                    input_count=0,
                    output_count=0,
                    rejection_reasons={"transcript_rescue_failed": 1},
                    note=str(rescue_exc),
                )
            return [], stats





def get_torch_embedding_diagnostics() -> dict[str, Any]:

    """PyTorch / sentence-transformers diagnostics for RTX pipeline panel."""

    out: dict[str, Any] = {

        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",

        "torch_installed": False,

        "torch_version": None,

        "torch_cuda_available": False,

        "torch_cuda_device_count": 0,

        "torch_cuda_device_name": None,

        "sentence_transformers_installed": False,

        "embeddings_device_selected": "cpu",

    }

    try:

        import torch



        out["torch_installed"] = True

        out["torch_version"] = getattr(torch, "__version__", "?")

        out["torch_cuda_available"] = bool(torch.cuda.is_available())

        out["torch_cuda_device_count"] = int(torch.cuda.device_count())

        if out["torch_cuda_available"] and out["torch_cuda_device_count"] > 0:

            out["torch_cuda_device_name"] = torch.cuda.get_device_name(0)

    except ImportError:

        pass

    except Exception as exc:

        out["torch_error"] = str(exc)[:200]

    try:

        import sentence_transformers  # noqa: F401



        out["sentence_transformers_installed"] = True

    except Exception as exc:

        out["sentence_transformers_error"] = str(exc)[:200]

    try:

        sem = semantic_pipeline_status()

        out["embeddings_device_selected"] = sem.get("device", "cpu")

    except Exception as exc:

        out["semantic_status_error"] = str(exc)[:200]

    return out





_RTX_STATUS_SAFE_DEFAULT: dict[str, Any] = {
    "cuda_available": False,
    "gpu_name": "unknown",
    "embeddings_on_gpu": False,
    "embeddings_available": False,
    "diarization_on_gpu": False,
    "pyannote_available": False,
    "faster_whisper_cuda": False,
    "local_ranking_enabled": False,
    "gpu_memory": None,
    "embedding_model": None,
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "torch_installed": False,
    "torch_version": None,
    "torch_cuda_available": False,
    "torch_cuda_device_count": 0,
    "torch_cuda_device_name": None,
    "sentence_transformers_installed": False,
    "embeddings_device_selected": "cpu",
    "_error": None,
}


def get_rtx_pipeline_status() -> dict[str, Any]:
    """Combined status for RTX 4090 AI Pipeline UI panel.

    Never raises — returns a safe default dict on any failure so the
    Streamlit sidebar never crashes the whole app.
    """
    try:
        sem = semantic_pipeline_status()
        spk = speaker_pipeline_status()
        torch_diag = get_torch_embedding_diagnostics()

        whisper_cuda = False
        try:
            from clip_engine.ffmpeg_gpu import faster_whisper_cuda_available
            whisper_cuda = faster_whisper_cuda_available()
        except Exception as exc:
            logger.debug("faster_whisper_cuda_available probe failed: %s", exc)

        gpu_mem = None
        if torch_diag.get("torch_cuda_available"):
            try:
                import torch
                gpu_mem = {
                    "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                }
            except Exception as exc:
                logger.debug("GPU memory stats unavailable: %s", exc)

        return {
            "cuda_available": sem.get("cuda_available", False),
            "gpu_name": sem.get("gpu_name", "unknown"),
            "embeddings_on_gpu": sem.get("device") == "cuda",
            "embeddings_available": sem.get("embeddings_available", False),
            "diarization_on_gpu": spk.get("device") == "cuda",
            "pyannote_available": spk.get("pyannote_available", False),
            "faster_whisper_cuda": whisper_cuda,
            "local_ranking_enabled": True,
            "gpu_memory": gpu_mem,
            "embedding_model": sem.get("model"),
            **torch_diag,
        }
    except Exception as exc:
        logger.warning("get_rtx_pipeline_status failed: %s", exc, exc_info=True)
        return {**_RTX_STATUS_SAFE_DEFAULT, "_error": str(exc)}


