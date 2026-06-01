# -*- coding: utf-8 -*-
"""
clip_engine/analysis_cache.py
Analysis result cache and partial-progress resume for clip pipeline.
Stdlib-only — JSON files under outputs/cache/analysis/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT

logger = logging.getLogger("clip_engine.analysis_cache")

CACHE_VERSION = "3"
ANALYSIS_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "analysis"


@dataclass
class AnalysisCacheKey:
    video_filename: str = ""
    transcript_hash: str = ""
    target_clips: int = 20
    clip_style: str = "Balanced"
    min_clip_seconds: float = 25.0
    max_clip_seconds: float = 160.0
    min_gap_seconds: float = 60.0
    similarity_threshold: float = 0.45
    token_saver_mode: bool = True
    model_fast: str = ""
    model_quality: str = ""
    context_before: float = 8.0
    context_after: float = 12.0
    discovery_mode: bool = False
    ai_profile_name: str = "SAFE"
    clip_strategy: str = "Balanced"
    platform_target: str = "TikTok/Reels/Shorts"
    title_style: str = "Curiosity"
    cache_version: str = CACHE_VERSION

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class AnalysisProgress:
    """Partial progress for resume after rate-limit failure."""

    cache_key: str = ""
    completed_steps: list[str] = field(default_factory=list)
    partial_candidates: list[dict] = field(default_factory=list)
    last_pass: str = ""
    last_region: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def step_key(self, pass_name: str, region_label: str) -> str:
        return f"{pass_name}:{region_label}"

    def is_done(self, pass_name: str, region_label: str) -> bool:
        return self.step_key(pass_name, region_label) in self.completed_steps

    def mark_done(self, pass_name: str, region_label: str) -> None:
        key = self.step_key(pass_name, region_label)
        if key not in self.completed_steps:
            self.completed_steps.append(key)
        self.last_pass = pass_name
        self.last_region = region_label
        self.updated_at = datetime.now(timezone.utc).isoformat()


def _cache_dir(cache_key: str) -> Path:
    return ANALYSIS_CACHE_DIR / cache_key


def hash_transcript(formatted: str, segments: list[dict] | None = None) -> str:
    """Hash transcript content for cache invalidation."""
    parts = [formatted[:200_000]]
    if segments:
        parts.append(str(len(segments)))
        parts.append(str(segments[0].get("start", 0)))
        parts.append(str(segments[-1].get("end", 0)))
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    *,
    video_filename: str,
    formatted: str,
    segments: list[dict],
    target_clips: int,
    clip_style: str,
    min_clip_seconds: float,
    max_clip_seconds: float,
    min_gap_seconds: float,
    similarity_threshold: float,
    token_saver_mode: bool,
    model_fast: str,
    model_quality: str,
    context_before: float,
    context_after: float,
    discovery_mode: bool = False,
    ai_profile_name: str = "SAFE",
    clip_strategy: str = "Balanced",
    platform_target: str = "TikTok/Reels/Shorts",
    title_style: str = "Curiosity",
) -> AnalysisCacheKey:
    return AnalysisCacheKey(
        video_filename=video_filename,
        transcript_hash=hash_transcript(formatted, segments),
        target_clips=target_clips,
        clip_style=clip_style,
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=max_clip_seconds,
        min_gap_seconds=min_gap_seconds,
        similarity_threshold=similarity_threshold,
        token_saver_mode=token_saver_mode,
        model_fast=model_fast,
        model_quality=model_quality,
        context_before=context_before,
        context_after=context_after,
        discovery_mode=discovery_mode,
        ai_profile_name=ai_profile_name,
        clip_strategy=clip_strategy,
        platform_target=platform_target,
        title_style=title_style,
    )


def load_cached_analysis(cache_key: str) -> dict[str, Any] | None:
    path = _cache_dir(cache_key) / "result.json"
    if not path.is_file():
        logger.info("[CACHE] miss: no file for %s", cache_key)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("cache_version") != CACHE_VERSION:
            logger.info("[CACHE] miss: version mismatch for %s", cache_key)
            return None
        logger.info("[CACHE] hit: %s", cache_key)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[CACHE] read failed %s: %s", cache_key, exc)
        return None


def save_cached_analysis(
    cache_key: str,
    *,
    clips: list[dict],
    stats: dict,
    token_usage: dict,
    raw_candidates: list[dict] | None = None,
    analysis_fingerprint: str = "",
) -> Path:
    d = _cache_dir(cache_key)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "analysis_fingerprint": analysis_fingerprint,
        "clips": clips,
        "stats": stats,
        "token_usage": token_usage,
        "raw_candidates_count": len(raw_candidates) if raw_candidates else 0,
    }
    result_path = d / "result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    token_path = d / "token_usage.json"
    token_path.write_text(json.dumps(token_usage, indent=2), encoding="utf-8")
    logger.info("[CACHE] saved analysis: %s", result_path)
    return result_path


def load_progress(cache_key: str) -> AnalysisProgress | None:
    path = _cache_dir(cache_key) / "progress.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AnalysisProgress(
            cache_key=cache_key,
            completed_steps=list(data.get("completed_steps", [])),
            partial_candidates=list(data.get("partial_candidates", [])),
            last_pass=str(data.get("last_pass", "")),
            last_region=str(data.get("last_region", "")),
            updated_at=str(data.get("updated_at", "")),
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Progress read failed: %s", exc)
        return None


def save_progress(progress: AnalysisProgress) -> None:
    d = _cache_dir(progress.cache_key)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "progress.json"
    path.write_text(
        json.dumps(
            {
                "cache_key": progress.cache_key,
                "completed_steps": progress.completed_steps,
                "partial_candidates": progress.partial_candidates,
                "last_pass": progress.last_pass,
                "last_region": progress.last_region,
                "updated_at": progress.updated_at,
                "cache_version": CACHE_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_progress(cache_key: str) -> None:
    path = _cache_dir(cache_key) / "progress.json"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def clear_all_analysis_cache() -> int:
    """Remove all cached analysis. Returns count of entries removed."""
    if not ANALYSIS_CACHE_DIR.is_dir():
        return 0
    count = 0
    for child in ANALYSIS_CACHE_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            count += 1
    return count
