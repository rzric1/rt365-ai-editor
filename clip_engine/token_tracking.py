"""
clip_engine/token_tracking.py
Per-run token usage tracking by pipeline stage and per-clip metadata calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("clip_engine.token_tracking")

# GPT-4o list pricing (USD per 1M tokens) - update if model changes
GPT4O_INPUT_PER_M = 2.50
GPT4O_OUTPUT_PER_M = 10.00
DEFAULT_MODEL = "gpt-4o"


@dataclass
class StageUsage:
    stage: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = DEFAULT_MODEL
    call_count: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.call_count += 1


@dataclass
class TokenTracker:
    """Tracks token usage for one video analysis/export run."""

    video_filename: str = ""
    stages: dict[str, StageUsage] = field(default_factory=dict)
    per_clip: dict[str, dict[str, int]] = field(default_factory=dict)
    retry_tokens: int = 0
    tokens_avoided_cache: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record(
        self,
        stage: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        model: str = DEFAULT_MODEL,
        clip_id: str | None = None,
    ) -> None:
        total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        if stage not in self.stages:
            self.stages[stage] = StageUsage(stage=stage, model=model)
        else:
            self.stages[stage].model = model
        self.stages[stage].add(prompt_tokens, completion_tokens, total)

        if clip_id:
            entry = self.per_clip.setdefault(clip_id, {"prompt": 0, "completion": 0, "total": 0})
            entry["prompt"] += prompt_tokens
            entry["completion"] += completion_tokens
            entry["total"] += total

        logger.debug(
            "Tokens [%s]: +%d (prompt=%d completion=%d)",
            stage, total, prompt_tokens, completion_tokens,
        )

    def record_retry(self, stage: str, estimated_tokens: int = 0, *, model: str = DEFAULT_MODEL) -> None:
        """Track tokens attributed to rate-limit retries (estimate when usage unavailable)."""
        self.retry_tokens += estimated_tokens
        retry_stage = f"{stage}_retry"
        self.record(retry_stage, prompt_tokens=estimated_tokens, completion_tokens=0, model=model)

    def record_cache_hit(self, tokens_avoided: int) -> None:
        self.tokens_avoided_cache += tokens_avoided

    def record_openai_usage(self, stage: str, usage: Any, *, model: str = DEFAULT_MODEL, clip_id: str | None = None) -> None:
        if usage is None:
            return
        self.record(
            stage,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            model=model,
            clip_id=clip_id,
        )

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.stages.values())

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.stages.values())

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.stages.values())

    def estimated_cost_usd(self) -> float:
        return (
            self.prompt_tokens * GPT4O_INPUT_PER_M + self.completion_tokens * GPT4O_OUTPUT_PER_M
        ) / 1_000_000

    def to_session_dict(self) -> dict:
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "total": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
        }

    def to_export_dict(
        self,
        *,
        target_clips: int,
        final_clip_count: int,
        model: str = DEFAULT_MODEL,
    ) -> dict:
        return {
            "video_filename": self.video_filename,
            "analysis_timestamp": self.started_at,
            "model": model,
            "target_clips": target_clips,
            "final_clip_count": final_clip_count,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retry_tokens": self.retry_tokens,
            "tokens_avoided_cache": self.tokens_avoided_cache,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "per_stage": {
                name: {
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "total_tokens": s.total_tokens,
                    "call_count": s.call_count,
                    "model": s.model,
                }
                for name, s in self.stages.items()
            },
            "per_clip": self.per_clip,
        }

    def write_json(self, path: Path, **export_kwargs) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_export_dict(**export_kwargs), indent=2),
            encoding="utf-8",
        )


# Module-level tracker for current run (Streamlit session / analysis batch)
_active_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    global _active_tracker
    if _active_tracker is None:
        _active_tracker = TokenTracker()
    return _active_tracker


def reset_tracker(video_filename: str = "") -> TokenTracker:
    global _active_tracker
    _active_tracker = TokenTracker(video_filename=video_filename)
    return _active_tracker


def get_session_tokens() -> dict:
    """Backward-compatible summary for UI."""
    return get_tracker().to_session_dict()
