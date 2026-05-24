"""
clip_engine/openai_resilience.py
OpenAI rate-limit retry/backoff, token estimation, and request guards.
Stdlib-only — no new dependencies.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from clip_engine.token_tracking import TokenTracker, get_tracker

logger = logging.getLogger("clip_engine.openai_resilience")

# Rough chars-per-token for English transcript + prompts
CHARS_PER_TOKEN = 4
DEFAULT_MAX_PROMPT_TOKENS = 12_000


@dataclass
class RateLimitConfig:
    max_retries: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_min_ms: int = 100
    jitter_max_ms: int = 500


@dataclass
class OpenAICallContext:
    """Thread/run-scoped settings for OpenAI calls in the clip pipeline."""

    token_saver_mode: bool = True
    rate_limit_safe: bool = True
    call_delay_seconds: float = 0.75
    max_chunk_chars: int = 8_000
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    status_callback: Callable[[str], None] | None = None
    tracker: TokenTracker | None = None


@dataclass
class PipelineTokenEstimate:
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_calls: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_completion_tokens": self.estimated_completion_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_calls": self.estimated_calls,
            "breakdown": self.breakdown,
        }


class OpenAIRateLimitError(RuntimeError):
    """Raised when all rate-limit retries are exhausted."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "",
        model: str = "",
        attempts: int = 0,
        mitigation: str = "",
    ):
        super().__init__(message)
        self.stage = stage
        self.model = model
        self.attempts = attempts
        self.mitigation = mitigation


_active_context: OpenAICallContext | None = None


def set_call_context(ctx: OpenAICallContext | None) -> None:
    global _active_context
    _active_context = ctx


def get_call_context() -> OpenAICallContext:
    return _active_context or OpenAICallContext()


def estimate_tokens_rough(text: str) -> int:
    """Fast token estimate without tiktoken."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Detect OpenAI 429 / TPM rate-limit errors."""
    name = type(exc).__name__.lower()
    if "ratelimit" in name or name == "rate_limit_error":
        return True
    msg = str(exc).lower()
    markers = (
        "rate limit",
        "rate_limit",
        "tokens per min",
        "tpm",
        "429",
        "too many requests",
        "please try again",
    )
    return any(m in msg for m in markers)


def parse_retry_after_seconds(exc: BaseException) -> float | None:
    """
    Parse OpenAI message like 'Please try again in 500ms' or 'in 6s'.
    Returns seconds or None.
    """
    msg = str(exc)
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|seconds)?", msg, re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit.startswith("ms"):
        return value / 1000.0
    return value


def _backoff_delay(attempt: int, config: RateLimitConfig, parsed_wait: float | None) -> float:
    if parsed_wait is not None and parsed_wait > 0:
        base = min(config.max_delay_seconds, parsed_wait)
    else:
        base = min(config.max_delay_seconds, config.base_delay_seconds * (2 ** (attempt - 1)))
    jitter = random.randint(config.jitter_min_ms, config.jitter_max_ms) / 1000.0
    return base + jitter


def _notify_status(message: str) -> None:
    ctx = get_call_context()
    if ctx.status_callback:
        try:
            ctx.status_callback(message)
        except Exception:
            pass
    logger.info(message)


def apply_call_delay() -> None:
    """Pause between OpenAI calls to reduce TPM spikes."""
    ctx = get_call_context()
    delay = ctx.call_delay_seconds if ctx.token_saver_mode or ctx.rate_limit_safe else 0.0
    if delay > 0:
        time.sleep(delay)


def truncate_text_safe(text: str, max_chars: int, *, label: str = "text") -> tuple[str, bool]:
    """Truncate oversized text with warning marker. Returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars] + "\n[section truncated for token budget]"
    logger.warning("%s truncated: %d -> %d chars", label, len(text), max_chars)
    return truncated, True


def split_text_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at line boundaries when over max_chars."""
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks or [text[:max_chars]]


def call_openai_with_backoff(
    client: Any,
    *,
    create_kwargs: dict[str, Any],
    stage: str,
    model: str,
    tracker: TokenTracker | None = None,
    config: RateLimitConfig | None = None,
    rate_limit_safe: bool | None = None,
    prompt_estimate: int = 0,
    clip_id: str | None = None,
) -> Any:
    """
    Call client.chat.completions.create with exponential backoff on 429.
    Records usage on success; tracks retry attempts on tracker.
    """
    cfg = config or RateLimitConfig()
    ctx = get_call_context()
    tracker = tracker or ctx.tracker or get_tracker()
    use_backoff = ctx.rate_limit_safe if rate_limit_safe is None else rate_limit_safe
    max_attempts = cfg.max_retries + 1 if use_backoff else 1
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            apply_call_delay()
            if prompt_estimate:
                logger.info(
                    "OpenAI call stage=%s model=%s attempt=%d est_prompt_tokens=%d",
                    stage, model, attempt, prompt_estimate,
                )
            response = client.chat.completions.create(**create_kwargs)
            usage = getattr(response, "usage", None)
            if usage:
                tracker.record_openai_usage(stage, usage, model=model, clip_id=clip_id)
            return response
        except Exception as exc:
            last_exc = exc
            if not use_backoff or not is_rate_limit_error(exc) or attempt >= max_attempts:
                if is_rate_limit_error(exc):
                    mitigation = (
                        "Enable Token Saver Mode, lower target clip count, "
                        "increase delay between calls, or wait a minute and retry."
                    )
                    raise OpenAIRateLimitError(
                        f"OpenAI rate limit at stage '{stage}' after {attempt} attempt(s): {exc}",
                        stage=stage,
                        model=model,
                        attempts=attempt,
                        mitigation=mitigation,
                    ) from exc
                raise

            parsed = parse_retry_after_seconds(exc)
            wait = _backoff_delay(attempt, cfg, parsed)
            retry_tokens = prompt_estimate  # rough attribution for retries
            if hasattr(tracker, "record_retry"):
                tracker.record_retry(stage, retry_tokens, model=model)
            msg = (
                f"OpenAI rate limit reached ({stage}). "
                f"Waiting {wait:.1f}s and retrying (attempt {attempt}/{cfg.max_retries})..."
            )
            _notify_status(msg)
            logger.warning(
                "Rate limit stage=%s model=%s attempt=%d/%d wait=%.2fs parsed=%s err=%s",
                stage, model, attempt, cfg.max_retries, wait, parsed, exc,
            )
            time.sleep(wait)

    raise OpenAIRateLimitError(
        f"OpenAI call failed at stage '{stage}': {last_exc}",
        stage=stage,
        model=model,
        attempts=max_attempts,
    )


def estimate_pipeline_tokens(
    formatted_transcript: str,
    *,
    target_count: int = 20,
    n_regions: int = 5,
    n_passes: int = 2,
    max_pass_rounds: int = 1,
    max_chunk_chars: int = 8_000,
    include_grounding: bool = True,
    include_split: bool = True,
    token_saver_mode: bool = True,
) -> PipelineTokenEstimate:
    """Estimate total tokens for a clip analysis run."""
    transcript_tokens = estimate_tokens_rough(formatted_transcript)
    region_tokens = min(max_chunk_chars // CHARS_PER_TOKEN, transcript_tokens // max(1, n_regions) + 500)
    system_overhead = 900
    per_call_prompt = region_tokens + system_overhead
    per_call_completion = 1200 if token_saver_mode else 1800

    analysis_calls = n_regions * n_passes * max(1, max_pass_rounds)
    if not token_saver_mode:
        analysis_calls += n_regions  # boost pass estimate

    breakdown: dict[str, int] = {}
    analysis_total = analysis_calls * (per_call_prompt + per_call_completion)
    breakdown["clip_analysis"] = analysis_total

    grounding_total = 0
    if include_grounding:
        g_calls = target_count
        grounding_total = g_calls * (800 + 400)
        breakdown["metadata_grounding"] = grounding_total

    split_total = 0
    if include_split:
        split_calls = max(1, target_count // 4)
        split_total = split_calls * (1500 + 800)
        breakdown["clip_split"] = split_total

    prompt_est = analysis_calls * per_call_prompt
    prompt_est += (target_count * 800 if include_grounding else 0)
    prompt_est += (split_total * 0.65 if include_split else 0)

    completion_est = analysis_total + grounding_total + split_total - int(prompt_est)

    return PipelineTokenEstimate(
        estimated_prompt_tokens=int(prompt_est),
        estimated_completion_tokens=int(max(0, completion_est)),
        estimated_total_tokens=int(prompt_est + max(0, completion_est)),
        estimated_calls=analysis_calls + (target_count if include_grounding else 0),
        breakdown={k: int(v) for k, v in breakdown.items()},
    )


def token_saver_pass_config(clip_style: str) -> tuple[int, int, int]:
    """
    Return (max_passes, max_pass_rounds, max_clips_per_region_scale).
    """
    if clip_style == "Long story clips":
        return 1, 1, 1
    if clip_style == "Micro clips":
        return 2, 1, 1
    return 2, 1, 1  # Balanced
