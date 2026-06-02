# -*- coding: utf-8 -*-
"""
clip_engine/openai_resilience.py
OpenAI rate-limit retry/backoff, token estimation, and request guards.
Stdlib-only — no new dependencies.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from clip_engine.token_tracking import TokenTracker, get_tracker

import os as _os

# ── API key masking ──────────────────────────────────────────────
def _mask_key(key: str) -> str:
    """Return a safely masked API key for log output."""
    if not key or len(key) < 8:
        return "***"
    return key[:7] + "..." + key[-4:]


# ── Session token cap ────────────────────────────────────────────
DEFAULT_SESSION_TOKEN_CAP = 80_000
SESSION_TOKEN_CAP = int(_os.environ.get("RT365_SESSION_TOKEN_CAP", DEFAULT_SESSION_TOKEN_CAP))


class TokenBudgetExceededError(RuntimeError):
    """Raised when the session OpenAI token budget is exhausted."""

    def __init__(self, used: int, cap: int) -> None:
        self.used = used
        self.cap = cap
        super().__init__(
            f"Session token budget exceeded: used {used:,} of {cap:,} tokens. "
            f"Set RT365_SESSION_TOKEN_CAP in your .env to raise the limit, "
            f"or switch to the SAFE profile to reduce usage."
        )


def check_token_budget() -> None:
    """Raise TokenBudgetExceededError if the session cap has been reached."""
    if SESSION_TOKEN_CAP <= 0:
        return
    try:
        from clip_engine.token_tracking import get_tracker

        tracker = get_tracker()
        used = getattr(tracker, "total_tokens", 0) or 0
        if used > SESSION_TOKEN_CAP:
            raise TokenBudgetExceededError(used, SESSION_TOKEN_CAP)
    except TokenBudgetExceededError:
        raise
    except Exception:
        pass


logger = logging.getLogger("clip_engine.openai_resilience")

# Runtime telemetry (GPT-5 JSON reliability)
GPT5_SUCCESS_COUNT = 0
GPT5_EMPTY_JSON_COUNT = 0
JSON_FALLBACK_COUNT = 0
_JSON_CALL_COUNT = 0
_TELEMETRY_LOG_EVERY = 10

# Rough chars-per-token for English transcript + prompts
CHARS_PER_TOKEN = 4
DEFAULT_MAX_PROMPT_TOKENS = 12_000

JSON_STRICT_RULES = """CRITICAL OUTPUT RULES:
- Return ONLY valid JSON.
- No markdown. No code fences. No explanations. No commentary.
- The first character must be { or [.
- The last character must be } or ].
- Use double quotes for all keys and strings.
- Do not include trailing commas."""


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
    json_fallback_model: str | None = None
    model_fast: str | None = None
    model_quality: str | None = None


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


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in short slices so cancel can stop rate-limit backoff."""
    from clip_engine.job_control import check_cancelled

    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while True:
        check_cancelled()
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.5, remaining))


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
        _interruptible_sleep(delay)


def truncate_text_safe(text: str, max_chars: int, *, label: str = "text") -> tuple[str, bool]:
    """Truncate oversized text with warning marker. Returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars] + "\n[section truncated for token budget]"
    logger.warning("%s truncated: %d -> %d chars", label, len(text), max_chars)
    return truncated, True


def model_is_gpt5_family(model: str) -> bool:
    """True for gpt-5* models that may return empty JSON completions."""
    return (model or "").lower().strip().startswith("gpt-5")


def get_json_telemetry() -> dict[str, int | float]:
    """Snapshot of GPT-5 JSON telemetry counters."""
    attempts = GPT5_SUCCESS_COUNT + GPT5_EMPTY_JSON_COUNT
    success_rate = (GPT5_SUCCESS_COUNT / attempts * 100.0) if attempts else 0.0
    fallback_rate = (JSON_FALLBACK_COUNT / attempts * 100.0) if attempts else 0.0
    return {
        "gpt5_success": GPT5_SUCCESS_COUNT,
        "gpt5_empty_json": GPT5_EMPTY_JSON_COUNT,
        "json_fallback": JSON_FALLBACK_COUNT,
        "gpt5_success_rate_pct": round(success_rate, 1),
        "fallback_rate_pct": round(fallback_rate, 1),
    }


def reset_json_telemetry() -> None:
    global GPT5_SUCCESS_COUNT, GPT5_EMPTY_JSON_COUNT, JSON_FALLBACK_COUNT, _JSON_CALL_COUNT
    GPT5_SUCCESS_COUNT = 0
    GPT5_EMPTY_JSON_COUNT = 0
    JSON_FALLBACK_COUNT = 0
    _JSON_CALL_COUNT = 0


def _record_gpt5_success() -> None:
    global GPT5_SUCCESS_COUNT
    GPT5_SUCCESS_COUNT += 1
    _maybe_log_telemetry_summary()


def _record_gpt5_empty() -> None:
    global GPT5_EMPTY_JSON_COUNT
    GPT5_EMPTY_JSON_COUNT += 1
    _maybe_log_telemetry_summary()


def _record_json_fallback(primary_model: str, fallback_model: str, reason: str) -> None:
    global JSON_FALLBACK_COUNT
    JSON_FALLBACK_COUNT += 1
    logger.warning(
        "GPT-5-mini fallback triggered (%s). Retrying with %s. Fallback count: %d",
        reason,
        fallback_model,
        JSON_FALLBACK_COUNT,
    )
    _maybe_log_telemetry_summary()


def _maybe_log_telemetry_summary() -> None:
    global _JSON_CALL_COUNT
    _JSON_CALL_COUNT += 1
    if _JSON_CALL_COUNT % _TELEMETRY_LOG_EVERY != 0:
        return
    tel = get_json_telemetry()
    if tel["gpt5_success"] or tel["gpt5_empty_json"] or tel["json_fallback"]:
        logger.info(
            "GPT-5-mini JSON success rate: %s%% | Fallback rate: %s%% "
            "(success=%d empty=%d fallback=%d)",
            tel["gpt5_success_rate_pct"],
            tel["fallback_rate_pct"],
            tel["gpt5_success"],
            tel["gpt5_empty_json"],
            tel["json_fallback"],
        )


def resolve_json_fallback_model(primary_model: str) -> str | None:
    """Return JSON fallback model from pipeline context or SAFE profile (never env GPT-5)."""
    ctx = get_call_context()
    if ctx.json_fallback_model:
        fb = ctx.json_fallback_model.strip()
        if fb and fb.lower() != (primary_model or "").lower().strip():
            return fb

    if not model_is_gpt5_family(primary_model):
        return None

    from clip_engine.effective_config import resolve_models_from_call_context

    resolved = resolve_models_from_call_context()
    fallback = resolved.json_fallback_model.strip()
    if not fallback or fallback.lower() == (primary_model or "").lower().strip():
        return None
    return fallback


def model_uses_max_completion_tokens(model: str) -> bool:
    """True for gpt-5* and similar models that prefer max_completion_tokens."""
    m = (model or "").lower().strip()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def model_supports_temperature(model: str) -> bool:
    """gpt-5* and reasoning models often reject non-default temperature."""
    m = (model or "").lower().strip()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return False
    return True


def append_json_rules_to_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages and append strict JSON rules to the system message."""
    out = [dict(m) for m in messages]
    for msg in out:
        if msg.get("role") == "system":
            content = str(msg.get("content", ""))
            if JSON_STRICT_RULES not in content:
                msg["content"] = content.rstrip() + "\n\n" + JSON_STRICT_RULES
            break
    return out


def _extract_balanced_json(text: str) -> str | None:
    """Extract first balanced JSON object or array from text."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json_from_text(text: str, *, stage: str = "") -> dict | list[Any]:
    """
    Parse JSON from model output with fence stripping and balanced-brace extraction.
    """
    if text is None or not str(text).strip():
        logger.error("JSON parse failed stage=%s: empty response", stage or "unknown")
        raise ValueError("Empty model response — no JSON to parse.")

    raw = str(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```\s*$", "", raw).strip()

    candidates: list[str] = []
    if raw:
        candidates.append(raw)
    balanced = _extract_balanced_json(raw)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    if "{" in raw or "[" in raw:
        start = raw.find("{") if "{" in raw else raw.find("[")
        end_obj = raw.rfind("}")
        end_arr = raw.rfind("]")
        end = max(end_obj, end_arr)
        if start != -1 and end != -1 and end > start:
            slice_json = raw[start : end + 1]
            if slice_json not in candidates:
                candidates.append(slice_json)

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue

    from clip_engine.telemetry import log_clip_reject

    log_clip_reject("invalid_json", stage=stage or "unknown", chars=len(raw))
    preview = raw[:500].replace("\n", "\\n")
    logger.error(
        "JSON parse failed stage=%s (%d chars). Preview: %s",
        stage or "unknown",
        len(raw),
        preview,
    )
    raise ValueError(
        f"No JSON object found in model response (stage={stage or 'unknown'})."
    ) from last_err


def get_chat_response_text(response: Any) -> str:
    """Extract text content from chat completion response."""
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        return ""


def _log_openai_request_params(params: dict[str, Any], *, stage: str) -> None:
    token_key = (
        "max_completion_tokens"
        if "max_completion_tokens" in params
        else "max_tokens"
        if "max_tokens" in params
        else "none"
    )
    token_val = params.get(token_key) if token_key != "none" else None
    logger.info(
        "OpenAI request stage=%s model=%s token_param=%s token_value=%s "
        "temperature=%s response_format=%s",
        stage,
        params.get("model", ""),
        token_key,
        token_val,
        params.get("temperature", "not_sent"),
        "yes" if "response_format" in params else "no",
    )


def build_openai_params(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    use_max_completion_tokens: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build kwargs for client.chat.completions.create with model-aware token limits.
    gpt-5* models use max_completion_tokens; older models use max_tokens.
    """
    params: dict[str, Any] = {"model": model, "messages": list(messages)}
    params.update(kwargs)

    if temperature is not None and model_supports_temperature(model):
        params["temperature"] = temperature

    if response_format is not None:
        params["response_format"] = response_format

    if max_tokens is not None:
        use_mct = (
            use_max_completion_tokens
            if use_max_completion_tokens is not None
            else model_uses_max_completion_tokens(model)
        )
        if use_mct:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    return params


def is_unsupported_parameter_error(exc: BaseException, param_name: str) -> bool:
    """Detect OpenAI unsupported/invalid parameter errors for a specific param."""
    msg = str(exc).lower()
    pname = param_name.lower()
    if pname not in msg:
        return False
    markers = (
        "unsupported",
        "not supported",
        "unsupported_parameter",
        "invalid_request",
        "unknown parameter",
        "unexpected parameter",
    )
    return any(m in msg for m in markers)


def is_unsupported_response_format_error(exc: BaseException) -> bool:
    """Detect when response_format is rejected."""
    msg = str(exc).lower()
    if "response_format" not in msg and "json_object" not in msg and "json_schema" not in msg:
        return False
    return is_unsupported_parameter_error(exc, "response_format") or "response_format" in msg


def is_unsupported_temperature_error(exc: BaseException) -> bool:
    """Detect when temperature is rejected by the model/API."""
    msg = str(exc).lower()
    if "temperature" not in msg:
        return False
    return is_unsupported_parameter_error(exc, "temperature") or (
        "only the default" in msg and "temperature" in msg
    )


def apply_param_compatibility_fix(params: dict[str, Any], exc: BaseException) -> bool:
    """
    Mutate params to recover from unsupported_parameter errors.
    Returns True if params were adjusted and the call should be retried.
    """
    changed = False

    if "max_tokens" in params and is_unsupported_parameter_error(exc, "max_tokens"):
        val = params.pop("max_tokens")
        params["max_completion_tokens"] = val
        logger.info("OpenAI compat: switched max_tokens -> max_completion_tokens (%s)", val)
        changed = True
    elif "max_completion_tokens" in params and is_unsupported_parameter_error(exc, "max_completion_tokens"):
        val = params.pop("max_completion_tokens")
        params["max_tokens"] = val
        logger.info("OpenAI compat: switched max_completion_tokens -> max_tokens (%s)", val)
        changed = True

    if "temperature" in params and is_unsupported_temperature_error(exc):
        params.pop("temperature", None)
        logger.info("OpenAI compat: removed temperature for model %s", params.get("model", ""))
        changed = True

    if "response_format" in params and is_unsupported_response_format_error(exc):
        params.pop("response_format", None)
        params["messages"] = append_json_rules_to_messages(params.get("messages", []))
        logger.info(
            "OpenAI compat: removed response_format, enforced JSON via prompt for %s",
            params.get("model", ""),
        )
        changed = True

    return changed


def repair_json_with_chat(
    client: Any,
    *,
    model: str,
    raw_text: str,
    schema_hint: str,
    stage: str,
    tracker: TokenTracker | None = None,
    prompt_estimate: int = 0,
    clip_id: str | None = None,
    max_tokens: int = 2000,
) -> dict | list[Any]:
    """One-shot repair call to convert malformed output into valid JSON."""
    if not (raw_text or "").strip():
        raise ValueError("Cannot repair empty model response.")
    repair_stage = f"{stage}_json_repair"
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Convert the following model output into valid JSON matching the required schema. "
                "Return only JSON. No markdown. No code fences. No explanations.\n\n"
                + JSON_STRICT_RULES
            ),
        },
        {
            "role": "user",
            "content": (
                f"Required schema:\n{schema_hint}\n\n"
                f"Model output to fix:\n{raw_text[:14_000]}"
            ),
        },
    ]
    response = call_openai_chat(
        client,
        model=model,
        messages=repair_messages,
        stage=repair_stage,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        tracker=tracker,
        prompt_estimate=prompt_estimate,
        clip_id=clip_id,
        enforce_json_prompt=True,
    )
    repaired_text = get_chat_response_text(response)
    return extract_json_from_text(repaired_text, stage=repair_stage)


def call_openai_chat(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    stage: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    tracker: TokenTracker | None = None,
    prompt_estimate: int = 0,
    clip_id: str | None = None,
    enforce_json_prompt: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Build model-compatible params, call with rate-limit backoff, and retry on
    unsupported max_tokens / max_completion_tokens / temperature / response_format errors.
    """
    check_token_budget()
    msg_list = append_json_rules_to_messages(messages) if (
        enforce_json_prompt or response_format is not None
    ) else list(messages)
    create_kwargs = build_openai_params(
        model=model,
        messages=msg_list,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        **kwargs,
    )
    compat_attempts = 0
    max_compat_attempts = 4

    while compat_attempts < max_compat_attempts:
        try:
            return call_openai_with_backoff(
                client,
                create_kwargs=dict(create_kwargs),
                stage=stage,
                model=model,
                tracker=tracker,
                prompt_estimate=prompt_estimate,
                clip_id=clip_id,
            )
        except OpenAIRateLimitError:
            raise
        except Exception as exc:
            compat_attempts += 1
            if apply_param_compatibility_fix(create_kwargs, exc) and compat_attempts < max_compat_attempts:
                logger.warning(
                    "OpenAI compat retry stage=%s model=%s attempt=%d: %s",
                    stage, model, compat_attempts, exc,
                )
                continue
            raise

    raise RuntimeError(f"OpenAI compatibility retries exhausted for stage '{stage}'")


def call_openai_chat_json(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    stage: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    schema_hint: str = "",
    tracker: TokenTracker | None = None,
    prompt_estimate: int = 0,
    clip_id: str | None = None,
    allow_repair: bool = True,
    **kwargs: Any,
) -> dict | list[Any]:
    """
    call_openai_chat + parse JSON.
    gpt-5* empty/invalid JSON: one full retry with OPENAI_MODEL_JSON_FALLBACK (no gpt-5 repair).
    Other models: optional single repair call when output is non-empty but unparseable.
    """
    primary_model = model
    json_fallback = resolve_json_fallback_model(primary_model)
    effective_model = primary_model
    attempted_fallback = False
    rf = response_format or {"type": "json_object"}

    def _fetch(chat_model: str, chat_stage: str) -> str:
        resp = call_openai_chat(
            client,
            model=chat_model,
            messages=messages,
            stage=chat_stage,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=rf,
            tracker=tracker,
            prompt_estimate=prompt_estimate,
            clip_id=clip_id,
            enforce_json_prompt=True,
            **kwargs,
        )
        return get_chat_response_text(resp)

    def _retry_with_json_fallback(reason: str) -> bool:
        nonlocal effective_model, attempted_fallback, text
        if attempted_fallback or not json_fallback:
            return False
        attempted_fallback = True
        effective_model = json_fallback
        if model_is_gpt5_family(primary_model):
            _record_json_fallback(primary_model, json_fallback, reason)
        from clip_engine.telemetry import record_json_fallback as _tel_fb

        _tel_fb(f"{stage}_json_fallback", primary_model, json_fallback)
        notice = (
            f"Empty JSON response from {primary_model}; retrying with {json_fallback}."
            if reason == "empty"
            else f"Invalid JSON from {primary_model}; retrying with {json_fallback}."
        )
        _notify_status(notice)
        text = _fetch(json_fallback, f"{stage}_json_fallback")
        return True

    text = _fetch(primary_model, stage)

    if not text.strip():
        if model_is_gpt5_family(primary_model):
            _record_gpt5_empty()
        if model_is_gpt5_family(primary_model) and _retry_with_json_fallback("empty"):
            if not text.strip():
                raise ValueError(
                    f"Empty model response after JSON fallback (stage={stage}, model={effective_model})."
                )
        else:
            raise ValueError(f"Empty model response — no JSON to parse (stage={stage}).")

    try:
        parsed = extract_json_from_text(text, stage=stage)
        if model_is_gpt5_family(primary_model) and not attempted_fallback:
            _record_gpt5_success()
        return parsed
    except (ValueError, json.JSONDecodeError) as parse_err:
        if (
            model_is_gpt5_family(primary_model)
            and not attempted_fallback
            and _retry_with_json_fallback("invalid")
        ):
            try:
                parsed = extract_json_from_text(text, stage=stage)
                return parsed
            except (ValueError, json.JSONDecodeError):
                pass
        elif not text.strip():
            raise ValueError(
                f"Empty model response — no JSON to parse (stage={stage})."
            ) from parse_err

        if not allow_repair or not text.strip():
            raise
        preview = text[:240].replace("\n", "\\n")
        logger.warning(
            "JSON repair stage=%s source_model=%s fallback_used=%s reason=%s preview=%s",
            stage,
            effective_model,
            attempted_fallback,
            parse_err,
            preview,
        )
        from clip_engine.telemetry import record_json_repair

        record_json_repair(stage, effective_model)
        hint = schema_hint or "Return a single valid JSON object."
        return repair_json_with_chat(
            client,
            model=effective_model,
            raw_text=text,
            schema_hint=hint,
            stage=stage,
            tracker=tracker,
            prompt_estimate=prompt_estimate,
            clip_id=clip_id,
            max_tokens=min(max_tokens or 2000, 2500),
        )


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
    check_token_budget()
    cfg = config or RateLimitConfig()
    ctx = get_call_context()
    tracker = tracker or ctx.tracker or get_tracker()
    use_backoff = ctx.rate_limit_safe if rate_limit_safe is None else rate_limit_safe
    max_attempts = cfg.max_retries + 1 if use_backoff else 1
    last_exc: BaseException | None = None

    from clip_engine.telemetry import record_openai_request

    for attempt in range(1, max_attempts + 1):
        t_req = time.perf_counter()
        try:
            apply_call_delay()
            _log_openai_request_params(create_kwargs, stage=stage)
            if prompt_estimate:
                logger.debug(
                    "OpenAI call stage=%s model=%s attempt=%d est_prompt_tokens=%d",
                    stage, model, attempt, prompt_estimate,
                )
            response = client.chat.completions.create(**create_kwargs)
            latency = time.perf_counter() - t_req
            usage = getattr(response, "usage", None)
            if usage:
                tracker.record_openai_usage(stage, usage, model=model, clip_id=clip_id)
            finish_reason = None
            response_empty = True
            try:
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                content = (choice.message.content or "").strip()
                response_empty = not bool(content)
            except (AttributeError, IndexError, TypeError):
                pass
            record_openai_request(
                stage=stage,
                model=str(create_kwargs.get("model", model)),
                create_kwargs=create_kwargs,
                latency_sec=latency,
                retry_count=max(0, attempt - 1),
                prompt_estimate=prompt_estimate,
                fallback_used=stage.endswith("_json_fallback"),
                json_repair="_json_repair" in stage,
                response_empty=response_empty,
                finish_reason=finish_reason,
                usage=usage,
            )
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
            _interruptible_sleep(wait)

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
