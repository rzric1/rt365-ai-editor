"""
clip_engine/telemetry.py

Production telemetry: structured logs, session accounting, timing, GPU snapshots.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("clip_engine.telemetry")
_openai_file_logger = logging.getLogger("clip_engine.telemetry.openai")
_gpu_file_logger = logging.getLogger("clip_engine.telemetry.gpu")
_export_file_logger = logging.getLogger("clip_engine.telemetry.exports")

_LOG_CONFIGURED = False
_MAX_RECENT = 40


@dataclass
class SessionTelemetry:
    """Accumulated diagnostics for one analysis / studio session."""

    openai_request_count: int = 0
    fallback_count: int = 0
    json_repair_count: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    rejection_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pipeline_timing: dict[str, float] = field(default_factory=dict)
    gpu_snapshots: list[dict[str, Any]] = field(default_factory=list)
    export_records: list[dict[str, Any]] = field(default_factory=list)
    recent_openai: list[dict[str, Any]] = field(default_factory=list)
    recent_rejects: list[dict[str, Any]] = field(default_factory=list)
    recent_exports: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        total_est = self.estimated_input_tokens + self.estimated_output_tokens
        total_actual = self.actual_prompt_tokens + self.actual_completion_tokens
        timing_total = sum(self.pipeline_timing.values())
        return {
            "openai_requests": self.openai_request_count,
            "fallbacks": self.fallback_count,
            "json_repairs": self.json_repair_count,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": total_est,
            "actual_prompt_tokens": self.actual_prompt_tokens,
            "actual_completion_tokens": self.actual_completion_tokens,
            "actual_total_tokens": total_actual,
            "rejection_summary": dict(self.rejection_counts),
            "pipeline_timing": dict(self.pipeline_timing),
            "pipeline_timing_total_sec": round(timing_total, 2),
            "gpu_snapshots": list(self.gpu_snapshots),
            "export_records": list(self.export_records),
            "recent_openai": list(self.recent_openai[-_MAX_RECENT:]),
            "recent_rejects": list(self.recent_rejects[-_MAX_RECENT:]),
            "recent_exports": list(self.recent_exports[-_MAX_RECENT:]),
        }


_session: SessionTelemetry | None = None


def configure_rotating_logs(logs_dir: Path | None = None) -> None:
    """Attach size-rotating file handlers under logs/ (idempotent)."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    if logs_dir is None:
        from config import LOGS_DIR

        logs_dir = LOGS_DIR
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _attach(name: str, filename: str, level: int = logging.INFO) -> None:
        log = logging.getLogger(name)
        log.setLevel(level)
        path = logs_dir / filename
        for h in log.handlers:
            if isinstance(h, logging.handlers.RotatingFileHandler) and getattr(
                h, "baseFilename", ""
            ).endswith(filename):
                return
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.propagate = True

    _attach("clip_studio", "app.log")
    _attach("clip_engine", "app.log")
    _attach("clip_engine.telemetry", "app.log")
    _attach("clip_engine.telemetry.openai", "openai.log")
    _attach("clip_engine.telemetry.gpu", "gpu.log")
    _attach("clip_engine.telemetry.exports", "exports.log")
    _attach("clip_engine.openai_resilience", "openai.log")
    _attach("clip_engine.gpu_pipeline", "gpu.log")
    _attach("clip_engine.export_vertical", "exports.log")

    _LOG_CONFIGURED = True
    logger.info("Rotating log files enabled in %s", logs_dir)


def reset_session_telemetry() -> SessionTelemetry:
    global _session
    _session = SessionTelemetry()
    return _session


def get_session_telemetry() -> SessionTelemetry:
    global _session
    if _session is None:
        _session = SessionTelemetry()
    return _session


@contextmanager
def pipeline_phase(name: str) -> Iterator[None]:
    """Time a pipeline stage and accumulate into session telemetry."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        sess = get_session_telemetry()
        sess.pipeline_timing[name] = sess.pipeline_timing.get(name, 0.0) + elapsed
        logger.info("[PIPELINE TIMING] %s=%.1fs", name, elapsed)


def _prompt_chars_from_messages(messages: list[dict[str, Any]] | None) -> int:
    if not messages:
        return 0
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(str(part.get("text", "")))
    return total


def _max_output_tokens(params: dict[str, Any]) -> int | None:
    for key in ("max_completion_tokens", "max_tokens"):
        val = params.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def record_openai_request(
    *,
    stage: str,
    model: str,
    create_kwargs: dict[str, Any],
    latency_sec: float,
    retry_count: int = 0,
    prompt_estimate: int = 0,
    fallback_used: bool = False,
    json_repair: bool = False,
    response_empty: bool = False,
    finish_reason: str | None = None,
    usage: Any = None,
) -> None:
    """Log structured OpenAI request telemetry and update session totals."""
    messages = create_kwargs.get("messages") or []
    prompt_chars = _prompt_chars_from_messages(messages)
    max_out = _max_output_tokens(create_kwargs)
    est_in = prompt_estimate or max(1, prompt_chars // 4)
    est_out = max_out or 0

    sess = get_session_telemetry()
    sess.openai_request_count += 1
    sess.estimated_input_tokens += est_in
    if est_out:
        sess.estimated_output_tokens += est_out
    if fallback_used:
        sess.fallback_count += 1
    if json_repair:
        sess.json_repair_count += 1

    if usage is not None:
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        sess.actual_prompt_tokens += pt
        sess.actual_completion_tokens += ct

    record = {
        "stage": stage,
        "model": model,
        "prompt_chars": prompt_chars,
        "est_input_tokens": est_in,
        "max_output_tokens": max_out,
        "latency_sec": round(latency_sec, 3),
        "retry_count": retry_count,
        "fallback_used": fallback_used,
        "json_repair": json_repair,
        "response_empty": response_empty,
        "finish_reason": finish_reason or "unknown",
    }
    sess.recent_openai.append(record)
    if len(sess.recent_openai) > _MAX_RECENT:
        sess.recent_openai = sess.recent_openai[-_MAX_RECENT:]

    block = (
        f"[OPENAI]\n"
        f"stage={stage}\n"
        f"model={model}\n"
        f"prompt_chars={prompt_chars}\n"
        f"est_input_tokens={est_in}\n"
        f"max_output_tokens={max_out}\n"
        f"latency_sec={latency_sec:.2f}\n"
        f"retry_count={retry_count}\n"
        f"fallback_used={fallback_used}\n"
        f"json_repair={json_repair}\n"
        f"response_empty={response_empty}\n"
        f"finish_reason={finish_reason or 'unknown'}"
    )
    _openai_file_logger.info(block)
    logger.info(
        "[OPENAI] stage=%s model=%s latency_sec=%.2f retry=%d fallback=%s repair=%s",
        stage,
        model,
        latency_sec,
        retry_count,
        fallback_used,
        json_repair,
    )


def record_json_fallback(stage: str, primary: str, fallback: str) -> None:
    """Log fallback decision; request counter incremented on the fallback API call."""
    logger.info(
        "[OPENAI] json_fallback stage=%s primary=%s fallback=%s",
        stage,
        primary,
        fallback,
    )


def record_json_repair(stage: str, model: str) -> None:
    get_session_telemetry().json_repair_count += 1
    logger.info("[OPENAI] json_repair stage=%s model=%s", stage, model)


def log_session_tokens_summary() -> None:
    sess = get_session_telemetry()
    total_est = sess.estimated_input_tokens + sess.estimated_output_tokens
    block = (
        f"[SESSION TOKENS]\n"
        f"estimated_input={sess.estimated_input_tokens}\n"
        f"estimated_output={sess.estimated_output_tokens}\n"
        f"estimated_total={total_est}\n"
        f"actual_input={sess.actual_prompt_tokens}\n"
        f"actual_output={sess.actual_completion_tokens}\n"
        f"openai_requests={sess.openai_request_count}\n"
        f"fallbacks={sess.fallback_count}\n"
        f"json_repairs={sess.json_repair_count}"
    )
    _openai_file_logger.info(block)
    logger.info(
        "[SESSION TOKENS] estimated_total=%d requests=%d fallbacks=%d repairs=%d",
        total_est,
        sess.openai_request_count,
        sess.fallback_count,
        sess.json_repair_count,
    )


def log_clip_reject(reason: str, **details: Any) -> None:
    """Structured clip rejection log."""
    sess = get_session_telemetry()
    sess.rejection_counts[reason] += 1
    entry = {"reason": reason, **details}
    sess.recent_rejects.append(entry)
    if len(sess.recent_rejects) > _MAX_RECENT:
        sess.recent_rejects = sess.recent_rejects[-_MAX_RECENT:]

    extra = " ".join(f"{k}={v}" for k, v in details.items())
    logger.info("[CLIP REJECT] reason=%s %s", reason, extra.strip())


def log_rejection_summary() -> None:
    sess = get_session_telemetry()
    if not sess.rejection_counts:
        return
    parts = [f"{k}={v}" for k, v in sorted(sess.rejection_counts.items())]
    block = "[REJECTION SUMMARY]\n" + "\n".join(parts)
    logger.info(block)


def log_pipeline_timing_summary() -> None:
    sess = get_session_telemetry()
    if not sess.pipeline_timing:
        return
    lines = [f"{k}={v:.1f}s" for k, v in sorted(sess.pipeline_timing.items())]
    total = sum(sess.pipeline_timing.values())
    lines.append(f"total_pipeline={total:.1f}s")
    block = "[PIPELINE TIMING]\n" + "\n".join(lines)
    logger.info(block)


def _collect_gpu_snapshot(label: str) -> dict[str, Any]:
    snap: dict[str, Any] = {"label": label, "cuda_available": False}
    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda_available"] = True
            snap["device_name"] = torch.cuda.get_device_name(0)
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            snap["allocated_gb"] = round(alloc, 2)
            snap["reserved_gb"] = round(reserved, 2)
            snap["peak_gb"] = round(peak, 2)
            try:
                import pynvml  # type: ignore[import-untyped]

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snap["gpu_utilization_pct"] = int(util.gpu)
            except Exception:
                snap["gpu_utilization_pct"] = None
    except ImportError:
        snap["note"] = "torch not installed"
    except Exception as exc:
        snap["error"] = str(exc)[:200]
    return snap


def log_gpu_memory(label: str) -> dict[str, Any]:
    """Snapshot CUDA memory and append to session telemetry."""
    snap = _collect_gpu_snapshot(label)
    get_session_telemetry().gpu_snapshots.append(snap)
    if snap.get("cuda_available"):
        util = snap.get("gpu_utilization_pct")
        util_s = f" gpu_util={util}%" if util is not None else ""
        block = (
            f"[GPU MEMORY] label={label}\n"
            f"allocated_gb={snap.get('allocated_gb')}\n"
            f"reserved_gb={snap.get('reserved_gb')}\n"
            f"peak_gb={snap.get('peak_gb')}{util_s}"
        )
        _gpu_file_logger.info(block)
        logger.info(
            "[GPU MEMORY] %s allocated_gb=%s peak_gb=%s",
            label,
            snap.get("allocated_gb"),
            snap.get("peak_gb"),
        )
    return snap


def log_export_event(
    *,
    clip_title: str,
    duration_sec: float,
    resolution: str,
    encoder: str,
    elapsed_sec: float,
    subtitle_burn: bool,
    size_mb: float | None = None,
    output_path: str | None = None,
) -> None:
    record = {
        "clip": clip_title,
        "duration": round(duration_sec, 1),
        "resolution": resolution,
        "encoder": encoder,
        "subtitle_burn": subtitle_burn,
        "elapsed_sec": round(elapsed_sec, 1),
        "size_mb": round(size_mb, 1) if size_mb is not None else None,
        "output_path": output_path,
    }
    sess = get_session_telemetry()
    sess.export_records.append(record)
    sess.recent_exports.append(record)
    if len(sess.recent_exports) > _MAX_RECENT:
        sess.recent_exports = sess.recent_exports[-_MAX_RECENT:]

    block = (
        f'[EXPORT]\n'
        f'clip="{clip_title}"\n'
        f"duration={duration_sec:.1f}\n"
        f"encoder={encoder}\n"
        f"subtitle_burn={subtitle_burn}\n"
        f"elapsed_sec={elapsed_sec:.1f}\n"
        f"size_mb={size_mb}\n"
        f"resolution={resolution}"
    )
    _export_file_logger.info(block)
    logger.info(
        '[EXPORT] clip="%s" encoder=%s elapsed_sec=%.1f',
        clip_title,
        encoder,
        elapsed_sec,
    )


# ---------------------------------------------------------------------------
# Error classification (user-facing + logs)
# ---------------------------------------------------------------------------

class OpenAITimeoutError(TimeoutError):
    """OpenAI or network timeout."""


class JSONParseError(ValueError):
    """Model response JSON could not be parsed."""


class StreamlitStateError(RuntimeError):
    """Streamlit widget/session state conflict."""


class CUDAUnavailableError(RuntimeError):
    """CUDA required but not available."""


class NVENCProbeError(RuntimeError):
    """NVENC probe or encode failed."""


class ExportFailure(RuntimeError):
    """Clip export failed."""


class TranscriptFailure(RuntimeError):
    """Transcription failed."""


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """
    Return (category, user_friendly_message).
    Full traceback should still be logged by the caller.
    """
    msg = str(exc).lower()
    name = type(exc).__name__

    from clip_engine.openai_resilience import OpenAIRateLimitError as _RL

    if isinstance(exc, _RL):
        mit = getattr(exc, "mitigation", "") or "Wait and retry."
        return (
            "OpenAIRateLimitError",
            f"OpenAI rate limit at {getattr(exc, 'stage', 'unknown')}: {mit}",
        )

    if isinstance(exc, (TimeoutError, OpenAITimeoutError)) or "timeout" in msg:
        return (
            "OpenAITimeoutError",
            "The AI request timed out. Try Token Saver mode, a shorter video, or retry.",
        )
    if isinstance(exc, (JSONParseError, json.JSONDecodeError)) or (
        "json" in msg and ("parse" in msg or "empty" in msg)
    ):
        return (
            "JSONParseError",
            "The model returned invalid or empty JSON. Retry analysis; SAFE profile uses gpt-4o-mini.",
        )
    if "session_state" in msg and "cannot be modified" in msg:
        return (
            "StreamlitStateError",
            "A UI state conflict occurred. Refresh the page and run the step again.",
        )
    if "cuda" in msg and ("not available" in msg or "unavailable" in msg):
        return (
            "CUDAUnavailableError",
            "CUDA is not available for this step. Check GPU diagnostics or use CPU fallback.",
        )
    if "nvenc" in msg or "h264_nvenc" in msg:
        return (
            "NVENCProbeError",
            "GPU export (NVENC) failed. Try Force GPU off or allow CPU fallback in settings.",
        )
    if "ffmpeg" in msg or "export" in msg and name in ("RuntimeError", "ExportFailure"):
        return (
            "ExportFailure",
            f"Export failed: {exc}",
        )
    if "transcri" in msg or isinstance(exc, TranscriptFailure):
        return (
            "TranscriptFailure",
            f"Transcription failed: {exc}",
        )
    if "openai" in msg or "api key" in msg:
        return ("OpenAIError", f"OpenAI error: {exc}")
    return (name, str(exc))


def render_telemetry_markdown(data: dict[str, Any] | None) -> str:
    """Format session telemetry for Streamlit markdown."""
    if not data:
        return "_No telemetry for this session yet. Run transcribe or analyze._"
    lines = [
        "### OpenAI Session Telemetry",
        f"- Requests: **{data.get('openai_requests', 0)}** | "
        f"Fallbacks: **{data.get('fallbacks', 0)}** | "
        f"JSON repairs: **{data.get('json_repairs', 0)}**",
        f"- Est. tokens: **{data.get('estimated_total_tokens', 0):,}** "
        f"(in {data.get('estimated_input_tokens', 0):,} / "
        f"out {data.get('estimated_output_tokens', 0):,})",
        f"- Actual tokens: **{data.get('actual_total_tokens', 0):,}**",
    ]
    timing = data.get("pipeline_timing") or {}
    if timing:
        lines.append("### Pipeline timing")
        for k, v in sorted(timing.items()):
            try:
                lines.append(f"- `{k}`: {float(v):.1f}s")
            except (TypeError, ValueError):
                lines.append(f"- `{k}`: {v}")
        try:
            total_sec = float(data.get("pipeline_timing_total_sec", 0))
        except (TypeError, ValueError):
            total_sec = 0.0
        lines.append(f"- **Total tracked:** {total_sec:.1f}s")
    rej = data.get("rejection_summary") or {}
    if rej:
        lines.append("### Clip rejection summary")
        for k, v in sorted(rej.items()):
            lines.append(f"- `{k}`: {v}")
    gpu = data.get("gpu_snapshots") or []
    if gpu:
        last = gpu[-1]
        lines.append("### GPU memory (latest)")
        lines.append(
            f"- `{last.get('label', '?')}`: alloc **{last.get('allocated_gb', '?')} GB**, "
            f"peak **{last.get('peak_gb', '?')} GB**"
        )
    exports = data.get("export_records") or []
    if exports:
        lines.append(f"### Exports ({len(exports)} clips)")
        for ex in exports[-5:]:
            lines.append(
                f"- **{ex.get('clip', '?')}** — {ex.get('encoder', '?')}, "
                f"{ex.get('elapsed_sec', '?')}s"
            )
    recent = data.get("recent_openai") or []
    if recent:
        lines.append("### Recent OpenAI calls")
        for r in recent[-5:]:
            if not isinstance(r, dict):
                continue
            try:
                lat = float(r.get("latency_sec", 0))
                lat_s = f"{lat:.2f}s"
            except (TypeError, ValueError):
                lat_s = str(r.get("latency_sec", "?"))
            lines.append(
                f"- `{r.get('stage', '?')}` **{r.get('model', '?')}** "
                f"{lat_s} (retry={r.get('retry_count', 0)})"
            )
    return "\n".join(lines)


__all__ = [
    "CUDAUnavailableError",
    "ExportFailure",
    "JSONParseError",
    "NVENCProbeError",
    "OpenAITimeoutError",
    "SessionTelemetry",
    "StreamlitStateError",
    "TranscriptFailure",
    "classify_exception",
    "configure_rotating_logs",
    "get_session_telemetry",
    "log_clip_reject",
    "log_export_event",
    "log_gpu_memory",
    "log_pipeline_timing_summary",
    "log_rejection_summary",
    "log_session_tokens_summary",
    "pipeline_phase",
    "record_json_fallback",
    "record_json_repair",
    "record_openai_request",
    "render_telemetry_markdown",
    "reset_session_telemetry",
]
