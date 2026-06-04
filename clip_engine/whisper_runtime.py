# -*- coding: utf-8 -*-
"""Shared faster-whisper model instance with explicit GPU release."""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("clip_engine.whisper_runtime")

logger.info(f"[env] pid={os.getpid()} executable={sys.executable} prefix={sys.prefix}")

_lock = threading.Lock()
_model: Any = None
_model_key: tuple[str, str, str] | None = None  # size, device, compute_type

_DEFAULT_TRANSCRIBE_TIMEOUT_SEC = float(os.environ.get("WHISPER_TRANSCRIBE_TIMEOUT", "600"))


def _transcribe_timeout_sec() -> float:
    try:
        v = float(os.environ.get("WHISPER_TRANSCRIBE_TIMEOUT", "600"))
        return max(30.0, v)
    except ValueError:
        return _DEFAULT_TRANSCRIBE_TIMEOUT_SEC


def allow_cpu_fallback() -> bool:
    from clip_engine.cuda_diagnostics import allow_cpu_fallback as _acf

    return _acf()


def get_whisper_cache_state() -> dict[str, Any]:
    """Snapshot for Runtime Debug panel."""
    with _lock:
        loaded = _model is not None
        key = _model_key
    return {
        "loaded": loaded,
        "model_size": key[0] if key else None,
        "device": key[1] if key else None,
        "compute_type": key[2] if key else None,
    }


def _validate_cuda_or_raise() -> None:
    """Raise RuntimeError if CUDA was requested but is unavailable."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA transcription requested but torch is not installed. "
            "Install torch in .venv311 or set ALLOW_CPU_FALLBACK=1."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA transcription requested but torch.cuda.is_available() is False "
            f"(executable={sys.executable}). Use launch_ai_clip_studio.ps1 with .venv311 "
            f"or set ALLOW_CPU_FALLBACK=1."
        )

    from clip_engine.cuda_diagnostics import ctranslate2_cuda_runtime_probe

    cuda_ok, cuda_msg = ctranslate2_cuda_runtime_probe()
    if not cuda_ok:
        raise RuntimeError(
            f"CUDA transcription requested but CTranslate2 CUDA probe failed: {cuda_msg}. "
            "Ensure cublas64_12.dll is on PATH (CUDA Toolkit 12.x bin) or set ALLOW_CPU_FALLBACK=1."
        )


def _whisper_model_kwargs(*, device: str, compute_type: str) -> dict[str, Any]:
    if device == "cuda":
        return {"num_workers": 4, "cpu_threads": 0}
    return {"num_workers": 1}


def _release_model() -> None:
    global _model, _model_key
    if _model is None:
        return
    try:
        del _model
    except Exception:
        pass
    _model = None
    _model_key = None
    from clip_engine.stability import release_gpu_memory

    release_gpu_memory("whisper_unload")


def get_whisper_model(*, model_size: str, device: str, compute_type: str) -> Any:
    """Return cached WhisperModel for (size, device, compute_type) or load once."""
    global _model, _model_key
    key = (model_size, device, compute_type)

    if device == "cuda" and not allow_cpu_fallback():
        _validate_cuda_or_raise()

    with _lock:
        if _model is not None and _model_key == key:
            logger.info(
                "[whisper] cache HIT pid=%s model=%s device=%s compute_type=%s",
                os.getpid(),
                model_size,
                device,
                compute_type,
            )
            return _model
        if _model is not None:
            logger.info("[whisper] releasing previous model (key change %s -> %s)", _model_key, key)
            _release_model()

        from faster_whisper import WhisperModel

        extra = _whisper_model_kwargs(device=device, compute_type=compute_type)
        load_start = time.perf_counter()
        load_ts = datetime.now(timezone.utc).isoformat()
        logger.info(
            "[whisper] load START pid=%s ts=%s model=%s device=%s compute_type=%s kwargs=%s",
            os.getpid(),
            load_ts,
            model_size,
            device,
            compute_type,
            extra,
        )
        try:
            _model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                **extra,
            )
        except Exception:
            logger.exception(
                "[whisper] load FAILED pid=%s model=%s device=%s compute_type=%s",
                os.getpid(),
                model_size,
                device,
                compute_type,
            )
            raise

        _model_key = key
        elapsed = time.perf_counter() - load_start
        actual_device = getattr(_model, "device", device)
        logger.info(
            "[whisper] load END pid=%s elapsed_sec=%.2f model=%s requested_device=%s "
            "actual_device=%s compute_type=%s",
            os.getpid(),
            elapsed,
            model_size,
            device,
            actual_device,
            compute_type,
        )

        from clip_engine.cuda_diagnostics import gpu_pid_check

        gpu_pid_check(context="after_model_load")

        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.info("[whisper] post-load nvidia-smi: %s", smi.stdout.strip())
        except Exception as exc:
            logger.debug("[whisper] nvidia-smi query failed: %s", exc)
        return _model


def _collect_transcription(
    model: Any,
    wav_path: Path,
    *,
    language: str | None,
) -> tuple[list[dict], str, Any]:
    """Run model.transcribe and drain segments (runs in worker thread for timeout)."""
    from clip_engine.job_control import check_cancelled

    t_start = time.perf_counter()
    ts_start = datetime.now(timezone.utc).isoformat()
    logger.info("[whisper] transcribe START pid=%s ts=%s wav=%s", os.getpid(), ts_start, wav_path)

    segs_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    first_seg = None
    try:
        first_seg = next(iter(segs_iter))
        elapsed_to_first = time.perf_counter() - t_start
        logger.info("[whisper] first_segment_latency_sec=%.2f", elapsed_to_first)
        segs_iter = itertools.chain([first_seg], segs_iter)
    except StopIteration:
        logger.warning("[whisper] transcription produced zero segments")

    segments: list[dict] = []
    text_parts: list[str] = []
    for seg in segs_iter:
        check_cancelled()
        t = seg.text.strip()
        if not t:
            continue
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": t})
        text_parts.append(t)

    full_text = " ".join(text_parts).strip()
    if not segments and full_text:
        segments = [{"start": 0.0, "end": 0.0, "text": full_text}]

    elapsed = time.perf_counter() - t_start
    ts_end = datetime.now(timezone.utc).isoformat()
    logger.info(
        "[whisper] transcribe END pid=%s ts=%s elapsed_sec=%.2f segments=%s",
        os.getpid(),
        ts_end,
        elapsed,
        len(segments),
    )
    return segments, full_text, info


def _cleanup_on_transcribe_failure() -> None:
    try:
        from clip_engine.subprocess_guard import terminate_orphan_job_processes

        n = terminate_orphan_job_processes()
        if n:
            logger.warning("[whisper] terminated %s orphan job child process(es) after failure", n)
    except Exception as exc:
        logger.debug("[whisper] orphan cleanup failed: %s", exc)


def transcribe_wav(
    wav_path: Path,
    *,
    language: str | None,
    model_size: str,
    device: str,
    compute_type: str,
    timeout_sec: float | None = None,
) -> tuple[list[dict], str, Any]:
    """Transcribe using shared model. Returns (segments, full_text, info)."""
    from clip_engine.cuda_diagnostics import gpu_pid_check
    from clip_engine.job_control import check_cancelled

    timeout = timeout_sec if timeout_sec is not None else _transcribe_timeout_sec()
    pid = os.getpid()

    logger.info(
        "[whisper] transcribe_wav pid=%s device=%s compute_type=%s model=%s timeout_sec=%.0f",
        pid,
        device,
        compute_type,
        model_size,
        timeout,
    )
    gpu_pid_check(context="transcription_start")

    model = get_whisper_model(
        model_size=model_size, device=device, compute_type=compute_type
    )
    check_cancelled()

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        logger.info("[whisper] mid-transcription nvidia-smi: %s", smi.stdout.strip())
    except Exception:
        pass

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper_xcribe") as pool:
            future = pool.submit(_collect_transcription, model, wav_path, language=language)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                logger.error(
                    "[whisper] transcribe TIMEOUT pid=%s after %.0fs wav=%s",
                    pid,
                    timeout,
                    wav_path,
                )
                _cleanup_on_transcribe_failure()
                raise RuntimeError(
                    f"Whisper transcription timed out after {timeout:.0f}s. "
                    "Try a shorter clip, increase WHISPER_TRANSCRIBE_TIMEOUT, or cancel and retry."
                ) from exc
    except Exception:
        logger.error("[whisper] transcribe failed pid=%s\n%s", pid, traceback.format_exc())
        _cleanup_on_transcribe_failure()
        raise


def unload_whisper() -> None:
    with _lock:
        _release_model()
