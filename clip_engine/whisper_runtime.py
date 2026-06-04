# -*- coding: utf-8 -*-
"""Shared faster-whisper model instance with explicit GPU release."""

from __future__ import annotations

import os
import sys


def _set_ct2_throughput_env() -> None:
    """CTranslate2 GPU throughput tuning — must run before ctranslate2 is imported."""
    if os.environ.get("CT2_VERBOSE"):
        return
    os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "1")
    os.environ.setdefault("CT2_CUDA_ALLOW_FP16", "1")
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "0,0,0,0")


_set_ct2_throughput_env()


def _torch_lib_dir() -> str | None:
    try:
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            torch_lib = os.path.join(list(torch_spec.submodule_search_locations)[0], "lib")
            if os.path.isdir(torch_lib):
                return torch_lib
    except Exception:
        pass
    return None


_dll_fix_torch_lib: str | None = None
_dll_fix_prepended: bool = False
_dll_fix_directory_added: bool = False


def _prepend_venv_cuda_dlls() -> None:
    """
    On Windows, torch bundles its own CUDA DLLs under .venv/Lib/site-packages/torch/lib/.
    If a newer system CUDA Toolkit is installed (e.g. 12.9 vs torch's 12.8), Windows
    DLL search picks up the system cublas64_12.dll first, causing WinError 127 symbol
    mismatch. Prepending the torch lib path to PATH forces torch's bundled DLLs to win.
    """
    global _dll_fix_torch_lib, _dll_fix_prepended
    if sys.platform != "win32":
        return
    try:
        torch_lib = _torch_lib_dir()
        if torch_lib:
            _dll_fix_torch_lib = torch_lib
            current = os.environ.get("PATH", "")
            if torch_lib.lower() not in current.lower():
                os.environ["PATH"] = torch_lib + os.pathsep + current
                _dll_fix_prepended = True
                import logging

                logging.getLogger("clip_engine.whisper_runtime").info(
                    "[cuda-dll-fix] prepended torch lib to PATH: %s",
                    torch_lib,
                )
    except Exception as exc:
        import logging

        logging.getLogger("clip_engine.whisper_runtime").warning(
            "[cuda-dll-fix] failed: %s", exc
        )


def _add_torch_dll_directory() -> None:
    global _dll_fix_torch_lib, _dll_fix_directory_added
    if sys.platform != "win32":
        return
    try:
        torch_lib = _torch_lib_dir()
        if torch_lib:
            _dll_fix_torch_lib = torch_lib
            os.add_dll_directory(torch_lib)
            _dll_fix_directory_added = True
            import logging

            logging.getLogger("clip_engine.whisper_runtime").info(
                "[cuda-dll-fix] os.add_dll_directory: %s",
                torch_lib,
            )
    except Exception as exc:
        import logging

        logging.getLogger("clip_engine.whisper_runtime").warning(
            "[cuda-dll-fix] add_dll_directory failed: %s", exc
        )


_prepend_venv_cuda_dlls()
_add_torch_dll_directory()

import itertools
import logging
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("clip_engine.whisper_runtime")


def get_cuda_dll_fix_startup_lines() -> list[str]:
    """Lines for logs/startup_diagnostics.txt (mirrors import-time DLL setup)."""
    lines: list[str] = []
    torch_lib = _dll_fix_torch_lib or _torch_lib_dir()
    if torch_lib:
        suffix = "" if _dll_fix_prepended else " (already on PATH)"
        lines.append(f"[cuda-dll-fix] prepended torch lib to PATH: {torch_lib}{suffix}")
        if _dll_fix_directory_added:
            lines.append(f"[cuda-dll-fix] os.add_dll_directory: {torch_lib}")
    return lines


def get_env_startup_line() -> str:
    return f"[env] pid={os.getpid()} executable={sys.executable} prefix={sys.prefix}"


logger.info(get_env_startup_line())

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
        inner_device = None
        if _model is not None:
            inner = getattr(_model, "model", None)
            dev = getattr(inner, "device", None) if inner is not None else None
            if dev is not None:
                inner_device = str(dev)
    return {
        "loaded": loaded,
        "model_size": key[0] if key else None,
        "device": key[1] if key else None,
        "compute_type": key[2] if key else None,
        "inner_device": inner_device,
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
    """Return cached WhisperModel for (size, device, compute_type) or load once.

    Default model_size should be config.DEFAULT_WHISPER_MODEL (large-v3 on RTX 4090).
    """
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
        inner = getattr(_model, "model", None)
        actual_device = getattr(inner, "device", None) or getattr(_model, "device", device)
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

    from clip_engine.cuda_diagnostics import query_nvidia_gpu_memory_and_util

    mem_before, util_before = query_nvidia_gpu_memory_and_util()

    model = get_whisper_model(
        model_size=model_size, device=device, compute_type=compute_type
    )
    check_cancelled()

    cache = get_whisper_cache_state()
    actual_device = cache.get("device") or device

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
                segments, full_text, info = future.result(timeout=timeout)
                mem_after, util_after = query_nvidia_gpu_memory_and_util()
                try:
                    from clip_engine.stability import append_gpu_transcription_session_result

                    append_gpu_transcription_session_result(
                        segment_count=len(segments),
                        requested_device=device,
                        actual_device=str(actual_device),
                        gpu_mem_before_mib=mem_before,
                        gpu_mem_after_mib=mem_after,
                        gpu_util_before_pct=util_before,
                        gpu_util_after_pct=util_after,
                    )
                except Exception as exc:
                    logger.debug("GPU transcription session diagnostics skipped: %s", exc)
                return segments, full_text, info
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
