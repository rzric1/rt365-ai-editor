# -*- coding: utf-8 -*-
"""Transcription: prefer faster-whisper on CUDA when enabled; else OpenAI Whisper API."""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI

from clip_engine.audio_extract import extract_audio_wav, ffmpeg_available
from clip_engine.cuda_diagnostics import cublas_missing_hint, ctranslate2_cuda_runtime_probe
from clip_engine.ffmpeg_gpu import faster_whisper_cuda_available
from clip_engine.stability import MAX_CLOUD_WHISPER_WAV_BYTES, release_gpu_memory

logger = logging.getLogger(__name__)


def _segments_from_response(tr: Any) -> tuple[list[dict[str, Any]], str]:
    segments: list[dict[str, Any]] = []
    full_text = str(getattr(tr, "text", "") or "").strip()

    raw_list = getattr(tr, "segments", None)
    if raw_list is None and hasattr(tr, "model_dump"):
        d = tr.model_dump()
        full_text = str(d.get("text") or full_text).strip()
        raw_list = d.get("segments")

    if not raw_list:
        return segments, full_text

    for s in raw_list:
        if isinstance(s, dict):
            start = float(s.get("start", 0))
            end = float(s.get("end", 0))
            t = (s.get("text") or "").strip()
        else:
            start = float(getattr(s, "start", 0))
            end = float(getattr(s, "end", 0))
            t = (getattr(s, "text", "") or "").strip()
        if t:
            segments.append({"start": start, "end": end, "text": t})

    return segments, full_text


def transcribe_with_faster_whisper_cuda(
    wav_path: Path,
    *,
    language: str | None,
    model_size: str = "base",
) -> tuple[list[dict[str, Any]], str, str] | None:
    """
    Local faster-whisper: CUDA when runtime probe passes; if CUDA DLLs/model fail but a GPU
    is present, fall back to CPU int8/float32 before returning None (caller may use OpenAI).

    Returns (segments, text, device_tag) where device_tag is \"cuda\" or \"cpu\".
    """
    from clip_engine.job_control import set_pipeline_step
    from clip_engine.whisper_runtime import transcribe_wav

    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        logger.info("faster-whisper not installed; pip install faster-whisper for local Whisper.")
        return None

    try:
        import ctranslate2 as ct

        n_cuda = int(ct.get_cuda_device_count())
    except Exception:
        n_cuda = 0

    cuda_ok, cuda_msg = ctranslate2_cuda_runtime_probe()
    cuda_attempted = False
    last_cuda_exc: Exception | None = None

    if n_cuda > 0 and cuda_ok and faster_whisper_cuda_available():
        cuda_attempted = True
        for comp in ("float16", "int8_float16", "int8"):
            try:
                set_pipeline_step(f"whisper_cuda_{comp}")
                segs, txt, _info = transcribe_wav(
                    wav_path,
                    language=language,
                    model_size=model_size,
                    device="cuda",
                    compute_type=comp,
                )
                return segs, txt, "cuda"
            except Exception as exc:  # noqa: BLE001
                last_cuda_exc = exc
                logger.warning("faster-whisper CUDA compute_type=%s failed: %s", comp, exc)
                if cublas_missing_hint(str(exc)):
                    logger.warning(
                        "cuBLAS/CUDA DLL load issue — add CUDA 12.x `bin` to PATH or set CUDA_PATH; "
                        "trying CPU int8 fallback."
                    )
    elif n_cuda > 0 and not cuda_ok:
        logger.warning(
            "faster-whisper CUDA skipped (CTranslate2 runtime probe failed): %s",
            cuda_msg,
        )
        if cublas_missing_hint(cuda_msg):
            logger.warning(
                "Likely missing cublas64_12.dll — install NVIDIA CUDA Toolkit 12.x or match "
                "ctranslate2 wheel to your CUDA stack."
            )

    use_cpu_fallback = n_cuda > 0 and (not cuda_ok or cuda_attempted)
    if use_cpu_fallback:
        logger.error(
            "[whisper] ⚠️  CUDA unavailable or probe failed — falling back to CPU int8. "
            "Transcription will be SLOW. Check cublas64_12.dll is on PATH. "
            "Run: where.exe cublas64_12.dll  to verify."
        )
        for comp in ("int8", "float32"):
            try:
                set_pipeline_step(f"whisper_cpu_{comp}")
                segs, txt, _info = transcribe_wav(
                    wav_path,
                    language=language,
                    model_size=model_size,
                    device="cpu",
                    compute_type=comp,
                )
                return segs, txt, "cpu"
            except Exception as exc:  # noqa: BLE001
                logger.warning("faster-whisper CPU compute_type=%s failed: %s", comp, exc)

    if last_cuda_exc and not use_cpu_fallback:
        logger.error("faster-whisper CUDA failed: %s", last_cuda_exc)
    return None


def transcribe_video(
    video_path: Path,
    api_key: str,
    *,
    work_dir: Path,
    language: str | None = None,
    prefer_gpu: bool = False,
    faster_whisper_model: str = "base",
) -> tuple[list[dict[str, Any]], str]:
    """
    Returns (segments, full_plain_text).

    If prefer_gpu and CUDA+faster-whisper work, uses local GPU.
    Otherwise uses OpenAI whisper-1 (requires api_key).
    """
    from clip_engine.job_control import check_cancelled, set_pipeline_step

    if not ffmpeg_available():
        from config import ENV_FFMPEG_BINARY  # noqa: PLC0415

        raise RuntimeError(
            f"ffmpeg not found. Install from https://ffmpeg.org/download.html or set {ENV_FFMPEG_BINARY} "
            "in .env to the full path to ffmpeg (e.g. WinGet Gyan.FFmpeg) and restart the app."
        )

    from clip_engine.telemetry import pipeline_phase

    wav_path = work_dir / "_whisper_input.wav"
    try:
        with pipeline_phase("audio_extract"):
            extract_audio_wav(video_path, wav_path)

        check_cancelled()

        if prefer_gpu and os.environ.get("FORCE_CPU_WHISPER", "").lower() not in ("1", "true", "yes"):
            with pipeline_phase("transcription"):
                local = transcribe_with_faster_whisper_cuda(
                    wav_path,
                    language=language,
                    model_size=faster_whisper_model,
                )
            if local is not None:
                segs, txt, dev_tag = local
                logger.info(
                    "[transcribe] backend=%s device=%s",
                    "faster-whisper (CUDA)" if dev_tag == "cuda" else "faster-whisper (CPU FALLBACK)",
                    dev_tag,
                )
                return segs, txt
            logger.info("Transcription backend: falling back to OpenAI whisper-1 API.")

        key = (api_key or "").strip()
        if not key:
            raise ValueError(
                "OpenAI API key missing for cloud Whisper. Set OPENAI_API_KEY in .env, "
                "or install faster-whisper + CUDA and enable GPU acceleration."
            )

        wav_size = wav_path.stat().st_size
        if wav_size > MAX_CLOUD_WHISPER_WAV_BYTES:
            raise ValueError(
                f"Extracted audio is {_mb(wav_size)} MB — too large for cloud Whisper in RAM. "
                "Enable GPU acceleration (local faster-whisper) or use a shorter clip."
            )

        set_pipeline_step("openai_whisper_api")
        client = OpenAI(api_key=key)

        def _request() -> Any:
            with wav_path.open("rb") as audio_file:
                kwargs: dict[str, Any] = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json",
                }
                if language:
                    kwargs["language"] = language
                return client.audio.transcriptions.create(**kwargs)

        with pipeline_phase("transcription"):
            try:
                tr = _request()
            except TypeError:
                language = None
                tr = _request()

        segments, full_text = _segments_from_response(tr)

        if not segments and full_text:
            segments = [{"start": 0.0, "end": 0.0, "text": full_text}]
            logger.warning("Whisper returned no segments; using single blob.")

        if not full_text and segments:
            full_text = " ".join(s["text"] for s in segments)

        logger.info("Transcription backend: OpenAI whisper-1 (API).")
        return segments, full_text
    finally:
        try:
            from clip_engine.gpu_cleanup import cleanup_gpu_after_phase

            cleanup_gpu_after_phase("transcription_complete", whisper=True)
        except Exception:
            release_gpu_memory("transcription_complete")


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f}"


def segments_to_prompt_transcript(segments: list[dict[str, Any]], max_chars: int = 100_000) -> str:
    """Format segments for the clip-scoring model."""
    lines: list[str] = []
    total = 0
    for s in segments:
        line = f"[{float(s['start']):.2f}-{float(s['end']):.2f}] {s.get('text', '').strip()}"
        if total + len(line) > max_chars:
            lines.append("\n[... transcript truncated for length ...]")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)
