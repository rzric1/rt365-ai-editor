# -*- coding: utf-8 -*-
"""Shared faster-whisper model instance with explicit GPU release."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("clip_engine.whisper_runtime")

_lock = threading.Lock()
_model: Any = None
_model_key: tuple[str, str, str] | None = None  # size, device, compute_type


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
    with _lock:
        if _model is not None and _model_key == key:
            return _model
        if _model is not None:
            logger.info("[whisper] releasing previous model (key change %s -> %s)", _model_key, key)
            _release_model()
        from faster_whisper import WhisperModel

        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_key = key
        logger.info(
            "[whisper] loaded model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )
        return _model


def transcribe_wav(
    wav_path: Path,
    *,
    language: str | None,
    model_size: str,
    device: str,
    compute_type: str,
) -> tuple[list[dict], str, Any]:
    """Transcribe using shared model. Returns (segments, full_text, info)."""
    from clip_engine.job_control import check_cancelled

    model = get_whisper_model(
        model_size=model_size, device=device, compute_type=compute_type
    )
    check_cancelled()
    segs_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )
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
    return segments, full_text, info


def unload_whisper() -> None:
    with _lock:
        _release_model()
