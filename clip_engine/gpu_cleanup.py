# -*- coding: utf-8 -*-
"""Unified GPU/RAM cleanup with VRAM logging to logs/gpu_cleanup.log."""

from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LOGS_DIR

logger = logging.getLogger("clip_engine.gpu_cleanup")

GPU_CLEANUP_LOG = LOGS_DIR / "gpu_cleanup.log"


def _vram_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            snap["allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            snap["reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
            snap["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        snap["torch_error"] = str(exc)
    try:
        import ctranslate2 as ct

        snap["ct2_cuda_devices"] = int(ct.get_cuda_device_count())
    except Exception:
        snap["ct2_cuda_devices"] = 0
    return snap


def _append_log(phase: str, before: dict[str, Any], after: dict[str, Any]) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = (
        f"{ts} phase={phase} "
        f"before_alloc={before.get('allocated_gb')} "
        f"after_alloc={after.get('allocated_gb')} "
        f"before_reserved={before.get('reserved_gb')} "
        f"after_reserved={after.get('reserved_gb')}"
    )
    GPU_CLEANUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GPU_CLEANUP_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.info("[gpu_cleanup] %s", line)


def release_yolo_model() -> None:
    try:
        from clip_engine.smart_crop import release_yolo_model as _release

        _release()
    except Exception as exc:  # noqa: BLE001
        logger.debug("yolo release: %s", exc)


def release_embedding_model() -> None:
    try:
        from clip_engine.semantic_ranking import release_embedding_model as _release

        _release()
    except Exception as exc:  # noqa: BLE001
        logger.debug("embedding release: %s", exc)


def cleanup_gpu_after_phase(
    phase: str,
    *,
    whisper: bool = False,
    yolo: bool = False,
    embeddings: bool = False,
) -> None:
    """
    Release ML models and CUDA cache after a pipeline phase.
    CTranslate2 memory requires dropping faster-whisper references (whisper=True).
    """
    before = _vram_snapshot()
    if whisper:
        try:
            from clip_engine.whisper_runtime import unload_whisper

            unload_whisper()
        except Exception as exc:  # noqa: BLE001
            logger.debug("whisper unload: %s", exc)
    if yolo:
        release_yolo_model()
    if embeddings:
        release_embedding_model()

    gc.collect()
    try:
        from clip_engine.stability import release_gpu_memory

        release_gpu_memory(phase)
    except Exception:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    after = _vram_snapshot()
    _append_log(phase, before, after)
