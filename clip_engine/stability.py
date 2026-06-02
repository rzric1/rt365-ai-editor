# -*- coding: utf-8 -*-
"""Crash logging, startup diagnostics, temp cleanup, GPU memory release."""

from __future__ import annotations

import gc
import logging
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    CLIP_STUDIO_OUTPUT_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    UPLOADS_DIR,
    ensure_directories,
)

logger = logging.getLogger("clip_engine.stability")

CRASH_REPORT_PATH = LOGS_DIR / "crash_report.txt"
STARTUP_DIAG_PATH = LOGS_DIR / "startup_diagnostics.txt"

# Cloud Whisper: refuse to load entire WAV into RAM above this size.
MAX_CLOUD_WHISPER_WAV_BYTES = 25 * 1024 * 1024

_TEMP_GLOBS = (
    "._tmp.ass",
    "*.partial.mp4",
    "*_preview_failed.mp4",
)


def _append_report(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(block)
        if not block.endswith("\n"):
            f.write("\n")


def _system_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["ram_total_gb"] = round(vm.total / 1e9, 2)
        snap["ram_available_gb"] = round(vm.available / 1e9, 2)
        disk = shutil.disk_usage(PROJECT_ROOT)
        snap["disk_free_gb"] = round(disk.free / 1e9, 2)
    except ImportError:
        snap["psutil"] = "not installed"
    except Exception as exc:  # noqa: BLE001
        snap["psutil_error"] = str(exc)

    try:
        from clip_engine.cuda_diagnostics import collect_ai_acceleration_diagnostics

        diag = collect_ai_acceleration_diagnostics()
        snap["nvidia_smi_ok"] = diag.nvidia_smi_ok
        snap["driver_cuda"] = diag.driver_reported_cuda
        snap["gpu_line"] = (diag.nvidia_gpu_line or "")[:200]
        snap["ctranslate2_cuda_devices"] = diag.ctranslate2_cuda_devices
        snap["cuda_runtime_probe_ok"] = diag.cuda_runtime_probe_ok
    except Exception as exc:  # noqa: BLE001
        snap["cuda_diag_error"] = str(exc)

    try:
        import torch

        if torch.cuda.is_available():
            snap["torch_cuda"] = True
            snap["torch_device"] = torch.cuda.get_device_name(0)
            snap["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            snap["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            snap["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
    except ImportError:
        snap["torch_cuda"] = False
    except Exception as exc:  # noqa: BLE001
        snap["torch_error"] = str(exc)

    try:
        from clip_engine.subprocess_guard import list_tracked_pids

        snap["tracked_child_pids"] = list_tracked_pids()
    except Exception:
        snap["tracked_child_pids"] = []

    try:
        from clip_engine.job_control import get_active_job, get_pipeline_step

        snap["active_job"] = get_active_job()
        snap["pipeline_step"] = get_pipeline_step()
    except Exception:
        pass

    return snap


def write_crash_report(
    exc: BaseException,
    *,
    context: str = "",
    ffmpeg_cmd: str = "",
) -> None:
    """Append a crash report block to logs/crash_report.txt."""
    ts = datetime.now(timezone.utc).isoformat()
    snap = _system_snapshot()
    lines = [
        "=" * 72,
        f"CRASH REPORT {ts}",
        f"Context: {context or 'unknown'}",
        f"Exception: {type(exc).__name__}: {exc}",
        f"Active job: {snap.get('active_job')}",
        f"Pipeline step: {snap.get('pipeline_step')}",
        f"FFmpeg cmd: {ffmpeg_cmd[:800] if ffmpeg_cmd else '(none)'}",
        "System snapshot:",
    ]
    for k, v in snap.items():
        lines.append(f"  {k}: {v}")
    lines.append("Traceback:")
    lines.append(traceback.format_exc())
    _append_report(CRASH_REPORT_PATH, "\n".join(lines))
    logger.error("Wrote crash report to %s", CRASH_REPORT_PATH)


def release_gpu_memory(label: str = "") -> None:
    """Best-effort GPU memory release after model use."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[stability] torch.cuda.empty_cache() after %s", label or "release")
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("empty_cache skipped: %s", exc)


def run_startup_diagnostics() -> str:
    """Write logs/startup_diagnostics.txt and return summary text."""
    ensure_directories()
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        f"RT365 AI Clip Studio — startup diagnostics {ts}",
        f"Project root: {PROJECT_ROOT}",
        "",
        "=== System ===",
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.split()[0]}",
        "",
        "=== RTX / NVIDIA (4090-class) ===",
    ]
    snap = _system_snapshot()
    lines.append(f"nvidia-smi OK: {snap.get('nvidia_smi_ok')}")
    lines.append(f"Driver CUDA (banner): {snap.get('driver_cuda')}")
    lines.append(f"GPU: {snap.get('gpu_line')}")
    lines.append(f"CTranslate2 CUDA devices: {snap.get('ctranslate2_cuda_devices')}")
    lines.append(f"CUDA runtime probe: {snap.get('cuda_runtime_probe_ok')}")
    lines.append(f"Torch CUDA available: {snap.get('torch_cuda')}")
    if snap.get("torch_device"):
        lines.append(f"Torch device: {snap.get('torch_device')}")
    if snap.get("vram_total_gb"):
        lines.append(
            f"VRAM total / allocated / reserved (GB): "
            f"{snap.get('vram_total_gb')} / {snap.get('vram_allocated_gb')} / {snap.get('vram_reserved_gb')}"
        )
    lines.append("")
    lines.append("=== Resources ===")
    lines.append(f"RAM total GB: {snap.get('ram_total_gb', '?')}")
    lines.append(f"RAM available GB: {snap.get('ram_available_gb', '?')}")
    lines.append(f"Disk free GB (project drive): {snap.get('disk_free_gb', '?')}")
    lines.append("")
    lines.append("=== Dependencies ===")
    ffmpeg = shutil.which("ffmpeg")
    lines.append(f"ffmpeg on PATH: {bool(ffmpeg)} ({ffmpeg or 'missing'})")
    try:
        from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path

        resolved = ensure_ffmpeg_on_path()
        lines.append(f"ffmpeg resolved: {resolved}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"ffmpeg resolve error: {exc}")

    for mod in ("streamlit", "openai", "faster_whisper", "torch", "ctranslate2"):
        try:
            __import__(mod)
            lines.append(f"import {mod}: OK")
        except ImportError:
            lines.append(f"import {mod}: MISSING (optional)" if mod in ("faster_whisper", "torch") else f"import {mod}: MISSING")

    lines.append("")
    lines.append("=== Folders ===")
    for d in (LOGS_DIR, UPLOADS_DIR, CLIP_STUDIO_OUTPUT_DIR):
        lines.append(f"{d}: exists={d.exists()}")

    lines.append("")
    lines.append(
        "RTX 4090 note: This build targets high-VRAM GPUs. Stability depends on "
        "releasing Whisper/NVENC stacks between phases and killing orphan ffmpeg.exe."
    )

    text = "\n".join(lines)
    STARTUP_DIAG_PATH.write_text(text, encoding="utf-8")
    logger.info("Startup diagnostics written to %s", STARTUP_DIAG_PATH)
    return text


def cleanup_temp_artifacts(*, max_preview_age_days: int = 7) -> dict[str, int]:
    """
    Remove safe temp artifacts. Never deletes user uploads or source videos.
    """
    stats = {"ass_removed": 0, "partial_removed": 0, "work_wav_cleared": 0, "preview_removed": 0}
    now = time.time()
    max_age = max_preview_age_days * 86400

    search_roots = [
        CLIP_STUDIO_OUTPUT_DIR,
        CLIP_STUDIO_OUTPUT_DIR / "_work",
        CLIP_STUDIO_OUTPUT_DIR / "previews",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name.endswith("._tmp.ass"):
                try:
                    path.unlink()
                    stats["ass_removed"] += 1
                except OSError:
                    pass
            elif name.endswith(".partial.mp4"):
                try:
                    path.unlink()
                    stats["partial_removed"] += 1
                except OSError:
                    pass
            elif "preview" in path.parts and name.endswith(".mp4"):
                try:
                    if now - path.stat().st_mtime > max_age:
                        path.unlink()
                        stats["preview_removed"] += 1
                except OSError:
                    pass

    work_wav = CLIP_STUDIO_OUTPUT_DIR / "_work" / "_whisper_input.wav"
    if work_wav.is_file():
        try:
            work_wav.unlink()
            stats["work_wav_cleared"] = 1
        except OSError:
            pass

    logger.info("[stability] temp cleanup: %s", stats)
    return stats


def log_resource_snapshot(*, label: str = "snapshot") -> dict[str, Any]:
    """
    Append a lightweight CPU/RAM/VRAM/disk snapshot to logs/resource_monitor.log.
    Safe to call from UI or pipeline phases; never raises.
    """
    snap = _system_snapshot()
    snap["label"] = label
    snap["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        import psutil

        snap["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        proc = psutil.Process()
        mem = proc.memory_info()
        snap["process_rss_mb"] = round(mem.rss / (1024 * 1024), 1)
        snap["process_threads"] = proc.num_threads()
    except Exception as exc:  # noqa: BLE001
        snap["process_metrics_error"] = str(exc)

    try:
        from clip_engine.subprocess_guard import find_orphan_ffmpeg_pids, list_tracked_pids

        snap["tracked_child_pids"] = list_tracked_pids()
        snap["orphan_ffmpeg_pids"] = find_orphan_ffmpeg_pids()
    except Exception:
        pass

    line = " ".join(f"{k}={v}" for k, v in snap.items())
    logger.info("[resource] %s", line[:2000])
    try:
        path = LOGS_DIR / "resource_monitor.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        _append_report(path, line)
    except OSError:
        pass
    return snap


def install_exception_hooks() -> None:
    """Log uncaught exceptions to crash_report.txt."""

    def _hook(exc_type, exc, tb):
        if exc is not None and issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        try:
            write_crash_report(
                exc if isinstance(exc, BaseException) else RuntimeError(str(exc)),
                context="uncaught_exception",
            )
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook
