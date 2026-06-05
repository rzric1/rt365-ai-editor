# -*- coding: utf-8 -*-
"""Extract mono WAV for Whisper using ffmpeg."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_executable

logger = logging.getLogger(__name__)

# Configurable via AUDIO_EXTRACT_TIMEOUT env var; default 900s (15 min).
_DEFAULT_EXTRACT_TIMEOUT_SEC = 900.0
_WAV_STALL_CHECK_INTERVAL_SEC = 30.0
_WAV_STALL_KILL_AFTER_SEC = 60.0


def _get_extract_timeout() -> float:
    try:
        return float(os.environ.get("AUDIO_EXTRACT_TIMEOUT", _DEFAULT_EXTRACT_TIMEOUT_SEC))
    except (TypeError, ValueError):
        return _DEFAULT_EXTRACT_TIMEOUT_SEC


def is_slow_drive(path: str | Path) -> bool:
    """Return True if path is on a non-C: drive (USB, network, HDD may be slow)."""
    drive = os.path.splitdrive(str(path))[0].upper().rstrip("\\").rstrip("/")
    return drive not in ("C:", "")


def ffmpeg_available() -> bool:
    return ensure_ffmpeg_on_path() is not None


def extract_audio_wav(video_path: Path, wav_out: Path, *, sample_rate: int = 16000) -> None:
    """16 kHz mono PCM WAV — good default for speech APIs."""
    from clip_engine.job_control import JobCancelledError, set_pipeline_step
    from clip_engine.subprocess_guard import run_subprocess

    ensure_ffmpeg_on_path()
    exe = get_ffmpeg_executable()
    wav_out.parent.mkdir(parents=True, exist_ok=True)

    if is_slow_drive(video_path):
        drive = os.path.splitdrive(str(video_path))[0].upper()
        logger.warning(
            "[audio_extract] WARNING: source file on non-C: drive (%s) — may be slow", drive
        )

    cmd = [
        exe,
        "-y",
        "-probesize",
        "50M",
        "-analyzeduration",
        "50M",
        "-i",
        str(video_path.resolve()),
        "-vn",
        "-threads",
        "0",
        "-acodec",
        "pcm_s16le",
        "-avoid_negative_ts",
        "make_zero",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_out.resolve()),
    ]
    set_pipeline_step("audio_extract")
    logger.info("[audio_extract] ffmpeg extract: %s", " ".join(cmd))
    t_start = time.perf_counter()
    timeout = _get_extract_timeout()

    # Stall monitor: kill ffmpeg if the output WAV stops growing for 60s.
    _stall_stop = threading.Event()
    _stall_reason: list[str] = []

    def _stall_monitor() -> None:
        last_size = -1
        last_growth_time = time.monotonic()
        while not _stall_stop.wait(_WAV_STALL_CHECK_INTERVAL_SEC):
            current_size = wav_out.stat().st_size if wav_out.exists() else 0
            if current_size > last_size:
                last_size = current_size
                last_growth_time = time.monotonic()
            elif time.monotonic() - last_growth_time > _WAV_STALL_KILL_AFTER_SEC:
                logger.error(
                    "[audio_extract] STALL detected — WAV not growing for %.0fs, killing ffmpeg",
                    _WAV_STALL_KILL_AFTER_SEC,
                )
                _stall_reason.append(
                    f"WAV output stalled (no growth for {_WAV_STALL_KILL_AFTER_SEC:.0f}s)"
                )
                try:
                    from clip_engine.job_control import request_cancel
                    request_cancel()
                except Exception:
                    pass
                break

    monitor = threading.Thread(target=_stall_monitor, daemon=True, name="ffmpeg_stall_monitor")
    monitor.start()

    try:
        run_subprocess(
            cmd,
            timeout=timeout,
            label="ffmpeg_audio_extract",
            check=True,
        )
    except JobCancelledError:
        if _stall_reason:
            msg = f"[audio_extract] ffmpeg stall: {_stall_reason[0]}"
            logger.error("[audio_extract] ERROR: %s", msg)
            raise TimeoutError(msg) from None
        raise
    except subprocess.TimeoutExpired:
        logger.error(
            "[audio_extract] ERROR: ffmpeg timed out after %.0fs (AUDIO_EXTRACT_TIMEOUT)", timeout
        )
        raise TimeoutError(
            f"Audio extraction timed out after {timeout:.0f}s. "
            "If reading from USB/network drive, copy the file to local storage first."
        ) from None
    except subprocess.CalledProcessError as exc:
        stderr = (getattr(exc, "output", "") or "").strip()
        logger.error("[audio_extract] ERROR: ffmpeg non-zero exit:\n%s", stderr[-2000:])
        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit {exc.returncode}):\n{stderr[-2000:]}"
        ) from exc
    finally:
        _stall_stop.set()
        monitor.join(timeout=2.0)

    elapsed = time.perf_counter() - t_start
    wav_size = os.path.getsize(wav_out) if wav_out.exists() else 0
    logger.info(
        "[audio_extract] complete: duration_sec=%.1f wav_size_mb=%.1f path=%s",
        elapsed,
        wav_size / 1_048_576,
        wav_out,
    )
    if wav_size == 0:
        raise RuntimeError(f"FFmpeg produced a zero-byte WAV file: {wav_out}")
