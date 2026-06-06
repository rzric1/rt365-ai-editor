# -*- coding: utf-8 -*-
"""Extract mono WAV for Whisper using ffmpeg."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_executable

logger = logging.getLogger(__name__)

# Configurable via AUDIO_EXTRACT_TIMEOUT env var; default 7200s (June 1 long-podcast value).
_DEFAULT_EXTRACT_TIMEOUT_SEC = 7200.0


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


_WORK_SUBDIR = "_work"
_WHISPER_INPUT_WAV = "_whisper_input.wav"


def whisper_input_wav_path(outputs_dir: Path | str | None = None) -> Path:
    """Canonical Whisper demux WAV: {outputs_dir}/_work/_whisper_input.wav."""
    if outputs_dir is None:
        from config import CLIP_STUDIO_OUTPUT_DIR

        outputs_dir = CLIP_STUDIO_OUTPUT_DIR
    base = Path(outputs_dir)
    return base / _WORK_SUBDIR / _WHISPER_INPUT_WAV


def _resolve_wav_out(
    wav_out: Path | str | None,
    *,
    outputs_dir: Path | str | None = None,
) -> Path:
    canonical = whisper_input_wav_path(outputs_dir)
    if wav_out is None:
        return canonical
    resolved = Path(os.fspath(wav_out))
    # Guard against collapsed paths from accidental "\_" escapes in string literals.
    if resolved.name == "clips_work_whisper_input.wav":
        return canonical
    return resolved


def extract_audio_wav(
    video_path: Path,
    wav_out: Path | str | None = None,
    *,
    outputs_dir: Path | str | None = None,
    sample_rate: int = 16000,
) -> None:
    """16 kHz mono PCM WAV — good default for speech APIs."""
    from clip_engine.job_control import set_pipeline_step
    from clip_engine.subprocess_guard import run_subprocess

    ensure_ffmpeg_on_path()
    exe = get_ffmpeg_executable()
    wav_out = _resolve_wav_out(wav_out, outputs_dir=outputs_dir)
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[audio_extract] START video=%s -> wav=%s", video_path.resolve(), wav_out.resolve())

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
        "-map",
        "0:a:0",
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
        os.fspath(wav_out.resolve()),
    ]
    set_pipeline_step("audio_extract")
    logger.info("[audio_extract] ffmpeg extract -> %s", wav_out.resolve())
    logger.debug("[audio_extract] ffmpeg cmd: %s", " ".join(cmd))
    t_start = time.perf_counter()
    timeout = _get_extract_timeout()

    try:
        run_subprocess(
            cmd,
            timeout=timeout,
            label="ffmpeg_audio_extract",
            check=True,
            capture_output=False,
        )
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
            f"FFmpeg audio extraction failed (exit {exc.returncode})\n"
            f"Output path used: {wav_out.resolve()}\n"
            f"{stderr[-2000:]}"
        ) from exc

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
