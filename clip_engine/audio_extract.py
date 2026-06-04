# -*- coding: utf-8 -*-
"""Extract mono WAV for Whisper using ffmpeg."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_executable

logger = logging.getLogger(__name__)

# Long podcasts: allow up to 2 hours for audio extract (no video re-encode).
FFMPEG_EXTRACT_TIMEOUT_SEC = 7200.0


def ffmpeg_available() -> bool:
    return ensure_ffmpeg_on_path() is not None


def extract_audio_wav(video_path: Path, wav_out: Path, *, sample_rate: int = 16000) -> None:
    """16 kHz mono PCM WAV — good default for speech APIs."""
    from clip_engine.job_control import set_pipeline_step
    from clip_engine.subprocess_guard import run_subprocess

    ensure_ffmpeg_on_path()
    exe = get_ffmpeg_executable()
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-y",
        "-i",
        str(video_path.resolve()),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_out.resolve()),
    ]
    set_pipeline_step("audio_extract")
    logger.info("ffmpeg extract: %s", " ".join(cmd))
    t_start = time.perf_counter()
    run_subprocess(
        cmd,
        timeout=FFMPEG_EXTRACT_TIMEOUT_SEC,
        label="ffmpeg_audio_extract",
        check=True,
    )
    elapsed = time.perf_counter() - t_start
    wav_size = os.path.getsize(wav_out) if wav_out.exists() else 0
    logger.info(
        "ffmpeg_audio_extract complete: duration_sec=%.1f wav_size_mb=%.1f path=%s",
        elapsed,
        wav_size / 1_048_576,
        wav_out,
    )
    if wav_size == 0:
        raise RuntimeError(f"FFmpeg produced a zero-byte WAV file: {wav_out}")
