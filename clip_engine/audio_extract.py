"""Extract mono WAV for Whisper using ffmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_executable

logger = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    return ensure_ffmpeg_on_path() is not None


def extract_audio_wav(video_path: Path, wav_out: Path, *, sample_rate: int = 16000) -> None:
    """16 kHz mono PCM WAV — good default for speech APIs."""
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
    logger.info("ffmpeg extract: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
