# -*- coding: utf-8 -*-
"""Media duration via ffprobe (same bin dir as ffmpeg)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from clip_engine.ffmpeg_resolve import get_ffmpeg_executable

logger = logging.getLogger(__name__)


def get_media_duration_seconds(media_path: Path) -> float:
    """
    Return container duration in seconds (float), or 0.0 if probe fails.
    Uses ffprobe next to resolved ffmpeg.
    """
    ffmpeg = Path(get_ffmpeg_executable())
    ffprobe = ffmpeg.parent / ("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    if not ffprobe.is_file():
        ffprobe = ffmpeg.parent / "ffprobe"
    if not ffprobe.is_file():
        logger.warning("ffprobe not found beside ffmpeg: %s", ffmpeg.parent)
        return 0.0
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path.resolve()),
    ]
    try:
        from clip_engine.subprocess_guard import run_subprocess

        r = run_subprocess(cmd, timeout=120.0, label="ffprobe_duration", text=True)
        if r.returncode != 0:
            logger.warning("ffprobe failed: %s", (r.stderr or r.stdout)[:400])
            return 0.0
        data = json.loads(r.stdout or "{}")
        dur = data.get("format", {}).get("duration")
        if dur is None:
            return 0.0
        return max(0.0, float(dur))
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffprobe exception: %s", exc)
        return 0.0
