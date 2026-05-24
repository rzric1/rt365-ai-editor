"""
clip_engine/captions.py
Modern burned-in caption styles + SRT/ASS sidecar file generation.
Presets: Clean | Bold Viral | Podcast | Minimal
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("clip_engine.captions")

CaptionPreset = Literal["Clean", "Bold Viral", "Podcast", "Minimal"]

# ---------------------------------------------------------------------------
# Caption style definitions
# ---------------------------------------------------------------------------

@dataclass
class CaptionStyle:
    preset: CaptionPreset
    font_name: str
    font_size: int
    primary_color: str      # ASS hex &H00BBGGRR
    outline_color: str
    back_color: str
    bold: bool
    outline_width: float
    shadow_depth: float
    margin_v: int           # bottom margin in pixels (for 1920h frame)
    margin_l: int
    margin_r: int
    all_caps: bool
    max_chars_per_line: int
    max_lines: int


CAPTION_PRESETS: dict[CaptionPreset, CaptionStyle] = {
    "Clean": CaptionStyle(
        preset="Clean",
        font_name="Arial",
        font_size=56,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&H80000000",
        bold=True,
        outline_width=2.0,
        shadow_depth=1.0,
        margin_v=160,
        margin_l=80,
        margin_r=80,
        all_caps=False,
        max_chars_per_line=32,
        max_lines=2,
    ),
    "Bold Viral": CaptionStyle(
        preset="Bold Viral",
        font_name="Impact",
        font_size=72,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&HA0000000",
        bold=True,
        outline_width=3.5,
        shadow_depth=2.0,
        margin_v=180,
        margin_l=60,
        margin_r=60,
        all_caps=True,
        max_chars_per_line=26,
        max_lines=2,
    ),
    "Podcast": CaptionStyle(
        preset="Podcast",
        font_name="Helvetica Neue",
        font_size=52,
        primary_color="&H00FFFFFF",
        outline_color="&H00101010",
        back_color="&H90000000",
        bold=False,
        outline_width=1.5,
        shadow_depth=1.0,
        margin_v=140,
        margin_l=100,
        margin_r=100,
        all_caps=False,
        max_chars_per_line=36,
        max_lines=2,
    ),
    "Minimal": CaptionStyle(
        preset="Minimal",
        font_name="Arial",
        font_size=48,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&H00000000",
        bold=False,
        outline_width=1.2,
        shadow_depth=0.0,
        margin_v=120,
        margin_l=100,
        margin_r=100,
        all_caps=False,
        max_chars_per_line=38,
        max_lines=2,
    ),
}

DEFAULT_PRESET: CaptionPreset = "Clean"


# ---------------------------------------------------------------------------
# Line breaking
# ---------------------------------------------------------------------------

def _wrap_text(text: str, max_chars: int, max_lines: int, all_caps: bool) -> str:
    """Wrap text into lines respecting max_chars and max_lines."""
    if all_caps:
        text = text.upper()
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return "\n".join(lines[:max_lines])


# ---------------------------------------------------------------------------
# SRT generation
# ---------------------------------------------------------------------------

def segments_to_srt(
    segments: list[dict],
    clip_start: float = 0.0,
    preset: CaptionPreset = DEFAULT_PRESET,
) -> str:
    """
    Generate SRT content from segments, offset by clip_start.
    Text is wrapped per the preset's line settings.
    """
    style = CAPTION_PRESETS.get(preset, CAPTION_PRESETS[DEFAULT_PRESET])
    lines: list[str] = []
    idx = 1
    for seg in segments:
        t0 = float(seg.get("start", 0)) - clip_start
        t1 = float(seg.get("end", 0)) - clip_start
        if t1 <= 0 or t0 < 0:
            continue
        t0 = max(0.0, t0)
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        wrapped = _wrap_text(text, style.max_chars_per_line, style.max_lines, style.all_caps)
        lines.append(str(idx))
        lines.append(f"{_srt_ts(t0)} --> {_srt_ts(t1)}")
        lines.append(wrapped)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def _srt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------

_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},{secondary},{outline},{back},{bold},0,0,0,100,100,0,0,1,{outw},{shadow},2,{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def segments_to_ass(
    segments: list[dict],
    clip_start: float = 0.0,
    preset: CaptionPreset = DEFAULT_PRESET,
    frame_h: int = 1920,
) -> str:
    """Generate ASS subtitle content from segments."""
    style = CAPTION_PRESETS.get(preset, CAPTION_PRESETS[DEFAULT_PRESET])

    header = _ASS_HEADER.format(
        font=style.font_name,
        size=style.font_size,
        primary=style.primary_color,
        secondary="&H00FFFFFF",
        outline=style.outline_color,
        back=style.back_color,
        bold="-1" if style.bold else "0",
        outw=style.outline_width,
        shadow=style.shadow_depth,
        ml=style.margin_l,
        mr=style.margin_r,
        mv=style.margin_v,
    )

    event_lines: list[str] = []
    for seg in segments:
        t0 = float(seg.get("start", 0)) - clip_start
        t1 = float(seg.get("end", 0)) - clip_start
        if t1 <= 0 or t0 < 0:
            continue
        t0 = max(0.0, t0)
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        wrapped = _wrap_text(text, style.max_chars_per_line, style.max_lines, style.all_caps)
        ass_text = wrapped.replace("\n", "\\N")
        event_lines.append(
            f"Dialogue: 0,{_ass_ts(t0)},{_ass_ts(t1)},Default,,0,0,0,,{ass_text}"
        )

    return header + "\n".join(event_lines)


def _ass_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# FFmpeg subtitle filter builder
# ---------------------------------------------------------------------------

def build_subtitles_filter(ass_path: Path) -> str:
    """Return an ffmpeg vf filter string to burn in an ASS subtitle file."""
    safe_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
    return f"subtitles='{safe_path}'"


def build_drawtext_fallback(
    style: CaptionStyle,
    text: str,
    frame_h: int = 1920,
) -> str:
    """
    Build a simple ffmpeg drawtext filter as fallback when ASS is unavailable.
    Only suitable for static text overlays (not time-synced captions).
    """
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    y_pos = frame_h - style.margin_v - style.font_size
    return (
        f"drawtext=text='{safe_text}'"
        f":fontfile='{style.font_name}'"
        f":fontsize={style.font_size}"
        f":fontcolor=white"
        f":x=(w-text_w)/2:y={y_pos}"
        f":borderw={int(style.outline_width)}"
        f":bordercolor=black"
    )


# ---------------------------------------------------------------------------
# Sidecar file writer
# ---------------------------------------------------------------------------

def write_sidecar_files(
    out_dir: Path,
    base_name: str,
    segments: list[dict],
    clip_start: float,
    preset: CaptionPreset = DEFAULT_PRESET,
) -> dict[str, Path]:
    """
    Write .srt and .ass sidecar files next to the exported clip.
    Returns dict with keys "srt" and "ass" pointing to written paths.
    """
    srt_path = out_dir / f"{base_name}.srt"
    ass_path = out_dir / f"{base_name}.ass"

    srt_content = segments_to_srt(segments, clip_start=clip_start, preset=preset)
    ass_content = segments_to_ass(segments, clip_start=clip_start, preset=preset)

    srt_path.write_text(srt_content, encoding="utf-8")
    ass_path.write_text(ass_content, encoding="utf-8")

    logger.info("Wrote sidecar files: %s, %s", srt_path.name, ass_path.name)
    return {"srt": srt_path, "ass": ass_path}
