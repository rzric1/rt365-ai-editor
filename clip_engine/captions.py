# -*- coding: utf-8 -*-
"""
clip_engine/captions.py
Modern burned-in caption styles + SRT/ASS sidecar file generation.
Presets: Clean | Bold Viral | Podcast | Minimal | Viral | Podcast Pro | Documentary | Gaming | Cinematic
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("clip_engine.captions")

CaptionPreset = Literal[
    "Clean", "Bold Viral", "Podcast", "Minimal",
    "Viral", "Podcast Pro", "Documentary", "Gaming", "Cinematic",
]

# Legacy presets preserved; advanced presets added for per-word/karaoke styles.

# ---------------------------------------------------------------------------
# Caption style definitions
# ---------------------------------------------------------------------------

@dataclass
class CaptionStyle:
    preset: str
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
    highlight_color: str = "&H0000FFFF"  # karaoke highlight (yellow)
    karaoke: bool = False


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
    "Viral": CaptionStyle(
        preset="Viral",
        font_name="Impact",
        font_size=78,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&HB0000000",
        bold=True,
        outline_width=4.0,
        shadow_depth=2.5,
        margin_v=200,
        margin_l=50,
        margin_r=50,
        all_caps=True,
        max_chars_per_line=24,
        max_lines=2,
        highlight_color="&H0000FFFF",
        karaoke=True,
    ),
    "Podcast Pro": CaptionStyle(
        preset="Podcast Pro",
        font_name="Segoe UI",
        font_size=54,
        primary_color="&H00FFFFFF",
        outline_color="&H00181818",
        back_color="&H90000000",
        bold=False,
        outline_width=2.0,
        shadow_depth=1.0,
        margin_v=150,
        margin_l=90,
        margin_r=90,
        all_caps=False,
        max_chars_per_line=34,
        max_lines=2,
        highlight_color="&H0000D7FF",
        karaoke=True,
    ),
    "Documentary": CaptionStyle(
        preset="Documentary",
        font_name="Georgia",
        font_size=50,
        primary_color="&H00F0F0F0",
        outline_color="&H00202020",
        back_color="&H70000000",
        bold=False,
        outline_width=1.5,
        shadow_depth=1.5,
        margin_v=130,
        margin_l=110,
        margin_r=110,
        all_caps=False,
        max_chars_per_line=40,
        max_lines=2,
        karaoke=False,
    ),
    "Gaming": CaptionStyle(
        preset="Gaming",
        font_name="Arial Black",
        font_size=64,
        primary_color="&H0000FF00",
        outline_color="&H00000000",
        back_color="&HC0000000",
        bold=True,
        outline_width=3.0,
        shadow_depth=2.0,
        margin_v=170,
        margin_l=70,
        margin_r=70,
        all_caps=True,
        max_chars_per_line=28,
        max_lines=2,
        highlight_color="&H00FFFF00",
        karaoke=True,
    ),
    "Cinematic": CaptionStyle(
        preset="Cinematic",
        font_name="Times New Roman",
        font_size=46,
        primary_color="&H00E8E8E8",
        outline_color="&H00080808",
        back_color="&H00000000",
        bold=False,
        outline_width=1.0,
        shadow_depth=2.0,
        margin_v=100,
        margin_l=120,
        margin_r=120,
        all_caps=False,
        max_chars_per_line=42,
        max_lines=2,
        karaoke=False,
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
    *,
    advanced: bool = False,
) -> str:
    """Generate ASS subtitle content from segments."""
    style = CAPTION_PRESETS.get(preset, CAPTION_PRESETS[DEFAULT_PRESET])
    if advanced and style.karaoke:
        try:
            return segments_to_ass_advanced(
                segments, clip_start=clip_start, preset=preset, frame_h=frame_h,
            )
        except Exception as e:
            logger.warning("Advanced ASS failed (%s) — falling back to phrase-level", e)

    header = _ass_header_for_style(style)

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


def _ass_header_for_style(style: CaptionStyle) -> str:
    return _ASS_HEADER.format(
        font=style.font_name,
        size=style.font_size,
        primary=style.primary_color,
        secondary=style.highlight_color,
        outline=style.outline_color,
        back=style.back_color,
        bold="-1" if style.bold else "0",
        outw=style.outline_width,
        shadow=style.shadow_depth,
        ml=style.margin_l,
        mr=style.margin_r,
        mv=style.margin_v,
    )


def _extract_word_timings(segments: list[dict]) -> list[dict]:
    """Collect word-level timings from segments or nested source_segments."""
    words: list[dict] = []
    for seg in segments:
        if seg.get("words"):
            words.extend(seg["words"])
            continue
        for src in seg.get("source_segments") or []:
            if src.get("words"):
                words.extend(src["words"])
    normalized: list[dict] = []
    for w in words:
        start = w.get("start")
        end = w.get("end")
        text = str(w.get("word", w.get("text", ""))).strip()
        if text and start is not None and end is not None:
            normalized.append({"start": float(start), "end": float(end), "word": text})
    return normalized


def _estimate_word_timings(seg: dict, clip_start: float) -> list[dict]:
    """Evenly distribute words across segment duration when no word timing exists."""
    text = str(seg.get("text", "")).strip()
    words = text.split()
    if not words:
        return []
    t0 = max(0.0, float(seg.get("start", 0)) - clip_start)
    t1 = max(t0 + 0.1, float(seg.get("end", 0)) - clip_start)
    dur = t1 - t0
    step = dur / len(words)
    return [
        {"start": t0 + i * step, "end": t0 + (i + 1) * step, "word": w}
        for i, w in enumerate(words)
    ]


def _build_karaoke_text(words: list[dict], clip_start: float, style: CaptionStyle) -> str:
    """Build ASS karaoke text with \\k tags (centiseconds per word)."""
    parts: list[str] = []
    for w in words:
        t0 = max(0.0, float(w["start"]) - clip_start)
        t1 = max(t0 + 0.05, float(w["end"]) - clip_start)
        cs = max(1, int(round((t1 - t0) * 100)))
        word = str(w["word"]).strip()
        if style.all_caps:
            word = word.upper()
        parts.append(f"{{\\k{cs}}}{word} ")
    return "".join(parts).strip()


def segments_to_ass_advanced(
    segments: list[dict],
    clip_start: float = 0.0,
    preset: CaptionPreset = DEFAULT_PRESET,
    frame_h: int = 1920,
) -> str:
    """
    Generate ASS with per-word or phrase-level karaoke highlighting.
    Uses pysubs2 when installed; falls back to built-in karaoke tags.
    """
    style = CAPTION_PRESETS.get(preset, CAPTION_PRESETS[DEFAULT_PRESET])

    try:
        import pysubs2  # type: ignore

        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = "1080"
        subs.info["PlayResY"] = str(frame_h)
        subs.styles["Default"] = pysubs2.SSAStyle(
            fontname=style.font_name,
            fontsize=style.font_size,
            primarycolor=_ass_color_to_pysubs(style.primary_color),
            secondarycolor=_ass_color_to_pysubs(style.highlight_color),
            outlinecolor=_ass_color_to_pysubs(style.outline_color),
            backcolor=_ass_color_to_pysubs(style.back_color),
            bold=style.bold,
            outline=int(style.outline_width),
            shadow=int(style.shadow_depth),
            marginl=style.margin_l,
            marginr=style.margin_r,
            marginv=style.margin_v,
            alignment=2,
        )
        for seg in segments:
            t0 = float(seg.get("start", 0)) - clip_start
            t1 = float(seg.get("end", 0)) - clip_start
            if t1 <= 0 or t0 < 0:
                continue
            words = _extract_word_timings([seg]) or _estimate_word_timings(seg, clip_start)
            if words and style.karaoke:
                text = _build_karaoke_text(words, clip_start, style)
            else:
                text = _wrap_text(
                    str(seg.get("text", "")).strip(),
                    style.max_chars_per_line, style.max_lines, style.all_caps,
                ).replace("\n", "\\N")
            if not text:
                continue
            subs.append(pysubs2.SSAEvent(
                start=max(0, int(t0 * 1000)),
                end=max(1, int(t1 * 1000)),
                text=text,
            ))
        return subs.to_string("ass")
    except ImportError:
        logger.debug("pysubs2 not installed — using built-in karaoke ASS")
    except Exception as e:
        logger.warning("pysubs2 ASS generation failed: %s", e)

    # Built-in fallback
    header = _ass_header_for_style(style)
    event_lines: list[str] = []
    for seg in segments:
        t0 = float(seg.get("start", 0)) - clip_start
        t1 = float(seg.get("end", 0)) - clip_start
        if t1 <= 0 or t0 < 0:
            continue
        t0 = max(0.0, t0)
        words = _extract_word_timings([seg]) or _estimate_word_timings(seg, clip_start)
        if words and style.karaoke:
            ass_text = _build_karaoke_text(words, clip_start, style)
        else:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            ass_text = _wrap_text(
                text, style.max_chars_per_line, style.max_lines, style.all_caps,
            ).replace("\n", "\\N")
        event_lines.append(
            f"Dialogue: 0,{_ass_ts(t0)},{_ass_ts(t1)},Default,,0,0,0,,{ass_text}"
        )
    return header + "\n".join(event_lines)


def _ass_color_to_pysubs(ass_hex: str) -> str:
    """Convert &H00BBGGRR to pysubs2 &HAABBGGRR format."""
    h = ass_hex.replace("&H", "").replace("&h", "")
    if len(h) == 8:
        return f"&H{h}"
    return ass_hex


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
    *,
    advanced: bool = False,
) -> dict[str, Path]:
    """
    Write .srt and .ass sidecar files next to the exported clip.
    Returns dict with keys "srt" and "ass" pointing to written paths.
    """
    srt_path = out_dir / f"{base_name}.srt"
    ass_path = out_dir / f"{base_name}.ass"

    srt_content = segments_to_srt(segments, clip_start=clip_start, preset=preset)
    ass_content = segments_to_ass(
        segments, clip_start=clip_start, preset=preset, advanced=advanced,
    )

    srt_path.write_text(srt_content, encoding="utf-8")
    ass_path.write_text(ass_content, encoding="utf-8")

    logger.info("Wrote sidecar files: %s, %s", srt_path.name, ass_path.name)
    return {"srt": srt_path, "ass": ass_path}
