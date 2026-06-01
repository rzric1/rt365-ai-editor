# -*- coding: utf-8 -*-
"""
Map AI marker types to Resolve marker colors and call Timeline.AddMarker only.

SAFETY:
  This module never deletes markers, never touches clips, and never opens the
  media pool. The only Resolve mutation is `timeline.AddMarker(...)`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from config import MARKER_DURATION
from openai_marker_engine import AiMarker
from resolve_client import ResolveContext, add_timeline_marker, seconds_to_timeline_frame

logger = logging.getLogger(__name__)

# Resolve marker colors (strings as documented by Blackmagic scripting API).
# See DaVinci Resolve scripting guide: timeline.AddMarker(frameId, "Blue", ...)
MARKER_TYPE_TO_COLOR: Dict[str, str] = {
    "HOT_TAKE": "Red",
    # Magenta is widely supported across Resolve versions (same family as Green/Blue in API docs).
    "STRONG_REACTION": "Magenta",
    "POSSIBLE_CUT": "Yellow",
    "SHORT_CLIP": "Cyan",
    "GOOD_QUOTE": "Green",
    "CHAPTER": "Blue",
    "AUDIO_DIP": "Purple",
}

_ILLEGAL_NAME_CHARS = re.compile(r"[\r\n\t]+")


def _sanitize_marker_text(value: str, *, max_len: int) -> str:
    cleaned = _ILLEGAL_NAME_CHARS.sub(" ", value).strip()
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1] + "…"
    return cleaned


def _custom_data_payload(marker: AiMarker) -> str:
    """Optional JSON blob stored in Resolve customData (not shown in main marker UI)."""
    payload = {
        "marker_type": marker.marker_type,
        "confidence": marker.confidence,
        "source": "RT365 AI Editor v1",
    }
    return json.dumps(payload, ensure_ascii=False)


def apply_markers_to_timeline(ctx: ResolveContext, markers: List[AiMarker]) -> int:
    """
    Place markers on the active timeline. Returns count successfully added.

    Markers with confidence outside [0, 1] are clamped for the note text only;
    Resolve still receives a valid marker.
    """
    added = 0
    for i, m in enumerate(markers):
        color = MARKER_TYPE_TO_COLOR.get(m.marker_type, "Blue")
        frame_id = seconds_to_timeline_frame(m.timestamp_seconds, ctx)
        if i == 0:
            logger.info(
                "First marker: timestamp_seconds=%.4f -> final frame=%s "
                "(alignment base=%s, fps=%.4f)",
                m.timestamp_seconds,
                frame_id,
                ctx.marker_alignment_frame,
                ctx.timeline_fps,
            )
        else:
            logger.debug(
                "Marker #%s: timestamp_seconds=%.4f -> final frame=%s",
                i + 1,
                m.timestamp_seconds,
                frame_id,
            )
        title = _sanitize_marker_text(m.title, max_len=120)
        note = _sanitize_marker_text(m.note, max_len=2000)
        custom = _custom_data_payload(m)

        ok = add_timeline_marker(
            ctx,
            frame_id=frame_id,
            color=color,
            name=f"[{m.marker_type}] {title}",
            note=note,
            duration=MARKER_DURATION,
            custom_data=custom,
        )
        if ok:
            added += 1
        else:
            logger.warning(
                "AddMarker returned False (frame=%s, type=%s, title=%s)",
                frame_id,
                m.marker_type,
                title,
            )
    return added


def markers_as_printable_dicts(markers: List[AiMarker]) -> List[Dict[str, Any]]:
    """Friendly structure for console output / logs."""
    return [
        {
            "timestamp_seconds": m.timestamp_seconds,
            "marker_type": m.marker_type,
            "title": m.title,
            "note": m.note,
            "confidence": m.confidence,
        }
        for m in markers
    ]
