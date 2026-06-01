# -*- coding: utf-8 -*-
"""
DaVinci Resolve Studio connection helpers (read-only + markers only).

SAFETY (v1):
  This module is intentionally narrow. It only:
    - Connects to Resolve
    - Reads the current project / timeline / frame rate
    - Adds timeline markers via Timeline.AddMarker

  It does NOT import or call Media Pool APIs, clip deletion, ripple delete,
  or any API that modifies source media. Keeping this file small reduces the
  chance of accidental destructive calls as the project grows.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, List, Optional, Union

from config import RESOLVE_SCRIPT_MODULE_PATHS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolveContext:
    """Everything we need to place markers on the active timeline."""

    resolve: Any
    project: Any
    timeline: Any
    project_name: str
    timeline_name: str
    timeline_fps: float
    timeline_start_frame: int
    # Smallest GetStart() among all timeline items on video + audio tracks, or None.
    earliest_clip_start_frame: Optional[int]
    # Transcript timestamp 0 seconds maps to this timeline frame (clip alignment).
    marker_alignment_frame: int


def _prepend_sys_path(module_dir: str) -> None:
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)


def _find_resolve_modules_path() -> Optional[str]:
    """Return the first existing Resolve 'Modules' directory, or None."""
    for p in RESOLVE_SCRIPT_MODULE_PATHS:
        if Path(p).is_dir():
            return p
    return None


def import_davinci_resolve_script() -> ModuleType:
    """
    Load Blackmagic's `DaVinciResolveScript` module.

    Resolve must be running with scripting enabled:
      DaVinci Resolve > Preferences > General > External scripting using > Local
    """
    modules_path = _find_resolve_modules_path()
    if not modules_path:
        searched = "\n  ".join(RESOLVE_SCRIPT_MODULE_PATHS)
        raise FileNotFoundError(
            "Could not find DaVinciResolveScript on this PC.\n"
            "Checked:\n  "
            + searched
            + "\nInstall DaVinci Resolve Studio, or update RESOLVE_SCRIPT_MODULE_PATHS in config.py."
        )

    _prepend_sys_path(modules_path)
    import DaVinciResolveScript as dvr_script  # type: ignore

    return dvr_script


def connect_resolve() -> Any:
    """Return the root Resolve application object."""
    dvr_script = import_davinci_resolve_script()
    resolve = dvr_script.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError(
            "scriptapp('Resolve') returned None.\n"
            "Open DaVinci Resolve Studio, load a project, and enable external scripting (Local)."
        )
    return resolve


def get_earliest_clip_start_frame(timeline: Any) -> Optional[int]:
    """
    Minimum timeline frame where any clip begins on video or audio tracks.

    Read-only: uses GetTrackCount + GetItemListInTrack + GetStart only.
    Returns None if there are no clips on those tracks.
    """
    earliest: Optional[int] = None

    for track_type in ("video", "audio"):
        try:
            track_count = timeline.GetTrackCount(track_type)
        except Exception:
            continue
        if track_count is None:
            continue
        try:
            n = int(track_count)
        except (TypeError, ValueError):
            continue
        for track_index in range(1, n + 1):
            try:
                items: List[Any] = timeline.GetItemListInTrack(track_type, track_index) or []
            except Exception:
                continue
            for item in items:
                try:
                    start = item.GetStart()
                except Exception:
                    continue
                if start is None:
                    continue
                try:
                    sf = int(start)
                except (TypeError, ValueError):
                    continue
                if earliest is None or sf < earliest:
                    earliest = sf

    return earliest


def _parse_timeline_frame_rate(setting_value: str) -> float:
    """
    Project setting 'timelineFrameRate' looks like '24', '23.976', '29.97 DF', etc.
    We only need the numeric rate for frame conversion; drop-frame is ignored here.
    """
    if not setting_value:
        raise ValueError("Empty timelineFrameRate from Resolve project settings.")
    cleaned = setting_value.strip()
    # Remove trailing "DF" drop-frame marker if present
    cleaned = re.sub(r"\s+DF\s*$", "", cleaned, flags=re.IGNORECASE)
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Could not parse timelineFrameRate: {setting_value!r}") from exc


def get_resolve_context(resolve: Any) -> ResolveContext:
    """
    Capture current project + timeline + timing information.
    """
    project_manager = resolve.GetProjectManager()
    if project_manager is None:
        raise RuntimeError("GetProjectManager() returned None.")

    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No current Resolve project. Open or create a project first.")

    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No current timeline. Open a timeline tab in Resolve.")

    project_name = project.GetName() or "(unnamed project)"
    timeline_name = timeline.GetName() or "(unnamed timeline)"

    rate_raw: Union[str, dict, None] = project.GetSetting("timelineFrameRate")
    if isinstance(rate_raw, dict):
        # Some API versions let you query all settings as a dict; prefer explicit key.
        rate_raw = rate_raw.get("timelineFrameRate", "")
    if rate_raw is None:
        rate_raw = ""
    if not isinstance(rate_raw, str):
        rate_raw = str(rate_raw)

    timeline_fps = _parse_timeline_frame_rate(rate_raw)
    timeline_start_frame = int(timeline.GetStartFrame())

    earliest_clip = get_earliest_clip_start_frame(timeline)
    if earliest_clip is not None:
        marker_alignment_frame = earliest_clip
    else:
        marker_alignment_frame = timeline_start_frame
        logger.warning(
            "No video/audio clips on the timeline — using timeline start frame "
            "as marker alignment base (transcript t=0)."
        )

    logger.info("Timeline start frame: %s", timeline_start_frame)
    if earliest_clip is not None:
        logger.info("Earliest video/audio clip start frame: %s", earliest_clip)
    else:
        logger.info("Earliest video/audio clip start frame: (none)")
    logger.info(
        "Marker alignment base frame (transcript t=0 lands here): %s",
        marker_alignment_frame,
    )

    return ResolveContext(
        resolve=resolve,
        project=project,
        timeline=timeline,
        project_name=project_name,
        timeline_name=timeline_name,
        timeline_fps=timeline_fps,
        timeline_start_frame=timeline_start_frame,
        earliest_clip_start_frame=earliest_clip,
        marker_alignment_frame=marker_alignment_frame,
    )


def seconds_to_timeline_frame(seconds: float, ctx: ResolveContext) -> int:
    """
    Convert transcript seconds to a Resolve timeline frame for AddMarker.

    Markers use: marker_alignment_frame + round(seconds * timeline_fps).

    marker_alignment_frame is the earliest clip start on video/audio tracks
    when clips exist; otherwise it matches timeline start frame.
    """
    if seconds < 0:
        seconds = 0.0
    elapsed_frames = int(round(seconds * ctx.timeline_fps))
    return ctx.marker_alignment_frame + elapsed_frames


def add_timeline_marker(
    ctx: ResolveContext,
    *,
    frame_id: int,
    color: str,
    name: str,
    note: str,
    duration: float,
    custom_data: str = "",
) -> bool:
    """
    Add a single marker on the current timeline.

    Signature matches Resolve: AddMarker(frameId, color, name, note, duration, customData)
    """
    add_marker: Callable[..., Any] = ctx.timeline.AddMarker
    if custom_data:
        return bool(add_marker(frame_id, color, name, note, duration, custom_data))
    # Some Resolve versions accept 5 args; others want empty string for customData.
    try:
        return bool(add_marker(frame_id, color, name, note, duration, ""))
    except TypeError:
        return bool(add_marker(frame_id, color, name, note, duration))
