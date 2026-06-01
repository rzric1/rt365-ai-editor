# -*- coding: utf-8 -*-
"""
Shared helpers for the RT365 AI Edit Companion (Streamlit) and other UIs.

Keeps path resolution, Resolve checks, and JSON logging in one place.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import DEFAULT_SRT_PATH, LOGS_DIR, PROJECT_ROOT, ensure_directories
from marker_writer import apply_markers_to_timeline
from openai_marker_engine import AiMarker
from resolve_client import connect_resolve, get_resolve_context

logger = logging.getLogger(__name__)


def resolve_transcript_file(path_str: str) -> Path:
    """
    Resolve a transcript path the same way as the CLI (cwd first, then project root).

    Raises FileNotFoundError if the path does not exist.
    """
    raw = (path_str or "").strip()
    if not raw:
        raw = str(DEFAULT_SRT_PATH.relative_to(PROJECT_ROOT)).replace("/", os.sep)

    p = Path(raw)
    if not p.is_absolute():
        cwd_try = Path.cwd() / p
        if cwd_try.exists():
            resolved = cwd_try.resolve()
        else:
            resolved = (PROJECT_ROOT / p).resolve()
    else:
        resolved = p.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Transcript not found:\n  {resolved}")
    return resolved


def format_timecode(seconds: float) -> str:
    """Human-readable H:MM:SS from seconds (for cards, not frame math)."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def save_companion_json(payload: Dict[str, Any]) -> Path:
    """
    Persist one companion turn to logs/ as JSON (audit + debugging).

    Filename includes UTC timestamp and intent slug.
    """
    ensure_directories()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    intent = str(payload.get("intent", "unknown"))
    safe = re.sub(r"[^\w\-]+", "_", intent)[:48].strip("_") or "unknown"
    path = LOGS_DIR / f"companion_{ts}_{safe}.json"
    out = dict(payload)
    out["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Companion result saved: %s", path)
    return path


def open_folder(path: Path | str) -> None:
    """Open a folder in the OS file manager (Windows: os.startfile)."""
    p = Path(path).resolve()
    ensure_directories()
    p.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        import subprocess

        subprocess.run(["open", str(p)], check=False)
    else:
        import subprocess

        subprocess.run(["xdg-open", str(p)], check=False)


def try_resolve_connection() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Try to connect to Resolve and read the current timeline.

    Returns (ok, message, details). Only read/connect APIs — no edits.
    """
    try:
        resolve = connect_resolve()
        ctx = get_resolve_context(resolve)
        details: Dict[str, Any] = {
            "project": ctx.project_name,
            "timeline": ctx.timeline_name,
            "timeline_fps": ctx.timeline_fps,
            "timeline_start_frame": ctx.timeline_start_frame,
            "marker_alignment_frame": ctx.marker_alignment_frame,
        }
        msg = (
            f"Connected to **{ctx.project_name}** — timeline **{ctx.timeline_name}** "
            f"(alignment frame {ctx.marker_alignment_frame})."
        )
        return True, msg, details
    except FileNotFoundError as exc:
        return False, f"Resolve scripting module not found:\n{exc}", {}
    except RuntimeError as exc:
        return False, str(exc), {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Resolve check failed")
        return False, f"Unexpected error: {exc}", {}


def apply_markers_to_resolve_safe(markers: List[AiMarker]) -> Tuple[bool, str, int]:
    """
    Add markers to the current Resolve timeline (AddMarker only).

    Returns (success, user_message, count_added).
    """
    if not markers:
        return False, "No markers to add.", 0
    try:
        resolve = connect_resolve()
        ctx = get_resolve_context(resolve)
        added = apply_markers_to_timeline(ctx, markers)
        if added == len(markers):
            return True, f"Added **{added}** marker(s) to **{ctx.timeline_name}**.", added
        return (
            True,
            f"Added **{added}** of **{len(markers)}** marker(s). Some AddMarker calls failed — see logs.",
            added,
        )
    except FileNotFoundError as exc:
        return False, f"Resolve scripting not available:\n{exc}", 0
    except RuntimeError as exc:
        return (
            False,
            f"Could not talk to Resolve. Is it open with a project and timeline?\n\n{exc}",
            0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("apply_markers_to_resolve_safe")
        return False, str(exc), 0
