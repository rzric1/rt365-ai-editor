"""
RT365 AI Editor — safe AI-assisted markers for DaVinci Resolve Studio (v1).

This version ONLY adds timeline markers. It never cuts clips, never ripple-deletes,
and never touches the media pool.

Typical flow (Windows, from this project folder):
  1) Confirm Resolve scripting:
       DaVinci Resolve > Preferences > General > External scripting using > Local
  2) py -m venv .venv
  3) .venv\\Scripts\\activate
  4) py -m pip install -r requirements.txt
  5) copy .env.example .env   (then add OPENAI_API_KEY)
  6) py main.py --test-marker
  7) py main.py --dry-run transcripts\\input.srt
  8) py main.py transcripts\\input.srt
  Or: py main.py --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import config
from config import (
    DEFAULT_JSON_PATH,
    DEFAULT_SRT_PATH,
    ENV_OPENAI_API_KEY,
    LOGS_DIR,
    PROJECT_ROOT,
)
from marker_writer import apply_markers_to_timeline, markers_as_printable_dicts
from openai_marker_engine import analyze_transcript
from resolve_client import connect_resolve, get_resolve_context, seconds_to_timeline_frame
from transcript_loader import load_transcript


def setup_logging() -> Path:
    """
    Log to logs/rt365_YYYYMMDD_HHMMSS.log and mirror important lines to the console.
    """
    config.ensure_directories()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"rt365_{stamp}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    logging.info("Log file: %s", log_path)
    return log_path


def resolve_transcript_path(user_path: Optional[str]) -> Path:
    """
    Pick a transcript file:
      - explicit CLI path (relative to CWD, else project root)
      - else transcripts/input.srt if present
      - else transcripts/input.json
    """
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            cwd_try = Path.cwd() / p
            if cwd_try.exists():
                return cwd_try.resolve()
            proj_try = PROJECT_ROOT / p
            return proj_try.resolve()
        return p.resolve()

    if DEFAULT_SRT_PATH.is_file():
        return DEFAULT_SRT_PATH.resolve()
    if DEFAULT_JSON_PATH.is_file():
        return DEFAULT_JSON_PATH.resolve()

    raise FileNotFoundError(
        "No transcript path was provided and defaults were not found.\n"
        f"  Expected: {DEFAULT_SRT_PATH}\n"
        f"  Or:      {DEFAULT_JSON_PATH}\n"
        "Pass a file explicitly, e.g.  py main.py transcripts\\input.srt"
    )


def cmd_test_marker() -> int:
    """Add a single marker at ~10 seconds to verify Resolve scripting works."""
    logging.info("TEST MODE: adding one marker at 10.0 seconds on the current timeline.")
    resolve = connect_resolve()
    ctx = get_resolve_context(resolve)
    logging.info("Project: %s | Timeline: %s", ctx.project_name, ctx.timeline_name)
    logging.info(
        "Timeline FPS: %s | timeline start frame: %s | marker alignment base: %s",
        ctx.timeline_fps,
        ctx.timeline_start_frame,
        ctx.marker_alignment_frame,
    )

    from resolve_client import add_timeline_marker

    frame_id = seconds_to_timeline_frame(10.0, ctx)
    ok = add_timeline_marker(
        ctx,
        frame_id=frame_id,
        color="Blue",
        name="RT365 test marker",
        note="If you see this, Resolve scripting + marker placement works.",
        duration=config.MARKER_DURATION,
        custom_data="rt365_test_marker",
    )
    if not ok:
        logging.error("AddMarker returned False — check that the timeline is writable.")
        return 2
    logging.info(
        "Success: marker at frame %s (10 transcript seconds from alignment base; "
        "see logs above for timeline vs clip start).",
        frame_id,
    )
    return 0


def cmd_debug_transcript(transcript_path: Path) -> int:
    """Parse transcript and print segments — handy for bracket / SRT / JSON checks."""
    logging.info("DEBUG transcript: %s", transcript_path)
    doc = load_transcript(transcript_path)
    logging.info("Total segments after parse: %s", len(doc.segments))
    preview = min(15, len(doc.segments))
    rows = []
    for seg in doc.segments[:preview]:
        rows.append(
            {
                "index": seg.index,
                "start_seconds": round(seg.start_seconds, 4),
                "end_seconds": round(seg.end_seconds, 4),
                "text_preview": seg.text[:200] + ("…" if len(seg.text) > 200 else ""),
            }
        )
    print(json.dumps({"segment_preview": rows, "total": len(doc.segments)}, indent=2, ensure_ascii=False))
    return 0


def cmd_analyze_and_maybe_apply(*, transcript_path: Path, dry_run: bool) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()

    logging.info("Transcript: %s", transcript_path)
    doc = load_transcript(transcript_path)
    logging.info("Loaded %s segments.", len(doc.segments))

    if not api_key:
        logging.error("OPENAI_API_KEY is missing. Copy .env.example to .env and set your key.")
        return 3

    try:
        markers = analyze_transcript(doc, api_key=api_key)
    except Exception:
        # Log full traceback to logs/ — beginners can paste this when asking for help.
        logging.exception("OpenAI Responses API call failed (check model name, billing, and network).")
        return 5

    logging.info("Model returned %s marker(s) after merge/dedupe.", len(markers))

    printable = markers_as_printable_dicts(markers)
    print(json.dumps({"markers": printable}, indent=2, ensure_ascii=False))

    if dry_run:
        logging.info("Dry run: NOT connecting to Resolve; no markers written.")
        return 0

    resolve = connect_resolve()
    ctx = get_resolve_context(resolve)
    logging.info("Applying markers to: %s / %s", ctx.project_name, ctx.timeline_name)

    added = apply_markers_to_timeline(ctx, markers)
    logging.info("Added %s / %s markers on the timeline.", added, len(markers))
    return 0 if added == len(markers) else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RT365 AI Editor — add Resolve timeline markers from a transcript (OpenAI).",
    )
    p.add_argument(
        "--test-marker",
        action="store_true",
        help="Add a single test marker at 10s on the current timeline (no OpenAI).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze transcript and print JSON markers; do not modify Resolve.",
    )
    p.add_argument(
        "--debug-transcript",
        action="store_true",
        help="Load transcript, print segment summary, and exit (no OpenAI / Resolve).",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive menu instead of running a single command.",
    )
    p.add_argument(
        "transcript",
        nargs="?",
        default=None,
        help="Path to .srt or .json (default: transcripts/input.srt or input.json).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")

    if args.interactive:
        from interactive_menu import run_interactive_menu

        return run_interactive_menu()

    logging.info("RT365 AI Editor | project root: %s", PROJECT_ROOT)
    logging.info("Safety: v1 only adds markers; no clip/media pool/timeline destructive APIs.")

    if args.test_marker:
        return cmd_test_marker()

    try:
        tpath = resolve_transcript_path(args.transcript)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 4

    if args.debug_transcript:
        return cmd_debug_transcript(tpath)

    return cmd_analyze_and_maybe_apply(transcript_path=tpath, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
