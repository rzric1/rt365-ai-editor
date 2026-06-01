# -*- coding: utf-8 -*-
"""
Interactive CLI menu for RT365 AI Editor (Windows-friendly).

Uses lazy imports of ``main`` inside ``run_interactive_menu`` so command helpers
stay in ``main.py`` without circular import issues at module load time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config import (
    DEFAULT_SRT_PATH,
    ENV_OPENAI_API_KEY,
    LOGS_DIR,
    PROJECT_ROOT,
    TRANSCRIPTS_DIR,
    ensure_directories,
)


def find_latest_log() -> Optional[Path]:
    """Return the most recently modified ``*.log`` under ``logs/``, or ``None``."""
    if not LOGS_DIR.is_dir():
        return None
    logs = [p for p in LOGS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".log"]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def open_folder(path: Path | str) -> None:
    """
    Open a folder in the system file manager.

    On Windows uses ``os.startfile`` as requested.
    """
    p = Path(path).resolve()
    if not p.exists():
        print(f"\nThat path does not exist yet:\n  {p}\n")
        return
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        import subprocess

        subprocess.run(["open", str(p)], check=False)
    else:
        import subprocess

        subprocess.run(["xdg-open", str(p)], check=False)


def prompt_transcript_path() -> Path:
    """
    Ask for a transcript path; Enter uses ``transcripts\\input.srt`` (project-relative).

    Resolution matches non-interactive CLI (cwd first, then project root).
    """
    import main as rt365_main

    try:
        default_rel = DEFAULT_SRT_PATH.relative_to(PROJECT_ROOT)
    except ValueError:
        default_rel = DEFAULT_SRT_PATH.name
    default_hint = str(default_rel).replace("/", os.sep)

    print()
    raw = input(f"Transcript path [default: {default_hint}]: ").strip()
    if not raw:
        raw = str(default_rel).replace("/", os.sep)

    return rt365_main.resolve_transcript_path(raw)


def _print_resolve_friendly_error(exc: BaseException) -> None:
    msg = str(exc).lower()
    print(f"\nResolve issue: {exc}\n")
    if "no current resolve project" in msg or "no current timeline" in msg:
        print(
            "Tip: In Resolve, open a project and double-click a timeline so it is the active edit.\n"
        )
    if "davinciresolvescript" in msg or "could not find" in msg:
        print(
            "Tip: Install DaVinci Resolve Studio and check that the scripting Modules path "
            "in config.py matches your install.\n"
        )
    elif "none" in msg and "scriptapp" in msg:
        print(
            "Tip: Open DaVinci Resolve Studio, open a project, and enable:\n"
            "  Preferences → General → External scripting using → Local\n"
        )
    elif "no current" in msg or "timeline" in msg:
        print(
            "Tip: In Resolve, open a project and click into a timeline tab so a timeline is active.\n"
        )


def _print_transcript_missing(path: Path) -> None:
    print(
        f"\nTranscript file not found:\n  {path.resolve()}\n\n"
        "Create the file or fix the path. Default location:\n"
        f"  {DEFAULT_SRT_PATH}\n"
    )


def run_interactive_menu() -> int:
    """
    Simple REPL menu. Version 1 safety: only actions that add markers or read data.

    Imports ``main`` lazily to reuse ``cmd_*`` and ``resolve_transcript_path``.
    """
    import main as rt365_main

    ensure_directories()

    print()
    print("RT365 AI Editor - interactive mode")
    print("Version 1 only adds timeline markers (no clip edits, ripple delete, or media pool changes).")
    print(f"Project folder: {PROJECT_ROOT}")

    while True:
        print()
        print("RT365 AI Editor")
        print("  1. Test Resolve connection / add test marker")
        print("  2. Debug transcript parsing")
        print("  3. Dry-run AI marker analysis")
        print("  4. Apply AI markers to Resolve")
        print("  5. Show latest log file")
        print("  6. Open transcripts folder")
        print("  7. Open logs folder")
        print("  8. Exit")
        print()
        choice = input("Choose an option (1-8): ").strip()

        if choice == "8":
            print("\nGoodbye.\n")
            return 0

        if choice == "1":
            print("\nAdding a test marker at 10 transcript seconds (see logs for alignment)…")
            try:
                code = rt365_main.cmd_test_marker()
                if code != 0:
                    print(f"\nFinished with exit code {code}. See messages above.\n")
            except FileNotFoundError as exc:
                print(f"\n{exc}\n")
            except RuntimeError as exc:
                _print_resolve_friendly_error(exc)
            input("\nPress Enter to return to the menu…")
            continue

        if choice == "2":
            try:
                tpath = prompt_transcript_path()
            except FileNotFoundError as exc:
                print(f"\n{exc}\n")
                input("\nPress Enter to return to the menu…")
                continue
            if not tpath.is_file():
                _print_transcript_missing(tpath)
                input("\nPress Enter to return to the menu…")
                continue
            try:
                rt365_main.cmd_debug_transcript(tpath)
            except Exception as exc:  # noqa: BLE001 — user-facing menu
                print(f"\nCould not parse transcript: {exc}\n")
            input("\nPress Enter to return to the menu…")
            continue

        if choice == "3":
            load_dotenv(PROJECT_ROOT / ".env")
            key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
            if not key:
                print(
                    "\nOpenAI API key is missing.\n"
                    "Add OPENAI_API_KEY to your .env file (copy from .env.example).\n"
                )
                input("\nPress Enter to return to the menu…")
                continue
            try:
                tpath = prompt_transcript_path()
            except FileNotFoundError as exc:
                print(f"\n{exc}\n")
                input("\nPress Enter to return to the menu…")
                continue
            if not tpath.is_file():
                _print_transcript_missing(tpath)
                input("\nPress Enter to return to the menu…")
                continue
            try:
                code = rt365_main.cmd_analyze_and_maybe_apply(transcript_path=tpath, dry_run=True)
                if code != 0:
                    print(f"\nDry-run finished with exit code {code}.\n")
            except Exception as exc:  # noqa: BLE001
                print(f"\nError during analysis: {exc}\n")
            input("\nPress Enter to return to the menu…")
            continue

        if choice == "4":
            load_dotenv(PROJECT_ROOT / ".env")
            key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
            if not key:
                print(
                    "\nOpenAI API key is missing.\n"
                    "Add OPENAI_API_KEY to your .env file (copy from .env.example).\n"
                )
                input("\nPress Enter to return to the menu…")
                continue
            try:
                tpath = prompt_transcript_path()
            except FileNotFoundError as exc:
                print(f"\n{exc}\n")
                input("\nPress Enter to return to the menu…")
                continue
            if not tpath.is_file():
                _print_transcript_missing(tpath)
                input("\nPress Enter to return to the menu…")
                continue

            print()
            confirm = input(
                "This will add markers to the current Resolve timeline. "
                "It will not cut or delete anything. Continue? y/n: "
            ).strip().lower()
            if confirm not in ("y", "yes"):
                print("\nCancelled.\n")
                input("Press Enter to return to the menu…")
                continue

            try:
                code = rt365_main.cmd_analyze_and_maybe_apply(transcript_path=tpath, dry_run=False)
                if code != 0:
                    print(f"\nApply finished with exit code {code}. Check logs for details.\n")
            except FileNotFoundError as exc:
                print(f"\n{exc}\n")
            except RuntimeError as exc:
                _print_resolve_friendly_error(exc)
            except Exception as exc:  # noqa: BLE001
                print(f"\nError: {exc}\n")
            input("\nPress Enter to return to the menu…")
            continue

        if choice == "5":
            latest = find_latest_log()
            if latest is None:
                print(f"\nNo log files found in:\n  {LOGS_DIR.resolve()}\n")
            else:
                print(f"\nLatest log file:\n  {latest.resolve()}\n")
                try:
                    text = latest.read_text(encoding="utf-8", errors="replace")
                    lines = text.splitlines()
                    tail = lines[-40:] if len(lines) > 40 else lines
                    print("--- Last lines (up to 40) ---\n")
                    print("\n".join(tail))
                except OSError as exc:
                    print(f"Could not read log: {exc}\n")
            input("\nPress Enter to return to the menu…")
            continue

        if choice == "6":
            ensure_directories()
            print(f"\nOpening: {TRANSCRIPTS_DIR.resolve()}\n")
            open_folder(TRANSCRIPTS_DIR)
            input("\nPress Enter to return to the menu…")
            continue

        if choice == "7":
            ensure_directories()
            print(f"\nOpening: {LOGS_DIR.resolve()}\n")
            open_folder(LOGS_DIR)
            input("\nPress Enter to return to the menu…")
            continue

        print("\nPlease enter a number from 1 to 8.\n")
        input("Press Enter to continue…")
