# -*- coding: utf-8 -*-
"""RT365 AI Clip Studio — environment self-check script."""
from __future__ import annotations

import os
import shutil
import sys

_OK = "  [OK]"
_WARN = "  [WARN]"
_ERR = "  [ERR]"

issues: list[str] = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    tag = _OK if ok else (_WARN if warn_only else _ERR)
    line = f"{tag} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok and not warn_only:
        issues.append(label)


def main() -> int:
    print("=" * 52)
    print(" RT365 AI Clip Studio — Environment Check")
    print("=" * 52)

    # Python version
    major, minor = sys.version_info[:2]
    check("Python version", major == 3 and minor >= 10, f"{major}.{minor}", warn_only=minor < 11)

    # ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    check("ffmpeg on PATH", bool(ffmpeg), ffmpeg or "not found", warn_only=True)

    # OpenAI key
    key = os.environ.get("OPENAI_API_KEY", "")
    check("OPENAI_API_KEY set", bool(key), "found" if key else "not set in environment", warn_only=True)

    # Core imports
    for mod in ("streamlit", "dotenv", "openai"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except ImportError as exc:
            check(f"import {mod}", False, str(exc))

    # Optional imports
    for mod in ("faster_whisper", "torch", "cv2", "pysubs2"):
        try:
            __import__(mod)
            check(f"import {mod} (optional)", True)
        except ImportError:
            check(f"import {mod} (optional)", True, "not installed — optional", warn_only=True)

    print("=" * 52)
    if issues:
        print(f"  {len(issues)} required item(s) missing — see [ERR] lines above.")
        return 1
    print("  All required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
