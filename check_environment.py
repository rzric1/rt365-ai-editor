# -*- coding: utf-8 -*-
"""RT365 AI Clip Studio — CLI environment self-check (use .venv311)."""
from __future__ import annotations

import sys

from clip_engine.environment_check import validate_startup_environment, write_environment_check_log


def main() -> int:
    status = validate_startup_environment()
    path = write_environment_check_log(status)
    print(path.read_text(encoding="utf-8"))
    if not status.ok:
        print("\nFAIL — use launch_ai_clip_studio.ps1 with Python 3.11 .venv311")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
