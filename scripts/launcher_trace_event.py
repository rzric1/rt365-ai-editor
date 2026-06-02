# -*- coding: utf-8 -*-
"""CLI helper for launch_ai_clip_studio.ps1 — one trace line per invocation."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clip_engine.startup_trace import trace

if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]).strip() or "event"
    trace(msg)
