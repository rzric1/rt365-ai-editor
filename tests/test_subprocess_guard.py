# -*- coding: utf-8 -*-
from __future__ import annotations

import sys


def test_orphan_ffmpeg_list_is_list():
    from clip_engine.subprocess_guard import find_orphan_ffmpeg_pids, terminate_orphan_ffmpeg

    assert isinstance(find_orphan_ffmpeg_pids(), list)
    assert isinstance(terminate_orphan_ffmpeg(), int)


def test_run_subprocess_echo():
    from clip_engine.subprocess_guard import run_subprocess

    if sys.platform == "win32":
        cmd = ["cmd", "/c", "echo", "ok"]
    else:
        cmd = ["echo", "ok"]
    r = run_subprocess(cmd, timeout=10.0, label="test_echo", text=True)
    assert r.returncode == 0


def test_tracked_registry_cleanup():
    from clip_engine.subprocess_guard import list_tracked_pids, terminate_all_tracked

    terminate_all_tracked()
    assert list_tracked_pids() == []
