# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path


def test_stale_lock_removed(tmp_path, monkeypatch):
    from clip_engine import app_lock

    monkeypatch.setattr(app_lock, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(app_lock, "LOCK_PATH", tmp_path / "rt365_app.lock")
    (tmp_path / "rt365_app.lock").write_text(
        json.dumps({"pid": 999999999}),
        encoding="utf-8",
    )
    assert app_lock.remove_stale_lock() is True
    assert not app_lock.LOCK_PATH.is_file()


def test_acquire_and_release(tmp_path, monkeypatch):
    from clip_engine import app_lock

    monkeypatch.setattr(app_lock, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(app_lock, "LOCK_PATH", tmp_path / "rt365_app.lock")
    app_lock._lock_held = False
    ok, _ = app_lock.acquire_app_lock()
    assert ok
    assert app_lock.LOCK_PATH.is_file()
    data = json.loads(app_lock.LOCK_PATH.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    app_lock.release_app_lock()
    assert not app_lock.LOCK_PATH.is_file()


def test_preflight_message_when_lock_held(tmp_path, monkeypatch):
    from clip_engine import app_lock

    monkeypatch.setattr(app_lock, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(app_lock, "LOCK_PATH", tmp_path / "rt365_app.lock")
    (tmp_path / "rt365_app.lock").write_text(
        json.dumps({"pid": os.getpid()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_lock, "_is_clip_studio_process", lambda _pid: True)
    monkeypatch.setattr(app_lock, "_pid_alive", lambda _pid: True)
    ok, msg = app_lock.preflight_single_instance()
    assert not ok
    assert "already running" in msg.lower()
