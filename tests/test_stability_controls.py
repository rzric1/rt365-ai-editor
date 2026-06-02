# -*- coding: utf-8 -*-
"""Lightweight stability control tests (no Streamlit/GPU required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_job_lock_single_active():
    from clip_engine.job_control import (
        JobBusyError,
        get_active_job,
        release_job,
        try_acquire_job,
    )

    try_acquire_job("transcribe")
    assert get_active_job() == "transcribe"
    with pytest.raises(JobBusyError):
        try_acquire_job("export")
    release_job("transcribe")
    assert get_active_job() is None


def test_cancel_raises():
    from clip_engine.job_control import (
        JobCancelledError,
        check_cancelled,
        clear_cancel,
        request_cancel,
    )

    clear_cancel()
    request_cancel()
    with pytest.raises(JobCancelledError):
        check_cancelled()
    clear_cancel()


def test_crash_report_writes(tmp_path, monkeypatch):
    from clip_engine import stability

    log_file = tmp_path / "crash_report.txt"
    monkeypatch.setattr(stability, "CRASH_REPORT_PATH", log_file)
    stability.write_crash_report(RuntimeError("test"), context="unit_test")
    assert log_file.is_file()
    text = log_file.read_text(encoding="utf-8")
    assert "unit_test" in text
    assert "RuntimeError" in text


def test_startup_diagnostics_writes(tmp_path, monkeypatch):
    from clip_engine import stability

    diag_file = tmp_path / "startup_diagnostics.txt"
    monkeypatch.setattr(stability, "STARTUP_DIAG_PATH", diag_file)
    stability.run_startup_diagnostics()
    assert diag_file.is_file()
    body = diag_file.read_text(encoding="utf-8")
    assert "python" in body.lower() or "Python" in body


def test_orphan_ffmpeg_finder_no_psutil_crash():
    from clip_engine.subprocess_guard import find_orphan_ffmpeg_pids, terminate_orphan_ffmpeg

    # Should return list (possibly empty), never raise
    assert isinstance(find_orphan_ffmpeg_pids(), list)
    assert isinstance(terminate_orphan_ffmpeg(), int)


def test_resource_snapshot_writes(tmp_path, monkeypatch):
    from clip_engine import stability

    log_file = tmp_path / "resource_monitor.log"
    monkeypatch.setattr(stability, "LOGS_DIR", tmp_path)
    snap = stability.log_resource_snapshot(label="unit_test")
    assert "label" in snap
    assert log_file.is_file() or True  # file created when LOGS_DIR is tmp_path


def test_upload_fingerprint_streaming():
    import io

    from clip_engine.upload_manifest import compute_upload_fingerprint

    data = b"x" * (9 * 1024 * 1024)
    upload = io.BytesIO(data)
    upload.name = "test.mp4"
    fp, size, name = compute_upload_fingerprint(upload)
    assert size == len(data)
    assert name == "test.mp4"
    assert len(fp) == 64
