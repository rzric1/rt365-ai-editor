# -*- coding: utf-8 -*-
from __future__ import annotations

import sys


def test_blocks_python_314_minor():
    from unittest.mock import patch

    from clip_engine.environment_check import _check_python_version

    with patch("sys.version_info", (3, 14, 0, "final", 0)):
        c = _check_python_version()
        assert not c.ok
        assert "3.14" in c.detail


def test_detects_missing_faster_whisper(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "faster_whisper":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from clip_engine.environment_check import validate_startup_environment

    status = validate_startup_environment(require_gpu_stack=True)
    assert any("faster-whisper" in e for e in status.errors)


def test_openai_api_key_loaded_from_dotenv(tmp_path, monkeypatch):
    from clip_engine import environment_check

    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-test-do-not-log-this-value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(environment_check, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment_check, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    check = environment_check._check_openai_api_key()
    assert check.ok
    assert check.detail == "present"
    log = environment_check.write_environment_check_log(
        environment_check.EnvironmentStatus(ok=True, checks=[check])
    )
    text = log.read_text(encoding="utf-8")
    assert "sk-test" not in text
    assert "[OK] OPENAI_API_KEY: present" in text


def test_openai_model_unrecognized_warns(tmp_path, monkeypatch):
    from clip_engine import environment_check

    (tmp_path / ".env").write_text("OPENAI_MODEL=gpt-5-mini\n", encoding="utf-8")
    monkeypatch.setattr(environment_check, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment_check, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    check = environment_check._check_openai_model()
    assert not check.ok
    assert "gpt-5-mini" in check.detail
    assert "not a recognized model name" in check.detail


def test_openai_api_key_missing_without_dotenv(tmp_path, monkeypatch):
    from clip_engine import environment_check

    monkeypatch.setattr(environment_check, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment_check, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    check = environment_check._check_openai_api_key()
    assert not check.ok
    assert check.detail == "missing — cloud Whisper/analyze need .env"


def test_environment_check_log_writes(tmp_path, monkeypatch):
    from clip_engine import environment_check

    monkeypatch.setattr(environment_check, "LOGS_DIR", tmp_path)
    path = environment_check.write_environment_check_log()
    assert path.is_file()
    assert "environment check" in path.read_text(encoding="utf-8").lower()


def test_check_environment_cli_runs():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "check_environment.py"],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=60,
    )
    assert r.returncode in (0, 1)
    assert "environment check" in (r.stdout + r.stderr).lower()
