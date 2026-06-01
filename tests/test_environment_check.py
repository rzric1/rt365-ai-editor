# -*- coding: utf-8 -*-
import subprocess
import sys

def test_check_environment_runs():
    result = subprocess.run(
        [sys.executable, "check_environment.py"],
        capture_output=True, text=True, cwd="C:/dev/rt365-ai-editor"
    )
    # Should exit 0 or 1 — never crash with a traceback
    assert result.returncode in (0, 1)
    assert "RT365 AI Clip Studio" in result.stdout
