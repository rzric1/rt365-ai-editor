# -*- coding: utf-8 -*-
"""Startup dependency validation for RT365 AI Clip Studio (project .venv / .venv311)."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import ENV_OPENAI_API_KEY, ENV_OPENAI_MODEL, LOGS_DIR, PROJECT_ROOT

VALID_OPENAI_MODELS = frozenset(
    {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}
)

ENV_CHECK_LOG = LOGS_DIR / "environment_check.txt"
DOTENV_PATH = PROJECT_ROOT / ".env"

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 11
BLOCKED_PYTHON_MINORS = frozenset({14})  # 3.14 — known crash risk with mixed stacks
EXPECTED_VENV_DIR = PROJECT_ROOT / ".venv311"
# RT365-GPU-FIX 2026-06-05: accept `.venv` launch path (user command) and `.venv311`.
PROJECT_VENV_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / ".venv",
    PROJECT_ROOT / ".venv311",
)


def is_project_venv_executable(exe: str | Path | None = None) -> bool:
    """True when executable lives under project .venv or .venv311 (not system Python)."""
    path = Path(exe or sys.executable).resolve()
    for venv_root in PROJECT_VENV_DIRS:
        try:
            if str(path).lower().startswith(str(venv_root.resolve()).lower()):
                return True
        except OSError:
            pass
    exe_norm = str(path).replace("\\", "/").lower()
    return "/.venv/" in exe_norm or "/.venv311/" in exe_norm


@dataclass
class DependencyCheck:
    name: str
    ok: bool
    detail: str = ""
    critical: bool = True


@dataclass
class EnvironmentStatus:
    ok: bool
    checks: list[DependencyCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail, "critical": c.critical}
                for c in self.checks
            ],
        }


def _in_expected_venv() -> bool:
    return is_project_venv_executable(sys.executable)


def _check_python_version() -> DependencyCheck:
    major, minor = sys.version_info[:2]
    micro = sys.version_info[2] if len(sys.version_info) > 2 else 0
    ver = f"{major}.{minor}.{micro}"
    if major != REQUIRED_PYTHON_MAJOR:
        return DependencyCheck(
            "Python version",
            False,
            f"{ver} — need Python {REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}.x",
        )
    if minor in BLOCKED_PYTHON_MINORS:
        if _in_expected_venv():
            # RT365-GPU-FIX 2026-06-05: allow `.venv` launch; warn instead of hard block.
            return DependencyCheck(
                "Python version",
                False,
                f"{ver} — Python 3.14 in project venv (warn): prefer 3.11 (.venv311) for stability.",
                critical=False,
            )
        return DependencyCheck(
            "Python version",
            False,
            f"{ver} — Python 3.14 is blocked (use .venv or .venv311 with Python 3.11).",
        )
    if minor != REQUIRED_PYTHON_MINOR:
        return DependencyCheck(
            "Python version",
            False,
            f"{ver} — require Python 3.11.x (launch via launch_ai_clip_studio.ps1).",
            critical=True,
        )
    return DependencyCheck("Python version", True, ver)


def _import_check(name: str, module: str, *, critical: bool = True) -> DependencyCheck:
    try:
        __import__(module)
        return DependencyCheck(name, True, "import OK")
    except ImportError as exc:
        return DependencyCheck(name, False, str(exc), critical=critical)


def _check_torch_cuda() -> list[DependencyCheck]:
    out: list[DependencyCheck] = []
    try:
        import torch

        out.append(DependencyCheck("torch installed", True, torch.__version__, critical=False))
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            out.append(
                DependencyCheck(
                    "torch CUDA",
                    True,
                    f"{name} ({total / (1024**3):.1f} GB VRAM)",
                    critical=False,
                )
            )
        else:
            out.append(
                DependencyCheck(
                    "torch CUDA",
                    True,
                    "not available (CPU-only torch)",
                    critical=False,
                )
            )
    except ImportError:
        out.append(
            DependencyCheck(
                "torch installed",
                True,
                "not installed — optional unless GPU prefilter/embeddings",
                critical=False,
            )
        )
    return out


def _load_project_dotenv() -> None:
    """Load PROJECT_ROOT/.env into os.environ. Never logs secret values."""
    if not DOTENV_PATH.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(DOTENV_PATH, override=False)
    except ImportError:
        pass


def _openai_api_key_present() -> bool:
    return bool((os.environ.get(ENV_OPENAI_API_KEY) or "").strip())


def _check_openai_model() -> DependencyCheck:
    _load_project_dotenv()
    model = (os.environ.get(ENV_OPENAI_MODEL) or "").strip()
    if not model:
        return DependencyCheck(
            ENV_OPENAI_MODEL,
            True,
            "unset (config default applies)",
            critical=False,
        )
    if model in VALID_OPENAI_MODELS:
        return DependencyCheck(ENV_OPENAI_MODEL, True, f"recognized: {model}", critical=False)
    return DependencyCheck(
        ENV_OPENAI_MODEL,
        False,
        f"'{model}' is not a recognized model name",
        critical=False,
    )


def _check_openai_api_key() -> DependencyCheck:
    _load_project_dotenv()
    if _openai_api_key_present():
        return DependencyCheck(
            ENV_OPENAI_API_KEY,
            True,
            "present",
            critical=False,
        )
    return DependencyCheck(
        ENV_OPENAI_API_KEY,
        False,
        "missing — cloud Whisper/analyze need .env",
        critical=False,
    )


def _check_ffmpeg() -> list[DependencyCheck]:
    out: list[DependencyCheck] = []
    which = shutil.which("ffmpeg")
    out.append(
        DependencyCheck(
            "ffmpeg on PATH",
            bool(which),
            which or "not found — install FFmpeg or set FFMPEG_BINARY in .env",
        )
    )
    try:
        from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_version_line

        resolved = ensure_ffmpeg_on_path()
        ver = get_ffmpeg_version_line() or "unknown"
        out.append(
            DependencyCheck(
                "ffmpeg resolved",
                bool(resolved),
                f"{resolved} — {ver[:120]}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(DependencyCheck("ffmpeg resolved", False, str(exc)))
    return out


def validate_startup_environment(*, require_gpu_stack: bool = True) -> EnvironmentStatus:
    """
    Run all startup checks. Critical failures set status.ok=False.
    GPU stack (faster-whisper, ctranslate2) is critical when require_gpu_stack=True.
    """
    _load_project_dotenv()
    checks: list[DependencyCheck] = []

    checks.append(
        DependencyCheck(
            "Python executable",
            True,
            sys.executable,
            critical=False,
        )
    )
    checks.append(_check_python_version())
    venv_ok = _in_expected_venv()
    venv_detail = (
        f"project venv (.venv or .venv311) — got {sys.prefix}"
        if venv_ok
        else f"expected {PROJECT_VENV_DIRS[0]} or {PROJECT_VENV_DIRS[1]} — got {sys.prefix}"
    )
    checks.append(
        DependencyCheck(
            "Virtual environment",
            venv_ok,
            venv_detail,
        )
    )
    existing_venvs = [str(p) for p in PROJECT_VENV_DIRS if p.is_dir()]
    if not existing_venvs:
        checks.append(
            DependencyCheck(
                "project venv exists",
                False,
                "Run setup_windows.bat or scripts\\setup_python311_ai_env.ps1",
            )
        )
    else:
        checks.append(
            DependencyCheck("project venv exists", True, ", ".join(existing_venvs))
        )

    checks.append(_import_check("streamlit", "streamlit"))
    checks.append(_import_check("openai", "openai"))
    checks.append(_import_check("numpy", "numpy"))
    checks.append(_import_check("psutil", "psutil"))
    checks.extend(_check_ffmpeg())

    gpu_critical = require_gpu_stack
    checks.append(_import_check("faster-whisper", "faster_whisper", critical=gpu_critical))
    checks.append(_import_check("ctranslate2", "ctranslate2", critical=gpu_critical))
    checks.append(
        _import_check("opencv-python", "cv2", critical=False),
    )
    checks.extend(_check_torch_cuda())

    checks.append(_check_openai_api_key())
    checks.append(_check_openai_model())

    errors: list[str] = []
    warnings: list[str] = []
    for c in checks:
        if c.ok:
            continue
        msg = f"{c.name}: {c.detail}"
        if c.critical:
            errors.append(msg)
        else:
            warnings.append(msg)

    ok = len(errors) == 0
    return EnvironmentStatus(ok=ok, checks=checks, errors=errors, warnings=warnings)


def write_environment_check_log(status: EnvironmentStatus | None = None) -> Path:
    """Write human-readable report to logs/environment_check.txt."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if status is None:
        status = validate_startup_environment()

    lines = [
        "RT365 AI Clip Studio — environment check",
        f"Python: {sys.executable}",
        f"Version: {sys.version.split()[0]}",
        f"Prefix: {sys.prefix}",
        f"Project: {PROJECT_ROOT}",
        f"Overall: {'PASS' if status.ok else 'FAIL'}",
        "",
    ]
    for c in status.checks:
        tag = "OK" if c.ok else ("WARN" if not c.critical else "FAIL")
        lines.append(f"[{tag}] {c.name}: {c.detail}")
    if status.errors:
        lines.append("")
        lines.append("CRITICAL:")
        lines.extend(f"  - {e}" for e in status.errors)
    if status.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"  - {w}" for w in status.warnings)
    if not status.ok:
        lines.append("")
        lines.append(
            "Fix: Run setup_windows.bat, then launch via "
            ".venv\\Scripts\\python.exe -m streamlit run clip_studio_app.py "
            "or launch_ai_clip_studio.ps1 (.venv311). Do not use bare system Python."
        )

    text = "\n".join(lines)
    ENV_CHECK_LOG.write_text(text, encoding="utf-8")
    return ENV_CHECK_LOG


def format_streamlit_error(status: EnvironmentStatus) -> str:
    if status.ok:
        return ""
    parts = [
        "**Environment check failed.** Clip Studio cannot start safely.",
        "",
        "Launch from the project virtual env: "
        "`.venv\\Scripts\\python.exe -m streamlit run clip_studio_app.py` "
        "or **launch_ai_clip_studio.ps1** (`.venv311`). "
        "Do **not** use bare system Python.",
        "",
    ]
    parts.extend(f"- {e}" for e in status.errors[:12])
    parts.append("")
    parts.append(f"Full report: `{ENV_CHECK_LOG}`")
    return "\n".join(parts)
