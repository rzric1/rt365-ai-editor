"""
Resolve ffmpeg.exe on Windows (Streamlit / IDE often lack WinGet PATH).

- Honors FFMPEG_BINARY / config.ENV_FFMPEG_BINARY
- shutil.which("ffmpeg")
- Common install paths + WinGet Gyan.FFmpeg layout
- Prepends ffmpeg's directory to os.environ["PATH"] and sets FFMPEG_BINARY for subprocess children.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHED_EXE: str | None = None
_CACHED_VERSION_LINE: str | None = None
_VERSION_LINE_EXE: str | None = None  # exe path _CACHED_VERSION_LINE belongs to


def _env_ffmpeg_binary() -> str | None:
    from config import ENV_FFMPEG_BINARY  # noqa: PLC0415

    raw = os.environ.get(ENV_FFMPEG_BINARY, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())
    logger.warning("%s is set but not a file: %s", ENV_FFMPEG_BINARY, raw)
    return None


def _win_common_paths() -> list[Path]:
    roots = [
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\FFmpeg\bin"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg" / "bin",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "ffmpeg" / "bin",
    ]
    return [p / "ffmpeg.exe" for p in roots]


def _win_winget_gyan_ffmpeg() -> str | None:
    la = os.environ.get("LOCALAPPDATA", "")
    if not la:
        return None
    base = Path(la) / "Microsoft" / "WinGet" / "Packages"
    if not base.is_dir():
        return None
    try:
        # Prefer newest-looking package folder name (lexicographic is OK for version suffixes)
        for d in sorted(base.glob("Gyan.FFmpeg_*"), key=lambda p: str(p), reverse=True):
            for exe in d.rglob("ffmpeg.exe"):
                return str(exe.resolve())
    except OSError as exc:
        logger.debug("WinGet FFmpeg scan failed: %s", exc)
    return None


def _scan_candidates() -> str | None:
    hit = _env_ffmpeg_binary()
    if hit:
        return hit
    w = shutil.which("ffmpeg")
    if w:
        return str(Path(w).resolve())
    if sys.platform == "win32":
        for p in _win_common_paths():
            try:
                if p.is_file():
                    return str(p.resolve())
            except OSError:
                continue
        return _win_winget_gyan_ffmpeg()
    return None


def _norm_path_key(p: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(p))
    except OSError:
        return p


def _ensure_path_has_ffmpeg_bin(bin_dir: str) -> None:
    """Prepend ffmpeg's directory so subprocess and shutil.which see it (Streamlit reruns, fresh env)."""
    if not bin_dir:
        return
    key = _norm_path_key(bin_dir)
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    parts = [p for p in parts if _norm_path_key(p) != key]
    os.environ["PATH"] = bin_dir + os.pathsep + os.pathsep.join(parts)
    logger.debug("FFmpeg bin dir first on PATH: %s", bin_dir)


def _ffmpeg_version_line(exe: str) -> str | None:
    global _CACHED_VERSION_LINE, _VERSION_LINE_EXE
    if _VERSION_LINE_EXE == exe and _CACHED_VERSION_LINE is not None:
        return _CACHED_VERSION_LINE
    try:
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run(
            [exe, "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            **kw,
        )
        if r.returncode != 0:
            return None
        line = (r.stdout or r.stderr or "").strip().splitlines()
        _VERSION_LINE_EXE = exe
        _CACHED_VERSION_LINE = line[0] if line else None
        return _CACHED_VERSION_LINE
    except Exception as exc:  # noqa: BLE001
        logger.debug("ffmpeg -version failed: %s", exc)
        return None


def ensure_ffmpeg_on_path(*, log: bool = False) -> str | None:
    """
    Resolve ffmpeg, prepend its bin dir to PATH, set os.environ['FFMPEG_BINARY'].
    Safe to call on every Streamlit rerun (idempotent).
    Returns absolute path to ffmpeg or None.
    """
    global _CACHED_EXE, _CACHED_VERSION_LINE, _VERSION_LINE_EXE
    if _CACHED_EXE and not Path(_CACHED_EXE).is_file():
        _CACHED_EXE = None
        _CACHED_VERSION_LINE = None
        _VERSION_LINE_EXE = None

    if _CACHED_EXE and Path(_CACHED_EXE).is_file():
        _ensure_path_has_ffmpeg_bin(str(Path(_CACHED_EXE).parent))
        from config import ENV_FFMPEG_BINARY  # noqa: PLC0415

        os.environ[ENV_FFMPEG_BINARY] = _CACHED_EXE
        if log:
            logger.info("[ffmpeg] using cached: %s", _CACHED_EXE)
            vl = _ffmpeg_version_line(_CACHED_EXE)
            if vl:
                logger.info("[ffmpeg] %s", vl)
            logger.info(
                "[ffmpeg] subprocess.run(-version): %s",
                "ok" if vl else "failed",
            )
            head = os.environ.get("PATH", "")[:240]
            logger.info("[ffmpeg] PATH visibility (first 240 chars): %s", head + ("…" if len(os.environ.get("PATH", "")) > 240 else ""))
        return _CACHED_EXE

    exe = _scan_candidates()
    if not exe:
        if log:
            logger.warning("[ffmpeg] not found (FFMPEG_BINARY, PATH, WinGet, common paths)")
        return None

    bin_dir = str(Path(exe).parent)
    _ensure_path_has_ffmpeg_bin(bin_dir)
    from config import ENV_FFMPEG_BINARY  # noqa: PLC0415

    os.environ[ENV_FFMPEG_BINARY] = exe
    _CACHED_EXE = exe

    try:
        from clip_engine.ffmpeg_gpu import invalidate_nvenc_cache  # noqa: PLC0415

        invalidate_nvenc_cache()
    except ImportError:
        pass

    if log:
        logger.info("[ffmpeg] resolved: %s", exe)
        vl = _ffmpeg_version_line(exe)
        if vl:
            logger.info("[ffmpeg] %s", vl)
        logger.info(
            "[ffmpeg] subprocess.run(-version): %s",
            "ok" if vl else "failed",
        )
        head = os.environ.get("PATH", "")[:240]
        logger.info("[ffmpeg] PATH visibility (first 240 chars): %s", head + ("…" if len(os.environ.get("PATH", "")) > 240 else ""))

    return exe


def get_ffmpeg_executable() -> str:
    """Absolute path to ffmpeg; raises if missing after resolution."""
    exe = ensure_ffmpeg_on_path()
    if not exe:
        from config import ENV_FFMPEG_BINARY  # noqa: PLC0415

        raise RuntimeError(
            "ffmpeg not found. Install FFmpeg (e.g. winget install Gyan.FFmpeg), "
            f"or set the full path in .env as {ENV_FFMPEG_BINARY}=C:/path/to/ffmpeg.exe "
            "and restart Streamlit."
        )
    return exe


def get_ffmpeg_version_line() -> str | None:
    exe = ensure_ffmpeg_on_path()
    if not exe:
        return None
    return _ffmpeg_version_line(exe)


def ffmpeg_available() -> bool:
    return ensure_ffmpeg_on_path() is not None
