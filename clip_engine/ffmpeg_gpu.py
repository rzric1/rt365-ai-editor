# -*- coding: utf-8 -*-
"""
NVIDIA NVENC detection and ffmpeg H.264 encode arguments (Ada / RTX 40-series friendly).

Falls back to libx264 when NVENC is unavailable or fails at runtime.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path

logger = logging.getLogger(__name__)

_nvenc_runtime_cached: bool | None = None
_nvenc_export_extras_cached: list[str] | None = None
_hwaccel_cuda_cached: bool | None = None
_last_nvenc_probe_log: str = ""
_NVENC_PROBE_CMD_LOGGED: bool = False


def _subprocess_kw() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _run(cmd: list[str], *, timeout: float = 25.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        **_subprocess_kw(),
    )


def invalidate_nvenc_cache() -> None:
    global _nvenc_runtime_cached, _nvenc_export_extras_cached, _hwaccel_cuda_cached
    _nvenc_runtime_cached = None
    _nvenc_export_extras_cached = None
    _hwaccel_cuda_cached = None


def get_last_nvenc_probe_log() -> str:
    """Human-readable log from the last NVENC probe cycle (for UI / diagnostics)."""
    return _last_nvenc_probe_log


def log_nvenc_probe_command_explicit() -> None:
    """Log the canonical NVENC self-test command line once at startup."""
    global _NVENC_PROBE_CMD_LOGGED
    if _NVENC_PROBE_CMD_LOGGED:
        return
    exe = ensure_ffmpeg_on_path()
    if not exe:
        return
    cmd = " ".join(_minimal_nvenc_probe_cmd(exe, extra_gpu=()))
    logger.info("[nvenc] Self-test command (lavfi null mux): %s", cmd)
    _NVENC_PROBE_CMD_LOGGED = True


def _minimal_nvenc_probe_cmd(
    exe: str,
    *,
    extra_gpu: tuple[str, ...] = (),
    preset: str = "p4",
) -> list[str]:
    """
    Short, driver-friendly probe at 640×360 (NVENC minimum frame size on many drivers).
    Heavy export flags (multipass fullres, etc.) are NOT used here — they often fail on
    tiny synthetic encodes even when full exports work.
    """
    return [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "info",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:d=0.04",
        "-c:v",
        "h264_nvenc",
        *extra_gpu,
        "-preset",
        preset,
        "-cq",
        "28",
        "-f",
        "null",
        "-",
    ]


def faster_whisper_cuda_available() -> bool:
    """True if CTranslate2 sees at least one CUDA device (for faster-whisper)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


@dataclass
class GpuAccelerationStatus:
    """Snapshot for UI + logging."""

    nvidia_smi_ok: bool
    nvidia_smi_line: str
    ffmpeg_nvenc_listed: bool
    nvenc_probe_ok: bool
    message: str


def nvidia_smi_summary() -> tuple[bool, str]:
    """Return (ok, one-line GPU summary or error hint)."""
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found (driver not installed?)"
    try:
        r = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            timeout=10.0,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "nvidia-smi failed").strip()[:200]
        line = (r.stdout or "").strip().splitlines()[0] if (r.stdout or "").strip() else "unknown GPU"
        return True, line
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


def ffmpeg_lists_h264_nvenc() -> bool:
    exe = ensure_ffmpeg_on_path()
    if not exe:
        return False
    try:
        r = _run([exe, "-hide_banner", "-encoders"], timeout=15.0)
        out = (r.stdout or "") + (r.stderr or "")
        return "h264_nvenc" in out
    except Exception:  # noqa: BLE001
        return False


def _probe_subprocess_flags() -> dict:
    """Optional visible console for NVENC probe (some driver stacks are picky)."""
    import os

    if os.environ.get("CLIP_STUDIO_NVENC_PROBE_VISIBLE", "").lower() in ("1", "true", "yes"):
        return {}
    return _subprocess_kw()


def probe_nvenc_encode() -> bool:
    """
    Try short NVENC encodes to null muxer until one succeeds.
    Uses a minimal flag set (export uses richer NVENC tuning separately).
    """
    global _last_nvenc_probe_log
    exe = ensure_ffmpeg_on_path()
    if not exe:
        _last_nvenc_probe_log = "ffmpeg not resolved; cannot probe NVENC."
        return False

    variants: list[tuple[str, tuple[str, ...], str]] = [
        ("minimal preset p4", (), "p4"),
        ("minimal p4 + -gpu 0", ("-gpu", "0"), "p4"),
        ("minimal p1 + -gpu 0", ("-gpu", "0"), "p1"),
    ]
    lines: list[str] = []
    for label, extra, preset in variants:
        cmd = _minimal_nvenc_probe_cmd(exe, extra_gpu=extra, preset=preset)
        cmd_line = " ".join(cmd)
        lines.append(f"--- {label} ---\n{cmd_line}")
        logger.info("[nvenc] probe try: %s", cmd_line)
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45.0,
                **_probe_subprocess_flags(),
            )
            tail_err = (r.stderr or r.stdout or "").strip()
            if tail_err:
                lines.append(f"exit={r.returncode} stderr/stdout tail:\n{tail_err[-2500:]}")
            ok = r.returncode == 0
            if ok:
                lines.append("RESULT: SUCCESS")
                _last_nvenc_probe_log = "\n".join(lines)
                logger.info("[nvenc] probe OK (%s)", label)
                return True
            logger.warning("[nvenc] probe failed (%s) rc=%s", label, r.returncode)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"EXCEPTION: {exc}")
            logger.warning("[nvenc] probe exception (%s): %s", label, exc)

    _last_nvenc_probe_log = "\n".join(lines)
    logger.warning("[nvenc] all probe variants failed; see get_last_nvenc_probe_log()")
    return False


def nvenc_runtime_available() -> bool:
    """Cached: whether we should attempt NVENC for this process."""
    if os.environ.get("FORCE_CPU_VIDEO", "").lower() in ("1", "true", "yes"):
        return False
    global _nvenc_runtime_cached
    if _nvenc_runtime_cached is not None:
        return _nvenc_runtime_cached
    if not ffmpeg_lists_h264_nvenc():
        _nvenc_runtime_cached = False
        return False
    _nvenc_runtime_cached = probe_nvenc_encode()
    return _nvenc_runtime_cached


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def should_attempt_nvenc_on_export(*, prefer_gpu: bool, force_gpu_mode: bool) -> bool:
    """
    True → first export pass uses h264_nvenc (+ export tuning).

    When **force_gpu_mode** (or env FORCE_NVENC_EXPORT) is set, we still attempt
    NVENC even if the lavfi probe failed (probe can be overly pessimistic vs real files).
    """
    if _env_truthy("FORCE_CPU_VIDEO"):
        return False
    if not prefer_gpu:
        return False
    if not ffmpeg_lists_h264_nvenc():
        return False
    if force_gpu_mode or _env_truthy("FORCE_NVENC_EXPORT"):
        return True
    return nvenc_runtime_available()


def get_gpu_acceleration_status() -> GpuAccelerationStatus:
    smi_ok, smi_line = nvidia_smi_summary()
    listed = ffmpeg_lists_h264_nvenc()
    probe = nvenc_runtime_available() if listed else False
    parts = [
        f"NVIDIA: {'OK — ' + smi_line if smi_ok else 'not detected'}",
        f"ffmpeg lists h264_nvenc: {listed}",
        f"NVENC runtime probe: {probe}",
    ]
    if _env_truthy("FORCE_NVENC_EXPORT"):
        parts.append("env FORCE_NVENC_EXPORT=1 (export tries NVENC even if probe had failed)")
    msg = " | ".join(parts)
    return GpuAccelerationStatus(
        nvidia_smi_ok=smi_ok,
        nvidia_smi_line=smi_line,
        ffmpeg_nvenc_listed=listed,
        nvenc_probe_ok=probe,
        message=msg,
    )


# Ada Lovelace (e.g. RTX 4090): p4 + tune hq + VBR CQ + AQ + multipass fullres
_NVENC_EXPORT_BASE = [
    "-c:v",
    "h264_nvenc",
    "-gpu",
    "0",
    "-preset",
    "p4",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "23",
    "-b:v",
    "0",
    "-spatial_aq",
    "1",
    "-temporal_aq",
    "1",
    "-multipass",
    "fullres",
    "-bf",
    "2",
    "-refs",
    "3",
]

# Optional RTX tuning — applied only when probe_nvenc_export_extras() succeeds.
_NVENC_EXPORT_OPTIONAL = [
    "-maxrate",
    "20M",
    "-bufsize",
    "40M",
    "-rc-lookahead",
    "32",
]

_CPU_X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]


def _export_nvenc_probe_cmd(exe: str, *, extra: tuple[str, ...] = ()) -> list[str]:
    """Probe full export NVENC stack on a short synthetic encode."""
    return [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:d=0.04",
        *_NVENC_EXPORT_BASE,
        *extra,
        "-pix_fmt",
        "yuv420p",
        "-f",
        "null",
        "-",
    ]


def probe_nvenc_export_extras() -> list[str]:
    """Return optional NVENC flags that pass a null-mux probe (cached)."""
    global _nvenc_export_extras_cached
    if _nvenc_export_extras_cached is not None:
        return list(_nvenc_export_extras_cached)
    exe = ensure_ffmpeg_on_path()
    if not exe:
        _nvenc_export_extras_cached = []
        return []
    ok_flags: list[str] = []
    try:
        r = subprocess.run(
            _export_nvenc_probe_cmd(exe, extra=tuple(_NVENC_EXPORT_OPTIONAL)),
            capture_output=True,
            text=True,
            timeout=60.0,
            **_probe_subprocess_flags(),
        )
        if r.returncode == 0:
            ok_flags = list(_NVENC_EXPORT_OPTIONAL)
            logger.info("[nvenc] export extras probe OK: %s", " ".join(ok_flags))
        else:
            logger.warning(
                "[nvenc] export extras probe failed; using base NVENC only. tail=%s",
                (r.stderr or "")[-800:],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[nvenc] export extras probe exception: %s", exc)
    _nvenc_export_extras_cached = ok_flags
    return list(ok_flags)


def probe_hwaccel_cuda_decode() -> bool:
    """True if ffmpeg can hwaccel-decode a short synthetic stream to null."""
    global _hwaccel_cuda_cached
    if _hwaccel_cuda_cached is not None:
        return _hwaccel_cuda_cached
    exe = ensure_ffmpeg_on_path()
    if not exe:
        _hwaccel_cuda_cached = False
        return False
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:d=0.04",
        "-f",
        "null",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=45.0,
            **_probe_subprocess_flags(),
        )
        _hwaccel_cuda_cached = r.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("[nvenc] hwaccel cuda probe exception: %s", exc)
        _hwaccel_cuda_cached = False
    if _hwaccel_cuda_cached:
        logger.info("[nvenc] CUDA hwaccel decode probe OK")
    else:
        logger.info("[nvenc] CUDA hwaccel decode probe failed or unavailable")
    return _hwaccel_cuda_cached


def ffmpeg_hwaccel_cuda_input_args() -> list[str]:
    """Input-side CUDA decode flags when probe passes (skip for -filter_complex paths)."""
    if probe_hwaccel_cuda_decode():
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []


def video_encode_args(*, use_nvenc: bool, preview_mode: bool = False) -> tuple[list[str], str]:
    """Return (ffmpeg args fragment, label for logging)."""
    if use_nvenc:
        args = list(_NVENC_EXPORT_BASE)
        if preview_mode:
            if "-cq" in args:
                cq_i = args.index("-cq")
                args[cq_i + 1] = "28"
            args.extend(["-maxrate", "8M", "-bufsize", "16M"])
        else:
            args.extend(probe_nvenc_export_extras())
        return args, "h264_nvenc"
    crf = "28" if preview_mode else "23"
    preset = "veryfast" if preview_mode else "fast"
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf], "libx264"


_NVENC_COMPRESS = [
    "-c:v",
    "h264_nvenc",
    "-gpu",
    "0",
    "-preset",
    "p4",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "30",
    "-b:v",
    "0",
    "-spatial_aq",
    "1",
    "-temporal_aq",
    "1",
]

_CPU_COMPRESS = ["-c:v", "libx264", "-preset", "medium", "-crf", "30"]


def video_encode_args_compress_crf30(*, use_nvenc: bool) -> tuple[list[str], str]:
    """Heavier compression for analysis proxy (matches prior CRF 30 intent)."""
    if use_nvenc:
        return list(_NVENC_COMPRESS), "h264_nvenc"
    return list(_CPU_COMPRESS), "libx264"


def run_ffmpeg_checked(cmd: list[str], *, cwd: str | None) -> None:
    r = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=7200,
        **_subprocess_kw(),
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(err[-4000:] if err else f"ffmpeg failed with code {r.returncode}")
