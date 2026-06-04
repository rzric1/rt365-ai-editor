# -*- coding: utf-8 -*-
"""
CUDA / cuBLAS / CTranslate2 diagnostics for faster-whisper and sidebar UI.

Detects common Windows failure: cublas64_12.dll missing while the driver
reports CUDA devices. Documents toolkit alignment for RTX 4090 + pip wheels.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

logger.info(f"[env] pid={os.getpid()} executable={sys.executable} prefix={sys.prefix}")

# --- Reference (shown in UI / logs; keep in sync with pip extras for faster-whisper) ------------
# RTX 4090 (Ada): needs a recent NVIDIA **driver** (not necessarily the full CUDA Toolkit).
# faster-whisper → CTranslate2; prebuilt wheels are tagged cuda12 / cuda11 (check PyPI).
# If you install the CUDA 12 wheel, CUDA **12.x** runtime DLLs (e.g. cuBLAS 12) must be loadable.
# Typical fix on Windows: install "CUDA Toolkit 12.x" from NVIDIA **or** add
#   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
# to PATH / set CUDA_PATH so cublas64_12.dll is found next to other CUDA DLLs.
CUDA_STACK_REFERENCE = (
    "**RTX 4090:** Use a current Game Ready / Studio driver (CUDA 12 user-mode is bundled with "
    "recent drivers; missing cuBLAS usually means the toolkit `bin` folder is not on PATH).\n\n"
    "**faster-whisper:** `pip install faster-whisper` then match **ctranslate2** to your CUDA "
    "major (e.g. `ctranslate2==4.x` + `cuda12` wheel from PyPI / OpenNMT).\n\n"
    "**ctranslate2 (CUDA 12 build):** requires **CUDA 12.x** DLLs such as **cublas64_12.dll** "
    "(from NVIDIA CUDA Toolkit 12.x `bin`, or a layout where those DLLs are discoverable).\n\n"
    "**Torch (optional):** if `torch` is installed, `torch.version.cuda` should match your "
    "CUDA toolkit / driver stack; `torch.cuda.is_available()` is independent of CTranslate2."
)

_RUNTIME_PROBE_CACHE: tuple[bool, str] | None = None
_STARTUP_DIAGNOSTICS_LOGGED: bool = False
_CACHED_DIAGNOSTICS: "AiAccelerationDiagnostics | None" = None


def invalidate_cuda_runtime_probe_cache() -> None:
    global _RUNTIME_PROBE_CACHE
    _RUNTIME_PROBE_CACHE = None


def _subprocess_kw() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _nvidia_smi_text() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe],
            capture_output=True,
            text=True,
            timeout=12,
            **_subprocess_kw(),
        )
        if r.returncode != 0:
            return None
        return r.stdout or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi failed: %s", exc)
        return None


def parse_driver_cuda_version(nvidia_smi_stdout: str | None) -> str | None:
    """Parse 'CUDA Version: 12.x' from nvidia-smi banner."""
    if not nvidia_smi_stdout:
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", nvidia_smi_stdout)
    return m.group(1) if m else None


def _first_gpu_line(nvidia_smi_stdout: str | None) -> str | None:
    if not nvidia_smi_stdout:
        return None
    for line in nvidia_smi_stdout.splitlines():
        if "|" in line and "NVIDIA" in line.upper() and "Driver Version" not in line:
            s = line.strip()
            if len(s) > 10:
                return s[:200]
    return None


def _venv_torch_lib_dir() -> Path | None:
    """Return site-packages/torch/lib without importing torch (avoids early DLL load)."""
    try:
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            torch_lib = Path(list(torch_spec.submodule_search_locations)[0]) / "lib"
            if torch_lib.is_dir():
                return torch_lib
    except Exception:
        pass
    return None


def _find_cublas_dlls() -> list[str]:
    """Best-effort search for cuBLAS DLLs (Windows + Linux-ish names)."""
    names = ("cublas64_12.dll", "cublas64_11.dll", "cublas.so.12", "cublas.so.11")
    seen: set[str] = set()
    out: list[str] = []

    def add(p: Path) -> None:
        try:
            r = str(p.resolve())
        except OSError:
            r = str(p)
        if r not in seen and p.is_file():
            seen.add(r)
            out.append(r)

    # Prefer venv-bundled torch CUDA DLLs (matches torch cu128 wheel, avoids system 12.9 mismatch).
    torch_lib = _venv_torch_lib_dir()
    if torch_lib is not None:
        for n in names:
            add(torch_lib / n)

    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        for n in names:
            add(Path(cuda_path) / "bin" / n)

    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        base = Path(pf) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if base.is_dir():
            for vdir in sorted(base.glob("v*"), key=lambda p: str(p), reverse=True):
                for n in names:
                    add(vdir / "bin" / n)

    path_env = os.environ.get("PATH", "")
    for d in path_env.split(os.pathsep):
        if not d.strip():
            continue
        for n in names:
            add(Path(d) / n)

    return out[:24]


def _try_load_cublas_windows(cublas_paths: list[str]) -> tuple[bool, str]:
    if sys.platform != "win32":
        return True, "(skipped on non-Windows)"
    try:
        import ctypes
    except ImportError:
        return False, "ctypes unavailable"

    venv_lib = _venv_torch_lib_dir()
    venv_paths: list[str] = []
    if venv_lib is not None:
        for name in ("cublas64_12.dll", "cublas64_11.dll"):
            candidate = venv_lib / name
            if candidate.is_file():
                venv_paths.append(str(candidate.resolve()))

    for p in venv_paths:
        try:
            ctypes.WinDLL(p)
            return True, f"loaded (venv torch/lib): {p}"
        except OSError as exc:
            logger.debug("venv cuBLAS load failed for %s: %s", p, exc)

    for p in cublas_paths:
        if not p.lower().endswith(".dll"):
            continue
        if p in venv_paths:
            continue
        try:
            ctypes.WinDLL(p)
            source = "venv torch/lib" if venv_lib and str(venv_lib).lower() in p.lower() else "system CUDA"
            return True, f"loaded ({source}): {p}"
        except OSError as exc:
            return False, f"failed to load {p}: {exc}"

    for name in ("cublas64_12.dll", "cublas64_11.dll"):
        try:
            ctypes.WinDLL(name)
            return True, f"loaded by name (PATH): {name}"
        except OSError:
            continue
    return False, "cublas64_*.dll not found or not loadable (check venv torch/lib or CUDA Toolkit bin on PATH)"


def ctranslate2_cuda_runtime_probe(*, use_cache: bool = True) -> tuple[bool, str]:
    """
    Lightweight GPU runtime check (CTranslate2): copy a tiny tensor to CUDA.
    Catches missing cuBLAS / broken CUDA user-mode even when device count > 0.
    """
    global _RUNTIME_PROBE_CACHE
    if use_cache and _RUNTIME_PROBE_CACHE is not None:
        return _RUNTIME_PROBE_CACHE
    try:
        import numpy as np
        import ctranslate2 as ct
        from ctranslate2 import Device, StorageView
    except Exception as exc:  # noqa: BLE001
        _RUNTIME_PROBE_CACHE = (False, f"import failed: {exc}")
        return _RUNTIME_PROBE_CACHE
    try:
        if ct.get_cuda_device_count() <= 0:
            _RUNTIME_PROBE_CACHE = (False, "get_cuda_device_count() == 0")
            return _RUNTIME_PROBE_CACHE
    except Exception as exc:  # noqa: BLE001
        _RUNTIME_PROBE_CACHE = (False, f"get_cuda_device_count failed: {exc}")
        return _RUNTIME_PROBE_CACHE
    try:
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        s = StorageView.from_array(a).to_device(Device.cuda)
        _ = s.shape
        _RUNTIME_PROBE_CACHE = (True, "StorageView.to_device(cuda): OK")
        return _RUNTIME_PROBE_CACHE
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip() or type(exc).__name__
        _RUNTIME_PROBE_CACHE = (False, msg[:800])
        return _RUNTIME_PROBE_CACHE


def _ctranslate2_version() -> str | None:
    try:
        from ctranslate2.version import __version__ as v

        return str(v)
    except Exception:
        return None


def _ctranslate2_cuda_devices() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _ctranslate2_cuda_compute_types() -> str | None:
    try:
        import ctranslate2

        types = ctranslate2.get_supported_compute_types("cuda")
        return ", ".join(sorted(types)) if types else None
    except Exception:
        return None


def _torch_cuda_summary() -> str:
    try:
        import torch
    except ImportError:
        return "torch: not installed"
    try:
        ver = getattr(torch.version, "cuda", None) or "?"
        ok = bool(torch.cuda.is_available())
        return f"torch CUDA: available={ok}, torch.version.cuda={ver}"
    except Exception as exc:  # noqa: BLE001
        return f"torch: error {exc}"


@dataclass
class AiAccelerationDiagnostics:
    """Snapshot for sidebar + expander (one-click panel)."""

    nvenc_listed: bool
    nvenc_probe_ok: bool
    nvidia_smi_ok: bool
    nvidia_gpu_line: str
    driver_reported_cuda: str | None
    cublas_paths: list[str] = field(default_factory=list)
    cublas_load_ok: bool = False
    cublas_load_detail: str = ""
    ctranslate2_version: str | None = None
    ctranslate2_cuda_devices: int = 0
    ctranslate2_cuda_compute_types: str | None = None
    cuda_runtime_probe_ok: bool = False
    cuda_runtime_probe_message: str = ""
    torch_summary: str = ""
    transcribe_hint: str = ""

    def to_sidebar_lines(self) -> str:
        lines = [
            f"NVENC listed: {self.nvenc_listed}",
            f"NVENC runtime probe: {self.nvenc_probe_ok}",
            f"nvidia-smi: {'OK' if self.nvidia_smi_ok else 'no/fail'} — {self.nvidia_gpu_line}",
            f"Driver-reported CUDA: {self.driver_reported_cuda or 'unknown'}",
            f"cuBLAS DLLs found: {len(self.cublas_paths)}",
            f"cuBLAS load test: {'OK' if self.cublas_load_ok else 'FAIL'} — {self.cublas_load_detail[:120]}",
            f"ctranslate2: {self.ctranslate2_version or '?'} | CUDA devices: {self.ctranslate2_cuda_devices}",
            f"CUDA compute types: {self.ctranslate2_cuda_compute_types or '?'}",
            f"CTranslate2 CUDA probe: {'OK' if self.cuda_runtime_probe_ok else 'FAIL'} — {self.cuda_runtime_probe_message[:160]}",
            self.torch_summary,
            f"Transcription hint: {self.transcribe_hint}",
        ]
        return "\n".join(lines)

    def to_detail_markdown(self) -> str:
        cub = "\n".join(f"- `{p}`" for p in self.cublas_paths[:12]) or "- _(none found)_"
        probe = self.cuda_runtime_probe_message or "—"
        return (
            f"**NVENC** — listed: `{self.nvenc_listed}`, runtime probe: `{self.nvenc_probe_ok}`\n\n"
            f"**NVIDIA** — smi: `{'OK' if self.nvidia_smi_ok else 'missing/fail'}`  \n"
            f"`{self.nvidia_gpu_line}`\n\n"
            f"**Driver CUDA version (banner):** `{self.driver_reported_cuda or 'unknown'}`\n\n"
            f"**cuBLAS paths (sample):**  \n{cub}\n\n"
            f"**cuBLAS load:** `{'OK' if self.cublas_load_ok else 'FAIL'}` — {self.cublas_load_detail}\n\n"
            f"**ctranslate2** `{self.ctranslate2_version or '?'}` — CUDA devices: `{self.ctranslate2_cuda_devices}`  \n"
            f"Supported CUDA compute types: `{self.ctranslate2_cuda_compute_types or '?'}`\n\n"
            f"**CTranslate2 CUDA runtime probe:** `{'OK' if self.cuda_runtime_probe_ok else 'FAIL'}`  \n"
            f"```\n{probe}\n```\n\n"
            f"**{self.torch_summary}**\n\n"
            f"**Transcription:** {self.transcribe_hint}\n\n"
            "---\n"
            + CUDA_STACK_REFERENCE
        )


def collect_ai_acceleration_diagnostics(*, refresh_cuda_probe: bool = False) -> AiAccelerationDiagnostics:
    """Gather NVENC + CUDA + cuBLAS + ctranslate2 + optional torch. Import ffmpeg_gpu lazily."""
    global _CACHED_DIAGNOSTICS
    from clip_engine.ffmpeg_gpu import (  # noqa: PLC0415
        ffmpeg_lists_h264_nvenc,
        nvenc_runtime_available,
    )

    if not refresh_cuda_probe and _CACHED_DIAGNOSTICS is not None:
        logger.debug("skipping duplicate diagnostics — using cached AI acceleration snapshot")
        return _CACHED_DIAGNOSTICS

    if refresh_cuda_probe:
        invalidate_cuda_runtime_probe_cache()

    smi = _nvidia_smi_text()
    nvidia_ok = bool(smi)
    gpu_line = _first_gpu_line(smi) or (smi.strip()[:200] if smi else "nvidia-smi not available")
    drv_cuda = parse_driver_cuda_version(smi)

    listed = ffmpeg_lists_h264_nvenc()
    probe_nvenc = nvenc_runtime_available() if listed else False

    cublas_paths = _find_cublas_dlls()
    if sys.platform == "win32":
        cubl_ok, cubl_detail = _try_load_cublas_windows(cublas_paths)
    else:
        cubl_ok, cubl_detail = (bool(cublas_paths), f"found {len(cublas_paths)} path(s)")

    ct_ver = _ctranslate2_version()
    ct_devs = _ctranslate2_cuda_devices()
    ct_types = _ctranslate2_cuda_compute_types()

    cuda_ok, cuda_msg = ctranslate2_cuda_runtime_probe(use_cache=True)

    torch_line = _torch_cuda_summary()

    # Transcription routing hint (mirrors transcribe_video logic)
    if ct_devs <= 0:
        hint = (
            "No CUDA devices for CTranslate2 — local faster-whisper **CUDA** is skipped. "
            "Use **OPENAI_API_KEY** for cloud Whisper (GPU mode does not force local CPU here)."
        )
    elif not cuda_ok:
        hint = (
            "CUDA runtime probe FAILED — local faster-whisper CUDA is blocked unless "
            "ALLOW_CPU_FALLBACK=1 (CPU int8, slow). Fix: run 'where.exe cublas64_12.dll'. "
            "If not found, install NVIDIA CUDA Toolkit 12.x and add "
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin to PATH, "
            "then restart via launch_ai_clip_studio.ps1 (.venv311)."
        )
    else:
        hint = "Local: **faster-whisper on CUDA** should work when GPU acceleration is on."

    _CACHED_DIAGNOSTICS = AiAccelerationDiagnostics(
        nvenc_listed=listed,
        nvenc_probe_ok=probe_nvenc,
        nvidia_smi_ok=nvidia_ok,
        nvidia_gpu_line=gpu_line,
        driver_reported_cuda=drv_cuda,
        cublas_paths=cublas_paths,
        cublas_load_ok=cubl_ok,
        cublas_load_detail=cubl_detail,
        ctranslate2_version=ct_ver,
        ctranslate2_cuda_devices=ct_devs,
        ctranslate2_cuda_compute_types=ct_types,
        cuda_runtime_probe_ok=cuda_ok,
        cuda_runtime_probe_message=cuda_msg,
        torch_summary=torch_line,
        transcribe_hint=hint,
    )
    return _CACHED_DIAGNOSTICS


def log_ai_acceleration_startup() -> None:
    """INFO lines once at process start (Streamlit / CLI)."""
    global _STARTUP_DIAGNOSTICS_LOGGED
    if _STARTUP_DIAGNOSTICS_LOGGED:
        logger.debug("skipping duplicate diagnostics — startup AI acceleration already logged")
        return
    _STARTUP_DIAGNOSTICS_LOGGED = True
    logger.info("diagnostics initialized — logging AI acceleration startup snapshot")
    d = collect_ai_acceleration_diagnostics(refresh_cuda_probe=True)
    logger.info("[ai-accel] NVENC listed=%s probe=%s", d.nvenc_listed, d.nvenc_probe_ok)
    logger.info("[ai-accel] nvidia-smi ok=%s cuda=%s gpu=%s", d.nvidia_smi_ok, d.driver_reported_cuda, d.nvidia_gpu_line[:120])
    logger.info(
        "[ai-accel] cuBLAS paths=%s load_ok=%s %s",
        len(d.cublas_paths),
        d.cublas_load_ok,
        d.cublas_load_detail[:200],
    )
    logger.info(
        "[ai-accel] ctranslate2=%s cuda_devices=%s runtime_probe_ok=%s msg=%s",
        d.ctranslate2_version,
        d.ctranslate2_cuda_devices,
        d.cuda_runtime_probe_ok,
        d.cuda_runtime_probe_message[:300],
    )
    logger.info("[ai-accel] %s", d.torch_summary)
    logger.info("[ai-accel] transcribe_hint=%s", d.transcribe_hint[:300])


def cublas_missing_hint(exc_message: str) -> bool:
    low = exc_message.lower()
    return "cublas" in low and ("not found" in low or "cannot be loaded" in low or "load" in low)


def allow_cpu_fallback() -> bool:
    """True only when ALLOW_CPU_FALLBACK=1|true|yes (default False)."""
    return os.environ.get("ALLOW_CPU_FALLBACK", "").lower() in ("1", "true", "yes")


def gpu_pid_check(*, context: str = "") -> tuple[bool, str]:
    """
    Verify the current process PID is using GPU compute (nvidia-smi).

    On Windows WDDM, --query-compute-apps often lists graphics clients; we also scan
    the default nvidia-smi process table for this PID with python in the process name.
    Returns (on_gpu, detail_message).
    """
    pid = os.getpid()
    prefix = f"[gpu_pid_check{(' ' + context) if context else ''}]"
    exe = shutil.which("nvidia-smi")
    if not exe:
        msg = f"{prefix} WARN nvidia-smi not found — cannot verify GPU compute PID"
        logger.warning(msg)
        return False, msg

    python_rows: list[str] = []
    try:
        r = subprocess.run(
            [
                exe,
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            **_subprocess_kw(),
        )
        if r.returncode == 0:
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    row_pid = int(parts[0])
                except ValueError:
                    continue
                proc_name = parts[1].lower()
                if "python" not in proc_name:
                    continue
                vram = parts[2] if len(parts) > 2 else "?"
                python_rows.append(f"{row_pid}({proc_name},{vram})")
                if row_pid == pid:
                    msg = (
                        f"{prefix} INFO pid={pid} in compute-apps (python): {parts[1]} "
                        f"VRAM={vram}; sys.executable={sys.executable} "
                        "(authoritative for .venv311 on Windows)"
                    )
                    logger.info(msg)
                    return True, msg
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("%s compute-apps query error: %s", prefix, exc)

    try:
        r2 = subprocess.run(
            [exe],
            capture_output=True,
            text=True,
            timeout=12,
            **_subprocess_kw(),
        )
        if r2.returncode == 0:
            for line in (r2.stdout or "").splitlines():
                if str(pid) not in line:
                    continue
                low = line.lower()
                if "python" in low:
                    msg = f"{prefix} INFO pid={pid} found in nvidia-smi process table"
                    logger.info(msg)
                    return True, msg
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("%s full smi parse error: %s", prefix, exc)

    if python_rows:
        others = ", ".join(python_rows)
        msg = (
            f"{prefix} WARN pid={pid} NOT in nvidia-smi python compute list; "
            f"other python GPU PIDs: {others}"
        )
    else:
        msg = (
            f"{prefix} WARN pid={pid} NOT using GPU compute (no python.exe in nvidia-smi). "
            "Whisper may be on CPU, not loaded yet, or launched from wrong python.exe."
        )
    logger.warning(msg)
    return False, msg


def query_nvidia_gpu_memory_and_util() -> tuple[int | None, int | None]:
    """
    Return (memory_used_mib, utilization_gpu_pct) for GPU 0 via nvidia-smi.
    Used for faster-whisper / CTranslate2 GPU checks (not torch.cuda memory).
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        r = subprocess.run(
            [
                exe,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            **_subprocess_kw(),
        )
        if r.returncode != 0:
            return None, None
        parts = [p.strip() for p in (r.stdout or "").strip().split(",")]
        if len(parts) < 2:
            return None, None
        return int(float(parts[0])), int(float(parts[1].rstrip(" %")))
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        logger.debug("nvidia-smi memory/util query failed: %s", exc)
        return None, None


def get_startup_cuda_log_lines() -> list[str]:
    """Lines written to logs/startup_diagnostics.txt for CUDA / DLL / venv checks."""
    from clip_engine.whisper_runtime import (  # noqa: PLC0415
        get_cuda_dll_fix_startup_lines,
        get_env_startup_line,
    )

    lines = list(get_cuda_dll_fix_startup_lines())
    lines.append(get_env_startup_line())
    try:
        d = collect_ai_acceleration_diagnostics(refresh_cuda_probe=True)
        lines.append(
            f"[ai-accel] cuBLAS paths={len(d.cublas_paths)} load_ok={d.cublas_load_ok} "
            f"{d.cublas_load_detail[:200]}"
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"[ai-accel] cuBLAS collect failed: {exc}")
    lines.append(
        "Windows venv note: nvidia-smi may list the base python.exe path while "
        "sys.executable / logs/rt365_app.lock show .venv311\\Scripts\\python.exe."
    )
    return lines


@dataclass
class GpuTranscriptionCheck:
    name: str
    passed: bool
    detail: str


def evaluate_gpu_transcription_checks(
    *,
    segment_count: int,
    requested_device: str,
    actual_device: str,
    sys_executable: str | None = None,
    gpu_mem_before_mib: int | None = None,
    gpu_mem_after_mib: int | None = None,
    gpu_util_before_pct: int | None = None,
    gpu_util_after_pct: int | None = None,
) -> list[GpuTranscriptionCheck]:
    """
    Pass criteria for local GPU transcription (CTranslate2 / faster-whisper).
    Does not use torch.cuda.memory_allocated() or WhisperModel.device.
    """
    exe = sys_executable or sys.executable
    venv_ok = ".venv311" in exe.replace("\\", "/")
    checks: list[GpuTranscriptionCheck] = []

    checks.append(
        GpuTranscriptionCheck(
            "sys.executable uses .venv311",
            venv_ok,
            exe,
        )
    )
    cuda_device = str(actual_device).lower() == "cuda" or (
        str(requested_device).lower() == "cuda"
        and "cuda" in str(actual_device).lower()
    )
    checks.append(
        GpuTranscriptionCheck(
            "whisper actual_device=cuda",
            cuda_device,
            f"requested={requested_device} actual={actual_device} "
            "(expect log line: actual_device=cuda)",
        )
    )
    checks.append(
        GpuTranscriptionCheck(
            "transcription segments > 0",
            segment_count > 0,
            str(segment_count),
        )
    )

    util_rise = False
    util_detail = f"before={gpu_util_before_pct}% after={gpu_util_after_pct}%"
    if gpu_util_before_pct is not None and gpu_util_after_pct is not None:
        util_rise = gpu_util_after_pct > max(gpu_util_before_pct, 10)
        util_detail += f" (rise>{max(gpu_util_before_pct, 10)}%={util_rise})"
    elif gpu_util_after_pct is not None:
        util_rise = gpu_util_after_pct > 10
        util_detail += f" (after-only >10%={util_rise})"
    checks.append(
        GpuTranscriptionCheck(
            "nvidia-smi GPU utilization during/after transcribe",
            util_rise,
            util_detail,
        )
    )

    vram_rise = False
    vram_detail = f"before={gpu_mem_before_mib} MiB after={gpu_mem_after_mib} MiB"
    if gpu_mem_before_mib is not None and gpu_mem_after_mib is not None:
        vram_rise = gpu_mem_after_mib > gpu_mem_before_mib
        vram_detail += f" delta={gpu_mem_after_mib - gpu_mem_before_mib} MiB"
    checks.append(
        GpuTranscriptionCheck(
            "nvidia-smi VRAM increased during transcribe",
            vram_rise,
            vram_detail,
        )
    )

    return checks


def format_gpu_transcription_check_report(checks: list[GpuTranscriptionCheck]) -> str:
    lines = ["--- GPU transcription pass criteria ---"]
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"[{status}] {c.name}: {c.detail}")
    return "\n".join(lines)
