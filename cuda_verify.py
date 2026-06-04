#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CUDA / cuBLAS / CTranslate2 / faster-whisper diagnostic for RT365 AI Clip Studio.

Run from project root:
  .venv311\\Scripts\\python.exe cuda_verify.py
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

# CUDA Toolkit path reported in logs (Windows default layout)
CUDA_129_CUBLAS = Path(
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublas64_12.dll"
)

WHISPER_SAMPLE_RATE = 16000
WHISPER_SILENCE_SECONDS = 1.0


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _status_label(status: str) -> str:
    return f"[{status}]"


def check_torch_cuda_build() -> CheckResult:
    name = "torch CUDA build"
    try:
        import torch
    except ImportError as exc:
        return CheckResult(name, "FAIL", f"torch not installed: {exc}")

    ver = getattr(torch, "__version__", "?")
    cuda_ver = getattr(torch.version, "cuda", None)
    if not cuda_ver:
        return CheckResult(
            name,
            "WARN",
            f"torch {ver} — no CUDA build (torch.version.cuda is empty)",
        )
    return CheckResult(
        name,
        "PASS",
        f"torch {ver}, compiled for CUDA {cuda_ver}",
    )


def check_torch_cuda_runtime() -> CheckResult:
    name = "torch.cuda runtime"
    try:
        import torch
    except ImportError as exc:
        return CheckResult(name, "FAIL", f"torch not installed: {exc}")

    available = bool(torch.cuda.is_available())
    if not available:
        return CheckResult(name, "FAIL", "torch.cuda.is_available() is False")

    try:
        gpu_name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, "WARN", f"CUDA available but device query failed: {exc}")

    return CheckResult(
        name,
        "PASS",
        f"available=True, devices={count}, GPU[0]={gpu_name}",
    )


def check_cublas_dll_load() -> CheckResult:
    name = "cuBLAS DLL (CUDA 12.9 path)"
    path = CUDA_129_CUBLAS

    if not path.is_file():
        return CheckResult(
            name,
            "FAIL",
            f"file missing: {path}",
        )

    if sys.platform != "win32":
        return CheckResult(name, "WARN", f"exists at {path} (WinDLL load skipped on non-Windows)")

    try:
        ctypes.WinDLL(str(path))
    except OSError as exc:
        winerror = getattr(exc, "winerror", None)
        errno = getattr(exc, "errno", None)
        return CheckResult(
            name,
            "FAIL",
            (
                f"exists but WinDLL load failed: {path} | "
                f"winerror={winerror} errno={errno} | {type(exc).__name__}: {exc}"
            ),
        )

    return CheckResult(name, "PASS", f"exists and WinDLL load OK: {path}")


def check_ctranslate2_cuda_probe() -> CheckResult:
    name = "ctranslate2 CUDA probe"
    try:
        import ctranslate2 as ct
        from ctranslate2 import Device, StorageView
        from ctranslate2.version import __version__ as ct_ver
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, "FAIL", f"import failed: {exc}")

    try:
        n_dev = int(ct.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, "FAIL", f"ctranslate2 {ct_ver} — get_cuda_device_count failed: {exc}")

    if n_dev <= 0:
        return CheckResult(
            name,
            "FAIL",
            f"ctranslate2 {ct_ver} — get_cuda_device_count() == 0",
        )

    try:
        import numpy as np

        a = np.array([[1.0, 2.0]], dtype=np.float32)
        s = StorageView.from_array(a).to_device(Device.cuda)
        _ = s.shape
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name,
            "FAIL",
            f"ctranslate2 {ct_ver}, devices={n_dev} — StorageView.to_device(cuda) failed: {exc}",
        )

    return CheckResult(
        name,
        "PASS",
        f"ctranslate2 {ct_ver}, devices={n_dev}, StorageView.to_device(cuda): OK",
    )


def check_faster_whisper_cuda() -> CheckResult:
    name = "faster-whisper CUDA transcribe"
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as exc:
        return CheckResult(name, "FAIL", f"import failed: {exc}")

    audio = np.zeros(int(WHISPER_SAMPLE_RATE * WHISPER_SILENCE_SECONDS), dtype=np.float32)

    cuda_errors: list[str] = []
    for compute_type in ("float16", "int8_float16", "int8"):
        model = None
        try:
            model = WhisperModel("base", device="cuda", compute_type=compute_type)
            segments, _info = model.transcribe(audio, language="en", vad_filter=False)
            list(segments)
            return CheckResult(
                name,
                "PASS",
                f"CUDA OK (compute_type={compute_type}, model=base, 1s silent audio)",
            )
        except Exception as exc:  # noqa: BLE001
            cuda_errors.append(f"{compute_type}: {exc}")
        finally:
            if model is not None:
                del model

    cpu_errors: list[str] = []
    for compute_type in ("int8", "float32"):
        model = None
        try:
            model = WhisperModel("base", device="cpu", compute_type=compute_type)
            segments, _info = model.transcribe(audio, language="en", vad_filter=False)
            list(segments)
            return CheckResult(
                name,
                "WARN",
                (
                    "CUDA failed; CPU fallback succeeded. "
                    f"CUDA errors: {' | '.join(cuda_errors[:3])}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            cpu_errors.append(f"{compute_type}: {exc}")
        finally:
            if model is not None:
                del model

    return CheckResult(
        name,
        "FAIL",
        (
            f"CUDA and CPU both failed. CUDA: {' | '.join(cuda_errors)}; "
            f"CPU: {' | '.join(cpu_errors)}"
        ),
    )


def _cross_check_torch_vs_cublas(
    torch_build: CheckResult, cublas: CheckResult
) -> CheckResult | None:
    """Optional alignment hint when torch cu121 vs toolkit 12.9 and cuBLAS fails."""
    if cublas.status != "FAIL":
        return None
    if "compiled for CUDA" not in torch_build.detail:
        return None
    import re

    m = re.search(r"compiled for CUDA ([\d.]+)", torch_build.detail)
    if not m:
        return None
    torch_cuda = m.group(1)
    if torch_cuda.startswith("12.1") and CUDA_129_CUBLAS.parent.parent.name == "v12.9":
        return CheckResult(
            "toolkit vs torch alignment",
            "WARN",
            (
                f"torch built for CUDA {torch_cuda} but toolkit DLL is 12.9 - "
                "WinError 127 on cuBLAS often means export/DLL mismatch; "
                "consider torch cu128 wheel or CUDA 12.1 runtime bin on PATH."
            ),
        )
    return None


def main() -> int:
    print("RT365 CUDA verification")
    print("Python:", sys.executable)
    print("Platform:", sys.platform)
    print("-" * 72)

    results: list[CheckResult] = []
    results.append(check_torch_cuda_build())
    results.append(check_torch_cuda_runtime())
    results.append(check_cublas_dll_load())
    results.append(check_ctranslate2_cuda_probe())
    results.append(check_faster_whisper_cuda())

    extra = _cross_check_torch_vs_cublas(results[0], results[2])
    if extra is not None:
        results.append(extra)

    for r in results:
        print(f"{_status_label(r.status):6} {r.name}")
        print(f"         {r.detail}")
        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    col_w = max(len(r.name) for r in results) + 2
    print(f"{'Check':<{col_w}} {'Status':<6} Explanation")
    print("-" * 72)
    for r in results:
        expl = r.detail.split("\n")[0]
        if len(expl) > 52:
            expl = expl[:49] + "..."
        print(f"{r.name:<{col_w}} {r.status:<6} {expl}")

    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    print("-" * 72)
    if fails:
        print(f"Overall: FAIL ({fails} failed, {warns} warnings)")
        return 1
    if warns:
        print(f"Overall: WARN ({warns} warning(s))")
        return 0
    print("Overall: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
