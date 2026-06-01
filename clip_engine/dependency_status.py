# -*- coding: utf-8 -*-
"""
clip_engine/dependency_status.py
Optional dependency checker for AI editing upgrades.
Reports installed/missing packages, feature availability, and fallback behavior.
"""

from __future__ import annotations

import importlib
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("clip_engine.dependency_status")


@dataclass
class PackageStatus:
    name: str
    installed: bool
    version: str | None = None
    import_error: str | None = None
    feature: str = ""
    enabled: bool = False
    fallback: str = ""


@dataclass
class DependencyReport:
    packages: list[PackageStatus] = field(default_factory=list)
    gpu_available: bool = False
    gpu_backend: str = "none"
    torch_cuda: bool = False
    torch_version: str | None = None
    cuda_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_available": self.gpu_available,
            "gpu_backend": self.gpu_backend,
            "torch_cuda": self.torch_cuda,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "packages": [
                {
                    "name": p.name,
                    "installed": p.installed,
                    "version": p.version,
                    "feature": p.feature,
                    "enabled": p.enabled,
                    "fallback": p.fallback,
                    "import_error": p.import_error,
                }
                for p in self.packages
            ],
        }


def _try_import(module: str) -> tuple[bool, str | None, str | None]:
    """Return (installed, version, error)."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", None)
        return True, str(version) if version else None, None
    except ImportError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def _gpu_info() -> tuple[bool, str, bool, str | None, str | None]:
    gpu_available = bool(shutil.which("nvidia-smi"))
    gpu_backend = "nvidia-smi" if gpu_available else "none"
    torch_cuda = False
    torch_version: str | None = None
    cuda_version: str | None = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        torch_cuda = bool(torch.cuda.is_available())
        if torch_cuda:
            cuda_version = getattr(torch.version, "cuda", None)
            gpu_backend = "torch+cuda"
        elif gpu_available:
            gpu_backend = "nvidia-smi (no torch)"
    except ImportError:
        pass
    return gpu_available, gpu_backend, torch_cuda, torch_version, cuda_version


def check_package(name: str, module: str, *, feature: str, fallback: str) -> PackageStatus:
    installed, version, err = _try_import(module)
    return PackageStatus(
        name=name,
        installed=installed,
        version=version,
        import_error=err,
        feature=feature,
        enabled=installed and err is None,
        fallback=fallback,
    )


def get_dependency_report() -> DependencyReport:
    """Full dependency status for optional AI upgrade packages."""
    gpu_available, gpu_backend, torch_cuda, torch_version, cuda_version = _gpu_info()

    packages = [
        check_package(
            "ultralytics",
            "ultralytics",
            feature="Dynamic YOLO smart crop",
            fallback="OpenCV Haar cascade or full-frame fit",
        ),
        check_package(
            "opencv-python-headless",
            "cv2",
            feature="OpenCV face detection smart crop",
            fallback="Full-frame fit with blurred background",
        ),
        check_package(
            "pyannote.audio",
            "pyannote.audio",
            feature="GPU speaker diarization (optional)",
            fallback="Transcript-based speaker/interruption heuristics",
        ),
        check_package(
            "transformers",
            "transformers",
            feature="Advanced NLP models (optional)",
            fallback="Local heuristic signal scoring",
        ),
        check_package(
            "librosa",
            "librosa",
            feature="Audio feature analysis (optional)",
            fallback="Transcript-based pacing/emotion heuristics",
        ),
        check_package(
            "pysubs2",
            "pysubs2",
            feature="Advanced ASS/karaoke captions",
            fallback="Built-in ASS generator",
        ),
        check_package(
            "sentence-transformers",
            "sentence_transformers",
            feature="GPU semantic ranking & embedding dedupe (RTX prefilter)",
            fallback="Jaccard word-set deduplication",
        ),
    ]

    return DependencyReport(
        packages=packages,
        gpu_available=gpu_available,
        gpu_backend=gpu_backend,
        torch_cuda=torch_cuda,
        torch_version=torch_version,
        cuda_version=cuda_version,
    )


def feature_enabled(feature_key: str) -> bool:
    """Check if a named feature's primary dependency is available."""
    mapping = {
        "dynamic_smart_crop": "ultralytics",
        "opencv_smart_crop": "cv2",
        "speaker_diarization": "pyannote.audio",
        "advanced_captions": "pysubs2",
        "semantic_models": "sentence_transformers",
        "audio_analysis": "librosa",
        "transformers_nlp": "transformers",
    }
    module = mapping.get(feature_key)
    if not module:
        return False
    installed, _, _ = _try_import(module)
    return installed


def render_status_markdown(report: DependencyReport | None = None) -> str:
    """Markdown summary for Streamlit sidebar."""
    report = report or get_dependency_report()
    lines = [
        f"**GPU:** {'available' if report.gpu_available else 'not detected'} ({report.gpu_backend})",
    ]
    if report.torch_version:
        lines.append(f"**Torch:** {report.torch_version} | CUDA: {report.torch_cuda}")
    for pkg in report.packages:
        status = "installed" if pkg.installed else "missing"
        ver = f" v{pkg.version}" if pkg.version else ""
        enabled = "enabled" if pkg.enabled else "disabled"
        lines.append(f"- **{pkg.name}**{ver}: {status} | {enabled}")
        lines.append(f"  - Feature: {pkg.feature}")
        if not pkg.installed:
            lines.append(f"  - Fallback: {pkg.fallback}")
        elif pkg.import_error:
            lines.append(f"  - Import error: {pkg.import_error}")
    return "\n".join(lines)
