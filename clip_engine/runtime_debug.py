# -*- coding: utf-8 -*-
"""Streamlit Runtime Debug panel (sidebar expander)."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from clip_engine.job_control import get_active_job, get_pipeline_step
from clip_engine.whisper_runtime import get_whisper_cache_state


def _get_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "not found"


def show_env_diagnostics() -> dict[str, Any]:
    """RT365-GPU-FIX 2026-06-05: startup GPU/env snapshot for sidebar diagnostics."""
    import ctranslate2

    diag: dict[str, Any] = {
        "sys.executable": sys.executable,
        "Python version": sys.version,
        "Venv path": os.environ.get("VIRTUAL_ENV", "NOT SET — may not be running in venv"),
        "faster-whisper version": _get_version("faster-whisper"),
        "ctranslate2 version": _get_version("ctranslate2"),
        "torch version": _get_version("torch"),
        "CUDA visible devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
        "ctranslate2 CUDA compute types": str(ctranslate2.get_supported_compute_types("cuda")),
    }
    try:
        import torch

        diag["torch.cuda.is_available()"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diag["CUDA device name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        diag["torch CUDA check"] = f"error: {e}"
    return diag


def render_env_diagnostics_panel() -> None:
    """Environment & GPU Diagnostics — runs once per Streamlit session."""
    with st.sidebar.expander("Environment & GPU Diagnostics", expanded=False):
        try:
            diag = show_env_diagnostics()
            for key, val in diag.items():
                st.markdown(f"**{key}:** `{val}`")
            try:
                from clip_engine.environment_check import is_project_venv_executable

                if is_project_venv_executable():
                    st.success("Running in project virtual environment (.venv or .venv311)")
                else:
                    st.warning(
                        "Not running in project venv — GPU transcription may use wrong interpreter."
                    )
            except Exception:
                pass
            exe_norm = sys.executable.replace("\\", "/").lower()
            if "/.venv/" in exe_norm or "/.venv311/" in exe_norm:
                st.caption(
                    "Windows note: nvidia-smi may show base Python311\\python.exe for this PID; "
                    "sys.executable above is authoritative."
                )
        except Exception as exc:
            st.error(f"Diagnostics failed: {exc}")


def _process_rss_mb() -> str:
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss
        return f"{rss / (1024 * 1024):.1f}"
    except Exception:
        return "n/a"


def _torch_info() -> dict[str, Any]:
    out: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "cuda_available": False,
        "device_name": None,
    }
    try:
        import torch

        out["installed"] = True
        out["version"] = torch.__version__
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def _nvidia_smi_gpu_line() -> str:
    from clip_engine.cuda_diagnostics import query_nvidia_gpu_memory_and_util

    mem, util = query_nvidia_gpu_memory_and_util()
    if mem is None and util is None:
        return "nvidia-smi unavailable"
    return f"memory.used={mem} MiB, utilization.gpu={util}% (CTranslate2 / Whisper)"


def _ctranslate2_version() -> str:
    try:
        import ctranslate2

        return str(getattr(ctranslate2, "__version__", "?"))
    except Exception:
        return "not installed"


def render_runtime_debug_panel() -> None:
    """Collapsible sidebar panel — updates on each Streamlit rerun."""
    with st.sidebar.expander("Runtime Debug", expanded=False):
        cache = get_whisper_cache_state()
        torch_i = _torch_info()
        active = get_active_job()
        step = get_pipeline_step()

        st.markdown(f"**Python executable:** `{sys.executable}`")
        st.markdown(f"**PID:** `{os.getpid()}`")
        st.markdown(f"**sys.prefix:** `{sys.prefix}`")
        if ".venv" in sys.executable.replace("\\", "/").lower():
            st.caption(
                "Windows note: nvidia-smi may show base Python311\\python.exe for this PID; "
                "sys.executable above is authoritative."
            )
        st.markdown(f"**ALLOW_CPU_FALLBACK:** `{os.environ.get('ALLOW_CPU_FALLBACK', '(unset → False)')}`")

        if torch_i["installed"]:
            st.markdown(f"**torch:** `{torch_i['version']}` (CUDA `{torch_i['cuda_version']}`)")
            st.markdown(f"**torch.cuda.is_available():** `{torch_i['cuda_available']}`")
            if torch_i["device_name"]:
                st.markdown(f"**GPU name:** `{torch_i['device_name']}`")
        else:
            st.markdown("**torch:** not installed")

        st.markdown(f"**GPU (nvidia-smi):** `{_nvidia_smi_gpu_line()}`")
        st.markdown(
            "Whisper VRAM/util: see nvidia-smi above (not `torch.cuda.memory_allocated`)."
        )

        st.markdown(f"**ctranslate2:** `{_ctranslate2_version()}`")
        st.markdown(
            f"**CT2_USE_EXPERIMENTAL_PACKED_GEMM:** "
            f"`{os.environ.get('CT2_USE_EXPERIMENTAL_PACKED_GEMM', '(unset)')}`"
        )
        st.markdown(
            f"**CT2_CUDA_ALLOW_FP16:** `{os.environ.get('CT2_CUDA_ALLOW_FP16', '(unset)')}`"
        )

        if cache["loaded"]:
            inner_dev = cache.get("inner_device") or "unknown"
            st.markdown(
                f"**Whisper model loaded:** Yes — size=`{cache['model_size']}` "
                f"requested_device=`{cache['device']}` inner_device=`{inner_dev}` "
                f"compute_type=`{cache['compute_type']}`"
            )
        else:
            st.markdown("**Whisper model loaded:** No (default size from config: large-v3)")

        st.markdown(f"**Active job:** `{active or '(idle)'}`")
        if step:
            st.markdown(f"**Pipeline step:** `{step}`")
        st.markdown(f"**Process RSS:** `{_process_rss_mb()}` MB")

        if st.button("Refresh GPU PID check", key="cs_runtime_debug_gpu_pid"):
            from clip_engine.cuda_diagnostics import gpu_pid_check

            on_gpu, detail = gpu_pid_check(context="debug_panel")
            if on_gpu:
                st.success(detail)
            elif ".venv" in sys.executable.replace("\\", "/").lower():
                st.info(detail)
            else:
                st.warning(detail)

        st.caption(
            "Benchmark (CLI): "
            f"`{sys.executable} -m clip_engine.transcription_benchmark`"
        )
        if st.button("Run 60s transcription benchmark", key="cs_runtime_debug_benchmark"):
            import subprocess

            with st.spinner("Running 60s CUDA transcription benchmark..."):
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "clip_engine.transcription_benchmark"],
                        capture_output=True,
                        text=True,
                        timeout=900,
                        cwd=str(Path(__file__).resolve().parent.parent),
                    )
                    out = (proc.stdout or "") + (proc.stderr or "")
                    if proc.returncode == 0:
                        st.success("Benchmark complete")
                    else:
                        st.error(f"Benchmark exit {proc.returncode}")
                    st.code(out[-8000:] if len(out) > 8000 else out)
                except subprocess.TimeoutExpired:
                    st.error("Benchmark timed out after 900s")
                except Exception as exc:
                    st.error(f"Benchmark failed: {exc}")
