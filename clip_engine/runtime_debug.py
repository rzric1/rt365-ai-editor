# -*- coding: utf-8 -*-
"""Streamlit Runtime Debug panel (sidebar expander)."""

from __future__ import annotations

import os
import sys
from typing import Any

import streamlit as st

from clip_engine.job_control import get_active_job, get_pipeline_step
from clip_engine.whisper_runtime import get_whisper_cache_state


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
        "vram_gb": None,
    }
    try:
        import torch

        out["installed"] = True
        out["version"] = torch.__version__
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["device_name"] = torch.cuda.get_device_name(0)
            out["vram_gb"] = f"{torch.cuda.memory_allocated(0) / 1e9:.3f}"
    except Exception:
        pass
    return out


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
        st.markdown(f"**ALLOW_CPU_FALLBACK:** `{os.environ.get('ALLOW_CPU_FALLBACK', '(unset → False)')}`")

        if torch_i["installed"]:
            st.markdown(f"**torch:** `{torch_i['version']}` (CUDA `{torch_i['cuda_version']}`)")
            st.markdown(f"**torch.cuda.is_available():** `{torch_i['cuda_available']}`")
            if torch_i["device_name"]:
                st.markdown(f"**GPU name:** `{torch_i['device_name']}`")
            if torch_i["vram_gb"] is not None:
                st.markdown(f"**VRAM allocated:** `{torch_i['vram_gb']}` GB")
        else:
            st.markdown("**torch:** not installed")

        st.markdown(f"**ctranslate2:** `{_ctranslate2_version()}`")

        if cache["loaded"]:
            st.markdown(
                f"**Whisper model loaded:** Yes — `{cache['model_size']}` "
                f"device=`{cache['device']}` compute_type=`{cache['compute_type']}`"
            )
        else:
            st.markdown("**Whisper model loaded:** No")

        st.markdown(f"**Active job:** `{active or '(idle)'}`")
        if step:
            st.markdown(f"**Pipeline step:** `{step}`")
        st.markdown(f"**Process RSS:** `{_process_rss_mb()}` MB")

        if st.button("Refresh GPU PID check", key="cs_runtime_debug_gpu_pid"):
            from clip_engine.cuda_diagnostics import gpu_pid_check

            on_gpu, detail = gpu_pid_check(context="debug_panel")
            if on_gpu:
                st.success(detail)
            else:
                st.warning(detail)
