# -*- coding: utf-8 -*-
"""DaVinci Resolve send panel for RT365 AI Clip Studio."""
from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import streamlit as st

from clip_engine.resolve_export import build_edl

logger = logging.getLogger("clip_studio")


def _clips_for_resolve_payload(clips: list) -> list[dict]:
    out: list[dict] = []
    for idx, c in enumerate(clips):
        wid = c.get("_wid", str(idx))
        t0 = float(st.session_state.get(f"start_{wid}", c.get("start_seconds", c.get("start", 0))))
        t1 = float(st.session_state.get(f"end_{wid}", c.get("end_seconds", c.get("end", 0))))
        hook = str(
            st.session_state.get(
                f"hook_widget_{wid}",
                st.session_state.get(f"hook_{wid}", c.get("hook_title", f"clip_{idx + 1}")),
            )
        ).strip()
        clip = dict(c)
        clip["start_time"] = t0
        clip["end_time"] = t1
        clip["hook_title"] = hook
        out.append(clip)
    return out


def render_resolve_panel() -> None:
    """Render the Send to DaVinci Resolve section."""
    if not (st.session_state.get("final_clips") and st.session_state.get("source_video_path")):
        return

    st.subheader("Send to DaVinci Resolve")
    fps_options = [24.0, 29.97, 30.0, 50.0, 60.0]
    fps_labels = ["24fps", "29.97fps (NTSC)", "30fps", "50fps", "60fps"]
    fps_to_project = {24.0: "24", 29.97: "29.97 DF", 30.0: "30", 50.0: "50", 60.0: "60"}
    resolve_col1, resolve_col2, resolve_col3 = st.columns(3)
    with resolve_col1:
        handle_secs = st.slider(
            "Handles (seconds each end)",
            min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            key="resolve_handle_secs",
        )
    with resolve_col2:
        fps_choice = st.selectbox(
            "Timeline FPS",
            options=fps_options,
            index=1,
            format_func=lambda x: fps_labels[fps_options.index(x)],
            key="resolve_fps_choice",
        )
    with resolve_col3:
        timeline_name = st.text_input(
            "Timeline name",
            value=Path(st.session_state["source_video_path"]).stem + "_AI",
            key="resolve_timeline_name",
        )

    if st.button("🚀 Send to DaVinci Resolve", type="primary", key="resolve_send_btn"):
        resolve_clips = _clips_for_resolve_payload(st.session_state["final_clips"])
        payload = {
            "source_path": st.session_state["source_video_path"],
            "clips": resolve_clips,
            "fps": fps_choice,
            "handle_seconds": handle_secs,
            "timeline_name": timeline_name,
            "project_fps": fps_to_project.get(fps_choice, "30"),
            "color_tag": "Blue",
        }
        candidates = [
            r"C:\Users\rzric\AppData\Local\Programs\Python\Python311\python.exe",
            r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Python\python.exe",
            r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Python3\python.exe",
            r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Python310\python.exe",
            sys.executable,
        ]
        resolve_python = next((p for p in candidates if Path(p).exists()), sys.executable)
        st.session_state["resolve_python_path"] = resolve_python
        try:
            from clip_engine.subprocess_guard import run_subprocess_with_input

            bridge = Path(__file__).parent.parent / "resolve_bridge.py"
            result = run_subprocess_with_input(
                [resolve_python, str(bridge)],
                input_text=json.dumps(payload),
                timeout=30.0,
                label="resolve_bridge",
                text=True,
            )
            if result.returncode == 0:
                try:
                    resp = json.loads(result.stdout.strip() or "{}")
                except json.JSONDecodeError:
                    resp = {"status": "error", "message": result.stdout or "Invalid JSON from bridge"}
                if resp.get("status") == "ok":
                    st.success(
                        f"Timeline **{resp.get('timeline_name', timeline_name)}** created with "
                        f"**{resp.get('clips_placed', len(resolve_clips))}** clip(s) in DaVinci Resolve."
                    )
                    with st.expander("📋 Build log", expanded=False):
                        for line in resp.get("log") or []:
                            st.text(line)
                else:
                    st.error(resp.get("message", "Unknown error from Resolve bridge."))
                    st.info(
                        "- DaVinci Resolve Studio is open\n"
                        "- Preferences → System → General → Enable Fusion page scripting is checked\n"
                        "- You are using DaVinci Resolve Studio (not the free version)"
                    )
            else:
                try:
                    resp = json.loads(result.stdout.strip() or result.stderr.strip() or "{}")
                    err_msg = resp.get("message", result.stderr or result.stdout or "Bridge failed")
                except json.JSONDecodeError:
                    err_msg = result.stderr or result.stdout or "Bridge failed"
                st.error(err_msg)
                st.info(
                    "- DaVinci Resolve Studio is open\n"
                    "- Preferences → System → General → Enable Fusion page scripting is checked\n"
                    "- You are using DaVinci Resolve Studio (not the free version)"
                )
        except subprocess.TimeoutExpired:
            st.error("Timed out connecting to Resolve (30s). Is Resolve open?")

    with st.expander("📄 Export EDL instead (DaVinci Resolve free version)"):
        if st.button("Generate EDL file", key="resolve_edl_generate"):
            st.session_state["resolve_edl_ready"] = False
            st.session_state.pop("resolve_edl_text", None)
            edl_clips = _clips_for_resolve_payload(st.session_state["final_clips"])
            st.write(f"Generating EDL for {len(edl_clips)} clips...")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        build_edl,
                        edl_clips,
                        st.session_state["source_video_path"],
                        fps=fps_choice,
                        handle_seconds=handle_secs,
                        title=timeline_name,
                    )
                    edl_text = future.result(timeout=5.0)
            except FuturesTimeoutError:
                st.error("EDL generation timed out after 5 seconds.")
            except Exception as exc:
                st.error(f"EDL generation failed: {exc}")
            else:
                st.session_state["resolve_edl_text"] = edl_text
                edl_stem = Path(st.session_state["source_video_path"]).stem
                st.session_state["resolve_edl_filename"] = f"{edl_stem}_resolve.edl"
                st.session_state["resolve_edl_ready"] = True
                st.success("EDL ready to download.")
        if st.session_state.get("resolve_edl_ready") and st.session_state.get("resolve_edl_text"):
            st.download_button(
                label="Download EDL file",
                data=st.session_state["resolve_edl_text"],
                file_name=st.session_state.get("resolve_edl_filename", "clips_resolve.edl"),
                mime="text/plain",
                key="resolve_edl_download",
            )
