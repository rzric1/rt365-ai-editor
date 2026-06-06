# -*- coding: utf-8 -*-
"""Job status and cancel controls for Clip Studio stability."""

from __future__ import annotations

import streamlit as st

from clip_engine.job_control import (
    get_active_job,
    request_cancel,
    release_job,
)
from clip_engine.stability import log_resource_snapshot
from clip_engine.subprocess_guard import (
    find_orphan_ffmpeg_pids,
    list_tracked_pids,
    terminate_all_tracked,
    terminate_orphan_ffmpeg,
)


def render_stability_controls() -> None:
    """Sidebar block: active job, cancel, child process warning."""
    active = get_active_job()
    if active:
        st.warning(f"Active job: **{active}**")
        if st.button("Cancel current job", type="secondary", key="cs_cancel_job"):
            request_cancel()
            n = terminate_all_tracked()
            orphans = terminate_orphan_ffmpeg()
            release_job(active)
            st.session_state.cs_status = (
                f"Cancelled {active} ({n} tracked + {orphans} orphan ffmpeg stopped)."
            )
            st.rerun()
    else:
        st.caption("No long-running job active.")

    if st.button("Clear session memory (transcript/clips)", key="cs_clear_session_ram"):
        from ui.session_memory import clear_session_heavy_data

        cleared = clear_session_heavy_data(keep_video_path=True)
        st.session_state.cs_status = f"Cleared {len(cleared)} in-memory session key(s)."
        st.rerun()

    if st.button("Refresh resource snapshot", key="cs_resource_snap"):
        snap = log_resource_snapshot(label="manual")
        st.session_state["_cs_resource_snap_data"] = snap
        st.caption(f"Logged snapshot (RSS {snap.get('process_rss_mb', '?')} MB).")

    snap = st.session_state.get("_cs_resource_snap_data")
    if snap:
        st.caption(
            f"Last snapshot: CPU {snap.get('cpu_percent', '?')}% | "
            f"RAM avail {snap.get('ram_available_gb', '?')} GB | "
            f"VRAM alloc {snap.get('vram_allocated_gb', '?')} GB"
        )

    pids = list_tracked_pids()
    if pids:
        st.error(f"Tracked child processes: {pids}. Click Cancel or restart the app.")

    orphan_pids = find_orphan_ffmpeg_pids()
    if orphan_pids:
        st.error(f"Orphan ffmpeg (untracked): {orphan_pids}.")
        if st.button("Kill orphan ffmpeg", key="cs_kill_orphan_ffmpeg"):
            k = terminate_orphan_ffmpeg()
            st.session_state.cs_status = f"Terminated {k} orphan ffmpeg process(es)."
            st.rerun()
