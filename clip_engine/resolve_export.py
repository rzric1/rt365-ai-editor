# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, math, re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def seconds_to_tc(seconds: float, fps: float = 25.0) -> str:
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    frames = total_frames % int(fps)
    total_secs = total_frames // int(fps)
    secs = total_secs % 60
    total_mins = total_secs // 60
    mins = total_mins % 60
    hours = total_mins // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

def sanitize_reel_name(path: str) -> str:
    stem = Path(path).stem
    clean = re.sub(r"[^A-Za-z0-9]", "", stem)
    return clean[:8].upper() or "SOURCE01"

def build_edl(clips, source_path, fps=25.0, handle_seconds=2.0, title="AI Clip Export"):
    reel = sanitize_reel_name(source_path)
    source_filename = Path(source_path).name
    lines = []
    lines.append(f"TITLE: {title}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")
    record_cursor = 3600.0
    kept_clips = [c for c in clips if c.get("finalizer_action","kept") not in ("rejected","rejected_before_ui") and c.get("start_time") is not None and c.get("end_time") is not None]
    if not kept_clips:
        logger.warning("[EDL] No kept clips to export")
    for i, clip in enumerate(kept_clips, start=1):
        src_in = max(0.0, float(clip["start_time"]) - handle_seconds)
        src_out = float(clip["end_time"]) + handle_seconds
        duration = src_out - src_in
        src_in_tc  = seconds_to_tc(src_in,  fps)
        src_out_tc = seconds_to_tc(src_out, fps)
        rec_in_tc  = seconds_to_tc(record_cursor, fps)
        rec_out_tc = seconds_to_tc(record_cursor + duration, fps)
        hook = clip.get("hook_title") or clip.get("title") or f"Clip {i:02d}"
        score = clip.get("score") or clip.get("virality_score") or 0
        action = clip.get("finalizer_action","kept")
        lines.append(f"* FROM CLIP NAME: {hook[:60]}")
        lines.append(f"* SCORE: {score:.1f}  ACTION: {action}  SRC: {source_filename}")
        lines.append(f"{i:03d}  {reel:<8}  V     C        {src_in_tc}  {src_out_tc}  {rec_in_tc}  {rec_out_tc}")
        lines.append("")
        record_cursor += duration
    logger.info(f"[EDL] Built {len(kept_clips)} events")
    return "\n".join(lines)

def save_edl(edl_text, output_path):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(edl_text, encoding="utf-8")
    logger.info(f"[EDL] Saved to {out}")
    return out

def default_edl_path(source_path, output_dir=None):
    src = Path(source_path)
    if output_dir:
        return Path(output_dir) / f"{src.stem}_resolve.edl"
    return src.parent / f"{src.stem}_resolve.edl"
