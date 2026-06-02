# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import re
import threading
import time
from pathlib import Path
from typing import Literal
from clip_engine.captions import CaptionPreset, DEFAULT_PRESET, segments_to_ass, write_sidecar_files
from clip_engine.smart_crop import ExportMode, OUTPUT_W, OUTPUT_H, build_vf_filter, validate_output_resolution

logger = logging.getLogger("clip_engine.export_vertical")

# Sequential export controls — prevents GPU/memory exhaustion from concurrent ffmpeg processes.
_EXPORT_LOCK = threading.Lock()
_EXPORT_INTER_DELAY = 2.0          # seconds between clip exports
_GPU_MEMORY_HEADROOM_GB = 18.0     # RTX 4090: wait only when VRAM is genuinely tight
_FFMPEG_TIMEOUT_SECONDS = 300      # per-clip ffmpeg hard timeout


def _check_gpu_headroom() -> None:
    """If GPU memory usage exceeds threshold, wait 5 seconds before proceeding."""
    try:
        from clip_engine.subprocess_guard import run_subprocess

        result = run_subprocess(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            label="nvidia-smi",
        )
        if result.returncode == 0:
            used_mb = int(result.stdout.strip().split("\n")[0].strip())
            used_gb = used_mb / 1024
            if used_gb > _GPU_MEMORY_HEADROOM_GB:
                logger.warning(
                    "GPU memory %.1fGB exceeds %.1fGB threshold — waiting 5s before export",
                    used_gb, _GPU_MEMORY_HEADROOM_GB,
                )
                time.sleep(5.0)
    except Exception:
        pass

EXPORT_MODE_LABELS: dict[str, ExportMode] = {
    "Full frame fit with blurred background": "full_fit",
    "Smart crop people/faces": "smart_crop",
    "Center crop": "center_crop",
}

PREVIEW_SCALE_W = 540
PREVIEW_SCALE_H = 960
PREVIEW_MAX_DURATION = 15.0  # seconds max for preview


def export_vertical_clip_with_captions(
    video_path: Path, output_path: Path, start: float, end: float, segments: list[dict], *,
    prefer_gpu: bool = True, force_gpu_export: bool = False, allow_cpu_fallback: bool = True,
    caption_preset: CaptionPreset = DEFAULT_PRESET, export_mode: ExportMode = "full_fit",
    write_sidecars: bool = True, smart_crop: bool = True,
    advanced_captions: bool = False, dynamic_smart_crop: bool = True,
    preview_mode: bool = False,
) -> dict:
    with _EXPORT_LOCK:
        _check_gpu_headroom()
        try:
            return _export_vertical_clip_impl(
                video_path, output_path, start, end, segments,
                prefer_gpu=prefer_gpu, force_gpu_export=force_gpu_export,
                allow_cpu_fallback=allow_cpu_fallback, caption_preset=caption_preset,
                export_mode=export_mode, write_sidecars=write_sidecars,
                smart_crop=smart_crop, advanced_captions=advanced_captions,
                dynamic_smart_crop=dynamic_smart_crop, preview_mode=preview_mode,
            )
        finally:
            if not preview_mode:
                time.sleep(_EXPORT_INTER_DELAY)


def _export_vertical_clip_impl(
    video_path: Path, output_path: Path, start: float, end: float, segments: list[dict], *,
    prefer_gpu: bool = True, force_gpu_export: bool = False, allow_cpu_fallback: bool = True,
    caption_preset: CaptionPreset = DEFAULT_PRESET, export_mode: ExportMode = "full_fit",
    write_sidecars: bool = True, smart_crop: bool = True,
    advanced_captions: bool = False, dynamic_smart_crop: bool = True,
    preview_mode: bool = False,
) -> dict:
    from clip_engine.telemetry import log_export_event, log_gpu_memory, pipeline_phase

    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid clip range: {start:.2f} to {end:.2f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_export = time.perf_counter()
    if not smart_crop and export_mode == "smart_crop":
        export_mode = "full_fit"

    effective_end = end
    if preview_mode:
        effective_end = min(end, start + PREVIEW_MAX_DURATION)
        duration = effective_end - start

    clip_segs = [s for s in segments if float(s.get("end", 0)) > start and float(s.get("start", 0)) < effective_end]
    ass_file = output_path.with_suffix("._tmp.ass")
    ass_file.write_text(
        segments_to_ass(
            clip_segs, clip_start=start, preset=caption_preset, advanced=advanced_captions,
        ),
        encoding="utf-8",
    )
    clip_title = output_path.stem
    try:
        log_gpu_memory("before_export")
        vf, uses_complex = build_vf_filter(
            video_path,
            mode=export_mode,
            ass_path=ass_file,
            clip_start=start,
            clip_end=effective_end,
            dynamic_crop=dynamic_smart_crop and export_mode == "smart_crop",
        )
        if preview_mode:
            vf = _apply_preview_scale(vf, uses_complex)
        encoder, cmd = _build_cmd(
            video_path, output_path, start, duration, vf, uses_complex,
            prefer_gpu, force_gpu_export, preview_mode=preview_mode,
        )
        with pipeline_phase("exports"):
            encoder_used = _run_with_fallback(
                cmd, video_path, output_path, start, duration, vf, uses_complex,
                encoder, allow_cpu_fallback, preview_mode=preview_mode,
            )
        out_w, out_h = validate_output_resolution(output_path)
        elapsed = time.perf_counter() - t_export
        size_mb = None
        try:
            size_mb = output_path.stat().st_size / (1024 * 1024)
        except OSError:
            pass
        if not preview_mode:
            log_export_event(
                clip_title=clip_title,
                duration_sec=duration,
                resolution=f"{out_w}x{out_h}",
                encoder=encoder_used,
                elapsed_sec=elapsed,
                subtitle_burn=True,
                size_mb=size_mb,
                output_path=str(output_path),
            )
        log_gpu_memory("after_exports")
        expected_w = PREVIEW_SCALE_W if preview_mode else OUTPUT_W
        expected_h = PREVIEW_SCALE_H if preview_mode else OUTPUT_H
        if out_w != expected_w or out_h != expected_h:
            logger.warning("Resolution mismatch: got %dx%d expected %dx%d", out_w, out_h, expected_w, expected_h)
    finally:
        try:
            ass_file.unlink(missing_ok=True)
        except Exception:
            pass
    sidecar_files = {}
    if write_sidecars and not preview_mode:
        sidecar_files = write_sidecar_files(
            out_dir=output_path.parent, base_name=output_path.stem,
            segments=clip_segs, clip_start=start, preset=caption_preset,
            advanced=advanced_captions,
        )
    return {
        "output_path": output_path,
        "export_mode": export_mode,
        "encoder_used": encoder_used,
        "resolution": f"{out_w}x{out_h}",
        "sidecar_files": sidecar_files,
        "preview": preview_mode,
    }


def export_clip_preview(
    video_path: Path,
    output_path: Path,
    start: float,
    end: float,
    segments: list[dict],
    *,
    caption_preset: CaptionPreset = DEFAULT_PRESET,
    export_mode: ExportMode = "full_fit",
    advanced_captions: bool = False,
    dynamic_smart_crop: bool = True,
    prefer_gpu: bool = True,
    allow_cpu_fallback: bool = True,
) -> dict:
    """Render a low-resolution short preview clip (faster than final export)."""
    return export_vertical_clip_with_captions(
        video_path, output_path, start, end, segments,
        prefer_gpu=prefer_gpu,
        allow_cpu_fallback=allow_cpu_fallback,
        caption_preset=caption_preset,
        export_mode=export_mode,
        write_sidecars=False,
        advanced_captions=advanced_captions,
        dynamic_smart_crop=dynamic_smart_crop,
        preview_mode=True,
    )


def _apply_preview_scale(vf: str, uses_complex: bool) -> str:
    """Downscale output to 540x960 for faster preview rendering."""
    scale = f"scale={PREVIEW_SCALE_W}:{PREVIEW_SCALE_H}"
    if uses_complex:
        if vf.endswith("[vout]"):
            return vf.replace("[vout]", f",{scale}[vout]")
        return f"{vf},{scale}[vout]"
    return f"{vf},{scale}"


def _build_cmd(video_path, output_path, start, duration, vf, uses_complex, prefer_gpu, force_gpu_export, *, preview_mode=False):
    from clip_engine.ffmpeg_gpu import should_attempt_nvenc_on_export
    use_nvenc = should_attempt_nvenc_on_export(prefer_gpu=prefer_gpu, force_gpu_mode=force_gpu_export)
    encoder = "h264_nvenc" if use_nvenc else "libx264"
    base = ["ffmpeg", "-y", "-ss", str(start), "-i", str(video_path), "-t", str(duration)]
    base += ["-filter_complex" if uses_complex else "-vf", vf]
    if uses_complex:
        base += ["-map", "[vout]", "-map", "0:a?"]
    base += ["-c:a", "aac", "-b:a", "128k" if preview_mode else "192k", "-ar", "44100"]
    if encoder == "h264_nvenc":
        cq = "28" if preview_mode else "23"
        bv = "2M" if preview_mode else "4M"
        base += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", cq,
                 "-b:v", bv, "-maxrate", bv, "-bufsize", bv, "-pix_fmt", "yuv420p"]
    else:
        crf = "28" if preview_mode else "23"
        preset = "veryfast" if preview_mode else "fast"
        base += ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]
    return encoder, base + [str(output_path)]


def export_filename_stem(clip: dict, sequence_index: int, title_slug: str) -> str:
    """
    Build export file stem (without _9x16.mp4 suffix).
    Series parts include Part1/Part2 in the slug segment.
    """
    if clip.get("is_part_of_series"):
        part_num = int(clip.get("part_number", 1))
        return f"{sequence_index:02d}_{title_slug}_Part{part_num}"
    return f"{sequence_index:02d}_{title_slug}"


def _run_with_fallback(cmd, video_path, output_path, start, duration, vf, uses_complex, encoder, allow_cpu_fallback, *, preview_mode=False):
    from clip_engine.job_control import set_pipeline_step
    from clip_engine.subprocess_guard import run_subprocess

    set_pipeline_step("ffmpeg_export")
    cmd = _resolve_ffmpeg_binary(list(cmd))
    result = run_subprocess(
        cmd,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
        label="ffmpeg_export",
        text=True,
    )
    if result.returncode == 0:
        return encoder
    if encoder == "h264_nvenc" and allow_cpu_fallback:
        logger.warning("NVENC failed, retrying with libx264.")
        _, cpu_cmd = _build_cmd(
            video_path, output_path, start, duration, vf, uses_complex,
            False, False, preview_mode=preview_mode,
        )
        cpu_cmd = _resolve_ffmpeg_binary(list(cpu_cmd))
        result2 = run_subprocess(
            cpu_cmd,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
            label="ffmpeg_export_cpu_fallback",
            text=True,
        )
        if result2.returncode == 0:
            return "libx264"
        raise RuntimeError(f"CPU fallback failed: {(result2.stderr or '')[-1000:]}")
    raise RuntimeError(f"FFmpeg failed: {(result.stderr or '')[-1000:]}")


def _resolve_ffmpeg_binary(cmd: list) -> list:
    """Replace leading 'ffmpeg' with resolved executable path."""
    if cmd and cmd[0] == "ffmpeg":
        from clip_engine.ffmpeg_resolve import get_ffmpeg_executable

        return [get_ffmpeg_executable(), *cmd[1:]]
    return cmd


def export_clips_sequential_safe(
    clips: list[dict],
    video_path: Path,
    output_dir: Path,
    segments: list[dict],
    *,
    prefer_gpu: bool = True,
    allow_cpu_fallback: bool = True,
    caption_preset: CaptionPreset = DEFAULT_PRESET,
    export_mode: ExportMode = "full_fit",
    advanced_captions: bool = False,
    dynamic_smart_crop: bool = True,
) -> list[dict]:
    """Export clips one at a time. Failed clips are logged and marked, not re-raised."""
    results: list[dict] = []
    for i, clip in enumerate(clips):
        title = str(clip.get("hook_title") or clip.get("export_title") or f"clip_{i+1}")
        slug = re.sub(r"[^\w\s-]", "", title)[:40].strip().replace(" ", "_")
        output_path = output_dir / f"{i+1:02d}_{slug}_9x16.mp4"
        start = float(clip.get("start_seconds", clip.get("start", 0)))
        end = float(clip.get("end_seconds", clip.get("end", 0)))
        try:
            result = export_vertical_clip_with_captions(
                video_path, output_path, start, end, segments,
                prefer_gpu=prefer_gpu, allow_cpu_fallback=allow_cpu_fallback,
                caption_preset=caption_preset, export_mode=export_mode,
                advanced_captions=advanced_captions, dynamic_smart_crop=dynamic_smart_crop,
            )
            result["clip_index"] = i
            result["failed"] = False
            results.append(result)
            logger.info("Export %d/%d complete: %s", i + 1, len(clips), output_path.name)
        except Exception as exc:
            logger.error("Export %d/%d failed (%s): %s", i + 1, len(clips), output_path.name, exc)
            results.append({
                "clip_index": i,
                "failed": True,
                "error": str(exc),
                "output_path": output_path,
            })
    return results
