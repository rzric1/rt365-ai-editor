from __future__ import annotations
import logging
import subprocess
from pathlib import Path
from typing import Literal
from clip_engine.captions import CaptionPreset, DEFAULT_PRESET, segments_to_ass, write_sidecar_files
from clip_engine.smart_crop import ExportMode, OUTPUT_W, OUTPUT_H, build_vf_filter, validate_output_resolution

logger = logging.getLogger("clip_engine.export_vertical")

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
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid clip range: {start:.2f} to {end:.2f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    try:
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
        encoder_used = _run_with_fallback(
            cmd, video_path, output_path, start, duration, vf, uses_complex,
            encoder, allow_cpu_fallback, preview_mode=preview_mode,
        )
        out_w, out_h = validate_output_resolution(output_path)
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


def _run_with_fallback(cmd, video_path, output_path, start, duration, vf, uses_complex, encoder, allow_cpu_fallback, *, preview_mode=False):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        return encoder
    if encoder == "h264_nvenc" and allow_cpu_fallback:
        logger.warning("NVENC failed, retrying with libx264.")
        _, cpu_cmd = _build_cmd(
            video_path, output_path, start, duration, vf, uses_complex,
            False, False, preview_mode=preview_mode,
        )
        result2 = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=600)
        if result2.returncode == 0:
            return "libx264"
        raise RuntimeError(f"CPU fallback failed: {result2.stderr[-1000:]}")
    raise RuntimeError(f"FFmpeg failed: {result.stderr[-1000:]}")
