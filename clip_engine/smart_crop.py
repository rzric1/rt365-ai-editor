"""
clip_engine/smart_crop.py
Three export modes for vertical (9:16) output:
  1. full_fit   — full 16:9 frame scaled to 1080 wide, blurred background fills 1080x1920
  2. smart_crop — face/people detection, crop window keeps everyone in frame
  3. center_crop — simple center 9:16 crop (user-selected fallback)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal, NamedTuple

logger = logging.getLogger("clip_engine.smart_crop")

ExportMode = Literal["full_fit", "smart_crop", "center_crop"]

OUTPUT_W = 1080
OUTPUT_H = 1920


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CropRegion(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    method: str   # "face" | "center" | "full_fit"


# ---------------------------------------------------------------------------
# Public API — build ffmpeg vf filter string for chosen mode
# ---------------------------------------------------------------------------

def build_vf_filter(
    video_path: Path,
    mode: ExportMode = "full_fit",
    ass_path: Path | None = None,
) -> tuple[str, bool]:
    """
    Return (filter_string, uses_complex) for the chosen export mode.
    Always outputs exactly 1080x1920.
    uses_complex=True -> caller must pass -filter_complex and -map [vout].
    uses_complex=False -> caller must pass -vf (simple filter chain).
    """
    if mode == "full_fit":
        vf = _full_fit_filter()
        uses_complex = True
    elif mode == "smart_crop":
        vf = _smart_crop_filter(video_path)
        uses_complex = "split=" in vf
    else:
        vf = _center_crop_filter(video_path)
        uses_complex = False

    if ass_path:
        safe = str(ass_path).replace("\\", "/").replace(":", "\\:")
        sub = f"subtitles='{safe}'"
        if uses_complex:
            vf = f"{vf}[vpre];[vpre]{sub}[vout]"
        else:
            vf = f"{vf},{sub}"
    elif uses_complex:
        vf = f"{vf}[vout]"

    logger.info("vf filter [mode=%s complex=%s]: %s", mode, uses_complex, vf[:200])
    return vf, uses_complex


# ---------------------------------------------------------------------------
# Mode 1: Full-frame fit with blurred background (DEFAULT / SAFE)
# ---------------------------------------------------------------------------

def _full_fit_filter() -> str:
    """
    Preserve entire 16:9 frame. Scale to 1080 wide.
    Fill 1080x1920 with a blurred/dimmed copy of the same frame as background.
    No cropping, no distortion.

    Filter graph:
      [bg]  scale to 1080x1920 (stretch + blur) → dimmed background
      [fg]  scale to 1080 wide keeping aspect ratio → foreground
      overlay fg centered on bg
    """
    bg = (
        f"[0:v]scale={OUTPUT_W}:{OUTPUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_W}:{OUTPUT_H},"
        f"boxblur=20:1,"
        f"eq=brightness=-0.15[bg]"
    )
    fg = f"[0:v]scale={OUTPUT_W}:-2[fg]"
    overlay = f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
    # Chain as filtergraph with split
    return (
        f"split=2[base][orig];"
        f"[base]scale={OUTPUT_W}:{OUTPUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_W}:{OUTPUT_H},"
        f"boxblur=20:1,"
        f"eq=brightness=-0.15[bg];"
        f"[orig]scale={OUTPUT_W}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )


# ---------------------------------------------------------------------------
# Mode 2: Smart crop — face/people detection
# ---------------------------------------------------------------------------

def _smart_crop_filter(video_path: Path) -> str:
    """
    Try face detection. Build a crop window that keeps all detected faces in frame.
    Falls back to full_fit if detection fails or confidence is low.
    """
    try:
        crop = _detect_people_crop(video_path)
        if crop is None:
            logger.info("Smart crop: no faces detected — falling back to full_fit")
            return _full_fit_filter()
        logger.info("Smart crop: crop=%dx%d at (%d,%d)", crop.w, crop.h, crop.x, crop.y)
        return f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},scale={OUTPUT_W}:{OUTPUT_H}"
    except ImportError:
        logger.info("Smart crop: OpenCV not installed — falling back to full_fit")
        return _full_fit_filter()
    except Exception as e:
        logger.warning("Smart crop failed (%s) — falling back to full_fit", e)
        return _full_fit_filter()


def _detect_people_crop(video_path: Path) -> CropRegion | None:
    """
    Sample frames, detect all faces, build a bounding box that contains ALL faces.
    Returns None if fewer than 3 frames had detections (low confidence).
    """
    import cv2  # type: ignore

    src_w, src_h = _probe_dimensions(video_path)
    if src_w == 0 or src_h == 0:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Sample every 3 seconds, up to 90 seconds of video
    step = max(1, int(fps * 3.0))
    max_frames = min(total_frames, int(fps * 90))

    # Collect per-frame bounding boxes that contain ALL faces
    frame_boxes: list[tuple[int, int, int, int]] = []  # (x1, y1, x2, y2) in source pixels

    for frame_idx in range(0, max_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            continue
        # Build bounding box covering ALL faces in this frame
        all_x1 = min(fx for fx, fy, fw, fh in faces)
        all_y1 = min(fy for fx, fy, fw, fh in faces)
        all_x2 = max(fx + fw for fx, fy, fw, fh in faces)
        all_y2 = max(fy + fh for fx, fy, fw, fh in faces)
        # Add generous padding so heads aren't cut off
        pad_x = int((all_x2 - all_x1) * 0.3)
        pad_y_top = int((all_y2 - all_y1) * 0.5)   # more headroom above
        pad_y_bot = int((all_y2 - all_y1) * 0.6)   # room for body below
        all_x1 = max(0, all_x1 - pad_x)
        all_y1 = max(0, all_y1 - pad_y_top)
        all_x2 = min(src_w, all_x2 + pad_x)
        all_y2 = min(src_h, all_y2 + pad_y_bot)
        frame_boxes.append((all_x1, all_y1, all_x2, all_y2))

    cap.release()

    # Require at least 3 confident detections
    if len(frame_boxes) < 3:
        return None

    # Median bounding box across all sampled frames (stable, not jumpy)
    def median_val(vals: list[int]) -> int:
        return sorted(vals)[len(vals) // 2]

    x1 = median_val([b[0] for b in frame_boxes])
    y1 = median_val([b[1] for b in frame_boxes])
    x2 = median_val([b[2] for b in frame_boxes])
    y2 = median_val([b[3] for b in frame_boxes])

    people_w = x2 - x1
    people_h = y2 - y1

    if people_w <= 0 or people_h <= 0:
        return None

    # Expand crop to exactly 9:16 ratio centered on the people bounding box
    target_ratio = OUTPUT_W / OUTPUT_H  # 9/16
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Start from the people region and expand to 9:16
    if people_w / people_h > target_ratio:
        # Width-constrained: set crop_w = people_w, derive crop_h
        crop_w = people_w
        crop_h = int(crop_w / target_ratio)
    else:
        # Height-constrained: set crop_h = people_h, derive crop_w
        crop_h = people_h
        crop_w = int(crop_h * target_ratio)

    # Clamp to source frame
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    cx = max(crop_w // 2, min(cx, src_w - crop_w // 2))
    cy = max(crop_h // 2, min(cy, src_h - crop_h // 2))

    x = cx - crop_w // 2
    y = cy - crop_h // 2

    return CropRegion(x=x, y=y, w=crop_w, h=crop_h, method="face")


# ---------------------------------------------------------------------------
# Mode 3: Center crop
# ---------------------------------------------------------------------------

def _center_crop_filter(video_path: Path) -> str:
    src_w, src_h = _probe_dimensions(video_path)
    target_ratio = OUTPUT_W / OUTPUT_H
    if src_w / src_h > target_ratio:
        crop_w = int(src_h * target_ratio)
        crop_h = src_h
    else:
        crop_w = src_w
        crop_h = int(src_w / target_ratio)
    x = (src_w - crop_w) // 2
    y = (src_h - crop_h) // 2
    return f"crop={crop_w}:{crop_h}:{x}:{y},scale={OUTPUT_W}:{OUTPUT_H}"


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _probe_dimensions(video_path: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception as e:
        logger.warning("ffprobe dimensions failed: %s", e)
    return 1920, 1080


def validate_output_resolution(output_path: Path) -> tuple[int, int]:
    """Use ffprobe to confirm exported file resolution. Returns (w, h)."""
    return _probe_dimensions(output_path)


# ---------------------------------------------------------------------------
# Legacy compat shims (used by old export_vertical imports)
# ---------------------------------------------------------------------------

def detect_speaker_crop(video_path: Path, target_w: int = OUTPUT_W, target_h: int = OUTPUT_H) -> CropRegion:
    """Legacy shim — returns a CropRegion for backward compatibility."""
    crop = _detect_people_crop(video_path)
    if crop:
        return crop
    src_w, src_h = _probe_dimensions(video_path)
    return CropRegion(x=(src_w - OUTPUT_W) // 2, y=(src_h - OUTPUT_H) // 2,
                      w=OUTPUT_W, h=OUTPUT_H, method="center")


def build_crop_filter(crop: CropRegion, output_w: int = OUTPUT_W, output_h: int = OUTPUT_H) -> str:
    """Legacy shim."""
    return f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},scale={output_w}:{output_h}"
