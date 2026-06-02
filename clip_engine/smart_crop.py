# -*- coding: utf-8 -*-
"""
clip_engine/smart_crop.py
Three export modes for vertical (9:16) output:
  1. full_fit   — full 16:9 frame scaled to 1080 wide, blurred background fills 1080x1920
  2. smart_crop — face/people detection (YOLO if available, else OpenCV), optional dynamic crop
  3. center_crop — simple center 9:16 crop (user-selected fallback)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
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
    method: str   # "yolo" | "face" | "center" | "full_fit"


@dataclass
class SmartCropResult:
    filter_string: str
    uses_complex: bool
    backend: str = "full_fit"
    confidence: float = 0.0
    fallback_reason: str = ""
    dynamic: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API — build ffmpeg vf filter string for chosen mode
# ---------------------------------------------------------------------------

def build_vf_filter(
    video_path: Path,
    mode: ExportMode = "full_fit",
    ass_path: Path | None = None,
    *,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    dynamic_crop: bool = True,
) -> tuple[str, bool]:
    """
    Return (filter_string, uses_complex) for the chosen export mode.
    Always outputs exactly 1080x1920.
    uses_complex=True -> caller must pass -filter_complex and -map [vout].
    uses_complex=False -> caller must pass -vf (simple filter chain).
    """
    crop_meta: SmartCropResult | None = None
    if mode == "full_fit":
        vf = _full_fit_filter()
        uses_complex = True
        crop_meta = SmartCropResult(vf, True, backend="full_fit")
    elif mode == "smart_crop":
        crop_meta = _smart_crop_filter(
            video_path,
            clip_start=clip_start,
            clip_end=clip_end,
            dynamic_crop=dynamic_crop,
        )
        vf = crop_meta.filter_string
        uses_complex = crop_meta.uses_complex
    else:
        vf = _center_crop_filter(video_path)
        uses_complex = False
        crop_meta = SmartCropResult(vf, False, backend="center_crop")

    if ass_path:
        safe = str(ass_path).replace("\\", "/").replace(":", "\\:")
        sub = f"subtitles='{safe}'"
        if uses_complex:
            vf = f"{vf}[vpre];[vpre]{sub}[vout]"
        else:
            vf = f"{vf},{sub}"
    elif uses_complex:
        vf = f"{vf}[vout]"

    if crop_meta:
        logger.info(
            "Smart crop backend=%s confidence=%.2f dynamic=%s fallback=%s",
            crop_meta.backend,
            crop_meta.confidence,
            crop_meta.dynamic,
            crop_meta.fallback_reason or "none",
        )
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

def _smart_crop_filter(
    video_path: Path,
    *,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    dynamic_crop: bool = True,
) -> SmartCropResult:
    """
    Try YOLO person detection, then OpenCV faces.
    Falls back to full_fit if detection fails or confidence is low.
    """
    try:
        trajectory, backend, confidence = _detect_people_trajectory(
            video_path,
            clip_start=clip_start,
            clip_end=clip_end,
            prefer_yolo=True,
        )
        if not trajectory:
            logger.info("Smart crop: no people detected — falling back to full_fit")
            return SmartCropResult(
                _full_fit_filter(), True,
                backend=backend or "opencv",
                confidence=confidence,
                fallback_reason="low detection confidence",
            )

        if dynamic_crop and len(trajectory) >= 3 and _trajectory_has_movement(trajectory):
            vf = _dynamic_crop_filter(trajectory, clip_start)
            logger.info(
                "Smart crop dynamic: %d keyframes backend=%s confidence=%.2f",
                len(trajectory), backend, confidence,
            )
            return SmartCropResult(
                vf, False, backend=backend, confidence=confidence, dynamic=True,
            )

        crop = _median_crop_from_trajectory(trajectory)
        logger.info("Smart crop static: crop=%dx%d at (%d,%d) backend=%s", crop.w, crop.h, crop.x, crop.y, backend)
        vf = f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},scale={OUTPUT_W}:{OUTPUT_H}"
        return SmartCropResult(vf, False, backend=backend, confidence=confidence)
    except ImportError as e:
        logger.info("Smart crop: dependency missing (%s) — falling back to full_fit", e)
        return SmartCropResult(
            _full_fit_filter(), True, backend="none",
            fallback_reason=f"import error: {e}",
        )
    except Exception as e:
        logger.warning("Smart crop failed (%s) — falling back to full_fit", e)
        return SmartCropResult(
            _full_fit_filter(), True, backend="error",
            fallback_reason=str(e),
        )


def _trajectory_has_movement(trajectory: list[tuple[float, CropRegion]], threshold: float = 40.0) -> bool:
    """True if crop center moves more than threshold pixels across trajectory."""
    if len(trajectory) < 2:
        return False
    centers = [(r.x + r.w // 2, r.y + r.h // 2) for _, r in trajectory]
    max_dx = max(abs(centers[i][0] - centers[i - 1][0]) for i in range(1, len(centers)))
    max_dy = max(abs(centers[i][1] - centers[i - 1][1]) for i in range(1, len(centers)))
    return max(max_dx, max_dy) >= threshold


def _median_crop_from_trajectory(trajectory: list[tuple[float, CropRegion]]) -> CropRegion:
    regions = [r for _, r in trajectory]

    def median_val(vals: list[int]) -> int:
        return sorted(vals)[len(vals) // 2]

    return CropRegion(
        x=median_val([r.x for r in regions]),
        y=median_val([r.y for r in regions]),
        w=median_val([r.w for r in regions]),
        h=median_val([r.h for r in regions]),
        method=regions[0].method,
    )


def _dynamic_crop_filter(
    trajectory: list[tuple[float, CropRegion]],
    clip_start: float,
) -> str:
    """
    Build ffmpeg crop filter with linear interpolation between keyframes.
    Uses relative time within clip (t offset by clip_start).
    """
    if len(trajectory) == 1:
        r = trajectory[0][1]
        return f"crop={r.w}:{r.h}:{r.x}:{r.y},scale={OUTPUT_W}:{OUTPUT_H}"

    # Smooth trajectory with 3-point moving average on centers
    smoothed: list[tuple[float, CropRegion]] = []
    for i, (t, r) in enumerate(trajectory):
        window = trajectory[max(0, i - 1): min(len(trajectory), i + 2)]
        avg_x = int(sum(w[1].x for w in window) / len(window))
        avg_y = int(sum(w[1].y for w in window) / len(window))
        smoothed.append((t, CropRegion(avg_x, avg_y, r.w, r.h, r.method)))

    rel_times = [max(0.0, t - clip_start) for t, _ in smoothed]
    regions = [r for _, r in smoothed]

    # Static crop if movement is small after smoothing
    if not _trajectory_has_movement(smoothed, threshold=25.0):
        r = _median_crop_from_trajectory(smoothed)
        return f"crop={r.w}:{r.h}:{r.x}:{r.y},scale={OUTPUT_W}:{OUTPUT_H}"

    # Build piecewise-linear x/y expressions
    def _interp_expr(values: list[int], times: list[float]) -> str:
        if len(values) == 1:
            return str(values[0])
        parts: list[str] = []
        for i in range(len(values) - 1):
            t0, t1 = times[i], times[i + 1]
            v0, v1 = values[i], values[i + 1]
            if t1 <= t0:
                continue
            # Linear interpolation: v0 + (v1-v0)*(t-t0)/(t1-t0)
            expr = f"{v0}+({v1}-{v0})*((t-{t0:.3f})/({t1-t0:.3f}))"
            if i == 0:
                parts.append(f"if(lt(t,{t1:.3f}),{expr},")
            elif i == len(values) - 2:
                parts.append(f"{expr})")
            else:
                parts.append(f"if(lt(t,{t1:.3f}),{expr},")
        result = "".join(parts)
        # Pad closing parens
        open_count = result.count("if(") - result.count(")")
        result += ")" * max(0, open_count)
        return result if result else str(values[0])

    w, h = regions[0].w, regions[0].h
    x_expr = _interp_expr([r.x for r in regions], rel_times)
    y_expr = _interp_expr([r.y for r in regions], rel_times)
    return f"crop={w}:{h}:x='{x_expr}':y='{y_expr}',scale={OUTPUT_W}:{OUTPUT_H}"


def _detect_people_trajectory(
    video_path: Path,
    *,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    prefer_yolo: bool = True,
) -> tuple[list[tuple[float, CropRegion]], str, float]:
    """
    Sample frames throughout clip, detect people, return trajectory + backend + confidence.
    """
    confidence = 0.0
    if prefer_yolo:
        try:
            traj, conf = _yolo_detect_trajectory(video_path, clip_start, clip_end)
            if traj and len(traj) >= 2:
                return traj, "yolo", conf
            confidence = conf
        except ImportError:
            logger.debug("ultralytics not installed — trying OpenCV")
        except Exception as e:
            logger.debug("YOLO detection failed: %s", e)

    traj, conf = _opencv_detect_trajectory(video_path, clip_start, clip_end)
    if traj:
        return traj, "opencv", conf
    return [], "opencv", confidence


_yolo_model: object | None = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO  # type: ignore

        _yolo_model = YOLO("yolov8n.pt")
        logger.info("[smart_crop] loaded YOLOv8n (cached for session)")
    return _yolo_model


def _yolo_detect_trajectory(
    video_path: Path,
    clip_start: float,
    clip_end: float | None,
) -> tuple[list[tuple[float, CropRegion]], float]:
    import cv2  # type: ignore
    from clip_engine.job_control import check_cancelled

    src_w, src_h = _probe_dimensions(video_path)
    if src_w == 0:
        return [], 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_time = clip_end if clip_end is not None else total_frames / fps
    end_time = min(end_time, total_frames / fps)
    duration = max(1.0, end_time - clip_start)

    model = _get_yolo_model()
    step_sec = max(2.0, duration / 12)  # ~12 samples across clip
    trajectory: list[tuple[float, CropRegion]] = []
    confidences: list[float] = []

    t = clip_start
    while t < end_time:
        check_cancelled()
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, classes=[0])  # class 0 = person
        boxes = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= 0.35:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

        if boxes:
            crop = _boxes_to_crop_region(boxes, src_w, src_h, method="yolo")
            avg_conf = sum(b[4] for b in boxes) / len(boxes)
            trajectory.append((t, crop))
            confidences.append(avg_conf)
        t += step_sec

    cap.release()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return trajectory, avg_conf


def _opencv_detect_trajectory(
    video_path: Path,
    clip_start: float,
    clip_end: float | None,
) -> tuple[list[tuple[float, CropRegion]], float]:
    import cv2  # type: ignore

    src_w, src_h = _probe_dimensions(video_path)
    if src_w == 0:
        return [], 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_time = clip_end if clip_end is not None else total_frames / fps
    end_time = min(end_time, total_frames / fps)
    duration = max(1.0, end_time - clip_start)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    step_sec = max(2.0, duration / 10)
    trajectory: list[tuple[float, CropRegion]] = []
    detections = 0

    t = clip_start
    while t < end_time:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            boxes = [(fx, fy, fx + fw, fy + fh, 0.6) for fx, fy, fw, fh in faces]
            crop = _boxes_to_crop_region(boxes, src_w, src_h, method="face")
            trajectory.append((t, crop))
            detections += 1
        t += step_sec

    cap.release()
    conf = min(1.0, detections / max(1, int(duration / step_sec)))
    if detections < 2:
        return [], conf
    return trajectory, conf


def _boxes_to_crop_region(
    boxes: list[tuple[int, int, int, int, float]],
    src_w: int,
    src_h: int,
    method: str,
) -> CropRegion:
    """Build 9:16 crop region containing all detected people, avoiding over-zoom."""
    all_x1 = min(b[0] for b in boxes)
    all_y1 = min(b[1] for b in boxes)
    all_x2 = max(b[2] for b in boxes)
    all_y2 = max(b[3] for b in boxes)

    people_w = all_x2 - all_x1
    people_h = all_y2 - all_y1
    pad_x = int(people_w * 0.35)
    pad_y_top = int(people_h * 0.55)
    pad_y_bot = int(people_h * 0.65)
    all_x1 = max(0, all_x1 - pad_x)
    all_y1 = max(0, all_y1 - pad_y_top)
    all_x2 = min(src_w, all_x2 + pad_x)
    all_y2 = min(src_h, all_y2 + pad_y_bot)

    people_w = all_x2 - all_x1
    people_h = all_y2 - all_y1
    target_ratio = OUTPUT_W / OUTPUT_H
    cx = (all_x1 + all_x2) // 2
    cy = (all_y1 + all_y2) // 2

    if people_w / max(people_h, 1) > target_ratio:
        crop_w = people_w
        crop_h = int(crop_w / target_ratio)
    else:
        crop_h = people_h
        crop_w = int(crop_h * target_ratio)

    # Avoid over-zoom: crop must cover at least 45% of frame width
    min_crop_w = int(src_w * 0.45)
    crop_w = max(crop_w, min_crop_w)
    crop_h = int(crop_w / target_ratio)

    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)
    cx = max(crop_w // 2, min(cx, src_w - crop_w // 2))
    cy = max(crop_h // 2, min(cy, src_h - crop_h // 2))

    return CropRegion(x=cx - crop_w // 2, y=cy - crop_h // 2, w=crop_w, h=crop_h, method=method)


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
