"""
CLI: compress a video for AI clip analysis (1280 wide, H.264 + AAC).
Uses NVENC when available, else libx264. Invoked from compress_video.bat.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

from clip_engine.ffmpeg_resolve import ensure_ffmpeg_on_path, get_ffmpeg_executable  # noqa: E402
from clip_engine.ffmpeg_gpu import (  # noqa: E402
    run_ffmpeg_checked,
    video_encode_args_compress_crf30,
    should_attempt_nvenc_on_export,
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: compress_encode.py <input_video> <output_mp4>")
        return 2
    ensure_ffmpeg_on_path()
    ff = get_ffmpeg_executable()
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not inp.is_file():
        print(f"Input not found: {inp}")
        return 1

    prefer_gpu = not _env_truthy("FORCE_CPU_VIDEO")
    force_nv = _env_truthy("FORCE_NVENC_EXPORT")
    want_nvenc = should_attempt_nvenc_on_export(prefer_gpu=prefer_gpu, force_gpu_mode=force_nv)
    vf = "scale=1280:-2:flags=lanczos,format=yuv420p"

    attempts: list[tuple[bool, bool]] = []
    if want_nvenc:
        if prefer_gpu:
            attempts.append((True, True))
        attempts.append((True, False))
    if prefer_gpu:
        attempts.append((False, True))
    attempts.append((False, False))

    last_err: str | None = None
    for use_nvenc, hwaccel in attempts:
        enc, label = video_encode_args_compress_crf30(use_nvenc=use_nvenc)
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "warning", "-stats"]
        if hwaccel and prefer_gpu:
            cmd += ["-hwaccel", "cuda"]
        cmd += [
            "-i",
            str(inp.resolve()),
            "-vf",
            vf,
            *enc,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(out.resolve()),
        ]
        dec = "cuda-decode" if hwaccel and prefer_gpu else "cpu-decode"
        print(f"Trying {label} ({dec}) …")
        print("[ffmpeg]", " ".join(cmd))
        try:
            run_ffmpeg_checked(cmd, cwd=None)
            print(f"Done ({label}, {dec}).")
            return 0
        except RuntimeError as e:
            last_err = str(e)
            print(f"Attempt failed: {str(e)[:800]}")

    print(f"All attempts failed. Last error: {last_err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
