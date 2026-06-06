# -*- coding: utf-8 -*-
"""
Transcription benchmark — validate faster-whisper CUDA throughput on RTX 4090.

Run from project root:
  .venv311\\Scripts\\python.exe -m clip_engine.transcription_benchmark

Optional:
  --seconds 60   audio length (default 60)
  --wav PATH     use existing WAV instead of generated silence
  --model NAME   whisper model (default config.DEFAULT_WHISPER_MODEL)
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _write_silence_wav(path: Path, *, seconds: float, sample_rate: int = 16000) -> None:
    n_frames = int(seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        chunk = struct.pack("<h", 0) * min(n_frames, sample_rate)
        remaining = n_frames
        while remaining > 0:
            n = min(remaining, sample_rate)
            wf.writeframes(chunk[: n * 2])
            remaining -= n


def run_benchmark(
    *,
    seconds: float = 60.0,
    wav_path: Path | None = None,
    model_size: str | None = None,
) -> int:
    from config import CLIP_STUDIO_OUTPUT_DIR, DEFAULT_WHISPER_MODEL
    from clip_engine.cuda_diagnostics import query_nvidia_gpu_memory_and_util
    from clip_engine.whisper_runtime import (
        _audio_duration_sec,
        _transcribe_call_kwargs,
        transcribe_wav,
    )

    model_size = model_size or DEFAULT_WHISPER_MODEL
    tmp_dir = CLIP_STUDIO_OUTPUT_DIR / "_work"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cleanup_tmp = False
    if wav_path is None:
        wav_path = tmp_dir / f"_benchmark_{int(seconds)}s.wav"
        _write_silence_wav(wav_path, seconds=seconds)
        cleanup_tmp = True
    elif not wav_path.is_file():
        logger.error("WAV not found: %s", wav_path)
        return 1

    audio_sec = _audio_duration_sec(wav_path) or seconds
    mem0, util0 = query_nvidia_gpu_memory_and_util()

    print("=== RT365 Transcription Benchmark ===")
    print(f"sys.executable: {sys.executable}")
    print(f"model: {model_size}")
    print(f"wav: {wav_path}")
    print(f"audio_sec: {audio_sec:.1f}")
    print(f"pre nvidia-smi: memory_mib={mem0} util_pct={util0}")

    kwargs = _transcribe_call_kwargs(device="cuda")
    print(
        f"transcribe settings: device=cuda compute_type=float16 "
        f"batch_size={kwargs.get('batch_size')} beam_size={kwargs.get('beam_size')} "
        f"vad_filter={kwargs.get('vad_filter')}"
    )

    t0 = time.perf_counter()
    try:
        segs, _txt, info = transcribe_wav(
            wav_path,
            language=None,
            model_size=model_size,
            device="cuda",
            compute_type="float16",
        )
    except Exception as exc:
        logger.exception("Benchmark failed")
        print(f"FAIL: {exc}")
        return 1
    finally:
        if cleanup_tmp and wav_path.is_file():
            try:
                wav_path.unlink()
            except OSError:
                pass

    elapsed = time.perf_counter() - t0
    duration = float(getattr(info, "duration", 0) or 0) or audio_sec
    rtf = elapsed / duration if duration > 0 else 0.0
    mem1, util1 = query_nvidia_gpu_memory_and_util()

    print(f"segments: {len(segs)}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"real_time_factor: {rtf:.3f} (lower is faster)")
    print(f"post nvidia-smi: memory_mib={mem1} util_pct={util1}")
    print(f"backend: faster-whisper CUDA float16")
    print("=== Benchmark complete ===")
    return 0


def main(argv: list[str] | None = None) -> int:
    from config import DEFAULT_WHISPER_MODEL

    parser = argparse.ArgumentParser(description="RT365 faster-whisper CUDA benchmark")
    parser.add_argument("--seconds", type=float, default=60.0, help="Generated audio length")
    parser.add_argument("--wav", type=Path, default=None, help="Existing 16 kHz mono WAV")
    parser.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL)
    args = parser.parse_args(argv)
    return run_benchmark(seconds=args.seconds, wav_path=args.wav, model_size=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
