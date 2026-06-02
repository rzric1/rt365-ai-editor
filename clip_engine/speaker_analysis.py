# -*- coding: utf-8 -*-
"""
clip_engine/speaker_analysis.py
Speaker turn detection using faster-whisper word-level timestamps.

Detects speaker changes based on silence gaps (>= 0.8 s) between words.
No pyannote, no HF_TOKEN required.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger("clip_engine.speaker_analysis")

# ---------------------------------------------------------------------------
# Pyannote stubs — kept for API compatibility; always returns False/None
# ---------------------------------------------------------------------------

_PYANNOTE_PIPELINE = None  # always None; kept so callers don't break


def pyannote_available() -> bool:
    """Always False — pyannote replaced by faster-whisper gap analysis."""
    return False


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_speaker_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def _load_pyannote_pipeline():
    """No-op stub — pyannote not used."""
    return None


def release_pyannote_pipeline() -> None:
    """No-op stub — nothing to release."""
    pass


# ---------------------------------------------------------------------------
# Core: gap-based speaker turn detection via faster-whisper
# ---------------------------------------------------------------------------

_GAP_THRESHOLD = 0.8  # seconds of silence that signals a speaker change


def _detect_turns_from_words(words: list[dict]) -> list[dict]:
    """
    Given a list of word dicts {start, end, word}, return
    [{start, end, speaker}, ...] by assigning a new speaker label
    each time the inter-word gap exceeds _GAP_THRESHOLD seconds.
    """
    if not words:
        return []

    turns: list[dict] = []
    speaker_idx = 0
    seg_start = float(words[0].get("start", 0.0))
    seg_end = float(words[0].get("end", 0.0))

    for i in range(1, len(words)):
        prev_end = float(words[i - 1].get("end", 0.0))
        curr_start = float(words[i].get("start", prev_end))
        gap = curr_start - prev_end

        if gap >= _GAP_THRESHOLD:
            # Close the current segment
            turns.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": f"SPEAKER_{speaker_idx:02d}",
                }
            )
            # Alternate speaker for a 2-speaker podcast heuristic;
            # cycle through more speakers if gaps accumulate differently
            speaker_idx = (speaker_idx + 1) % 2
            seg_start = curr_start

        seg_end = float(words[i].get("end", seg_end))

    # Flush last segment
    turns.append(
        {
            "start": seg_start,
            "end": seg_end,
            "speaker": f"SPEAKER_{speaker_idx:02d}",
        }
    )

    return turns


def diarize_audio_file(audio_path: str) -> list[dict]:
    """
    Detect speaker turns using faster-whisper word-level timestamps.

    Returns [{start, end, speaker}, ...]
    Results are cached as audio_path + "_diarization.json".
    """
    cache_path = pathlib.Path(audio_path + "_diarization.json")
    if cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info("Loaded diarization from cache: %s", cache_path)
            return cached
        except Exception as exc:
            logger.warning("Failed to read diarization cache: %s", exc)

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        logger.warning(
            "faster-whisper not installed. "
            "Run: pip install faster-whisper"
        )
        return []

    try:
        from clip_engine.job_control import check_cancelled
        from clip_engine.whisper_runtime import get_whisper_model

        device = "cuda" if cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        logger.info("faster-whisper diarization: device=%s compute=%s", device, compute)

        model = get_whisper_model(model_size="base", device=device, compute_type=compute)
        segments_iter, _info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
        )

        words: list[dict] = []
        for seg in segments_iter:
            check_cancelled()
            if seg.words:
                for w in seg.words:
                    words.append(
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": w.word,
                        }
                    )

        turns = _detect_turns_from_words(words)
        logger.info(
            "Detected %d speaker turns from %d words (gap >= %.1fs)",
            len(turns),
            len(words),
            _GAP_THRESHOLD,
        )

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(turns, f)
        except Exception as exc:
            logger.warning("Failed to write diarization cache: %s", exc)

        return turns

    except Exception as exc:
        logger.warning("faster-whisper diarization failed: %s", exc)
        return []
    finally:
        try:
            from clip_engine.gpu_cleanup import cleanup_gpu_after_phase

            cleanup_gpu_after_phase("diarize", whisper=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Score boosting helpers (unchanged API)
# ---------------------------------------------------------------------------

def boost_candidates_from_transcript_speakers(
    candidates: list[dict],
    segments: list[dict],
) -> list[dict]:
    """Boost scores where transcript shows speaker alternation / interruptions."""
    from clip_engine.speaker_signals import apply_speaker_signals_to_clips

    if not candidates or not segments:
        return candidates
    boosted = apply_speaker_signals_to_clips(candidates, segments, enabled=True)
    for c in boosted:
        alt = int(c.get("speaker_alternation_score", 0) or 0)
        intr = int(c.get("interruption_score", 0) or 0)
        if alt >= 55 or intr >= 50:
            c["composite_score"] = min(90, int(c.get("composite_score", 50)) + 8)
            c.setdefault("warnings", []).append("speaker_moment_boost")
    return boosted


def boost_candidates_from_diarization(
    candidates: list[dict],
    turns: list[dict],
    *,
    min_score_boost: int = 10,
) -> list[dict]:
    """Boost candidates overlapping speaker-change boundaries."""
    if not turns or not candidates:
        return candidates
    change_times: list[float] = []
    last_sp = None
    for t in sorted(turns, key=lambda x: x["start"]):
        sp = t.get("speaker")
        if last_sp is not None and sp != last_sp:
            change_times.append(float(t["start"]))
        last_sp = sp

    for c in candidates:
        t0 = float(c.get("start_seconds", 0))
        t1 = float(c.get("end_seconds", t0))
        for ct in change_times:
            if t0 <= ct <= t1:
                c["composite_score"] = min(92, int(c.get("composite_score", 50)) + min_score_boost)
                c["speaker_diarization_boost"] = True
                break
    return candidates


def speaker_pipeline_status() -> dict[str, Any]:
    return {
        "pyannote_available": False,
        "faster_whisper_diarization": True,
        "gap_threshold_seconds": _GAP_THRESHOLD,
        "cuda_available": cuda_available(),
        "device": get_speaker_device(),
        "pipeline_loaded": False,  # no persistent pipeline; model loaded on demand
    }
