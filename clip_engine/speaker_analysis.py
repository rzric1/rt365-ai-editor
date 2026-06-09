# -*- coding: utf-8 -*-
"""
clip_engine/speaker_analysis.py
Speaker diarization via pyannote/speaker-diarization-3.1 with faster-whisper gap fallback.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import warnings
from typing import Any

from config import PROJECT_ROOT

logger = logging.getLogger("clip_engine.speaker_analysis")

_PYANNOTE_PIPELINE = None
_PYANNOTE_PIPELINE_CLASS: Any = None
_PYANNOTE_IMPORT_ATTEMPTED = False
_GAP_THRESHOLD = 0.8  # seconds of silence that signals a speaker change (fallback)
_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def _emergency_log(msg: str) -> None:
    try:
        log_path = PROJECT_ROOT / "logs" / "emergency_repair_log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def _load_hf_token() -> str | None:
    """Load HF token from .env / process env. Never logs the token value."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
    return token or None


def _silence_torchcodec_import_noise() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*torchcodec.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")


def _import_pyannote_pipeline_class():
    """Import pyannote Pipeline once, silencing torchcodec UserWarning/OSError."""
    global _PYANNOTE_PIPELINE_CLASS, _PYANNOTE_IMPORT_ATTEMPTED
    if _PYANNOTE_PIPELINE_CLASS is not None:
        return _PYANNOTE_PIPELINE_CLASS
    if _PYANNOTE_IMPORT_ATTEMPTED:
        return None
    _PYANNOTE_IMPORT_ATTEMPTED = True
    _silence_torchcodec_import_noise()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                from pyannote.audio import Pipeline
            except OSError:
                return None
        _PYANNOTE_PIPELINE_CLASS = Pipeline
        _emergency_log("pyannote.audio.Pipeline import: OK (torchcodec noise suppressed)")
        return Pipeline
    except Exception as exc:
        logger.debug("pyannote import unavailable: %s", exc)
        _emergency_log(f"pyannote.audio import failed: {type(exc).__name__}")
        return None


def pyannote_available() -> bool:
    return _import_pyannote_pipeline_class() is not None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_speaker_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def _load_pyannote_pipeline():
    """Load and cache pyannote diarization pipeline. Returns None on any failure."""
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is not None:
        return _PYANNOTE_PIPELINE

    Pipeline = _import_pyannote_pipeline_class()
    if Pipeline is None:
        return None

    token = _load_hf_token()
    if not token:
        logger.info("HF token not set — pyannote diarization skipped")
        return None

    try:
        pipeline = Pipeline.from_pretrained(_PYANNOTE_MODEL, token=token)
        device = get_speaker_device()
        if device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        _PYANNOTE_PIPELINE = pipeline
        logger.info("pyannote pipeline loaded (device=%s)", device)
        _emergency_log(f"pyannote pipeline loaded: device={device}")
        return pipeline
    except Exception as exc:
        logger.warning("pyannote pipeline load failed (gap fallback): %s", exc)
        _emergency_log(f"pyannote pipeline load failed: {type(exc).__name__}")
        return None


def release_pyannote_pipeline() -> None:
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is None:
        return
    try:
        import torch

        if cuda_available():
            _PYANNOTE_PIPELINE.to(torch.device("cpu"))
    except Exception:
        pass
    _PYANNOTE_PIPELINE = None
    try:
        from clip_engine.gpu_cleanup import cleanup_gpu_after_phase

        cleanup_gpu_after_phase("diarize", whisper=False)
    except Exception:
        pass


def _load_wav_for_pyannote(audio_path: str) -> dict[str, Any]:
    """Load WAV in-memory for pyannote (avoids broken torchcodec file decoding)."""
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _turns_from_pyannote_annotation(diarization: Any) -> list[dict]:
    turns: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    turns.sort(key=lambda x: x["start"])
    return turns


def _diarize_with_pyannote(audio_path: str) -> list[dict]:
    pipeline = _load_pyannote_pipeline()
    if pipeline is None:
        return []

    try:
        from clip_engine.job_control import check_cancelled

        check_cancelled()
        audio = _load_wav_for_pyannote(audio_path)
        check_cancelled()
        diarization = pipeline(audio)
        turns = _turns_from_pyannote_annotation(diarization)
        logger.info("pyannote diarization: %d turns", len(turns))
        return turns
    except Exception as exc:
        logger.warning("pyannote diarization failed (gap fallback): %s", exc)
        _emergency_log(f"pyannote diarization failed: {type(exc).__name__}")
        return []


def _detect_turns_from_words(words: list[dict]) -> list[dict]:
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
            turns.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": f"SPEAKER_{speaker_idx:02d}",
                }
            )
            speaker_idx = (speaker_idx + 1) % 2
            seg_start = curr_start

        seg_end = float(words[i].get("end", seg_end))

    turns.append(
        {
            "start": seg_start,
            "end": seg_end,
            "speaker": f"SPEAKER_{speaker_idx:02d}",
        }
    )
    return turns


def _diarize_with_faster_whisper_gap(audio_path: str) -> list[dict]:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        logger.warning("faster-whisper not installed — gap diarization unavailable")
        return []

    try:
        from clip_engine.job_control import check_cancelled
        from clip_engine.whisper_runtime import get_whisper_model

        device = "cuda" if cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        logger.info("faster-whisper gap diarization: device=%s compute=%s", device, compute)

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
            "Gap diarization: %d turns from %d words (gap >= %.1fs)",
            len(turns),
            len(words),
            _GAP_THRESHOLD,
        )
        return turns
    except Exception as exc:
        logger.warning("faster-whisper gap diarization failed: %s", exc)
        return []


def _read_diarization_cache(cache_path: pathlib.Path) -> list[dict] | None:
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict) and isinstance(cached.get("turns"), list):
            return cached["turns"]
        if isinstance(cached, list):
            return cached
    except Exception as exc:
        logger.warning("Failed to read diarization cache: %s", exc)
    return None


def _write_diarization_cache(cache_path: pathlib.Path, turns: list[dict], method: str) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"method": method, "turns": turns}, f)
    except Exception as exc:
        logger.warning("Failed to write diarization cache: %s", exc)


def diarize_audio_file(audio_path: str) -> list[dict]:
    """
    Detect speaker turns. Tries pyannote first, falls back to faster-whisper gaps.
    Returns [{start, end, speaker}, ...]
    """
    cache_path = pathlib.Path(audio_path + "_diarization.json")
    cached = _read_diarization_cache(cache_path)
    if cached is not None:
        logger.info("Loaded diarization from cache: %s", cache_path)
        return cached

    method = "none"
    turns: list[dict] = []

    if pyannote_available() and _load_hf_token():
        turns = _diarize_with_pyannote(audio_path)
        if turns:
            method = "pyannote"

    if not turns:
        turns = _diarize_with_faster_whisper_gap(audio_path)
        if turns:
            method = "gap"

    if turns:
        _write_diarization_cache(cache_path, turns, method)

    return turns


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
        mono = int(c.get("monologue_score", 0) or 0)
        if mono >= 70:
            c["composite_score"] = min(92, int(c.get("composite_score", 50)) + 10)
            c.setdefault("warnings", []).append("expert_monologue_boost")
        elif alt >= 55 or intr >= 50:
            c["composite_score"] = min(90, int(c.get("composite_score", 50)) + 8)
            c.setdefault("warnings", []).append("speaker_moment_boost")
    return boosted


def _window_speaker_stats(
    t0: float,
    t1: float,
    turns: list[dict],
) -> tuple[float, int, str | None]:
    """Return (dominance_ratio, speaker_changes, dominant_speaker) for [t0, t1]."""
    window_dur = max(0.0, t1 - t0)
    if window_dur <= 0 or not turns:
        return 0.0, 0, None

    overlap_by_speaker: dict[str, float] = {}
    window_turns: list[dict] = []
    for turn in turns:
        ts = float(turn.get("start", 0))
        te = float(turn.get("end", ts))
        if te <= t0 or ts >= t1:
            continue
        overlap = max(0.0, min(te, t1) - max(ts, t0))
        if overlap <= 0:
            continue
        sp = str(turn.get("speaker", "UNKNOWN"))
        overlap_by_speaker[sp] = overlap_by_speaker.get(sp, 0.0) + overlap
        window_turns.append({"start": ts, "end": te, "speaker": sp})

    if not overlap_by_speaker:
        return 0.0, 0, None

    window_turns.sort(key=lambda x: x["start"])
    changes = sum(
        1
        for i in range(1, len(window_turns))
        if window_turns[i]["speaker"] != window_turns[i - 1]["speaker"]
    )
    dominant = max(overlap_by_speaker, key=overlap_by_speaker.get)
    ratio = overlap_by_speaker[dominant] / window_dur
    return ratio, changes, dominant


def boost_candidates_from_diarization(
    candidates: list[dict],
    turns: list[dict],
    *,
    min_score_boost: int = 10,
) -> list[dict]:
    """Boost clips with a single dominant uninterrupted speaker (expert monologue)."""
    if not turns or not candidates:
        return candidates

    for c in candidates:
        t0 = float(c.get("start_seconds", c.get("start", 0)))
        t1 = float(c.get("end_seconds", c.get("end", t0)))
        ratio, changes, dominant = _window_speaker_stats(t0, t1, turns)
        c["diarization_dominance"] = round(ratio, 3)
        c["diarization_speaker_changes"] = changes
        if dominant:
            c["dominant_speaker"] = dominant

        if ratio >= 0.82 and changes <= 1:
            boost = min_score_boost + (4 if ratio >= 0.92 else 0)
            c["composite_score"] = min(94, int(c.get("composite_score", 50)) + boost)
            c["speaker_diarization_boost"] = True
            c.setdefault("warnings", []).append("expert_monologue_diarization")

    return candidates


def speaker_pipeline_status() -> dict[str, Any]:
    pipeline_loaded = _PYANNOTE_PIPELINE is not None
    token_set = bool(_load_hf_token())
    return {
        "pyannote_available": pyannote_available(),
        "pyannote_token_set": token_set,
        "faster_whisper_diarization": True,
        "gap_threshold_seconds": _GAP_THRESHOLD,
        "cuda_available": cuda_available(),
        "device": get_speaker_device(),
        "pipeline_loaded": pipeline_loaded,
        "model": _PYANNOTE_MODEL if pipeline_loaded else None,
    }
