"""
clip_engine/clip_split.py
Split overly long clips into sharper micro-clips using transcript analysis.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from clip_engine.transcription_utils import detect_pauses, segments_to_prompt_transcript
from clip_engine.token_tracking import TokenTracker, get_tracker

logger = logging.getLogger("clip_engine.clip_split")


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON in split response")
    return json.loads(raw[start : end + 1])


def _heuristic_split_by_pauses(
    clip: dict,
    segments: list[dict],
    *,
    max_duration: float,
    min_duration: float,
) -> list[dict]:
    """Fallback: split long clip at natural pauses into 2-3 sub-windows."""
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    dur = t1 - t0
    if dur <= max_duration:
        return [clip]

    window_segs = [
        s for s in segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]
    if len(window_segs) < 4:
        return [clip]

    pauses = detect_pauses(window_segs, min_pause_seconds=0.9)
    pause_times = [p.after_seconds for p in pauses if t0 + min_duration < p.after_seconds < t1 - min_duration]
    if len(pause_times) < 1:
        mid = t0 + dur / 2
        pause_times = [mid]

    n_parts = min(3, max(2, int(round(dur / max_duration))))
    if len(pause_times) >= n_parts - 1:
        boundaries = sorted(pause_times[: n_parts - 1])
    else:
        step = dur / n_parts
        boundaries = [t0 + step * i for i in range(1, n_parts)]

    cuts = [t0] + boundaries + [t1]
    out: list[dict] = []
    for i in range(len(cuts) - 1):
        s0, s1 = cuts[i], cuts[i + 1]
        part_dur = s1 - s0
        if part_dur < min_duration or part_dur > max_duration + 5:
            continue
        nc = dict(clip)
        nc["start_seconds"] = round(s0, 3)
        nc["end_seconds"] = round(s1, 3)
        nc["original_start"] = s0
        nc["original_end"] = s1
        nc.setdefault("warnings", [])
        nc["warnings"].append(f"Heuristic split from parent {t0:.0f}s-{t1:.0f}s.")
        nc["split_from_parent"] = True
        out.append(nc)

    if len(out) >= 2:
        logger.info("Heuristic split %.1f-%.1f into %d sub-clips", t0, t1, len(out))
        return out
    return [clip]


def _split_one_clip(
    client: Any,
    clip: dict,
    segments: list[dict],
    *,
    max_duration: float,
    sub_clip_max: float,
    min_duration: float,
    tracker: TokenTracker,
) -> list[dict]:
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    dur = t1 - t0
    if dur <= max_duration:
        return [clip]

    window_segs = [
        s for s in segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]
    if not window_segs:
        return [clip]

    ts_transcript = segments_to_prompt_transcript(window_segs)
    if len(ts_transcript) > 12_000:
        ts_transcript = ts_transcript[:12_000] + "\n[truncated]"

    sub_max = min(sub_clip_max, max_duration)
    system = f"""You split one long podcast clip into 2-3 shorter standalone micro-clips.
Parent window: {t0:.1f}s to {t1:.1f}s ({dur:.0f}s total).
Rules:
- Each sub-clip MUST be {min_duration:.0f}s to {sub_max:.0f}s
- Each needs strong hook, clear point, clean ending
- No repeated setup across sub-clips
- Use EXACT timestamps from transcript
- Do NOT overlap sub-clips in time
- Prefer ONE powerful idea per sub-clip
Output ONLY JSON:
{{"sub_clips": [{{"start_seconds": 0, "end_seconds": 0, "hook_title": "", "composite_score": 70, "selection_reason": "", "dominant_signal": "educational"}}]}}"""

    user = f"TRANSCRIPT INSIDE LONG CLIP:\n{ts_transcript}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.25,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        clip_id = str(clip.get("_wid") or f"split-{t0:.0f}")
        tracker.record_openai_usage("clip_split", response.usage, clip_id=clip_id)
        data = _extract_json(response.choices[0].message.content or "{}")
        subs = data.get("sub_clips", [])
        if not isinstance(subs, list) or len(subs) < 2:
            return _heuristic_split_by_pauses(
                clip, segments, max_duration=sub_max, min_duration=min_duration
            )
    except Exception as e:
        logger.warning("Split failed for %.1f-%.1f: %s", t0, t1, e)
        return _heuristic_split_by_pauses(
            clip, segments, max_duration=sub_max, min_duration=min_duration
        )

    out: list[dict] = []
    for sub in subs:
        s0 = float(sub.get("start_seconds", 0))
        s1 = float(sub.get("end_seconds", 0))
        if s1 - s0 < min_duration or s1 - s0 > sub_max + 5:
            continue
        if s0 < t0 - 1 or s1 > t1 + 1:
            continue
        nc = dict(clip)
        nc["start_seconds"] = round(s0, 3)
        nc["end_seconds"] = round(s1, 3)
        nc["original_start"] = s0
        nc["original_end"] = s1
        nc["hook_title"] = str(sub.get("hook_title") or nc.get("hook_title", "Sub-clip"))
        nc["composite_score"] = int(sub.get("composite_score", nc.get("composite_score", 65)))
        if sub.get("selection_reason"):
            nc["selection_reason"] = str(sub["selection_reason"])
        if sub.get("dominant_signal"):
            nc["dominant_signal"] = str(sub["dominant_signal"])
        nc.setdefault("warnings", [])
        nc["warnings"].append(f"Split from parent clip {t0:.0f}s-{t1:.0f}s.")
        nc["split_from_parent"] = True
        out.append(nc)

    if len(out) >= 2:
        logger.info("Split %.1f-%.1f into %d sub-clips", t0, t1, len(out))
        return out
    return _heuristic_split_by_pauses(
        clip, segments, max_duration=sub_max, min_duration=min_duration
    )


def split_long_clips(
    clips: list[dict],
    segments: list[dict],
    api_key: str,
    *,
    max_duration: float = 90.0,
    sub_clip_max: float | None = None,
    min_duration: float = 25.0,
    tracker: TokenTracker | None = None,
) -> list[dict]:
    """Split clips longer than max_duration into smaller standalone clips."""
    import openai

    if not clips:
        return []

    sub_max = sub_clip_max if sub_clip_max is not None else max_duration
    tracker = tracker or get_tracker()
    client = openai.OpenAI(api_key=api_key)
    result: list[dict] = []

    for clip in clips:
        t0 = float(clip.get("start_seconds", 0))
        t1 = float(clip.get("end_seconds", t0))
        if t1 - t0 > max_duration and not clip.get("split_from_parent"):
            result.extend(
                _split_one_clip(
                    client, clip, segments,
                    max_duration=max_duration,
                    sub_clip_max=sub_max,
                    min_duration=min_duration,
                    tracker=tracker,
                )
            )
        else:
            result.append(clip)

    logger.info("split_long_clips: %d in -> %d out", len(clips), len(result))
    return result
