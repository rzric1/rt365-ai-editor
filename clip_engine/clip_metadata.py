"""
clip_engine/clip_metadata.py
Ground clip titles/reasons against the final export transcript window.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from clip_engine.transcription_utils import extract_transcript_excerpt, extract_transcript_window
from clip_engine.token_tracking import TokenTracker, get_tracker
from clip_engine.openai_resilience import (
    JSON_STRICT_RULES,
    call_openai_chat_json,
    estimate_tokens_rough,
    get_call_context,
    truncate_text_safe,
)
from clip_engine.clip_scoring import assess_hook_quality, repair_hook_title_local
from clip_engine.effective_config import ResolvedModels, resolve_models_from_call_context

logger = logging.getLogger("clip_engine.clip_metadata")

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "was", "are", "were", "be", "been", "i", "you", "he", "she", "we", "they",
    "it", "this", "that", "so", "just", "like", "about", "what", "how", "when", "why",
    "his", "her", "their", "my", "your", "our",
}


def _normalize_words(text: str) -> set[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _extract_key_terms(text: str) -> set[str]:
    """Significant content words (4+ chars) for topic matching."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def metadata_grounding_score(title: str, reason: str, window_text: str) -> float:
    """
    Fraction of significant metadata words found in transcript window.
    Returns 0.0-1.0.
    """
    meta_words = _normalize_words(f"{title} {reason}")
    key_terms = _extract_key_terms(f"{title} {reason}")
    window_words = _normalize_words(window_text)
    window_keys = _extract_key_terms(window_text)
    if not meta_words and not key_terms:
        return 0.0
    if not window_words:
        return 0.0

    hits = meta_words & window_words
    base = len(hits) / len(meta_words) if meta_words else 0.0

    if key_terms:
        key_hits = key_terms & window_keys
        key_ratio = len(key_hits) / len(key_terms)
        # Titles with specific nouns must have at least one key term in transcript
        if len(key_terms) >= 2 and len(key_hits) == 0:
            return min(base, 0.1)
        return max(base, key_ratio * 0.85)

    return base


def _fallback_title_from_transcript(window_text: str, max_words: int = 8) -> str:
    """Build a conservative title from the first strong sentence in the window."""
    sentences = re.split(r"(?<=[.!?])\s+", window_text.strip())
    for sent in sentences:
        words = sent.split()
        if len(words) >= 4:
            title = " ".join(words[:max_words]).strip(".,!?")
            if title:
                return title[:60]
    words = window_text.split()
    return " ".join(words[:max_words]) if words else "Clip moment"


def _regenerate_metadata_from_window(
    client: Any,
    window_text: str,
    clip: dict,
    tracker: TokenTracker,
    clip_id: str,
    *,
    resolved_models: ResolvedModels | None = None,
) -> dict | None:
    """Call GPT to rewrite metadata using ONLY the final transcript window."""
    system = f"""You rewrite short-form clip metadata so it matches ONLY the provided transcript.
Rules:
- hook_title: max 8 words, must describe content actually spoken in the transcript
- selection_reason: one sentence, only facts from the transcript
- ai_context_reason: one sentence on why this window works as a standalone clip
- dominant_signal: one of educational | emotional | story | debate | funny | inspirational
- platform_fit: list from TikTok | YouTube Shorts | Instagram Reels | LinkedIn
- grounding_confidence: 0-100 how well the title matches the transcript
Return ONLY valid JSON. No markdown. No code fences. No explanations.
{JSON_STRICT_RULES}
Schema keys: hook_title, selection_reason, ai_context_reason, dominant_signal, platform_fit, grounding_confidence"""

    user = f"FINAL CLIP TRANSCRIPT (only source of truth):\n{window_text[:8000]}"
    user, _ = truncate_text_safe(user, 8200, label="grounding_window")
    prompt_estimate = estimate_tokens_rough(system + user)
    models = resolved_models or resolve_models_from_call_context()
    model = models.quality_model

    schema_hint = (
        '{"hook_title": "", "selection_reason": "", "ai_context_reason": "", '
        '"dominant_signal": "", "platform_fit": [], "grounding_confidence": 0}'
    )
    try:
        data = call_openai_chat_json(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
            stage="metadata_grounding",
            schema_hint=schema_hint,
            tracker=tracker,
            prompt_estimate=prompt_estimate,
            clip_id=clip_id,
        )
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        logger.warning("Metadata regeneration failed: %s", e)
        return None


def ground_clip_metadata_against_window(
    clip: dict,
    segments: list[dict],
    api_key: str,
    *,
    tracker: TokenTracker | None = None,
    force_regenerate: bool = False,
    min_confidence: float = 0.20,
    resolved_models: ResolvedModels | None = None,
) -> dict:
    """
    Verify and correct clip metadata against final start_seconds/end_seconds window.
    Mutates and returns clip dict with grounding fields.
    """
    import openai

    tracker = tracker or get_tracker()
    c = dict(clip)

    t0 = float(c.get("start_seconds", c.get("start", 0)))
    t1 = float(c.get("end_seconds", c.get("end", t0)))
    window_text = extract_transcript_window(segments, t0, t1)
    excerpt = extract_transcript_excerpt(segments, t0, t1, max_chars=900)

    c["grounded_transcript_excerpt"] = excerpt
    c["title_before_grounding"] = str(c.get("hook_title", ""))
    c["reason_before_grounding"] = str(c.get("selection_reason", c.get("reason", "")))

    o_start = float(c.get("original_start", t0))
    o_end = float(c.get("original_end", t1))
    timing_shifted = abs(t0 - o_start) > 3.0 or abs(t1 - o_end) > 3.0

    if not window_text.strip():
        c.setdefault("warnings", [])
        c["warnings"].append("No transcript text in final export window.")
        c["grounding_confidence"] = 0
        c["metadata_grounded"] = False
        return c

    title = str(c.get("hook_title", ""))
    reason = str(c.get("selection_reason", c.get("reason", "")))
    score = metadata_grounding_score(title, reason, window_text)
    c["grounding_confidence"] = int(round(score * 100))

    needs_regen = (
        force_regenerate
        or timing_shifted
        or score < min_confidence
        or len(window_text.split()) < 8
    )
    ctx = get_call_context()
    if ctx.token_saver_mode and not force_regenerate and score >= min_confidence and not timing_shifted:
        needs_regen = False

    clip_id = str(c.get("_wid") or f"{t0:.1f}-{t1:.1f}")

    if needs_regen:
        client = openai.OpenAI(api_key=api_key)
        regen = _regenerate_metadata_from_window(
            client, window_text, c, tracker, clip_id, resolved_models=resolved_models
        )
        if regen:
            if regen.get("hook_title"):
                new_title = str(regen["hook_title"]).strip()
                hq, hw = assess_hook_quality(new_title)
                if hq < 55:
                    new_title = repair_hook_title_local(new_title, window_text)
                    c["hook_title_repaired"] = True
                c["hook_title"] = new_title
                c["hook_quality_score"] = assess_hook_quality(new_title)[0]
                if hw:
                    c["hook_warning"] = hw
            if regen.get("selection_reason"):
                c["selection_reason"] = str(regen["selection_reason"]).strip()
            if regen.get("ai_context_reason"):
                c["ai_context_reason"] = str(regen["ai_context_reason"]).strip()
            if regen.get("dominant_signal"):
                c["dominant_signal"] = str(regen["dominant_signal"]).strip()
            if regen.get("platform_fit"):
                c["platform_fit"] = regen["platform_fit"]
            if regen.get("grounding_confidence") is not None:
                c["grounding_confidence"] = int(regen["grounding_confidence"])

            new_score = metadata_grounding_score(
                c.get("hook_title", ""), c.get("selection_reason", ""), window_text
            )
            c["grounding_confidence"] = max(c["grounding_confidence"], int(round(new_score * 100)))
            c["metadata_grounded"] = True
            c["title_after_grounding"] = c.get("hook_title", "")
            c["reason_after_grounding"] = c.get("selection_reason", "")
        else:
            c.setdefault("warnings", [])
            c["warnings"].append("Metadata may not match final clip window (regeneration failed).")
            c["metadata_grounded"] = False

        # Hard validation: if title still unsupported, replace with transcript-derived title
        post_score = metadata_grounding_score(
            c.get("hook_title", ""), c.get("selection_reason", ""), window_text
        )
        if post_score < 0.20:
            fallback = _fallback_title_from_transcript(window_text)
            c["hook_title"] = fallback
            c["selection_reason"] = f"Key moment: {fallback[:80]}"
            c["grounding_confidence"] = max(
                int(c.get("grounding_confidence", 0)),
                int(round(metadata_grounding_score(fallback, "", window_text) * 100)),
            )
            c.setdefault("warnings", [])
            c["warnings"].append("Title replaced with transcript-derived fallback (weak grounding).")
            c["title_after_grounding"] = c.get("hook_title", "")
            c["reason_after_grounding"] = c.get("selection_reason", "")
    else:
        c["metadata_grounded"] = True
        c["title_after_grounding"] = c.get("hook_title", "")
        c["reason_after_grounding"] = c.get("selection_reason", "")

    if c.get("grounding_confidence", 0) < 25:
        c.setdefault("warnings", [])
        if "Metadata may not match final clip window." not in c["warnings"]:
            c["warnings"].append("Metadata may not match final clip window.")

    return c


def ground_all_clips_metadata(
    clips: list[dict],
    segments: list[dict],
    api_key: str,
    *,
    tracker: TokenTracker | None = None,
    force_regenerate: bool = False,
    skip_strong_grounding: bool = False,
    resolved_models: ResolvedModels | None = None,
) -> list[dict]:
    """Ground metadata for every clip in the list."""
    tracker = tracker or get_tracker()
    ctx = get_call_context()
    effective_force = force_regenerate and not (ctx.token_saver_mode and skip_strong_grounding)
    out: list[dict] = []
    for c in clips:
        grounded = ground_clip_metadata_against_window(
            c,
            segments,
            api_key,
            tracker=tracker,
            force_regenerate=effective_force,
            resolved_models=resolved_models,
        )
        out.append(grounded)
    logger.info("Grounded metadata for %d clips", len(out))
    return out


def write_clip_audit_json(clip: dict, path: Path, *, index: int) -> None:
    """Write per-clip audit file for export session."""
    audit = {
        "clip_index": index,
        "clip_title": clip.get("hook_title"),
        "start_seconds": clip.get("start_seconds"),
        "end_seconds": clip.get("end_seconds"),
        "original_ai_start": clip.get("original_start"),
        "original_ai_end": clip.get("original_end"),
        "final_start": clip.get("start_seconds"),
        "final_end": clip.get("end_seconds"),
        "transcript_excerpt": clip.get("grounded_transcript_excerpt"),
        "grounding_confidence": clip.get("grounding_confidence"),
        "title_before_grounding": clip.get("title_before_grounding"),
        "title_after_grounding": clip.get("title_after_grounding", clip.get("hook_title")),
        "reason_before_grounding": clip.get("reason_before_grounding"),
        "reason_after_grounding": clip.get("reason_after_grounding", clip.get("selection_reason")),
        "warnings": clip.get("warnings", []),
        "token_usage": clip.get("_token_usage"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
