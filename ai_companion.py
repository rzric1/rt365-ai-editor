# -*- coding: utf-8 -*-
"""
RT365 AI Edit Companion — intent routing and OpenAI structured responses.

Version 1 safety: outputs are suggestions only. Applying anything to Resolve
goes through marker_writer (AddMarker only). No cuts, ripple, or media pool.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from config import get_openai_model
from openai_marker_engine import AiMarker, analyze_transcript
from marker_writer import markers_as_printable_dicts
from transcript_loader import TranscriptDocument

logger = logging.getLogger(__name__)

# Intent labels returned by classification (must match OpenAI schema enum).
INTENT_ENUM: List[str] = [
    "ANALYZE_MARKERS",
    "FIND_CLIP",
    "CHAPTERS",
    "POSSIBLE_CUTS",
    "GOOD_QUOTES",
    "AUDIO_ISSUES",
    "GENERAL_EDIT_ADVICE",
]

COMPANION_MAX_TRANSCRIPT_CHARS = 28000

SAFETY_PREAMBLE = (
    "You help human editors working in DaVinci Resolve on YouTube podcasts and reaction videos. "
    "You never instruct cutting, deleting, ripple edits, or media pool changes. "
    "You only produce timestamps and text suggestions; timeline changes are markers only."
)


def _transcript_excerpt(doc: TranscriptDocument) -> str:
    plain = doc.plain_text_with_timestamps()
    if len(plain) <= COMPANION_MAX_TRANSCRIPT_CHARS:
        return plain
    return (
        plain[:COMPANION_MAX_TRANSCRIPT_CHARS]
        + "\n\n[Transcript truncated for this request — focus on the visible portion.]\n"
    )


def _responses_json(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    user_input: str,
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        store=False,
    )
    raw = getattr(response, "output_text", None) or ""
    if not raw.strip():
        raise RuntimeError("OpenAI returned empty output_text.")
    return json.loads(raw)


# --- Intent classification ---

_CLASSIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": INTENT_ENUM,
            "description": "Single best intent for this user message.",
        },
        "brief_reason": {
            "type": "string",
            "description": "One short sentence for the UI.",
        },
    },
    "required": ["intent", "brief_reason"],
    "additionalProperties": False,
}

_CLASSIFY_INSTRUCTIONS = (
    SAFETY_PREAMBLE
    + " Classify the user's message into exactly one intent:\n"
    "- ANALYZE_MARKERS: broad pass — find many useful markers (hot takes, reactions, quotes, etc.).\n"
    "- FIND_CLIP: user wants a specific vertical/short clip segment with start/end and social copy.\n"
    "- CHAPTERS: chapter titles and timestamps for YouTube or the edit.\n"
    "- POSSIBLE_CUTS: tangents, repeats, or trims to *consider* (hints only).\n"
    "- GOOD_QUOTES: memorable lines worth marking.\n"
    "- AUDIO_ISSUES: possible level dips, noise, mic issues (hint markers).\n"
    "- GENERAL_EDIT_ADVICE: pacing, structure, or workflow tips without timestamp lists.\n"
)


def classify_intent(client: OpenAI, model: str, user_message: str) -> Tuple[str, str]:
    payload = _responses_json(
        client,
        model=model,
        instructions=_CLASSIFY_INSTRUCTIONS,
        user_input=f"User message:\n{user_message.strip()}",
        schema_name="rt365_intent",
        schema=_CLASSIFY_SCHEMA,
    )
    return str(payload["intent"]), str(payload["brief_reason"])


# --- Structured outputs per intent ---

_FIND_CLIP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "clip_title": {"type": "string"},
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "hook": {"type": "string"},
        "caption": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["clip_title", "start_seconds", "end_seconds", "hook", "caption", "reason"],
    "additionalProperties": False,
}

_CHAPTERS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "summary": {"type": "string"},
                },
                "required": ["title", "start_seconds", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}

_ITEMS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp_seconds": {"type": "number"},
                    "title": {"type": "string"},
                    "note": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["timestamp_seconds", "title", "note", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_GENERAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "advice": {"type": "string"},
    },
    "required": ["advice"],
    "additionalProperties": False,
}


def _run_find_clip(
    client: OpenAI, model: str, user_message: str, excerpt: str, source_name: str
) -> Dict[str, Any]:
    instructions = (
        SAFETY_PREAMBLE
        + " Pick one compelling clip from the transcript that matches the user's ask. "
        "Times must match the [HH:MM:SS] timestamps in the transcript. "
        "end_seconds must be greater than start_seconds unless the moment is a single beat (then use ~15s span minimum)."
    )
    user_input = (
        f"Source file: {source_name}\nUser request:\n{user_message}\n\nTranscript:\n{excerpt}"
    )
    return _responses_json(
        client,
        model=model,
        instructions=instructions,
        user_input=user_input,
        schema_name="rt365_find_clip",
        schema=_FIND_CLIP_SCHEMA,
    )


def _run_chapters(
    client: OpenAI, model: str, user_message: str, excerpt: str, source_name: str
) -> Dict[str, Any]:
    instructions = (
        SAFETY_PREAMBLE
        + " Propose YouTube-style chapters: clear titles, start_seconds aligned with transcript timestamps. "
        "Order chapters by time. Summaries should help an editor skim."
    )
    user_input = (
        f"Source file: {source_name}\nUser request:\n{user_message}\n\nTranscript:\n{excerpt}"
    )
    return _responses_json(
        client,
        model=model,
        instructions=instructions,
        user_input=user_input,
        schema_name="rt365_chapters",
        schema=_CHAPTERS_SCHEMA,
    )


def _run_items(
    client: OpenAI,
    model: str,
    user_message: str,
    excerpt: str,
    source_name: str,
    *,
    focus: str,
) -> Dict[str, Any]:
    instructions = SAFETY_PREAMBLE + " " + focus
    user_input = (
        f"Source file: {source_name}\nUser request:\n{user_message}\n\nTranscript:\n{excerpt}"
    )
    return _responses_json(
        client,
        model=model,
        instructions=instructions,
        user_input=user_input,
        schema_name="rt365_items",
        schema=_ITEMS_SCHEMA,
    )


def _run_general(
    client: OpenAI, model: str, user_message: str, excerpt: str, source_name: str
) -> Dict[str, Any]:
    instructions = (
        SAFETY_PREAMBLE
        + " Give practical editing advice for this episode. No fabricated events; "
        "if the transcript is thin, say so. Plain language, bullet style inside the advice string is OK."
    )
    user_input = (
        f"Source file: {source_name}\nUser request:\n{user_message}\n\nTranscript:\n{excerpt}"
    )
    return _responses_json(
        client,
        model=model,
        instructions=instructions,
        user_input=user_input,
        schema_name="rt365_general",
        schema=_GENERAL_SCHEMA,
    )


# --- Convert companion payloads to Resolve-safe AiMarker lists ---


def clip_payload_to_markers(data: Dict[str, Any]) -> List[AiMarker]:
    """START + END markers for a suggested clip (SHORT_CLIP type)."""
    title = str(data.get("clip_title", "Clip")).strip() or "Clip"
    start = float(data["start_seconds"])
    end = float(data["end_seconds"])
    if end < start:
        start, end = end, start
    hook = str(data.get("hook", ""))
    caption = str(data.get("caption", ""))
    reason = str(data.get("reason", ""))
    note_body = "\n".join(x for x in (hook, caption, reason) if x).strip() or "Companion clip suggestion"
    conf = 0.9
    return [
        AiMarker(
            timestamp_seconds=start,
            marker_type="SHORT_CLIP",
            title=f"{title} - START",
            note=note_body[:1900],
            confidence=conf,
        ),
        AiMarker(
            timestamp_seconds=end,
            marker_type="SHORT_CLIP",
            title=f"{title} - END",
            note=note_body[:1900],
            confidence=conf,
        ),
    ]


def chapters_payload_to_markers(data: Dict[str, Any]) -> List[AiMarker]:
    out: List[AiMarker] = []
    for ch in data.get("chapters", []) or []:
        if not isinstance(ch, dict):
            continue
        title = str(ch.get("title", "Chapter")).strip() or "Chapter"
        start = float(ch["start_seconds"])
        summary = str(ch.get("summary", ""))
        out.append(
            AiMarker(
                timestamp_seconds=start,
                marker_type="CHAPTER",
                title=title[:110],
                note=summary[:1900],
                confidence=0.88,
            )
        )
    out.sort(key=lambda m: m.timestamp_seconds)
    return out


def items_payload_to_markers(data: Dict[str, Any], marker_type: str) -> List[AiMarker]:
    out: List[AiMarker] = []
    for it in data.get("items", []) or []:
        if not isinstance(it, dict):
            continue
        conf = float(it.get("confidence", 0.85))
        conf = max(0.0, min(1.0, conf))
        out.append(
            AiMarker(
                timestamp_seconds=float(it["timestamp_seconds"]),
                marker_type=marker_type,
                title=str(it.get("title", "Item"))[:110],
                note=str(it.get("note", ""))[:1900],
                confidence=conf,
            )
        )
    out.sort(key=lambda m: m.timestamp_seconds)
    return out


def run_companion_turn(
    *,
    user_message: str,
    doc: TranscriptDocument,
    api_key: str,
) -> Dict[str, Any]:
    """
    Classify intent, run the right analysis, return a JSON-serializable dict for UI + logs.

    Keys: intent, brief_reason, data (intent-specific), transcript_path (string).
    """
    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    excerpt = _transcript_excerpt(doc)
    source_name = doc.source_path.name

    intent, brief_reason = classify_intent(client, model, user_message)
    logger.info("Companion intent=%s (%s)", intent, brief_reason)

    data: Any
    if intent == "ANALYZE_MARKERS":
        markers = analyze_transcript(doc, api_key=api_key)
        data = {"markers": markers_as_printable_dicts(markers)}
    elif intent == "FIND_CLIP":
        data = _run_find_clip(client, model, user_message, excerpt, source_name)
    elif intent == "CHAPTERS":
        data = _run_chapters(client, model, user_message, excerpt, source_name)
    elif intent == "POSSIBLE_CUTS":
        data = _run_items(
            client,
            model,
            user_message,
            excerpt,
            source_name,
            focus=(
                "List timestamped POSSIBLE_CUT style suggestions: tangents, repeats, "
                "or moments an editor might shorten. Hints only — never instruct to delete."
            ),
        )
    elif intent == "GOOD_QUOTES":
        data = _run_items(
            client,
            model,
            user_message,
            excerpt,
            source_name,
            focus="List timestamped memorable quotes or strong lines worth marking.",
        )
    elif intent == "AUDIO_ISSUES":
        data = _run_items(
            client,
            model,
            user_message,
            excerpt,
            source_name,
            focus=(
                "List timestamped moments that may need an audio pass "
                "(levels, noise, mic bumps). Hint-only."
            ),
        )
    elif intent == "GENERAL_EDIT_ADVICE":
        data = _run_general(client, model, user_message, excerpt, source_name)
    else:
        intent = "GENERAL_EDIT_ADVICE"
        data = _run_general(client, model, user_message, excerpt, source_name)

    return {
        "intent": intent,
        "brief_reason": brief_reason,
        "user_message": user_message.strip(),
        "transcript_path": str(doc.source_path.resolve()),
        "data": data,
    }


def markers_for_resolve(intent: str, data: Dict[str, Any]) -> Optional[List[AiMarker]]:
    """
    Build AiMarker list for Apply-to-Resolve buttons, or None if not applicable.
    """
    if intent == "ANALYZE_MARKERS":
        raw = data.get("markers", [])
        out: List[AiMarker] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            conf = max(0.0, min(1.0, float(m.get("confidence", 0.85))))
            out.append(
                AiMarker(
                    timestamp_seconds=float(m["timestamp_seconds"]),
                    marker_type=str(m["marker_type"]),
                    title=str(m.get("title", ""))[:120],
                    note=str(m.get("note", ""))[:2000],
                    confidence=conf,
                )
            )
        return out
    if intent == "FIND_CLIP":
        return clip_payload_to_markers(data)
    if intent == "CHAPTERS":
        return chapters_payload_to_markers(data)
    if intent == "GOOD_QUOTES":
        return items_payload_to_markers(data, "GOOD_QUOTE")
    if intent == "POSSIBLE_CUTS":
        return items_payload_to_markers(data, "POSSIBLE_CUT")
    if intent == "AUDIO_ISSUES":
        return items_payload_to_markers(data, "AUDIO_DIP")
    return None
