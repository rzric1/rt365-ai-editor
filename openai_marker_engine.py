# -*- coding: utf-8 -*-
"""
Call OpenAI (Responses API) to propose timeline markers from transcript text.

We request **structured JSON** via `text.format` + `json_schema` so the model
must return machine-readable markers your script can validate and apply.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config import (
    TRANSCRIPT_CHUNK_MAX_CHARS,
    TRANSCRIPT_CHUNK_OVERLAP_CHARS,
    get_openai_model,
)
from transcript_loader import TranscriptDocument

logger = logging.getLogger(__name__)

# All marker types requested for podcast / reaction workflows.
MARKER_TYPES: Tuple[str, ...] = (
    "HOT_TAKE",
    "STRONG_REACTION",
    "POSSIBLE_CUT",
    "SHORT_CLIP",
    "GOOD_QUOTE",
    "CHAPTER",
    "AUDIO_DIP",
)

# JSON Schema for OpenAI Structured Outputs (strict).
# See: https://platform.openai.com/docs/guides/structured-outputs
MARKERS_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "markers": {
            "type": "array",
            "description": "Suggested Resolve timeline markers for an editor.",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp_seconds": {
                        "type": "number",
                        "description": "Time in seconds from timeline start (matches transcript).",
                    },
                    "marker_type": {
                        "type": "string",
                        "enum": list(MARKER_TYPES),
                        "description": "Category of editorial interest.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short marker title shown in Resolve.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Longer note for the editor; may include why this moment matters.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0 - 1.0 confidence for this suggestion.",
                    },
                },
                "required": [
                    "timestamp_seconds",
                    "marker_type",
                    "title",
                    "note",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["markers"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AiMarker:
    """One marker proposal from the model (timeline-neutral — seconds only)."""

    timestamp_seconds: float
    marker_type: str
    title: str
    note: str
    confidence: float


INSTRUCTIONS = """You are an assistant editor for long YouTube podcasts and reaction videos.

You ONLY suggest **non-destructive timeline markers** for DaVinci Resolve. You never
instruct anyone to cut, delete, ripple-delete, or move media. Markers are hints for a human editor.

Given a transcript chunk with [HH:MM:SS] timestamps, return JSON matching the schema:
  - Pick moments that help editing: hot takes, strong reactions, possible dead air / tangents,
    short vertical clips, memorable quotes, chapter boundaries, and places that may need audio level fixes.
  - timestamp_seconds must align with the transcript (use the bracket time nearest the moment).
  - Keep titles short (Resolve marker names are visible in a small UI).
  - Notes should be practical for an editor (what to check, why it is interesting).
  - confidence: how sure you are this marker is useful (0.0 - 1.0).
  - Do not invent content that is not grounded in the transcript text for this chunk.
If the chunk has no strong moments, return an empty markers array.
"""


def _chunk_plain_text(plain: str) -> List[str]:
    """Split large transcripts into overlapping windows by character count."""
    plain = plain.strip()
    if not plain:
        return []
    max_chars = max(TRANSCRIPT_CHUNK_MAX_CHARS, 1000)
    overlap = min(TRANSCRIPT_CHUNK_OVERLAP_CHARS, max_chars // 2)
    chunks: List[str] = []
    start = 0
    n = len(plain)
    while start < n:
        end = min(n, start + max_chars)
        chunk = plain[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def analyze_transcript(doc: TranscriptDocument, *, api_key: str) -> List[AiMarker]:
    """
    Send transcript chunks to OpenAI and merge marker suggestions.

    Long podcasts are processed in multiple API calls for reliability.
    """
    if not api_key.strip():
        raise ValueError("Missing OPENAI_API_KEY (set it in .env).")

    plain = doc.plain_text_with_timestamps()
    chunks = _chunk_plain_text(plain)
    if not chunks:
        logger.warning("Transcript is empty — no chunks to analyze.")
        return []

    client = OpenAI(api_key=api_key)
    model = get_openai_model()

    merged: List[AiMarker] = []
    for i, chunk in enumerate(chunks, start=1):
        logger.info("OpenAI chunk %s / %s (%s chars)", i, len(chunks), len(chunk))
        user_prompt = (
            f"Transcript chunk {i} of {len(chunks)} from file {doc.source_path.name}.\n\n"
            f"{chunk}"
        )
        response = client.responses.create(
            model=model,
            instructions=INSTRUCTIONS,
            input=user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rt365_timeline_markers",
                    "strict": True,
                    "schema": MARKERS_RESPONSE_SCHEMA,
                }
            },
            # Avoid storing customer transcripts on OpenAI servers by default.
            store=False,
        )
        raw_text = getattr(response, "output_text", None) or ""
        if not raw_text.strip():
            logger.warning("Empty output_text from OpenAI for chunk %s", i)
            continue
        payload = json.loads(raw_text)
        markers = payload.get("markers", [])
        if not isinstance(markers, list):
            logger.warning("Unexpected markers payload in chunk %s", i)
            continue
        for m in markers:
            conf = float(m["confidence"])
            # Keep scores in-range even if the model drifts slightly.
            conf = max(0.0, min(1.0, conf))
            merged.append(
                AiMarker(
                    timestamp_seconds=float(m["timestamp_seconds"]),
                    marker_type=str(m["marker_type"]),
                    title=str(m["title"]),
                    note=str(m["note"]),
                    confidence=conf,
                )
            )

    merged.sort(key=lambda m: (m.timestamp_seconds, m.marker_type, m.title))
    return _dedupe_markers(merged)


def _dedupe_markers(markers: List[AiMarker]) -> List[AiMarker]:
    """Remove near-duplicate markers that appear at chunk boundaries."""
    seen: set[tuple[float, str, str]] = set()
    out: List[AiMarker] = []
    for m in markers:
        key = (round(m.timestamp_seconds, 2), m.marker_type, m.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out
