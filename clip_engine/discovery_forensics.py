"""
clip_engine/discovery_forensics.py
Stage-by-stage discovery pipeline forensics (input/output/rejections).
No UI dependency — consumed by clip_pipeline stats and logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("clip_engine.discovery_forensics")


@dataclass
class DiscoveryForensics:
    """Aggregated discovery-path diagnostics for candidate starvation analysis."""

    stages: list[dict[str, Any]] = field(default_factory=list)
    windows_scanned: int = 0
    windows_rejected: int = 0
    emotion_hits: int = 0
    curiosity_hits: int = 0
    story_turn_hits: int = 0
    trauma_hits: int = 0
    keyword_hits: int = 0
    gpu_candidates_generated: int = 0
    gpu_candidates_rejected: int = 0
    gpu_rejection_reasons: dict[str, int] = field(default_factory=dict)
    fallback_candidates_generated: int = 0
    first_zero_stage: str = ""
    notes: list[str] = field(default_factory=list)

    def record_stage(
        self,
        stage: str,
        *,
        input_count: int,
        output_count: int,
        rejected_count: int | None = None,
        rejection_reasons: dict[str, int] | None = None,
        note: str = "",
    ) -> None:
        rejected = (
            rejected_count
            if rejected_count is not None
            else max(0, input_count - output_count)
        )
        entry = {
            "stage": stage,
            "input_count": int(input_count),
            "output_count": int(output_count),
            "rejected_count": int(rejected),
            "rejection_reasons": dict(rejection_reasons or {}),
            "note": note,
        }
        self.stages.append(entry)
        if rejection_reasons:
            for reason, count in rejection_reasons.items():
                if count <= 0:
                    continue
                logger.info(
                    "[DISCOVERY FORENSIC] %s reject %s: %d",
                    stage,
                    reason,
                    count,
                )
        logger.info(
            "[DISCOVERY FORENSIC] %s in=%d out=%d rejected=%d%s",
            stage,
            input_count,
            output_count,
            rejected,
            f" ({note})" if note else "",
        )
        if not self.first_zero_stage and output_count == 0:
            self.first_zero_stage = stage

    def merge_scan_stats(self, scan: dict[str, Any]) -> None:
        """Merge transcript scanner / discovery_scan dict into aggregate counters."""
        if not scan:
            return
        self.windows_scanned += int(scan.get("windows_scanned", 0))
        self.windows_rejected += int(scan.get("windows_rejected", 0))
        self.emotion_hits += int(scan.get("emotion_triggers", scan.get("emotion_hits", 0)))
        self.curiosity_hits += int(
            scan.get("curiosity_triggers", scan.get("curiosity_hits", 0))
        )
        self.story_turn_hits += int(
            scan.get("story_phrase_triggers", scan.get("story_turn_hits", 0))
        )
        self.trauma_hits += int(scan.get("trauma_triggers", scan.get("trauma_hits", 0)))
        self.keyword_hits += int(scan.get("keyword_hits", 0))
        self.fallback_candidates_generated += int(
            scan.get("fallback_generated", scan.get("transcript_only_candidates", 0))
        )

    def record_gpu_rejection(self, reason: str, count: int = 1) -> None:
        if count <= 0:
            return
        self.gpu_candidates_rejected += count
        self.gpu_rejection_reasons[reason] = (
            self.gpu_rejection_reasons.get(reason, 0) + count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": list(self.stages),
            "windows_scanned": self.windows_scanned,
            "windows_rejected": self.windows_rejected,
            "emotion_hits": self.emotion_hits,
            "curiosity_hits": self.curiosity_hits,
            "story_turn_hits": self.story_turn_hits,
            "trauma_hits": self.trauma_hits,
            "keyword_hits": self.keyword_hits,
            "gpu_candidates_generated": self.gpu_candidates_generated,
            "gpu_candidates_rejected": self.gpu_candidates_rejected,
            "gpu_rejection_reasons": dict(self.gpu_rejection_reasons),
            "fallback_candidates_generated": self.fallback_candidates_generated,
            "first_zero_stage": self.first_zero_stage,
            "notes": list(self.notes),
        }


def count_lexicon_hits(text: str) -> dict[str, int]:
    """Count signal lexicon hits in text (forensics only)."""
    from clip_engine.clip_signals import (
        CURIOSITY_HOOKS,
        DRAMATIC_TURNS,
        EMOTION_WORDS,
    )
    from clip_engine.transcript_candidate_scanner import (
        STORY_TURN_PHRASES,
        TRAUMA_PHRASES,
    )

    import re

    lower = text.lower()
    words = set(re.findall(r"\b[a-z]{3,}\b", lower))
    emotion = len(words & EMOTION_WORDS)
    curiosity = sum(1 for p in CURIOSITY_HOOKS if p in lower)
    story = sum(1 for p in STORY_TURN_PHRASES if p in lower)
    trauma = sum(1 for p in TRAUMA_PHRASES if p in lower)
    dramatic = sum(1 for p in DRAMATIC_TURNS if p in lower)
    keyword = emotion + curiosity + story + trauma + dramatic
    return {
        "emotion_hits": emotion,
        "curiosity_hits": curiosity,
        "story_turn_hits": story,
        "trauma_hits": trauma,
        "keyword_hits": keyword,
    }


__all__ = [
    "DiscoveryForensics",
    "count_lexicon_hits",
]
