"""
clip_engine/clip_style.py
Clip style profiles: Balanced, Micro clips, Long story clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClipStyle = Literal["Balanced", "Micro clips", "Long story clips"]

CLIP_STYLE_OPTIONS: list[ClipStyle] = ["Balanced", "Micro clips", "Long story clips"]


@dataclass(frozen=True)
class ClipStyleProfile:
    """Duration and generation targets for a clip style."""

    ideal_min_seconds: float
    ideal_max_seconds: float
    ai_max_clip_seconds: float
    expansion_max_seconds: float
    split_threshold_seconds: float
    sub_clip_max_seconds: float
    hard_max_export_seconds: float
    context_before: float
    context_after: float
    max_clips_per_region: int
    min_score: int
    micro_focus: bool = False


def get_clip_style_profile(
    style: ClipStyle,
    *,
    user_min_seconds: float = 25.0,
    user_max_seconds: float = 160.0,
) -> ClipStyleProfile:
    """Map UI clip style to generation/expansion parameters."""
    user_min = max(5.0, float(user_min_seconds))
    user_max = max(user_min, min(float(user_max_seconds), 120.0))

    if style == "Micro clips":
        return ClipStyleProfile(
            ideal_min_seconds=max(user_min, 30.0),
            ideal_max_seconds=min(user_max, 75.0),
            ai_max_clip_seconds=min(user_max, 90.0),
            expansion_max_seconds=min(user_max, 90.0),
            split_threshold_seconds=75.0,
            sub_clip_max_seconds=75.0,
            hard_max_export_seconds=min(user_max, 90.0),
            context_before=4.0,
            context_after=8.0,
            max_clips_per_region=18,
            min_score=48,
            micro_focus=True,
        )

    if style == "Long story clips":
        return ClipStyleProfile(
            ideal_min_seconds=max(user_min, 90.0),
            ideal_max_seconds=min(user_max, 160.0),
            ai_max_clip_seconds=user_max,
            expansion_max_seconds=user_max,
            split_threshold_seconds=160.0,
            sub_clip_max_seconds=120.0,
            hard_max_export_seconds=user_max,
            context_before=10.0,
            context_after=15.0,
            max_clips_per_region=12,
            min_score=55,
            micro_focus=False,
        )

    # Balanced (default)
    return ClipStyleProfile(
        ideal_min_seconds=max(user_min, 45.0),
        ideal_max_seconds=min(user_max, 90.0),
        ai_max_clip_seconds=min(user_max, 90.0),
        expansion_max_seconds=min(user_max, 90.0),
        split_threshold_seconds=90.0,
        sub_clip_max_seconds=90.0,
        hard_max_export_seconds=min(user_max, 90.0),
        context_before=6.0,
        context_after=10.0,
        max_clips_per_region=15,
        min_score=50,
        micro_focus=False,
    )
