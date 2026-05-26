"""

clip_engine/ai_profiles.py

Centralized AI reliability profiles for Clip Studio (models, budget, modes).

"""



from __future__ import annotations



from dataclasses import dataclass

from typing import Any



PROFILE_SAFE = "SAFE"

PROFILE_BALANCED = "BALANCED"

PROFILE_MAX_QUALITY = "MAX_QUALITY"



DEFAULT_PROFILE = PROFILE_SAFE



PROFILE_LABELS: dict[str, str] = {

    PROFILE_SAFE: "SAFE (Recommended)",

    PROFILE_BALANCED: "BALANCED",

    PROFILE_MAX_QUALITY: "MAX QUALITY (Experimental)",

}



LABEL_TO_PROFILE: dict[str, str] = {v: k for k, v in PROFILE_LABELS.items()}





@dataclass(frozen=True)

class AIProfile:

    name: str

    fast_model: str

    quality_model: str

    json_fallback_model: str

    max_tokens: int

    token_saver: bool

    discovery_mode: bool

    description: str

    warning: str | None = None

    prefer_gpu_prefilter: bool = True

    max_gpt_passes: int = 1

    max_gpt_regions: int = 5

    max_clip_length: float = 160.0

    context_before: float = 8.0

    context_after: float = 12.0

    target_gpu_shortlist_min: int = 10

    target_gpu_shortlist_max: int = 25

    max_active_gpt_regions: int = 5



    @property

    def max_active_regions(self) -> int:

        """Alias for pipeline / logging."""

        return self.max_active_gpt_regions





_PROFILES: dict[str, AIProfile] = {

    PROFILE_SAFE: AIProfile(

        name=PROFILE_SAFE,

        fast_model="gpt-4o-mini",

        quality_model="gpt-4o-mini",

        json_fallback_model="gpt-4o-mini",

        max_tokens=35_000,

        token_saver=True,

        discovery_mode=True,

        description="Most stable configuration for long podcasts and production workflows.",

        prefer_gpu_prefilter=True,

        max_gpt_passes=1,

        max_gpt_regions=5,

        max_clip_length=90.0,

        context_before=4.0,

        context_after=5.0,

        target_gpu_shortlist_min=10,

        target_gpu_shortlist_max=25,

        max_active_gpt_regions=5,

    ),

    PROFILE_BALANCED: AIProfile(

        name=PROFILE_BALANCED,

        fast_model="gpt-4o-mini",

        quality_model="gpt-5-mini",

        json_fallback_model="gpt-4o-mini",

        max_tokens=60_000,

        token_saver=True,

        discovery_mode=True,

        description="Balanced quality and reliability.",

        prefer_gpu_prefilter=True,

        max_gpt_passes=2,

        max_gpt_regions=5,

        max_clip_length=120.0,

        context_before=6.0,

        context_after=8.0,

        target_gpu_shortlist_min=15,

        target_gpu_shortlist_max=40,

        max_active_gpt_regions=5,

    ),

    PROFILE_MAX_QUALITY: AIProfile(

        name=PROFILE_MAX_QUALITY,

        fast_model="gpt-5-mini",

        quality_model="gpt-5-mini",

        json_fallback_model="gpt-4o-mini",

        max_tokens=70_000,

        token_saver=False,

        discovery_mode=True,

        description="Highest quality analysis mode.",

        warning="May increase retries and JSON fallback frequency during clip extraction.",

        prefer_gpu_prefilter=True,

        max_gpt_passes=3,

        max_gpt_regions=5,

        max_clip_length=160.0,

        context_before=8.0,

        context_after=12.0,

        target_gpu_shortlist_min=25,

        target_gpu_shortlist_max=60,

        max_active_gpt_regions=8,

    ),

}





def get_ai_profile(profile_name: str) -> AIProfile:

    """Return profile by key; unknown names fall back to SAFE."""

    key = (profile_name or "").strip().upper().replace(" ", "_")

    if key in ("SAFE_(RECOMMENDED)", "SAFE_RECOMMENDED"):

        key = PROFILE_SAFE

    if key not in _PROFILES:

        for alias, prof in _PROFILES.items():

            if key == alias or key in PROFILE_LABELS.get(alias, "").upper():

                return prof

        return _PROFILES[PROFILE_SAFE]

    return _PROFILES[key]





def profile_from_ui_label(label: str) -> AIProfile:

    """Map Streamlit selectbox label to profile."""

    key = LABEL_TO_PROFILE.get(label, PROFILE_SAFE)

    return get_ai_profile(key)





def apply_profile_to_session(session_state: Any, profile: AIProfile) -> None:

    """Apply profile settings to Streamlit session_state (never touches widget-bound keys)."""

    from clip_engine.effective_config import (
        apply_profile_non_widget_keys,
        apply_profile_widget_defaults,
    )

    apply_profile_non_widget_keys(session_state, profile)
    apply_profile_widget_defaults(session_state, profile)





def get_profile_help_text() -> str:

    return (

        "**SAFE** = most stable and production-ready (gpt-4o-mini, 35k token budget, GPU pre-filter).\n\n"

        "**BALANCED** = better reasoning with moderate fallback risk.\n\n"

        "**MAX QUALITY** = highest reasoning quality but may retry more often."

    )


