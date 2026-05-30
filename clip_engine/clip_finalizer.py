"""
clip_engine/clip_finalizer.py

Final quality pass before clips reach UI, cache, or export.
Expands incomplete boundaries, merges same-story fragments, rejects weak clips,
and repairs hook titles for watchable standalone shorts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from clip_engine.clip_boundaries import (
    ends_with_dangling_word,
    hook_title_is_incomplete,
    starts_mid_sentence,
)
from clip_engine.clip_scoring import assess_hook_quality, repair_hook_title_local
from clip_engine.transcription_utils import extract_transcript_window

logger = logging.getLogger("clip_engine.clip_finalizer")

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

DANGLING_END_WORDS = frozenset({
    "and", "but", "so", "because", "the", "a", "an", "to", "of", "with",
    "that", "which", "when", "while", "or", "if", "as", "at", "for", "in",
    "on", "by", "from", "into", "about", "than", "then", "also", "just",
    "even", "still", "yet", "nor", "though", "although", "unless", "until",
    "we", "they", "he", "she", "it", "i", "you", "my", "your", "their",
    "would", "could", "should", "have", "had", "was", "were", "been",
})

FILLER_START_RE = re.compile(
    r"^\s*(?:yeah|yes|yep|okay|ok|um+|uh+|i mean|you know|like|so|well|right|"
    r"honestly|literally|basically|actually)\b",
    re.IGNORECASE,
)

HOST_QUESTION_START_RE = re.compile(
    r"^\s*(?:"
    r"do you|did you|can you|could you|would you|will you|have you|had you|"
    r"are you|were you|is it|was it|what |how |why |when |where |who |"
    r"tell me|talk to me about|you ever|did she|did he|did they|"
    r"what's|what is|how's|how is|why's|why is"
    r")\b",
    re.IGNORECASE,
)

QUESTION_SENTENCE_RE = re.compile(
    r"\?\s*$|^\s*(?:do|does|did|can|could|would|will|have|has|had|is|are|were|was)\s+",
    re.IGNORECASE,
)

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")

GENERIC_TITLES = frozenset({
    "emotional moment",
    "key moment",
    "key moment from this episode",
    "untitled clip",
    "untitled clip moment",
    "clip moment",
    "podcast moment",
    "important moment",
    "powerful moment",
})

STOP_WORDS = frozenset({
    "that", "this", "with", "from", "they", "have", "been", "were", "was",
    "are", "for", "and", "the", "but", "not", "you", "all", "can", "her",
    "his", "she", "him", "our", "out", "just", "about", "into", "over",
    "think", "know", "take", "when", "your", "what", "there", "would", "like",
    "them", "then", "some", "could", "other", "than", "very", "also", "really",
    "because", "said", "well", "even", "back", "after", "most", "made",
    "being", "through", "where", "much", "before", "right", "going", "those",
    "something", "still", "such", "only", "never", "here", "more", "these",
    "same", "being", "yeah", "okay",
})

EMOTION_KEYWORDS = frozenset({
    "love", "hate", "cry", "cried", "crying", "afraid", "scared", "trauma",
    "abuse", "alcoholic", "alcoholism", "drunk", "drinking", "heartbreaking",
    "devastating", "broken", "hurt", "pain", "grief", "mother", "father", "dad",
    "mom", "brother", "sister", "childhood", "cancer", "death", "died", "suicide",
    "divorce", "abandoned", "betrayed", "shattered", "traumatic", "addiction",
})

TOKEN_RE = re.compile(r"[A-Za-z']+")

# Patterns for hook titles that are raw transcript fragments
FRAGMENT_TITLE_START_RE = re.compile(
    r"^\s*(?:uh+|um+|yeah|so i'?m gonna|to make the|and then|you know|i mean|"
    r"like i|so basically|honestly)\b",
    re.IGNORECASE,
)

# Trailing words that signal an incomplete title
_FRAGMENT_TRAIL_PHRASES = frozenset({
    "the", "and it's", "for the", "it's uh", "it's um", "and", "or", "but",
    "a", "an", "of", "with", "that", "which", "when", "while", "if", "as",
})

HOST_QUESTION_TITLE_RE = re.compile(
    r"^\s*(?:do you|did you|can you|could you|would you|will you|have you|"
    r"are you|were you|what |how |why |when |where |who |tell me|"
    r"what's|what is|how's|how is)\b.*\?",
    re.IGNORECASE,
)

# Production rejection / warning thresholds (finalizer-only)
FINALIZER_HARD_HOOK_THRESHOLD = 55
FINALIZER_SOFT_HOOK_THRESHOLD = 70
FINALIZER_NORMAL_MIN_DURATION = 20.0
FINALIZER_SOFT_SHORT_MAX_DURATION = 25.0
FINALIZER_HARD_BROKEN_MAX_DURATION = 10.0
GUEST_ANSWER_WINDOW_SECONDS = 15.0
# Bump when rejection/warning rules change so cached clips re-finalize.
FINALIZER_LOGIC_VERSION = 3


@dataclass
class FinalizerReport:
    checked: int = 0
    expanded: int = 0
    merged: int = 0
    rejected: int = 0
    hard_rejections: int = 0
    soft_warnings: int = 0
    hooks_repaired: int = 0
    kept: int = 0
    low_hook_warning: int = 0
    short_duration_warning: int = 0
    dangling_ending_warning: int = 0
    host_question_warning: int = 0
    metadata_grounding_warning: int = 0
    incomplete_beginning_warning: int = 0
    merge_pairs: list[tuple[int, int]] = field(default_factory=list)
    expanded_indices: list[int] = field(default_factory=list)
    rejected_indices: list[tuple[int, str]] = field(default_factory=list)
    hook_repairs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "expanded": self.expanded,
            "merged": self.merged,
            "rejected": self.rejected,
            "hard_rejections": self.hard_rejections,
            "soft_warnings": self.soft_warnings,
            "hooks_repaired": self.hooks_repaired,
            "kept": self.kept,
            "low_hook_warning": self.low_hook_warning,
            "short_duration_warning": self.short_duration_warning,
            "dangling_ending_warning": self.dangling_ending_warning,
            "host_question_warning": self.host_question_warning,
            "metadata_grounding_warning": self.metadata_grounding_warning,
            "incomplete_beginning_warning": self.incomplete_beginning_warning,
            "finalizer_logic_version": FINALIZER_LOGIC_VERSION,
            "merge_pairs": self.merge_pairs,
            "expanded_indices": self.expanded_indices,
            "rejected_indices": self.rejected_indices,
            "hook_repairs": self.hook_repairs,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _hook_title_is_filler_fragment(title: str) -> bool:
    """True when a hook title is a raw transcript fragment or host question, not a real title."""
    t = (title or "").strip()
    if not t:
        return True
    if FRAGMENT_TITLE_START_RE.match(t):
        return True
    if HOST_QUESTION_TITLE_RE.match(t):
        return True
    # Ends with any of the trailing incomplete phrases
    t_lower = t.lower().rstrip(".!?,;:")
    for phrase in _FRAGMENT_TRAIL_PHRASES:
        if t_lower == phrase or t_lower.endswith(" " + phrase):
            return True
    # Mid-sentence filler: contains "uh" or "um" surrounded by words
    words = TOKEN_RE.findall(t.lower())
    if len(words) >= 3 and any(w in {"uh", "um", "uhh", "umm"} for w in words[1:-1]):
        return True
    return False


def _clip_times(clip: dict) -> tuple[float, float]:
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    return t0, max(t1, t0 + 0.01)


def _clip_duration(clip: dict) -> float:
    t0, t1 = _clip_times(clip)
    return max(0.0, t1 - t0)


def _clip_has_invalid_timestamps(clip: dict) -> bool:
    try:
        t0 = float(clip.get("start_seconds", clip.get("start", 0)))
        t1 = float(clip.get("end_seconds", clip.get("end", 0)))
    except (TypeError, ValueError):
        return True
    return t1 <= t0


def _clip_has_empty_transcript(clip: dict, segments: list[dict] | None) -> bool:
    if segments:
        t0, t1 = _clip_times(clip)
        return not bool(extract_transcript_window(segments, t0, t1).strip())
    excerpt = str(
        clip.get("grounded_transcript_excerpt")
        or clip.get("selection_reason")
        or ""
    ).strip()
    return not excerpt


def clip_has_dangling_ending(clip: dict, segments: list[dict] | None = None) -> bool:
    """True when the clip window ends on a dangling phrase (not full incomplete-beginning)."""
    text = _window_text(clip, segments).strip()
    if not text:
        return False
    if ends_with_dangling_word(text):
        return True
    words = [w.lower() for w in TOKEN_RE.findall(text)]
    if words and not SENTENCE_END_RE.search(text):
        if words[-1] in {
            "was", "were", "is", "are", "am", "be", "been", "being",
            "had", "have", "has", "did", "do", "does",
        }:
            return True
        if len(words) >= 2 and words[-2] in {"and", "but", "so", "or", "the", "a", "an"}:
            return True
    if not SENTENCE_END_RE.search(text):
        tail = text[-100:]
        return ends_with_dangling_word(tail)
    return False


def _guest_answer_follows_within(
    clip: dict,
    segments: list[dict],
    *,
    within_seconds: float = GUEST_ANSWER_WINDOW_SECONDS,
) -> bool:
    if not segments:
        return True
    t0, t1 = _clip_times(clip)
    window_segs = _segments_in_range(segments, t0, min(t1, t0 + within_seconds))
    for seg in window_segs:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        if HOST_QUESTION_START_RE.search(seg_text) and "?" in seg_text[: min(60, len(seg_text))]:
            continue
        if FILLER_START_RE.match(seg_text) and len(TOKEN_RE.findall(seg_text)) < 5:
            continue
        if len(TOKEN_RE.findall(seg_text)) >= 4:
            return True
    return False


def _host_question_only_without_guest(clip: dict, segments: list[dict] | None) -> bool:
    if not clip_starts_with_host_question(clip, segments):
        return False
    if not segments:
        return False
    return not _guest_answer_follows_within(clip, segments)


def _hook_score(clip: dict) -> int:
    return int(clip.get("hook_quality_score", 0) or 0)


def _resolve_hook_score(clip: dict) -> int:
    score = _hook_score(clip)
    if score <= 0:
        title = str(clip.get("hook_title", ""))
        score, _ = assess_hook_quality(title)
        clip["hook_quality_score"] = score
    return score


def _hook_below_hard_threshold(clip: dict) -> bool:
    """Hard-reject pairing threshold only (55). Never use FINALIZER_SOFT_HOOK_THRESHOLD (70) here."""
    return _resolve_hook_score(clip) < FINALIZER_HARD_HOOK_THRESHOLD


def _hook_in_soft_warning_band(clip: dict) -> bool:
    score = _resolve_hook_score(clip)
    return FINALIZER_HARD_HOOK_THRESHOLD <= score < FINALIZER_SOFT_HOOK_THRESHOLD


def _has_metadata_grounding_warning(clip: dict) -> bool:
    if clip.get("boundary_warning") or clip.get("ungrounded_metadata"):
        return True
    for w in clip.get("warnings") or []:
        w_l = str(w).lower()
        if "metadata" in w_l or "ground" in w_l or "transcript" in w_l:
            return True
    return False


def _production_hard_reject_conditions(
    clip: dict,
    segments: list[dict] | None,
) -> list[str]:
    """Compound production failures; any one triggers hard rejection (after always-broken checks)."""
    conditions: list[str] = []
    if clip_has_dangling_ending(clip, segments) and _hook_below_hard_threshold(clip):
        conditions.append("dangling ending with hook below 55")
    if _host_question_only_without_guest(clip, segments):
        conditions.append("host-question-only start without guest answer within 15s")
    dur = _clip_duration(clip)
    if dur < FINALIZER_NORMAL_MIN_DURATION and _hook_below_hard_threshold(clip):
        conditions.append("duration under 20 seconds with hook below 55")
    if _clip_has_empty_transcript(clip, segments):
        conditions.append("empty transcript window")
    return conditions


def _always_hard_reject_reasons(clip: dict, segments: list[dict] | None) -> list[str]:
    reasons: list[str] = []
    if _clip_has_invalid_timestamps(clip):
        reasons.append("invalid timestamps")
    dur = _clip_duration(clip)
    if dur < FINALIZER_HARD_BROKEN_MAX_DURATION:
        reasons.append("duration under 10 seconds")
    if _clip_has_empty_transcript(clip, segments):
        reasons.append("empty transcript window")
    if clip.get("hook_fragment_unrepairable"):
        reasons.append("fragment hook title could not be repaired above quality threshold 55")
    return reasons


def _collect_soft_warnings(clip: dict, segments: list[dict] | None) -> list[str]:
    """
    Single-issue quality problems become warnings — never hard-reject alone.
    Hook scores 55–69 are warned; only scores < 55 pair with other failures for hard reject.
    """
    warnings: list[str] = []
    dur = _clip_duration(clip)

    hook_score = _resolve_hook_score(clip)
    if hook_score < FINALIZER_SOFT_HOOK_THRESHOLD:
        warnings.append("Hook quality below ideal threshold")

    if FINALIZER_NORMAL_MIN_DURATION <= dur < FINALIZER_SOFT_SHORT_MAX_DURATION:
        if not (
            clip_has_dangling_ending(clip, segments)
            and _hook_below_hard_threshold(clip)
        ):
            warnings.append("Short clip between 20–25 seconds")

    if clip_has_dangling_ending(clip, segments) and not _hook_below_hard_threshold(clip):
        warnings.append("Possible dangling ending")

    if clip_has_incomplete_ending(clip, segments) and not clip_has_dangling_ending(clip, segments):
        warnings.append("Possible incomplete ending")

    if clip_has_incomplete_beginning(clip, segments):
        if _host_question_only_without_guest(clip, segments):
            pass
        elif clip_starts_with_host_question(clip, segments):
            warnings.append("Host-question-heavy opening")
        elif clip_starts_with_filler(clip, segments):
            warnings.append("Clip may start mid-thought or with filler")
        else:
            warnings.append("Possible incomplete beginning")

    if _has_metadata_grounding_warning(clip):
        warnings.append("Metadata may not match final clip window")

    return list(dict.fromkeys(warnings))


def _window_text(clip: dict, segments: list[dict] | None) -> str:
    if segments:
        t0, t1 = _clip_times(clip)
        return extract_transcript_window(segments, t0, t1).strip()
    return str(
        clip.get("grounded_transcript_excerpt")
        or clip.get("selection_reason")
        or clip.get("hook_title", "")
        or ""
    ).strip()


def _keywords(text: str, *, min_len: int = 3) -> set[str]:
    return {
        w
        for w in TOKEN_RE.findall((text or "").lower())
        if len(w) >= min_len and w not in STOP_WORDS
    }


def _named_entities(text: str) -> set[str]:
    """Lightweight capitalized-phrase entities (no spaCy dependency)."""
    entities: set[str] = set()
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text or ""):
        phrase = match.group(1).strip()
        words = phrase.split()
        if words and words[0].lower() in {"i", "the", "a", "an", "so", "and", "but"}:
            continue
        if len(words) == 1 and words[0].lower() in STOP_WORDS:
            continue
        entities.add(phrase.lower())
    return entities


def clip_starts_with_filler(clip: dict, segments: list[dict] | None = None) -> bool:
    text = _window_text(clip, segments)
    if not text:
        return False
    first = text.split(None, 1)[0] if text.split() else ""
    if FILLER_START_RE.match(first + " "):
        return True
    words = TOKEN_RE.findall(text.lower())
    if len(words) <= 3 and words and words[0] in {
        "yeah", "yes", "yep", "okay", "ok", "um", "uh", "so", "well", "right",
    }:
        return True
    return False


def clip_starts_with_host_question(clip: dict, segments: list[dict] | None = None) -> bool:
    text = _window_text(clip, segments)
    if not text:
        return False
    if HOST_QUESTION_START_RE.search(text):
        return True
    first_chunk = text[:120]
    if "?" in first_chunk[:80] and QUESTION_SENTENCE_RE.search(first_chunk):
        return True
    speaker = str(clip.get("speaker", "")).lower()
    if speaker in ("host", "interviewer", "moderator"):
        return True
    return False


def clip_has_incomplete_beginning(clip: dict, segments: list[dict] | None = None) -> bool:
    text = _window_text(clip, segments)
    if not text:
        return True
    if clip_starts_with_host_question(clip, segments):
        return True
    if clip_starts_with_filler(clip, segments):
        return True
    if starts_mid_sentence(text):
        return True
    first = text[: min(80, len(text))]
    if not SENTENCE_END_RE.search(first) and len(TOKEN_RE.findall(first)) < 8:
        return True
    return False


def clip_has_incomplete_ending(clip: dict, segments: list[dict] | None = None) -> bool:
    text = _window_text(clip, segments)
    if not text:
        return True
    if ends_with_dangling_word(text):
        return True
    if not SENTENCE_END_RE.search(text.strip()):
        tail = text.strip()[-80:]
        if ends_with_dangling_word(tail):
            return True
        if hook_title_is_incomplete(str(clip.get("hook_title", ""))):
            return True
    return False


def clip_has_low_payoff(clip: dict, segments: list[dict] | None = None) -> bool:
    dur = _clip_duration(clip)
    text = _window_text(clip, segments).lower()
    if dur < 18 and not any(k in text for k in EMOTION_KEYWORDS):
        return True
    if clip_has_incomplete_ending(clip, segments) and dur < 35:
        return True
    if clip_starts_with_host_question(clip, segments) and dur < 40:
        return True
    return False


def _title_copied_from_transcript(title: str, window: str) -> bool:
    t = (title or "").strip().lower()
    w = (window or "").strip().lower()
    if not t or not w or len(t) < 12:
        return False
    if t in GENERIC_TITLES:
        return True
    t_words = TOKEN_RE.findall(t)
    if len(t_words) >= 4 and " ".join(t_words[:6]) in w[: max(len(w), len(t) + 40)]:
        return True
    return False


def _embedding_similarity(text_a: str, text_b: str) -> float | None:
    try:
        from clip_engine.semantic_ranking import embeddings_available, generate_embeddings

        if not embeddings_available():
            return None
        emb = generate_embeddings([text_a[:2000], text_b[:2000]])
        if emb.shape[0] < 2:
            return None
        import numpy as np

        a, b = emb[0], emb[1]
        return float(np.dot(a, b))
    except Exception as exc:
        logger.debug("Embedding similarity unavailable: %s", exc)
        return None


def clips_are_same_story_beat(
    a: dict,
    b: dict,
    segments: list[dict] | None = None,
    *,
    merge_gap_seconds: float = 20.0,
) -> bool:
    t0_a, t1_a = _clip_times(a)
    t0_b, t1_b = _clip_times(b)
    gap = max(0.0, t0_b - t1_a)
    if gap > merge_gap_seconds:
        return False

    text_a = _window_text(a, segments)
    text_b = _window_text(b, segments)
    kw_a = _keywords(text_a)
    kw_b = _keywords(text_b)
    overlap = kw_a & kw_b
    if len(overlap) >= 2:
        return True

    entity_overlap = _named_entities(text_a) & _named_entities(text_b)
    if entity_overlap:
        return True

    emb_sim = _embedding_similarity(text_a, text_b)
    if emb_sim is not None and emb_sim >= 0.72:
        return True

    return False


# ---------------------------------------------------------------------------
# Expansion / host shift
# ---------------------------------------------------------------------------


def _segments_in_range(
    segments: list[dict],
    t0: float,
    t1: float,
) -> list[dict]:
    return [
        s
        for s in segments
        if float(s.get("end", 0)) > t0 and float(s.get("start", 0)) < t1
    ]


def _sentence_spans(segments: list[dict], t0: float, t1: float) -> list[tuple[float, float, str]]:
    window = _segments_in_range(segments, t0, t1)
    if not window:
        return []

    spans: list[tuple[float, float, str]] = []
    buf_start = float(window[0].get("start", t0))
    buf_end = float(window[0].get("end", t0))
    buf_text = str(window[0].get("text", "")).strip()

    for seg in window[1:]:
        seg_start = float(seg.get("start", buf_end))
        seg_end = float(seg.get("end", seg_start))
        seg_text = str(seg.get("text", "")).strip()
        gap = seg_start - buf_end
        if gap > 1.2 or SENTENCE_END_RE.search(buf_text):
            if buf_text:
                spans.append((buf_start, buf_end, buf_text))
            buf_start = seg_start
            buf_text = seg_text
        else:
            buf_text = f"{buf_text} {seg_text}".strip()
        buf_end = seg_end

    if buf_text:
        spans.append((buf_start, buf_end, buf_text))
    return spans


def _shift_start_past_host_question(
    clip: dict,
    segments: list[dict],
) -> float:
    t0, t1 = _clip_times(clip)
    window_segs = _segments_in_range(segments, t0, min(t1, t0 + 45))
    if not window_segs:
        return t0

    for seg in window_segs:
        seg_start = float(seg.get("start", t0))
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        if HOST_QUESTION_START_RE.search(seg_text) or (
            "?" in seg_text[: min(60, len(seg_text))]
        ):
            continue
        if FILLER_START_RE.match(seg_text) and len(TOKEN_RE.findall(seg_text)) < 5:
            continue
        if len(TOKEN_RE.findall(seg_text)) >= 4:
            return max(t0, seg_start - 0.05)
    return t0


def expand_clip_to_sentence_or_thought_boundary(
    clip: dict,
    transcript_segments: list[dict],
    max_duration: float,
    *,
    min_duration: float = 25.0,
    growth_slack_seconds: float = 8.0,
) -> dict:
    c = dict(clip)
    t0, t1 = _clip_times(c)
    initial_dur = t1 - t0
    if not transcript_segments:
        return c

    # Do not grow an already-long clip toward the hard cap (reduces timeline overlap).
    if initial_dur >= max_duration - 2.0:
        growth_slack_seconds = 2.0
    elif initial_dur >= max_duration * 0.85:
        growth_slack_seconds = min(growth_slack_seconds, 4.0)
    expand_ceiling = min(max_duration, initial_dur + growth_slack_seconds)

    spans = _sentence_spans(transcript_segments, t0 - 45, t1 + 45)
    if not spans:
        return c

    if clip_starts_with_host_question(c, transcript_segments):
        t0 = _shift_start_past_host_question(c, transcript_segments)

    overlap_idx = [
        i
        for i, (start, end, _text) in enumerate(spans)
        if end > t0 and start < t1
    ]
    if not overlap_idx:
        return c

    first_i = overlap_idx[0]
    last_i = overlap_idx[-1]

    while first_i > 0:
        prev_text = spans[first_i - 1][2]
        cur_text = spans[first_i][2]
        if not (starts_mid_sentence(cur_text) or clip_starts_with_filler(c, transcript_segments)):
            break
        if spans[last_i][1] - spans[first_i - 1][0] > expand_ceiling:
            break
        if SENTENCE_END_RE.search(prev_text):
            first_i -= 1
            break
        first_i -= 1

    while last_i < len(spans) - 1:
        cur_text = spans[last_i][2]
        if SENTENCE_END_RE.search(cur_text) and not ends_with_dangling_word(cur_text):
            break
        if spans[last_i + 1][1] - spans[first_i][0] > expand_ceiling:
            break
        last_i += 1

    new_t0 = spans[first_i][0]
    new_t1 = spans[last_i][1]

    if new_t1 - new_t0 > expand_ceiling:
        trimmed_end = new_t0
        for start, end, text in spans[first_i : last_i + 1]:
            if end - new_t0 > expand_ceiling:
                break
            if SENTENCE_END_RE.search(text) and not ends_with_dangling_word(text):
                trimmed_end = end
        if trimmed_end > new_t0 + min_duration * 0.6:
            new_t1 = trimmed_end

    if new_t1 - new_t0 > expand_ceiling:
        new_t1 = new_t0 + expand_ceiling

    if new_t1 - new_t0 < min_duration * 0.85:
        return c

    if abs(new_t0 - t0) > 0.2 or abs(new_t1 - t1) > 0.2:
        c["start_seconds"] = round(new_t0, 3)
        c["end_seconds"] = round(new_t1, 3)
        c["finalizer_action"] = "expanded"
        c["finalizer_reason"] = "expanded to sentence boundaries"
        c["expansion_reason"] = "finalizer_sentence_expand"
    return c


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _merge_two_clips(
    a: dict,
    b: dict,
    segments: list[dict] | None,
) -> dict:
    t0 = min(_clip_times(a)[0], _clip_times(b)[0])
    t1 = max(_clip_times(a)[1], _clip_times(b)[1])
    score_a = int(a.get("composite_score", a.get("virality_score", 0)))
    score_b = int(b.get("composite_score", b.get("virality_score", 0)))
    base = a if score_a >= score_b else b
    other = b if base is a else a

    merged = dict(base)
    merged["start_seconds"] = round(t0, 3)
    merged["end_seconds"] = round(t1, 3)
    merged["composite_score"] = max(score_a, score_b)
    merged["virality_score"] = max(
        int(a.get("virality_score", 0)),
        int(b.get("virality_score", 0)),
    )
    merged["finalizer_action"] = "merged"
    merged["finalizer_reason"] = "same story beat"
    merged["merge_source_count"] = int(a.get("merge_source_count", 1) or 1) + int(
        b.get("merge_source_count", 1) or 1
    )
    merged["merged_from"] = list(dict.fromkeys(
        (merged.get("merged_from") or [])
        + [str(a.get("_wid", a.get("clip_id", ""))), str(b.get("_wid", b.get("clip_id", "")))]
    ))
    if segments:
        merged["grounded_transcript_excerpt"] = extract_transcript_window(segments, t0, t1)[:900]
    merged.pop("boundary_warning", None)
    merged.pop("boundary_repaired", None)
    return merged


def merge_adjacent_story_fragments(
    clips: list[dict],
    transcript_segments: list[dict] | None,
    merge_gap_seconds: float,
    max_duration: float,
    *,
    min_duration: float = 25.0,
    log: logging.Logger | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    if len(clips) < 2:
        return clips, []

    from clip_engine.clip_duration_governor import (
        SOFT_CAP_SECONDS,
        merge_allowed_max_duration,
        refresh_expansion_diagnostics,
    )

    log = log or logger
    sorted_clips = sorted(clips, key=lambda c: _clip_times(c)[0])
    merged_pairs: list[tuple[int, int]] = []
    if not sorted_clips:
        return [], merged_pairs

    out: list[dict] = []
    current = refresh_expansion_diagnostics(sorted_clips[0])
    for j in range(1, len(sorted_clips)):
        nxt = refresh_expansion_diagnostics(sorted_clips[j])
        cur_t0, cur_t1 = _clip_times(current)
        nxt_t0, nxt_t1 = _clip_times(nxt)
        combined_dur = nxt_t1 - cur_t0
        cur_dur = cur_t1 - cur_t0
        nxt_dur = nxt_t1 - nxt_t0
        merge_cap = merge_allowed_max_duration(current, nxt, max_duration)
        # Skip merge when either fragment is already long or union would dominate timeline.
        already_long = cur_dur > SOFT_CAP_SECONDS * 0.82 or nxt_dur > SOFT_CAP_SECONDS * 0.82
        if (
            clips_are_same_story_beat(
                current, nxt, transcript_segments, merge_gap_seconds=merge_gap_seconds
            )
            and combined_dur <= merge_cap
            and not already_long
        ):
            current = _merge_two_clips(current, nxt, transcript_segments)
            current = refresh_expansion_diagnostics(current)
            merged_pairs.append((j - 1, j))
            log.info(
                '[CLIP FINALIZER] merged clips=%d,%d combined=%.1fs cap=%.1fs sources=%d',
                j - 1,
                j,
                combined_dur,
                merge_cap,
                int(current.get("merge_source_count", 2)),
            )
        else:
            out.append(current)
            current = nxt
    out.append(current)
    return out, merged_pairs


# ---------------------------------------------------------------------------
# Rejection / hooks / watchability
# ---------------------------------------------------------------------------


def _watchability_score(
    clip: dict,
    segments: list[dict] | None,
    *,
    soft_warnings: list[str] | None = None,
) -> int:
    score = 70
    text = _window_text(clip, segments)
    dur = _clip_duration(clip)

    if clip_has_dangling_ending(clip, segments):
        score -= 12
    elif clip_has_incomplete_ending(clip, segments):
        score -= 8
    if clip_has_incomplete_beginning(clip, segments):
        score -= 10
    if clip_starts_with_host_question(clip, segments):
        score -= 8
    if clip_starts_with_filler(clip, segments):
        score -= 5
    if dur < FINALIZER_NORMAL_MIN_DURATION:
        score -= 15
    elif dur < FINALIZER_SOFT_SHORT_MAX_DURATION:
        score -= 6
    if SENTENCE_END_RE.search(text) and not starts_mid_sentence(text):
        score += 8
    if any(k in text.lower() for k in EMOTION_KEYWORDS):
        score += 6
    hook_q = _hook_score(clip)
    if hook_q:
        score = int(round((score + hook_q) / 2))
    for warn in soft_warnings or []:
        if "Hook quality" in warn:
            score -= 6
        elif "Short clip" in warn:
            score -= 4
        elif "dangling" in warn.lower():
            score -= 8
        elif "Host-question" in warn:
            score -= 5
        elif "Metadata" in warn:
            score -= 4
    return max(0, min(100, score))


def repair_final_hook_title(
    clip: dict,
    transcript_text: str | None = None,
    *,
    segments: list[dict] | None = None,
    log: logging.Logger | None = None,
) -> dict:
    log = log or logger
    c = dict(clip)
    old = str(c.get("hook_title", "")).strip()
    window = transcript_text or _window_text(c, segments)

    was_fragment = _hook_title_is_filler_fragment(old)
    needs_repair = (
        not old
        or hook_title_is_incomplete(old)
        or _title_copied_from_transcript(old, window)
        or old.lower() in GENERIC_TITLES
        or was_fragment
    )

    if needs_repair:
        new_title = repair_hook_title_local(old, window)
        if _title_copied_from_transcript(new_title, window) or hook_title_is_incomplete(new_title):
            new_title = _declarative_title_from_window(window)
        c["hook_title_before_finalizer"] = old
        c["hook_title"] = new_title
        c["finalizer_action"] = "hook_repaired"
        c["finalizer_reason"] = "hook title repaired for complete thought"
        new_score, _ = assess_hook_quality(new_title)
        c["hook_quality_score"] = new_score
        # Fragment titles that still can't reach quality threshold 55 after repair are flagged
        # for hard rejection — they cannot be passed to the UI as watchable clips.
        if was_fragment and new_score < FINALIZER_HARD_HOOK_THRESHOLD:
            c["hook_fragment_unrepairable"] = True
            log.info(
                '[HOOK REPAIR] clip=%s fragment title unrepairable after repair old="%s" new="%s" score=%d',
                c.get("_wid", c.get("clip_id", "?")),
                old[:60],
                new_title[:60],
                new_score,
            )
        else:
            log.info(
                '[HOOK REPAIR] clip=%s old="%s" new="%s" score=%s',
                c.get("_wid", c.get("clip_id", "?")),
                old[:60],
                new_title[:60],
                new_score,
            )
    else:
        score, _ = assess_hook_quality(old)
        c["hook_quality_score"] = score

    return c


def _declarative_title_from_window(window: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", (window or "").strip())
    for sent in sentences:
        words = TOKEN_RE.findall(sent)
        if len(words) < 5:
            continue
        title = " ".join(words[:10]).strip(".,;:!? ")
        title = title.strip()
        if len(title) > 70:
            title = " ".join(words[:8])
        if title:
            return title[0].upper() + title[1:] if len(title) > 1 else title.upper()
    words = TOKEN_RE.findall(window)
    if len(words) >= 4:
        title = " ".join(words[:8])
        return title[0].upper() + title[1:]
    return "A Story Worth Hearing"


def _apply_warning_counts(report: FinalizerReport | None, warnings: list[str]) -> None:
    if report is None or not warnings:
        return
    for warn in warnings:
        w = warn.lower()
        if "hook quality" in w:
            report.low_hook_warning += 1
        elif "short clip" in w:
            report.short_duration_warning += 1
        elif "dangling" in w:
            report.dangling_ending_warning += 1
        elif "host-question" in w:
            report.host_question_warning += 1
        elif "metadata" in w:
            report.metadata_grounding_warning += 1
        elif "incomplete beginning" in w:
            report.incomplete_beginning_warning += 1


def reject_unwatchable_clips(
    clips: list[dict],
    min_duration: float,
    *,
    segments: list[dict] | None = None,
    log: logging.Logger | None = None,
    report: FinalizerReport | None = None,
    **kwargs: Any,
) -> tuple[list[dict], list[tuple[int, str]]]:
    """
    Keep clips unless genuinely broken or a compound production failure applies.
    Single-issue problems become soft warnings on the clip.

    Hard hook threshold: FINALIZER_HARD_HOOK_THRESHOLD (55) — never 70.
    Soft hook band 55–69: kept with warning only.
    """
    if kwargs.get("min_hook_quality") is not None:
        logger.warning(
            "[CLIP FINALIZER] min_hook_quality=%s ignored; hard threshold is %s",
            kwargs["min_hook_quality"],
            FINALIZER_HARD_HOOK_THRESHOLD,
        )
    log = log or logger
    kept: list[dict] = []
    rejected: list[tuple[int, str]] = []
    _ = min_duration  # expansion uses caller min; rejection uses FINALIZER_* constants

    for idx, clip in enumerate(clips):
        c = dict(clip)
        hard_reasons = _always_hard_reject_reasons(c, segments)
        if not hard_reasons:
            production = _production_hard_reject_conditions(c, segments)
            if production:
                hard_reasons = production

        if hard_reasons:
            reason = "; ".join(hard_reasons)
            rejected.append((idx, reason))
            if report:
                report.hard_rejections += 1
            log.info(
                '[CLIP FINALIZER] rejected clip=%d reason="%s"',
                idx,
                hard_reasons[0],
            )
            continue

        soft = _collect_soft_warnings(c, segments)
        watch = _watchability_score(c, segments, soft_warnings=soft)
        c["watchability_score"] = watch
        if soft:
            c["finalizer_warnings"] = soft
            c.setdefault("warnings", [])
            for w in soft:
                entry = f"Finalizer: {w}"
                if entry not in c["warnings"]:
                    c["warnings"].append(entry)
            if report:
                report.soft_warnings += 1
                _apply_warning_counts(report, soft)
            c["finalizer_action"] = "kept_with_warnings"
            c["finalizer_reason"] = soft[0]
            log.info(
                '[CLIP FINALIZER] kept clip=%d with warnings: %s',
                idx,
                "; ".join(soft[:2]),
            )
        else:
            c["finalizer_action"] = "kept"
            c["finalizer_reason"] = "watchable story moment"
            log.info('[CLIP FINALIZER] kept clip=%d reason="watchable story moment"', idx)

        c["finalizer_checked"] = True
        c["finalizer_logic_version"] = FINALIZER_LOGIC_VERSION
        kept.append(c)

    return kept, rejected


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _finalize_clips_core(
    clips: list[dict],
    transcript_segments: list[dict] | None = None,
    transcript_text: str | None = None,
    min_duration: float = 20.0,
    max_duration: float = 90.0,
    merge_gap_seconds: float = 20.0,
    log: logging.Logger | None = None,
) -> tuple[list[dict], FinalizerReport]:
    log = log or logging.getLogger("clip_engine.clip_finalizer")
    if not clips:
        return [], FinalizerReport()

    report = FinalizerReport(checked=len(clips))

    try:
        working = [dict(c) for c in clips]

        # 1) Expand boundaries
        expanded: list[dict] = []
        for idx, clip in enumerate(working):
            old_t0, old_t1 = _clip_times(clip)
            if transcript_segments:
                expanded_clip = expand_clip_to_sentence_or_thought_boundary(
                    clip,
                    transcript_segments,
                    max_duration,
                    min_duration=min_duration,
                )
            else:
                expanded_clip = dict(clip)
            if expanded_clip.get("finalizer_action") == "expanded":
                report.expanded += 1
                report.expanded_indices.append(idx)
                log.info(
                    "[CLIP FINALIZER] expanded clip=%d start old=%.1f new=%.1f end old=%.1f new=%.1f",
                    idx,
                    old_t0,
                    float(expanded_clip.get("start_seconds", old_t0)),
                    old_t1,
                    float(expanded_clip.get("end_seconds", old_t1)),
                )
            expanded.append(expanded_clip)
        working = expanded

        # 2) Merge adjacent story beats
        working, merge_pairs = merge_adjacent_story_fragments(
            working,
            transcript_segments,
            merge_gap_seconds,
            max_duration,
            min_duration=min_duration,
            log=log,
        )
        report.merged = len(merge_pairs)
        report.merge_pairs = merge_pairs

        # 3) Re-expand merged clips (growth-limited; pipeline governor clamps after)
        from clip_engine.clip_duration_governor import (
            clamp_clip_to_duration_policy,
            clip_virality_score,
            refresh_expansion_diagnostics,
        )

        re_expanded: list[dict] = []
        for clip in working:
            if clip.get("finalizer_action") == "merged" and transcript_segments:
                clip = expand_clip_to_sentence_or_thought_boundary(
                    clip,
                    transcript_segments,
                    max_duration,
                    min_duration=min_duration,
                    growth_slack_seconds=4.0,
                )
                clip, _ = clamp_clip_to_duration_policy(
                    clip,
                    0.0,
                    pre_virality=clip_virality_score(clip) <= 90,
                )
                clip = refresh_expansion_diagnostics(clip)
            re_expanded.append(clip)
        working = re_expanded

        # 4) Hook repair before rejection
        repaired_clips: list[dict] = []
        for idx, clip in enumerate(working):
            clip = repair_final_hook_title(
                clip,
                segments=transcript_segments,
                log=log,
            )
            if clip.get("finalizer_action") == "hook_repaired":
                report.hooks_repaired += 1
                report.hook_repairs.append({
                    "index": idx,
                    "old": clip.get("hook_title_before_finalizer", ""),
                    "new": clip.get("hook_title", ""),
                    "score": clip.get("hook_quality_score", 0),
                })
            repaired_clips.append(clip)
        working = repaired_clips

        # 5) Reject unwatchable
        kept, rejected = reject_unwatchable_clips(
            working,
            min_duration,
            segments=transcript_segments,
            log=log,
            report=report,
        )
        report.rejected = len(rejected)
        report.hard_rejections = len(rejected)
        report.rejected_indices = rejected
        report.kept = len(kept)

        log.info(
            "[CLIP FINALIZER] checked clips=%d expanded=%d merged=%d "
            "hard_rejected=%d soft_warnings=%d hooks_repaired=%d kept=%d",
            report.checked,
            report.expanded,
            report.merged,
            report.hard_rejections,
            report.soft_warnings,
            report.hooks_repaired,
            report.kept,
        )
        return kept, report

    except Exception as exc:
        log.exception("[CLIP FINALIZER] failed — returning original clips: %s", exc)
        report.errors.append(str(exc))
        return [dict(c) for c in clips], report


def finalize_clips_for_ui(
    clips: list[dict],
    transcript_segments: list[dict] | None = None,
    transcript_text: str | None = None,
    min_duration: float = 20.0,
    max_duration: float = 90.0,
    merge_gap_seconds: float = 20.0,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Final quality pass. On failure, returns the original clips unchanged."""
    out, _report = _finalize_clips_core(
        clips,
        transcript_segments=transcript_segments,
        transcript_text=transcript_text,
        min_duration=min_duration,
        max_duration=max_duration,
        merge_gap_seconds=merge_gap_seconds,
        log=logger,
    )
    return out


def finalize_clips_with_report(
    clips: list[dict],
    transcript_segments: list[dict] | None = None,
    transcript_text: str | None = None,
    **kwargs: Any,
) -> tuple[list[dict], FinalizerReport]:
    """Like finalize_clips_for_ui but always returns a report object."""
    log = kwargs.pop("logger", None)
    return _finalize_clips_core(
        clips,
        transcript_segments=transcript_segments,
        transcript_text=transcript_text,
        log=log,
        **kwargs,
    )


def validate_clip_for_export(
    clip: dict,
    *,
    min_duration: float = 1.0,
    max_duration: float = 600.0,
) -> tuple[bool, str]:
    """Return (ok, reason) for a single export candidate."""
    title = str(
        clip.get("hook_title")
        or clip.get("export_title")
        or clip.get("grounded_hook_title")
        or ""
    ).strip()
    if not title:
        return False, "missing title"
    try:
        t0 = float(clip.get("start_seconds", clip.get("start", 0)))
        t1 = float(clip.get("end_seconds", clip.get("end", 0)))
    except (TypeError, ValueError):
        return False, "invalid start/end times"
    if t1 <= t0:
        return False, "end must be after start"
    dur = t1 - t0
    if dur < min_duration:
        return False, f"duration {dur:.1f}s below minimum"
    if dur > max_duration:
        return False, f"duration {dur:.1f}s above maximum"
    if hook_title_is_incomplete(title):
        return False, "incomplete hook title"
    return True, ""


def ensure_clips_finalized(
    clips: list[dict],
    transcript_segments: list[dict] | None = None,
    *,
    min_duration: float = 20.0,
    max_duration: float = 90.0,
    merge_gap_seconds: float = 20.0,
    logger: logging.Logger | None = None,
) -> tuple[list[dict], bool]:
    """Run finalizer only when clips lack finalizer_checked metadata."""
    if not clips:
        return [], False
    needs_finalize = any(
        not bool(c.get("finalizer_checked"))
        or int(c.get("finalizer_logic_version", 0)) < FINALIZER_LOGIC_VERSION
        for c in clips
    )
    if not needs_finalize:
        return clips, False
    out = finalize_clips_for_ui(
        clips,
        transcript_segments=transcript_segments,
        min_duration=min_duration,
        max_duration=max_duration,
        merge_gap_seconds=merge_gap_seconds,
        logger=logger,
    )
    return out, True


__all__ = [
    "FINALIZER_HARD_HOOK_THRESHOLD",
    "FINALIZER_SOFT_HOOK_THRESHOLD",
    "FINALIZER_LOGIC_VERSION",
    "FinalizerReport",
    "clip_starts_with_filler",
    "clip_starts_with_host_question",
    "clip_has_incomplete_beginning",
    "clip_has_incomplete_ending",
    "clip_has_dangling_ending",
    "clip_has_low_payoff",
    "clips_are_same_story_beat",
    "expand_clip_to_sentence_or_thought_boundary",
    "merge_adjacent_story_fragments",
    "reject_unwatchable_clips",
    "repair_final_hook_title",
    "finalize_clips_for_ui",
    "finalize_clips_with_report",
    "ensure_clips_finalized",
    "validate_clip_for_export",
]
