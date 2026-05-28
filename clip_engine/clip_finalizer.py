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


@dataclass
class FinalizerReport:
    checked: int = 0
    expanded: int = 0
    merged: int = 0
    rejected: int = 0
    hooks_repaired: int = 0
    kept: int = 0
    merge_pairs: list[tuple[int, int]] = field(default_factory=list)
    expanded_indices: list[int] = field(default_factory=list)
    rejected_indices: list[int] = field(default_factory=list)
    hook_repairs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "expanded": self.expanded,
            "merged": self.merged,
            "rejected": self.rejected,
            "hooks_repaired": self.hooks_repaired,
            "kept": self.kept,
            "merge_pairs": self.merge_pairs,
            "expanded_indices": self.expanded_indices,
            "rejected_indices": self.rejected_indices,
            "hook_repairs": self.hook_repairs,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _clip_times(clip: dict) -> tuple[float, float]:
    t0 = float(clip.get("start_seconds", clip.get("start", 0)))
    t1 = float(clip.get("end_seconds", clip.get("end", t0)))
    return t0, max(t1, t0 + 0.01)


def _clip_duration(clip: dict) -> float:
    t0, t1 = _clip_times(clip)
    return max(0.0, t1 - t0)


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
) -> dict:
    c = dict(clip)
    t0, t1 = _clip_times(c)
    if not transcript_segments:
        return c

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
        if spans[last_i][1] - spans[first_i - 1][0] > max_duration:
            break
        if SENTENCE_END_RE.search(prev_text):
            first_i -= 1
            break
        first_i -= 1

    while last_i < len(spans) - 1:
        cur_text = spans[last_i][2]
        if SENTENCE_END_RE.search(cur_text) and not ends_with_dangling_word(cur_text):
            break
        if spans[last_i + 1][1] - spans[first_i][0] > max_duration:
            break
        last_i += 1

    new_t0 = spans[first_i][0]
    new_t1 = spans[last_i][1]

    if new_t1 - new_t0 > max_duration:
        trimmed_end = new_t0
        for start, end, text in spans[first_i : last_i + 1]:
            if end - new_t0 > max_duration:
                break
            if SENTENCE_END_RE.search(text) and not ends_with_dangling_word(text):
                trimmed_end = end
        if trimmed_end > new_t0 + min_duration * 0.6:
            new_t1 = trimmed_end

    if new_t1 - new_t0 < min_duration * 0.85:
        return c

    if abs(new_t0 - t0) > 0.2 or abs(new_t1 - t1) > 0.2:
        c["start_seconds"] = round(new_t0, 3)
        c["end_seconds"] = round(new_t1, 3)
        c["finalizer_action"] = "expanded"
        c["finalizer_reason"] = "expanded to sentence boundaries"
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

    log = log or logger
    sorted_clips = sorted(clips, key=lambda c: _clip_times(c)[0])
    merged_pairs: list[tuple[int, int]] = []
    if not sorted_clips:
        return [], merged_pairs

    out: list[dict] = []
    current = sorted_clips[0]
    for j in range(1, len(sorted_clips)):
        nxt = sorted_clips[j]
        combined_dur = _clip_times(nxt)[1] - _clip_times(current)[0]
        if (
            clips_are_same_story_beat(
                current, nxt, transcript_segments, merge_gap_seconds=merge_gap_seconds
            )
            and combined_dur <= max_duration
        ):
            current = _merge_two_clips(current, nxt, transcript_segments)
            merged_pairs.append((j - 1, j))
            log.info(
                '[CLIP FINALIZER] merged clips=%d,%d reason="same story beat"',
                j - 1,
                j,
            )
        else:
            out.append(current)
            current = nxt
    out.append(current)
    return out, merged_pairs


# ---------------------------------------------------------------------------
# Rejection / hooks / watchability
# ---------------------------------------------------------------------------


def _watchability_score(clip: dict, segments: list[dict] | None) -> int:
    score = 70
    text = _window_text(clip, segments)
    dur = _clip_duration(clip)

    if clip_has_incomplete_ending(clip, segments):
        score -= 25
    if clip_has_incomplete_beginning(clip, segments):
        score -= 20
    if clip_starts_with_host_question(clip, segments):
        score -= 15
    if clip_starts_with_filler(clip, segments):
        score -= 8
    if dur < min(25.0, float(clip.get("_min_duration", 25))):
        score -= 10
    if SENTENCE_END_RE.search(text) and not starts_mid_sentence(text):
        score += 8
    if any(k in text.lower() for k in EMOTION_KEYWORDS):
        score += 6
    hook_q = int(clip.get("hook_quality_score", 0))
    if hook_q:
        score = int(round((score + hook_q) / 2))
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

    needs_repair = (
        not old
        or hook_title_is_incomplete(old)
        or _title_copied_from_transcript(old, window)
        or old.lower() in GENERIC_TITLES
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


def reject_unwatchable_clips(
    clips: list[dict],
    min_duration: float,
    *,
    segments: list[dict] | None = None,
    min_hook_quality: float = 70.0,
    log: logging.Logger | None = None,
) -> tuple[list[dict], list[tuple[int, str]]]:
    log = log or logger
    kept: list[dict] = []
    rejected: list[tuple[int, str]] = []

    for idx, clip in enumerate(clips):
        reasons: list[str] = []
        if clip_has_incomplete_ending(clip, segments):
            reasons.append("incomplete ending")
        if clip_has_incomplete_beginning(clip, segments):
            reasons.append("incomplete beginning")
        if clip_starts_with_host_question(clip, segments) and _clip_duration(clip) < 45:
            reasons.append("host-question-only start")
        if clip_has_low_payoff(clip, segments):
            reasons.append("low payoff")
        if _clip_duration(clip) < min_duration and not any(
            k in _window_text(clip, segments).lower() for k in EMOTION_KEYWORDS
        ):
            reasons.append("too short without standalone moment")

        watch = _watchability_score(clip, segments)
        clip["watchability_score"] = watch
        hook_score = int(clip.get("hook_quality_score", 0))
        virality = int(clip.get("virality_score", clip.get("composite_score", 0)))
        strong_moment = virality >= 75 and watch >= 55
        if hook_score and hook_score < min_hook_quality and not strong_moment:
            reasons.append(f"hook quality {hook_score}<{min_hook_quality:.0f}")
        elif hook_title_is_incomplete(str(clip.get("hook_title", ""))) and not strong_moment:
            reasons.append("incomplete hook title")
        if watch < 45 and reasons:
            reasons.append("low watchability")

        if reasons:
            rejected.append((idx, "; ".join(reasons)))
            log.info(
                '[CLIP FINALIZER] rejected clip=%d reason="%s"',
                idx,
                reasons[0],
            )
            continue

        clip["finalizer_checked"] = True
        clip.setdefault("finalizer_action", "kept")
        clip.setdefault("finalizer_reason", "watchable story moment")
        kept.append(clip)
        log.info('[CLIP FINALIZER] kept clip=%d reason="watchable story moment"', idx)

    return kept, rejected


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _finalize_clips_core(
    clips: list[dict],
    transcript_segments: list[dict] | None = None,
    transcript_text: str | None = None,
    min_duration: float = 25.0,
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

        # 3) Re-expand merged clips
        re_expanded: list[dict] = []
        for clip in working:
            if clip.get("finalizer_action") == "merged" and transcript_segments:
                clip = expand_clip_to_sentence_or_thought_boundary(
                    clip,
                    transcript_segments,
                    max_duration,
                    min_duration=min_duration,
                )
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
        )
        report.rejected = len(rejected)
        report.rejected_indices = rejected
        report.kept = len(kept)

        log.info(
            "[CLIP FINALIZER] checked clips=%d expanded=%d merged=%d rejected=%d hooks_repaired=%d kept=%d",
            report.checked,
            report.expanded,
            report.merged,
            report.rejected,
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
    min_duration: float = 25.0,
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
    min_duration: float = 25.0,
    max_duration: float = 90.0,
    merge_gap_seconds: float = 20.0,
    logger: logging.Logger | None = None,
) -> tuple[list[dict], bool]:
    """Run finalizer only when clips lack finalizer_checked metadata."""
    if not clips:
        return [], False
    if all(bool(c.get("finalizer_checked")) for c in clips):
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
    "FinalizerReport",
    "clip_starts_with_filler",
    "clip_starts_with_host_question",
    "clip_has_incomplete_beginning",
    "clip_has_incomplete_ending",
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
