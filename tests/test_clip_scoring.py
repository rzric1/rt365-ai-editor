# -*- coding: utf-8 -*-
import pytest
from clip_engine.clip_scoring import compute_virality_score, assess_hook_quality

SEGS = [
    {"start": 0.0, "end": 3.0, "text": "When she was diagnosed with cancer everything changed."},
    {"start": 3.0, "end": 6.0, "text": "The doctor said she had three months to live."},
]

def _clip(title="How cancer changed our lives forever", start=0.0, end=45.0):
    return {"hook_title": title, "start_seconds": start, "end_seconds": end,
            "composite_score": 75, "dominant_signal": "story",
            "boundary_repaired": False, "boundary_warning": False}

def test_score_in_range():
    score, breakdown, explanation = compute_virality_score(_clip(), SEGS)
    assert 0 <= score <= 100
    assert isinstance(breakdown, dict)
    assert isinstance(explanation, str)

def test_emotional_scores_higher_than_neutral():
    s_e, _, _ = compute_virality_score(_clip("She cried at the terrifying diagnosis"), SEGS)
    s_n, _, _ = compute_virality_score(_clip("A discussion about quarterly planning"), SEGS)
    assert s_e >= s_n

def test_strong_hook_scores_well():
    score, warning = assess_hook_quality("Why getting nervous before a fight is essential")
    assert score >= 55

def test_fragment_hook_scores_poorly():
    score, _ = assess_hook_quality("and so I")
    assert score < 55

def test_empty_hook():
    score, _ = assess_hook_quality("")
    assert score == 0

@pytest.mark.parametrize("strategy", ["Balanced", "Emotional", "Educational", "Debate"])
def test_all_strategies_accepted(strategy):
    score, _, _ = compute_virality_score(_clip(), SEGS, clip_strategy=strategy)
    assert 0 <= score <= 100
