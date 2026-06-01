# -*- coding: utf-8 -*-
import pytest
from clip_engine.captions import segments_to_srt, segments_to_ass, CAPTION_PRESETS

SEGS = [
    {"start": 0.0,  "end": 2.5, "text": "Hello world this is a test"},
    {"start": 2.5,  "end": 5.0, "text": "Second line of the caption"},
    {"start": 5.0,  "end": 8.3, "text": "Third line here"},
]

@pytest.mark.parametrize("preset", list(CAPTION_PRESETS.keys()))
def test_srt_all_presets(preset):
    srt = segments_to_srt(SEGS, clip_start=0.0, preset=preset)
    assert "00:00" in srt
    assert any(w in srt for w in ("Hello world", "HELLO WORLD"))

@pytest.mark.parametrize("preset", list(CAPTION_PRESETS.keys()))
def test_ass_all_presets(preset):
    ass = segments_to_ass(SEGS, clip_start=0.0, preset=preset)
    assert "[Script Info]" in ass
    assert "Dialogue:" in ass

def test_srt_empty():
    assert segments_to_srt([], 0.0).strip() == ""

def test_srt_offset_drops_before_start():
    srt = segments_to_srt(SEGS, clip_start=100.0)
    assert srt.strip() == ""
