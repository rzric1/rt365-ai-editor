# -*- coding: utf-8 -*-
import pytest
from unittest.mock import patch, MagicMock
from clip_engine.clip_analysis import _chunk_transcript
from clip_engine.clip_boundaries import ends_with_dangling_word, hook_title_is_incomplete

def test_chunk_transcript_five_parts():
    lines = [f"Line {i}" for i in range(100)]
    chunks = _chunk_transcript("\n".join(lines), n_chunks=5)
    assert len(chunks) == 5
    assert chunks[0][0] == "beginning"
    assert chunks[-1][0] == "ending"

def test_chunk_transcript_single_line():
    chunks = _chunk_transcript("Only one line", n_chunks=5)
    assert len(chunks) >= 1

def test_dangling_word_true():
    assert ends_with_dangling_word("I went to the") is True

def test_dangling_word_false():
    assert ends_with_dangling_word("I went to the store.") is False

def test_dangling_word_empty():
    assert ends_with_dangling_word("") is False

def test_hook_incomplete_fragment():
    assert hook_title_is_incomplete("and so") is True

def test_hook_complete():
    assert hook_title_is_incomplete("How I overcame cancer") is False

def test_hook_empty():
    assert hook_title_is_incomplete("") is True

@patch("openai.OpenAI")
def test_suggest_clips_returns_list_on_empty(mock_openai):
    from clip_engine.clip_analysis import suggest_clips_from_transcript
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"clips": []}'))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5)
    )
    result = suggest_clips_from_transcript("", api_key="sk-test", target_count=5)
    assert isinstance(result, list)
