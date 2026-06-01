# -*- coding: utf-8 -*-
"""Targeted tests for clip finalizer rejection thresholds."""

from __future__ import annotations

import unittest

from clip_engine.clip_finalizer import (
    FINALIZER_HARD_HOOK_THRESHOLD,
    FINALIZER_NORMAL_MIN_DURATION,
    reject_unwatchable_clips,
)


def _clip(
    *,
    start: float = 0.0,
    end: float = 30.0,
    hook: str = "A Complete Story Title",
    hook_score: int = 75,
    window: str = "This is a complete sentence about childhood and trauma.",
) -> dict:
    return {
        "start_seconds": start,
        "end_seconds": end,
        "hook_title": hook,
        "hook_quality_score": hook_score,
        "grounded_transcript_excerpt": window,
    }


class TestClipFinalizerThresholds(unittest.TestCase):
    def _reject(self, clip: dict, segments: list | None = None) -> tuple[list[dict], list]:
        kept, rejected = reject_unwatchable_clips([clip], 20.0, segments=segments)
        return kept, rejected

    def test_hook_60_kept_with_warning(self) -> None:
        kept, rejected = self._reject(_clip(hook_score=60, end=30.0))
        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(any("Hook quality" in w for w in kept[0].get("finalizer_warnings", [])))

    def test_22_second_clip_kept_with_warning(self) -> None:
        kept, rejected = self._reject(_clip(end=22.0, hook_score=75))
        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(
            any("Short clip" in w for w in kept[0].get("finalizer_warnings", []))
        )

    def test_dangling_ending_alone_kept_with_warning(self) -> None:
        kept, rejected = self._reject(
            _clip(
                hook_score=75,
                window="She talked about her mother and he was",
            )
        )
        self.assertEqual(len(rejected), 0)
        self.assertTrue(
            any("dangling" in w.lower() for w in kept[0].get("finalizer_warnings", []))
        )

    def test_hook_50_alone_kept_with_warning(self) -> None:
        kept, rejected = self._reject(_clip(hook_score=50))
        self.assertEqual(len(rejected), 0)
        self.assertLess(_hook_score(kept[0]), FINALIZER_HARD_HOOK_THRESHOLD + 15)
        self.assertTrue(any("Hook quality" in w for w in kept[0].get("finalizer_warnings", [])))

    def test_hook_50_plus_dangling_hard_rejected(self) -> None:
        _, rejected = self._reject(
            _clip(
                hook_score=50,
                window="It was devastating and he was",
            )
        )
        self.assertEqual(len(rejected), 1)
        self.assertIn("dangling", rejected[0][1].lower())

    def test_duration_18_hook_50_hard_rejected(self) -> None:
        _, rejected = self._reject(_clip(start=0, end=18, hook_score=50))
        self.assertEqual(len(rejected), 1)
        self.assertTrue(
            "20" in rejected[0][1] or "hook" in rejected[0][1].lower()
        )

    def test_empty_transcript_hard_rejected(self) -> None:
        clip = _clip(window="", hook="Title Only")
        clip["grounded_transcript_excerpt"] = ""
        clip["selection_reason"] = ""
        _, rejected = self._reject(clip)
        self.assertEqual(len(rejected), 1)
        self.assertIn("empty", rejected[0][1].lower())

    def test_invalid_timestamps_hard_rejected(self) -> None:
        _, rejected = self._reject(_clip(start=40, end=10))
        self.assertEqual(len(rejected), 1)
        self.assertIn("timestamp", rejected[0][1].lower())

    def test_under_10_seconds_hard_rejected(self) -> None:
        _, rejected = self._reject(_clip(start=0, end=8, hook_score=80))
        self.assertEqual(len(rejected), 1)
        self.assertIn("10", rejected[0][1])

    def test_hook_67_kept_with_warning_not_hard_reject(self) -> None:
        kept, rejected = self._reject(_clip(hook_score=67, end=35.0))
        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(any("Hook quality" in w for w in kept[0].get("finalizer_warnings", [])))
        self.assertNotIn("70", " ".join(kept[0].get("finalizer_warnings", [])))

    def test_incomplete_beginning_alone_kept_with_warning(self) -> None:
        kept, rejected = self._reject(
            _clip(
                hook_score=75,
                window="and then she told me about her childhood growing up in Ohio",
            )
        )
        self.assertEqual(len(rejected), 0)
        self.assertTrue(
            any(
                "incomplete beginning" in w.lower() or "mid-thought" in w.lower()
                for w in kept[0].get("finalizer_warnings", [])
            )
        )


def _hook_score(clip: dict) -> int:
    return int(clip.get("hook_quality_score", 0))


if __name__ == "__main__":
    unittest.main()
