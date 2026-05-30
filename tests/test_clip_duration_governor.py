"""Tests for clip duration governor (soft 90 / hard 120 / virality exception)."""

import unittest

from clip_engine.clip_duration_governor import (
    HARD_CAP_SECONDS,
    MAX_GROWTH_PERCENT,
    SOFT_CAP_SECONDS,
    apply_duration_policy_batch,
    clamp_clip_to_duration_policy,
    compute_timeline_occupancy,
    effective_max_duration,
    merge_allowed_max_duration,
    pin_ai_core_window,
    refresh_expansion_diagnostics,
    scaled_context_padding,
)


class TestDurationGovernor(unittest.TestCase):
    def test_effective_max_low_virality(self) -> None:
        clip = {"virality_score": 70}
        self.assertEqual(effective_max_duration(clip), SOFT_CAP_SECONDS)

    def test_effective_max_high_virality(self) -> None:
        clip = {"virality_score": 95}
        self.assertEqual(effective_max_duration(clip), HARD_CAP_SECONDS)

    def test_clamp_soft_when_not_viral(self) -> None:
        clip = pin_ai_core_window({
            "ai_core_start": 100.0,
            "ai_core_end": 150.0,
            "start_seconds": 90.0,
            "end_seconds": 210.0,
            "virality_score": 60,
        })
        fixed, actions = clamp_clip_to_duration_policy(clip, media_duration=600.0)
        self.assertLessEqual(fixed["end_seconds"] - fixed["start_seconds"], SOFT_CAP_SECONDS + 0.5)
        self.assertTrue(actions)
        self.assertLessEqual(float(fixed["growth_percent"]), MAX_GROWTH_PERCENT + 0.5)

    def test_growth_100_percent_clamp(self) -> None:
        """24s core expanded to 131s must shrink (production regression)."""
        clip = pin_ai_core_window({
            "ai_core_start": 100.0,
            "ai_core_end": 124.0,
            "start_seconds": 50.0,
            "end_seconds": 181.0,
            "virality_score": 55,
        })
        fixed, actions = clamp_clip_to_duration_policy(clip, media_duration=600.0)
        dur = fixed["end_seconds"] - fixed["start_seconds"]
        self.assertLessEqual(dur, 48.0 + 1.0)
        self.assertTrue(any("growth_clamp" in a for a in actions))
        self.assertIn("expansion_reason", fixed)

    def test_high_virality_allows_up_to_hard(self) -> None:
        clip = {
            "original_start": 0.0,
            "original_end": 80.0,
            "start_seconds": 0.0,
            "end_seconds": 115.0,
            "virality_score": 92,
        }
        fixed, actions = clamp_clip_to_duration_policy(clip, media_duration=600.0)
        dur = fixed["end_seconds"] - fixed["start_seconds"]
        self.assertLessEqual(dur, HARD_CAP_SECONDS + 0.5)
        self.assertGreater(dur, SOFT_CAP_SECONDS)
        self.assertFalse(actions)

    def test_pre_virality_always_soft(self) -> None:
        clip = {
            "original_start": 0.0,
            "original_end": 70.0,
            "start_seconds": 0.0,
            "end_seconds": 110.0,
            "virality_score": 95,
        }
        fixed, actions = clamp_clip_to_duration_policy(
            clip, media_duration=600.0, pre_virality=True,
        )
        self.assertLessEqual(
            fixed["end_seconds"] - fixed["start_seconds"],
            SOFT_CAP_SECONDS + 0.5,
        )
        self.assertTrue(actions)

    def test_diagnostics_fields(self) -> None:
        clip = {
            "original_start": 10.0,
            "original_end": 40.0,
            "start_seconds": 0.0,
            "end_seconds": 95.0,
        }
        d = refresh_expansion_diagnostics(clip)
        self.assertEqual(d["expanded_start"], 0.0)
        self.assertEqual(d["expanded_end"], 95.0)
        self.assertEqual(d["original_duration"], 30.0)
        self.assertEqual(d["expanded_duration"], 95.0)
        self.assertEqual(d["growth_seconds"], 65.0)
        self.assertEqual(d["merge_source_count"], 1)
        self.assertIn("expansion_justification", d)

    def test_timeline_occupancy_overlap(self) -> None:
        clips = [
            {"start_seconds": 0.0, "end_seconds": 100.0},
            {"start_seconds": 50.0, "end_seconds": 150.0},
        ]
        occ = compute_timeline_occupancy(clips, 600.0)
        self.assertEqual(occ["clip_count"], 2)
        self.assertGreater(occ["overlap_seconds"], 40.0)
        self.assertEqual(occ["over_soft_cap"], 2)

    def test_merge_allowed_cap(self) -> None:
        low = {"virality_score": 50}
        high = {"virality_score": 95}
        self.assertEqual(merge_allowed_max_duration(low, low, 120.0), SOFT_CAP_SECONDS)
        self.assertEqual(merge_allowed_max_duration(low, high, 120.0), HARD_CAP_SECONDS)

    def test_scaled_context_large_core(self) -> None:
        b, a = scaled_context_padding(80.0, 8.0, 12.0)
        self.assertEqual(b, 2.0)
        self.assertEqual(a, 4.0)

    def test_batch_stats(self) -> None:
        clips = [
            {
                "original_start": 0.0,
                "original_end": 50.0,
                "start_seconds": 0.0,
                "end_seconds": 120.0,
                "virality_score": 50,
            },
        ]
        out, stats = apply_duration_policy_batch(clips, 600.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(stats.clamped_soft, 1)
        self.assertLessEqual(out[0]["duration"], SOFT_CAP_SECONDS + 0.5)


if __name__ == "__main__":
    unittest.main()
