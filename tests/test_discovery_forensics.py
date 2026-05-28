"""Tests for discovery forensic tracing."""

import unittest

from clip_engine.discovery_forensics import DiscoveryForensics, count_lexicon_hits


class TestDiscoveryForensics(unittest.TestCase):
    def test_first_zero_stage_recorded(self) -> None:
        f = DiscoveryForensics()
        f.record_stage("a", input_count=10, output_count=5)
        f.record_stage("b", input_count=5, output_count=0)
        self.assertEqual(f.first_zero_stage, "b")

    def test_merge_scan_stats(self) -> None:
        f = DiscoveryForensics()
        f.merge_scan_stats(
            {
                "windows_scanned": 100,
                "windows_rejected": 80,
                "emotion_triggers": 3,
                "curiosity_triggers": 2,
                "story_phrase_triggers": 1,
                "trauma_triggers": 1,
                "keyword_hits": 7,
                "fallback_generated": 4,
            }
        )
        self.assertEqual(f.windows_scanned, 100)
        self.assertEqual(f.emotion_hits, 3)
        self.assertEqual(f.fallback_candidates_generated, 4)

    def test_lexicon_hits(self) -> None:
        hits = count_lexicon_hits("I was terrified and my mom said the truth about trauma")
        self.assertGreater(hits["emotion_hits"], 0)
        self.assertGreater(hits["trauma_hits"], 0)


if __name__ == "__main__":
    unittest.main()
