import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import question_stem
from sanctifier import (
    bounded_interval,
    note_is_relevant,
    paired_cluster_interval,
    parse_contract,
    quote_is_attributed,
    should_accept_update,
    trigger_decision,
)


class SanctifierTests(unittest.TestCase):
    def test_question_stem_hides_choices(self):
        question = "What was said?\n(1) red\n(2) blue\n(3) green\n(4) black"
        self.assertEqual(question_stem(question), "What was said?")

    def test_cross_view_trigger_is_hard_union(self):
        self.assertTrue(
            trigger_decision(
                "cross_view", disagreement=True, sparse=False, sample_id="a"
            )
        )
        self.assertTrue(
            trigger_decision(
                "cross_view", disagreement=False, sparse=True, sample_id="a"
            )
        )
        self.assertFalse(
            trigger_decision(
                "cross_view", disagreement=False, sparse=False, sample_id="a"
            )
        )

    def test_interval_obeys_clip_and_per_example_sar_caps(self):
        interval = bounded_interval(
            sample_id="a",
            duration=60.0,
            proposal={"start": 0.0, "end": 60.0, "score": 1.0},
            clip_seconds=20.0,
            sar_cap=0.20,
            selection_policy="query",
        )
        self.assertAlmostEqual(interval["end"] - interval["start"], 12.0)
        self.assertGreaterEqual(interval["start"], 0.0)
        self.assertLessEqual(interval["end"], 60.0)

    def test_contract_requires_verbatim_normalized_attribution(self):
        response = "CHOICE: 2\nEVIDENCE: blue bicycle"
        choice, quote = parse_contract(response)
        note = "HEARD: A blue, bicycle passed. RELEVANCE: relevant"
        self.assertEqual(choice, "2")
        self.assertTrue(quote_is_attributed(quote, note))
        self.assertTrue(
            should_accept_update(
                triggered=True,
                note=note,
                prior="1",
                candidate=choice,
                quote=quote,
                require_attribution=True,
            )
        )

    def test_uncertain_note_preserves_prior(self):
        note = "HEARD: unclear speech; RELEVANCE: uncertain"
        self.assertFalse(note_is_relevant(note))
        self.assertFalse(
            should_accept_update(
                triggered=True,
                note=note,
                prior="1",
                candidate="2",
                quote="unclear speech",
                require_attribution=True,
            )
        )

    def test_zero_delta_cluster_interval_is_zero(self):
        rows = [
            {
                "audio_sha256": "audio-a",
                "prior_correct": True,
                "final_correct": True,
            },
            {
                "audio_sha256": "audio-b",
                "prior_correct": False,
                "final_correct": False,
            },
        ]
        self.assertEqual(
            paired_cluster_interval(rows, resamples=100, seed=25), [0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()
