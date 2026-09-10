import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import js_divergence, minmax, select_under_budget


class CoreTests(unittest.TestCase):
    def test_js_identity(self):
        self.assertAlmostEqual(js_divergence([0.25] * 4, [0.25] * 4), 0.0)

    def test_minmax_constant(self):
        self.assertEqual(minmax([3.0, 3.0]), [0.5, 0.5])

    def test_budget_and_overlap(self):
        candidates = [
            {"clip_start": 0, "clip_end": 10, "value_score": 3},
            {"clip_start": 5, "clip_end": 15, "value_score": 2},
            {"clip_start": 20, "clip_end": 30, "value_score": 1},
        ]
        selected = select_under_budget(candidates, 20)
        self.assertEqual([(x["clip_start"], x["clip_end"]) for x in selected], [(0, 10), (20, 30)])


if __name__ == "__main__":
    unittest.main()
