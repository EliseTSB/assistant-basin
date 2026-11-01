import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import secondary165 as s


class SecondaryTests(unittest.TestCase):
    def test_pair_fraction_collapses_on_repetition(self):
        self.assertEqual(s.distinct_pair_fraction("one two three four"), 1.0)
        self.assertAlmostEqual(s.distinct_pair_fraction("go go go go go"), 0.25)
        self.assertIsNone(s.distinct_pair_fraction("word"))

    def test_refusal_reads_only_the_first_sentence(self):
        self.assertTrue(s.opens_with_refusal("I'm sorry, but I don't have a name."))
        self.assertTrue(s.opens_with_refusal("I cannot help with that."))
        self.assertTrue(s.opens_with_refusal("  Sorry about that."))
        self.assertFalse(s.opens_with_refusal("I am Qwen. I'm sorry you asked."))
        self.assertFalse(s.opens_with_refusal("I am Qwen."))

    def test_summary_counts_each_arm(self):
        rows = [{"arm": "plain", "answer": "I am Qwen, made by Alibaba."},
                {"arm": "full", "answer": "I'm sorry."},
                {"arm": "full", "answer": "go go go go go"}]
        result = s.summarise(rows)
        self.assertEqual(result["plain"]["refusal_openers"], 0)
        self.assertEqual(result["full"]["refusal_openers"], 1)
        self.assertEqual(result["full"]["answers"], 2)
        self.assertEqual(result["n+"]["answers"], 0)
        self.assertIsNone(result["n+"]["refusal_rate"])


if __name__ == "__main__":
    unittest.main()
