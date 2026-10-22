import unittest
from pathlib import Path

import sweep_sideeffects as sweep


class SweepSideEffectsTests(unittest.TestCase):
    def test_answer_containing_word_counts_once_even_if_repeated(self):
        answers = [
            ["magic"] * 20,
            ["magic", "wizard"],
            ["wizard"],
        ]
        counts = sweep.answer_presence_counts(answers)
        self.assertEqual(counts["magic"], 2)
        self.assertEqual(counts["wizard"], 2)
        self.assertEqual(sweep.count_answers_with_word(answers, "magic"), 2)

    def test_up_threshold_edges(self):
        self.assertEqual(sweep.flag_direction(5, 0), "UP")
        self.assertIsNone(sweep.flag_direction(4, 0))
        self.assertEqual(sweep.flag_direction(6, 1), "UP")
        self.assertIsNone(sweep.flag_direction(5, 1))
        self.assertEqual(sweep.flag_direction(15, 4), "UP")
        self.assertIsNone(sweep.flag_direction(14, 4))

    def test_down_threshold_edges(self):
        self.assertEqual(sweep.flag_direction(0, 5), "DOWN")
        self.assertIsNone(sweep.flag_direction(0, 4))
        self.assertEqual(sweep.flag_direction(1, 6), "DOWN")
        self.assertIsNone(sweep.flag_direction(1, 5))
        self.assertEqual(sweep.flag_direction(4, 15), "DOWN")
        self.assertIsNone(sweep.flag_direction(4, 14))

    def test_prompt_words_and_stopwords_are_removed(self):
        self.assertEqual(
            sweep.eligible_words("Wizard waits beside the tower", "wizard tower"),
            {"waits", "beside"},
        )

    def test_control_detection_on_two_real_result_schemas(self):
        results = sweep.DEFAULT_RESULTS

        # JSONL header arms + result rows, as in the Part 49b push run.
        part49 = sweep.extract_file(results / "part49b-xpu" / "push49b.jsonl", results)
        part49_arms = {row["arm"] for row in part49["rows"]}
        part49_control, _ = sweep.choose_controls(part49_arms, sweep.metadata_hints([part49]))
        self.assertEqual(part49_control, "no_clamp")

        # Nested JSON conditions/replies, as in the Part 16b judged run.
        part16 = sweep.extract_file(results / "part16b" / "judged.json", results)
        part16_arms = {row["arm"] for row in part16["rows"]}
        part16_control, _ = sweep.choose_controls(part16_arms, sweep.metadata_hints([part16]))
        self.assertEqual(part16_control, "unchanged")

        # The exact "none" cache arm is control; a reencoded_unclamped arm is
        # still a separate intervention and must not win by substring match.
        part31 = sweep.extract_file(results / "part31-full-base" / "samples.jsonl", results)
        part31_arms = {row["arm"] for row in part31["rows"]}
        part31_control, _ = sweep.choose_controls(part31_arms, sweep.metadata_hints([part31]))
        self.assertEqual(part31_control, "none")


if __name__ == "__main__":
    unittest.main()
