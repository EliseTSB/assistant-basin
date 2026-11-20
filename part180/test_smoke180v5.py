"""One end-to-end smoke: generation -> margins -> reader (stub) -> classification -> winner.json.

A tiny randomly-initialised OLMo-2 config built in memory (no download) and a stub reader call stand in for the
1B model and DeepSeek. The point is that the frozen plumbing runs in order and the winner file is written from the
real rule, not that the tiny model's answers mean anything.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import classify180v5 as classify
import reader180v5 as reader
import texts180v5 as v5
import v5_support as support


def stub_reader(prompt: str) -> str:
    """Deterministic label from the answer text: no model, no network, a valid tuple every time."""
    answer = prompt.rsplit("Answer:", 1)[-1].casefold()
    persona = prompt.split("Target persona: ", 1)[1].split(". ", 1)[0].casefold()
    mentioned = persona in answer
    described = mentioned and " like " in answer
    return json.dumps({"claim": "yes" if mentioned else "no", "mention": mentioned, "theme": False,
                       "description": described, "negated": False, "quoted": False})


class EndToEndSmokeTests(unittest.TestCase):
    def test_screen_runs_from_generation_to_a_written_winner_file(self):
        tokenizer = support.TinyTokenizer()
        model = support.tiny_model()
        neurons = support.tiny_neurons(3)
        pushes = support.tiny_pushes(neurons, hidden=model.config.hidden_size)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # 1. generation
            tasks = v5.plan_tasks(neurons)
            info = v5.generate_texts(tasks, tokenizer, model, pushes, out / "texts.jsonl",
                                     batch_size=8, max_new_tokens=40)
            text_rows = v5.read_rows(out / "texts.jsonl")
            self.assertEqual(len(text_rows), len(tasks))
            self.assertEqual(len({tuple(row["batch_shape"]) for row in text_rows}), 1)
            self.assertTrue(all(row["token_ids"] for row in text_rows), "no scored text may be empty")

            # 2. development margins
            margin_tasks = v5.margin_tasks(text_rows, neurons, tokenizer=tokenizer)
            v5.score_margins(margin_tasks, tokenizer, model, pushes, out / "scores.jsonl", batch_size=8)
            margins = v5.compute_margins(v5.read_rows(out / "scores.jsonl"), neurons)
            v5.append_rows(out / "margins.jsonl", margins)
            self.assertEqual(len(margins), 6 * len(neurons))
            self.assertTrue(any(margin["self"] > 0 for margin in margins))

            # 3. reader, replaced by a stub
            items = reader.build_items(v5.read_rows(out / "texts.jsonl"), neurons, tokenizer)
            with (out / "reader.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                def sink(row):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                reader.label_items(items, stub_reader, sink=sink)
            labels = v5.read_rows(out / "reader.jsonl")
            self.assertEqual(len(labels), 6 * len(neurons) * len(reader.ARMS))
            self.assertTrue(all(row["label"] is not None for row in labels))

            # 4. classification and selection
            outcomes = classify.classify(margins, labels, neurons, shared_ids=[neurons[0]["id"]])
            winner = classify.select(outcomes)
            report = classify.winner_report(outcomes, winner, shared_ids=[neurons[0]["id"]], neurons=neurons)
            classify._write_json(out / "winner.json", report)
            written = json.loads((out / "winner.json").read_text(encoding="utf-8"))

        self.assertEqual(sorted(int(key) for key in outcomes), list(v5.CONFIG_NUMBERS))
        for number in v5.CONFIG_NUMBERS:
            outcome = outcomes[str(number)]
            self.assertEqual(outcome["of"], 3)
            self.assertLessEqual(outcome["joint_passes"], 3)
            self.assertEqual(outcome["cohort"], [neuron_id for neuron_id in outcome["cohort"]])
        self.assertIn(written["selected_configuration"], v5.CONFIG_NUMBERS)
        self.assertEqual(written["development_cohort"],
                         outcomes[str(written["selected_configuration"])]["cohort"])
        self.assertEqual(written["joint_passes"], len(written["development_cohort"]))
        self.assertFalse(written["bars"]["applied"])
        self.assertIn(written["bars"]["status_if_applied"], ("PASS", "PARK"))
        self.assertEqual(written["shared_eight"]["of"], 1)
        self.assertEqual(written["self_prompt"],
                         v5.CONFIGURATIONS[written["selected_configuration"] - 1]["self_prompt"])
        self.assertEqual(info["generated"], len(tasks))


if __name__ == "__main__":
    unittest.main()
