"""Part 180 v5 generation and development-margin checks. Fixtures only: no real model, no network."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import torch

import texts180 as v4
import texts180v5 as v5
import v5_support as support

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / "PART180-V5-DESIGN.md"


def design_configurations():
    """Pull the six numbered pairs out of the design itself, so the test cannot agree with a stale copy."""
    text = DESIGN.read_text(encoding="utf-8")
    section = text.split("## 3. One bounded screen")[1].split("### 3a")[0]
    return [(int(number), self_prompt, content_prompt)
            for number, self_prompt, content_prompt
            in re.findall(r'^(\d)\. "(.*?)" / "(.*?)"', section, re.M)]


class ConfigurationPromptTests(unittest.TestCase):
    def test_six_pairs_equal_the_design_exactly(self):
        rows = design_configurations()
        self.assertEqual(len(rows), 6)
        self.assertEqual([config["number"] for config in v5.CONFIGURATIONS], [row[0] for row in rows])
        for (_, self_prompt, content_prompt), config in zip(rows, v5.CONFIGURATIONS):
            self.assertEqual(config["self_prompt"], self_prompt)
            self.assertEqual(config["content_prompt"], content_prompt)

    def test_list_order_is_the_tie_break_order(self):
        self.assertEqual([config["number"] for config in v5.CONFIGURATIONS], [1, 2, 3, 4, 5, 6])

    def test_no_configuration_prompt_carries_a_concept_or_role_word(self):
        for config in v5.CONFIGURATIONS:
            for prompt in (config["self_prompt"], config["content_prompt"]):
                self.assertFalse(re.search(r"\b(act as|pretend|roleplay|you are an?)\b", prompt, re.I), prompt)


class ConfirmationGuardTests(unittest.TestCase):
    def test_guard_refuses_each_confirmation_prompt(self):
        for prompt in v5.CONFIRMATION.values():
            with self.assertRaises(ValueError):
                v5.assert_no_confirmation([{"prompt": prompt}])

    def test_guard_ignores_case_whitespace_and_quotes(self):
        prompt = '  "' + v5.CONFIRMATION["content_prompt"].upper() + '"  '
        with self.assertRaises(ValueError):
            v5.assert_no_confirmation([{"prompt": prompt}])

    def test_guard_passes_a_development_list(self):
        tasks = v5.plan_tasks(support.tiny_neurons(2))
        self.assertIs(v5.assert_no_confirmation(tasks), tasks)

    def test_plan_contains_no_confirmation_text_at_all(self):
        blob = json.dumps(v5.plan_tasks(support.tiny_neurons(2))).casefold()
        for word in ("postcard", "to a child"):
            self.assertNotIn(word, blob)

    def test_generate_and_margin_tasks_both_refuse(self):
        neurons = support.tiny_neurons(2)
        bad = {"kind": "S", "config": 1, "neuron": "p92-00", "key": "S",
               "prompt": v5.CONFIRMATION["self_prompt"], "frame": "chat", "pushed": True, "push_layer": 0}
        with self.assertRaises(ValueError):
            v5.generate_texts([bad], None, None, None, Path("/nonexistent/never-written.jsonl"))
        with self.assertRaises(ValueError):
            v5.score_margins([bad], None, None, None, Path("/nonexistent/never-written.jsonl"))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.neurons = support.tiny_neurons(3)

    def test_task_counts(self):
        tasks = v5.plan_tasks(self.neurons)
        kinds = [task["kind"] for task in tasks]
        self.assertEqual(kinds.count("A"), 2 * 3)
        self.assertEqual(kinds.count("S"), 6 * 3)
        self.assertEqual(kinds.count("C"), 6 * 3)
        self.assertEqual(kinds.count("B"), 6)
        self.assertEqual(len(tasks), 6 + 6 * 3 + 6 * 3 + 6)

    def test_about_texts_are_written_once_per_neuron_and_never_per_configuration(self):
        about = [task for task in v5.plan_tasks(self.neurons) if task["kind"] == "A"]
        self.assertEqual({task["config"] for task in about}, {None})
        self.assertEqual([task["prompt"] for task in about if task["neuron"] == "p92-00"],
                         ["Tell me about a boulder.", "Describe a boulder."])
        self.assertEqual([task["prompt"] for task in about if task["neuron"] == "p92-02"],
                         ["Tell me about an 80s cover singer-songwriter.",
                          "Describe an 80s cover singer-songwriter."])

    def test_one_shared_b_row_per_configuration_and_no_pushed_share(self):
        tasks = v5.plan_tasks(self.neurons)
        shared = [task for task in tasks if task["kind"] == "B"]
        self.assertEqual(sorted(task["config"] for task in shared), [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(task["neuron"] == "unpushed" and task["pushed"] is False for task in shared))
        self.assertTrue(all(task["pushed"] for task in tasks if task["kind"] in ("S", "C")))

    def test_task_row_keys_are_unique(self):
        keys = [v5.task_row_key(task) for task in v5.plan_tasks(self.neurons)]
        self.assertEqual(len(keys), len(set(keys)))


class MarginTaskTests(unittest.TestCase):
    def setUp(self):
        self.neurons = support.tiny_neurons(3)
        self.rows = support.fake_text_rows(self.neurons)
        self.tokenizer = support.TinyTokenizer()
        self.tasks = v5.margin_tasks(self.rows, self.neurons, tokenizer=self.tokenizer)

    def test_six_rows_per_configuration_and_neuron(self):
        self.assertEqual(len(self.tasks), 6 * 3 * 6)
        per = {}
        for task in self.tasks:
            per.setdefault((task["config"], task["neuron"]), set()).add(task["case"])
        self.assertEqual(len(per), 18)
        self.assertTrue(all(cases == set(v5.MARGIN_CASES) for cases in per.values()))

    def test_each_task_carries_its_configurations_question(self):
        by_number = {config["number"]: config for config in v5.CONFIGURATIONS}
        for task in self.tasks:
            config = by_number[task["config"]]
            expected = config["self_prompt"] if task["case"].startswith("self_") else config["content_prompt"]
            self.assertEqual(task["prompt"], expected, task["case"])
            self.assertEqual(task["prompt_ids"], v4._prompt_ids(self.tokenizer, expected, "chat"))

    def test_about_texts_are_scored_under_the_configuration_not_their_own_prompt(self):
        a0 = [task for task in self.tasks if task["case"] == "self_A0"]
        self.assertEqual({task["prompt"] for task in a0},
                         {config["self_prompt"] for config in v5.CONFIGURATIONS})
        self.assertEqual([task["token_ids"] for task in a0],
                         [[row["token_ids"] for row in self.rows
                           if (row["neuron"], row["key"], row["config"]) == (task["neuron"], "A0", None)][0]
                          for task in a0])

    def test_k_is_min_of_the_pair_and_capped_at_forty(self):
        pairs = {(task["config"], task["neuron"], task["case"]): task["k_tokens"] for task in self.tasks}
        lengths = support.TOKEN_LENGTHS
        for neuron in self.neurons:
            neuron_id = str(neuron["id"])
            for number in range(1, 7):
                self.assertEqual(pairs[(number, neuron_id, "self_S_A0")],
                                 min(lengths["S"], lengths["A0"], v5.K_CAP))
                self.assertEqual(pairs[(number, neuron_id, "self_A0")],
                                 min(lengths["S"], lengths["A0"], v5.K_CAP))
                self.assertEqual(pairs[(number, neuron_id, "self_S_A1")],
                                 min(lengths["S"], lengths["A1"], v5.K_CAP))
                self.assertEqual(pairs[(number, neuron_id, "self_A1")],
                                 min(lengths["S"], lengths["A1"], v5.K_CAP))
                self.assertEqual(pairs[(number, neuron_id, "cont_C")],
                                 min(lengths["C"], lengths["B"], v5.K_CAP))
                self.assertEqual(pairs[(number, neuron_id, "cont_B")],
                                 min(lengths["C"], lengths["B"], v5.K_CAP))
        self.assertNotEqual(pairs[(1, self.neurons[0]["id"], "self_A0")],
                            pairs[(1, self.neurons[0]["id"], "self_A1")])

    def test_every_margin_row_is_scored_with_its_neurons_push(self):
        layers = {str(neuron["id"]): int(neuron["layer"]) for neuron in self.neurons}
        self.assertTrue(all(task["pushed"] for task in self.tasks))
        for task in self.tasks:
            self.assertEqual(task["push_layer"], layers[task["neuron"]])

    def test_missing_frozen_row_is_an_error(self):
        rows = [row for row in self.rows
                if not (row["neuron"] == self.neurons[1]["id"] and row["key"] == "C" and row["config"] == 4)]
        with self.assertRaisesRegex(ValueError, "frozen text row missing"):
            v5.margin_tasks(rows, self.neurons, tokenizer=self.tokenizer)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = support.TinyTokenizer()
        self.model = support.tiny_model()
        self.neurons = support.tiny_neurons(2)
        self.pushes = support.tiny_pushes(self.neurons, hidden=self.model.config.hidden_size)
        self.tasks = v5.plan_tasks(self.neurons)

    def test_one_fixed_batch_shape_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "texts.jsonl"
            info = v5.generate_texts(self.tasks, self.tokenizer, self.model, self.pushes, out,
                                     batch_size=5, max_new_tokens=6)
            rows = v5.read_rows(out)
            self.assertEqual(len(rows), len(self.tasks))
            self.assertEqual({tuple(row["batch_shape"]) for row in rows}, {tuple(info["fixed_batch_shape"])})
            self.assertEqual(info["fixed_batch_shape"][0], 5)
            self.assertGreaterEqual(info["padded_rows"], 0)
            again = v5.generate_texts(self.tasks, self.tokenizer, self.model, self.pushes, out,
                                      batch_size=5, max_new_tokens=6)
            self.assertEqual(again["generated"], 0)
            self.assertEqual(len(v5.read_rows(out)), len(self.tasks))

    def test_width_is_fixed_from_the_whole_plan_not_the_pending_rows(self):
        width = v5.fixed_width(self.tasks, self.tokenizer)
        expected = max(len(v4._prompt_ids(self.tokenizer, task["prompt"], task["frame"]))
                       for task in self.tasks)
        self.assertEqual(width, expected)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "texts.jsonl"
            v5.generate_texts(self.tasks, self.tokenizer, self.model, self.pushes, out,
                              batch_size=4, max_new_tokens=4)
            self.assertEqual({row["batch_shape"][1] for row in v5.read_rows(out)}, {width})

    def test_pushes_reach_exactly_the_pushed_rows(self):
        rows_in = [{"neuron": "p92-00", "push_layer": 0, "pushed": True},
                   {"neuron": "unpushed", "push_layer": None, "pushed": False},
                   {"neuron": "p92-01", "push_layer": 1, "pushed": True}]
        vectors = v5._push_vectors(rows_in, self.pushes, self.model)
        self.assertEqual(sorted(vectors), [0, 1])
        self.assertTrue(torch.equal(vectors[0][0], self.pushes["p92-00"]))
        self.assertFalse(bool(vectors[0][1].any()))
        self.assertFalse(bool(vectors[1][0].any()))
        self.assertTrue(torch.equal(vectors[1][2], self.pushes["p92-01"]))
        self.assertEqual(v5._push_vectors([rows_in[1]], self.pushes, self.model), {})


class MarginScoreTests(unittest.TestCase):
    def test_margins_are_the_pair_differences_and_self_is_the_minimum(self):
        tokenizer = support.TinyTokenizer()
        model = support.tiny_model()
        neurons = support.tiny_neurons(2)
        pushes = support.tiny_pushes(neurons, hidden=model.config.hidden_size)
        tasks = v5.plan_tasks(neurons)
        with tempfile.TemporaryDirectory() as tmp:
            texts_path = Path(tmp) / "texts.jsonl"
            v5.generate_texts(tasks, tokenizer, model, pushes, texts_path, batch_size=8, max_new_tokens=40)
            rows = v5.read_rows(texts_path)
            self.assertTrue(all(row["token_ids"] for row in rows
                                if row["key"] in ("S", "C", "A0", "A1", "B")))
            margin_tasks = v5.margin_tasks(rows, neurons, tokenizer=tokenizer)
            scores_path = Path(tmp) / "scores.jsonl"
            v5.score_margins(margin_tasks, tokenizer, model, pushes, scores_path, batch_size=8)
            score_rows = v5.read_rows(scores_path)
            self.assertEqual(len(score_rows), 6 * len(neurons) * 6)
            margins = v5.compute_margins(score_rows, neurons)
        values = {(row["config"], row["neuron"], row["case"]): float(row["lp_k"]) for row in score_rows}
        for margin in margins:
            number, neuron_id = margin["config"], margin["neuron"]
            a0 = values[(number, neuron_id, "self_S_A0")] - values[(number, neuron_id, "self_A0")]
            a1 = values[(number, neuron_id, "self_S_A1")] - values[(number, neuron_id, "self_A1")]
            cont = values[(number, neuron_id, "cont_C")] - values[(number, neuron_id, "cont_B")]
            self.assertAlmostEqual(margin["self_a0"], a0, places=6)
            self.assertAlmostEqual(margin["self_a1"], a1, places=6)
            self.assertAlmostEqual(margin["self"], min(a0, a1), places=6)
            self.assertAlmostEqual(margin["self_mean"], (a0 + a1) / 2.0, places=6)
            self.assertAlmostEqual(margin["cont"], cont, places=6)

    def test_scoring_resumes_by_whole_group(self):
        tokenizer = support.TinyTokenizer()
        model = support.tiny_model()
        neurons = support.tiny_neurons(1)
        pushes = support.tiny_pushes(neurons, hidden=model.config.hidden_size)
        with tempfile.TemporaryDirectory() as tmp:
            texts_path = Path(tmp) / "texts.jsonl"
            v5.generate_texts(v5.plan_tasks(neurons), tokenizer, model, pushes, texts_path,
                              batch_size=8, max_new_tokens=8)
            rows = v5.read_rows(texts_path)
            margin_tasks = v5.margin_tasks(rows, neurons, tokenizer=tokenizer)
            scores_path = Path(tmp) / "scores.jsonl"
            v5.score_margins(margin_tasks, tokenizer, model, pushes, scores_path, batch_size=6)
            first = v5.read_rows(scores_path)
            info = v5.score_margins(margin_tasks, tokenizer, model, pushes, scores_path, batch_size=6)
            self.assertEqual(info["scored"], 0)
            self.assertEqual(len(v5.read_rows(scores_path)), len(first))
            # a half-written group is dropped and rescored, never merged
            partial = [row for row in first if not (row["config"] == 3 and row["case"] == "cont_B")]
            v5.append_rows(Path(tmp) / "rewritten.jsonl", partial)
            (Path(tmp) / "rewritten.jsonl").replace(scores_path)
            info = v5.score_margins(margin_tasks, tokenizer, model, pushes, scores_path, batch_size=6)
            self.assertEqual(info["scored"], 6 * len(neurons))
            self.assertEqual(len(v5.read_rows(scores_path)), len(first))


if __name__ == "__main__":
    unittest.main()
