"""Part 180 v5 reader items: the frozen grid, K, the shuffle and the reader binding. No network, no model."""
from __future__ import annotations

import hashlib
import json
import unittest

import reader180v5 as reader
import texts180v5 as v5
import v5_support as support


def stub_call(prompt: str) -> str:
    """A deterministic stand-in for the DeepSeek call: labels from the answer text alone."""
    answer = prompt.rsplit("Answer:", 1)[-1].casefold()
    persona = prompt.split("Target persona: ", 1)[1].split(". ", 1)[0].casefold()
    present = persona in answer
    return json.dumps({"claim": "yes" if present else "no", "mention": present, "theme": False,
                       "description": present, "negated": False, "quoted": False})


class ReaderItemTests(unittest.TestCase):
    def setUp(self):
        self.neurons = support.tiny_neurons(3)
        self.rows = support.fake_text_rows(self.neurons)
        self.tokenizer = support.TinyTokenizer()
        self.items = reader.build_items(self.rows, self.neurons, self.tokenizer)

    def test_one_item_per_configuration_neuron_and_arm(self):
        self.assertEqual(len(self.items), 6 * 3 * len(reader.ARMS))
        cells = {(item["config"], item["neuron"], item["key"]) for item in self.items}
        self.assertEqual(len(cells), 6 * 3 * len(reader.ARMS))
        self.assertEqual({item["key"] for item in self.items}, set(reader.ARMS))

    def test_b_is_read_once_per_neuron_against_that_personas_label(self):
        b_items = [item for item in self.items if item["key"] == "B"]
        self.assertEqual(len(b_items), 6 * 3)
        personas = {item["neuron"]: item["persona"] for item in b_items}
        self.assertEqual(personas["p92-00"], "Boulder")
        self.assertEqual({item["config"] for item in b_items}, set(v5.CONFIG_NUMBERS))

    def test_k_is_min_of_the_pair_and_s_is_read_inside_both_pairs(self):
        by_cell = {(item["config"], item["neuron"], item["key"]): item for item in self.items}
        for config in v5.CONFIG_NUMBERS:
            for neuron in self.neurons:
                neuron_id = str(neuron["id"])
                lengths = support.TOKEN_LENGTHS
                a0 = by_cell[(config, neuron_id, "A0")]
                a1 = by_cell[(config, neuron_id, "A1")]
                s = by_cell[(config, neuron_id, "S")]
                c = by_cell[(config, neuron_id, "C")]
                b = by_cell[(config, neuron_id, "B")]
                self.assertEqual(a0["k"], min(lengths["S"], lengths["A0"], 40))
                self.assertEqual(a1["k"], min(lengths["S"], lengths["A1"], 40))
                self.assertEqual(s["k"], min(a0["k"], a1["k"]))
                self.assertEqual(c["k"], min(lengths["C"], lengths["B"], 40))
                self.assertEqual(b["k"], c["k"])
                self.assertEqual(s["k_by_pair"], {"A0": a0["k"], "A1": a1["k"]})

    def test_prefix_token_ids_are_the_scored_prefix(self):
        for item in self.items:
            self.assertEqual(len(item["prefix_token_ids"]), item["k"])
            self.assertEqual(item["prefix_tokens"], item["k"])
            self.assertEqual(item["prefix_text"], self.tokenizer.decode(item["prefix_token_ids"],
                                                                       skip_special_tokens=True))

    def test_each_item_carries_its_own_question(self):
        by_cell = {(item["config"], item["neuron"], item["key"]): item for item in self.items}
        self.assertEqual(by_cell[(3, "p92-00", "S")]["question"], "Who are you?")
        self.assertEqual(by_cell[(3, "p92-00", "C")]["question"], "Describe your perfect day.")
        self.assertEqual(by_cell[(3, "p92-00", "A0")]["question"], "Tell me about a boulder.")
        self.assertEqual(by_cell[(3, "p92-00", "A1")]["question"], "Describe a boulder.")
        self.assertEqual(by_cell[(3, "p92-00", "B")]["question"], "Describe your perfect day.")

    def test_persona_label_and_description_come_from_part_92(self):
        item = next(item for item in self.items if item["neuron"] == "p92-01")
        self.assertEqual(item["persona"], "Whiskey")
        self.assertEqual(item["desc"], "A whiskey speaking in the first person.")

    def test_no_identifier_reaches_the_prompt_text(self):
        text = json.dumps([{key: item[key] for key in ("persona", "desc", "question", "prefix_text")}
                           for item in self.items])
        self.assertNotIn("p92-", text)
        self.assertNotIn("v5-c", text)

    def test_shuffle_is_fixed_by_seed(self):
        again = reader.build_items(self.rows, self.neurons, self.tokenizer)
        self.assertEqual([item["id"] for item in self.items], [item["id"] for item in again])
        other = reader.build_items(self.rows, self.neurons, self.tokenizer, seed=181)
        self.assertNotEqual([item["id"] for item in self.items], [item["id"] for item in other])
        self.assertEqual({item["id"] for item in self.items}, {item["id"] for item in other})

    def test_missing_text_row_is_an_error(self):
        rows = [row for row in self.rows if not (row["neuron"] == "p92-01" and row["key"] == "A1")]
        with self.assertRaisesRegex(ValueError, "frozen text row missing"):
            reader.build_items(rows, self.neurons, self.tokenizer)

    def test_duplicate_text_row_is_an_error(self):
        rows = self.rows + [dict(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "duplicate frozen text row"):
            reader.build_items(rows, self.neurons, self.tokenizer)


class ReaderLabelTests(unittest.TestCase):
    def setUp(self):
        self.neurons = support.tiny_neurons(2)
        self.rows = support.fake_text_rows(self.neurons)
        self.items = reader.build_items(self.rows, self.neurons, support.TinyTokenizer())

    def test_rows_keep_the_grid_cell_k_and_prefix_ids(self):
        written = []
        rows = reader.label_items(self.items, stub_call, sink=written.append)
        self.assertEqual(len(rows), len(self.items))
        self.assertEqual([row["id"] for row in written], [row["id"] for row in rows])
        for row, item in zip(rows, self.items):
            self.assertEqual(row["id"], item["id"])
            self.assertEqual(row["k"], item["k"])
            self.assertEqual(row["prefix_token_ids"], item["prefix_token_ids"])
            self.assertEqual(set(row["label"]), {"claim", "mention", "theme", "description",
                                                 "negated", "quoted"})

    def test_stub_labels_are_parsed_not_echoed(self):
        rows = reader.label_items(self.items, stub_call)
        self.assertTrue(all(row["label"]["claim"] in ("yes", "no", "unresolved") for row in rows))
        self.assertTrue(all(row["attempts"] == 1 for row in rows))


class ReaderBindingTests(unittest.TestCase):
    def test_binding_records_model_route_and_prompt_hash(self):
        binding = reader.reader_binding("deepseek")
        self.assertEqual(binding["model"], "deepseek-v4.1-flash")
        self.assertEqual(binding["route"], "ocgo")
        self.assertEqual(binding["reader"], "deepseek")
        digest = hashlib.sha256((reader.HERE / "prompt180v5.txt").read_bytes()).hexdigest()
        self.assertEqual(binding["prompt_sha256"], digest)
        self.assertEqual(len(binding["prompt_sha256"]), 64)

    def test_beside_reader_is_a_different_call_never_a_chooser(self):
        deepseek = reader.reader_binding("deepseek")
        haiku = reader.reader_binding("haiku")
        self.assertNotEqual(deepseek["model"], haiku["model"])
        self.assertEqual(haiku["route"], deepseek["route"])
        self.assertEqual(haiku["prompt_sha256"], deepseek["prompt_sha256"])


class ExistingFileTests(unittest.TestCase):
    def test_duplicate_id_on_disk_is_rejected(self):
        expected = {"a", "b"}
        with self.assertRaisesRegex(ValueError, "duplicate reader row id"):
            reader.validate_existing([{"id": "a"}, {"id": "a"}], expected)

    def test_id_outside_the_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the frozen grid"):
            reader.validate_existing([{"id": "zzz"}], {"a", "b"})

    def test_valid_file_resumes(self):
        self.assertEqual(reader.validate_existing([{"id": "a"}], {"a", "b"}), {"a"})


if __name__ == "__main__":
    unittest.main()
