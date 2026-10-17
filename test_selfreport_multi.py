"""Offline checks for Part 12g multiclass self-report mechanics."""
import unittest
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from selfreport import install_lora
from test_selfreport import FakeTokenizer, tiny_model, tiny_spec
from selfreport_multi import (
    ANSWERS, CONDITIONS, MultiExample, MultiNeuronHooks, assign_conditions,
    build_evaluation_examples, condition_label, first_token_ids, make_groups, padded_multi_batch,
    shuffle_training_labels, train_multi,
)
from selfreport import build_examples


class MultiSelfReportTests(unittest.TestCase):
    def test_chat_template_answer_ids_are_single_and_distinct(self):
        ids = first_token_ids(FakeTokenizer())
        self.assertEqual(set(ids), set(ANSWERS))
        self.assertEqual(len(set(ids.values())), 5)

    def test_groups_disjoint_layer_matched_and_seeded(self):
        g1 = [(layer, i) for layer, count in ((0, 2), (1, 3), (2, 2), (3, 2), (4, 0), (5, 2)) for i in range(count)]
        a = make_groups(g1, 100, seed=19)
        b = make_groups(g1, 100, seed=19)
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"G1", "G2", "G3", "G4", "G5"})
        all_units = [u for group in a.values() for u in group]
        self.assertEqual(len(all_units), len(set(all_units)))
        layers = sorted(l for l, _ in g1)
        for name in ("G2", "G3", "G4", "G5"):
            self.assertEqual(sorted(l for l, _ in a[name]), layers)
            self.assertFalse(set(g1) & set(a[name]))

    def test_training_conditions_balance_and_label_mapping(self):
        examples = list(range(53))
        assignment = assign_conditions(examples, seed=42)
        counts = {condition: sum(row.count(condition) for row in assignment) for condition in CONDITIONS}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(len(row) == 2 and len(set(row)) == 2 for row in assignment))
        self.assertEqual([condition_label(c) for c in CONDITIONS], ["A", "B", "C", "D", "none", "none"])

    def test_group_clamp_changes_only_selected_units_at_message_positions(self):
        model = tiny_model(seed=101, layers=2)
        groups = {"G1": [(0, 1)], "G2": [(1, 2)], "G3": [(0, 3)], "G4": [(1, 4)], "G5": [(0, 5)]}
        highs = {key: np.array([100.0 + i], dtype=np.float32) for i, key in enumerate(groups)}
        ids = torch.tensor([[1, 3, 5, 7, 9]])
        positions = [[1, 3]]
        target_module = model.model.layers[0].mlp.down_proj
        baseline_seen, clamped_seen = [], []
        h = target_module.register_forward_pre_hook(lambda _m, ins: baseline_seen.append(ins[0].detach().clone()))
        with torch.inference_mode(): model(input_ids=ids, use_cache=False)
        h.remove()
        hooks = MultiNeuronHooks(model, groups, highs)
        h = target_module.register_forward_pre_hook(lambda _m, ins: clamped_seen.append(ins[0].detach().clone()))
        hooks.prepare(positions, ["G3"])
        with torch.inference_mode(): model(input_ids=ids, use_cache=False)
        h.remove(); hooks.close()
        base, changed = baseline_seen[0], clamped_seen[0]
        expected = torch.zeros_like(base, dtype=torch.bool)
        expected[0, torch.tensor([1, 3]), 3] = True
        self.assertTrue(torch.equal(changed[expected], torch.tensor([102.0, 102.0])))
        self.assertTrue(torch.equal(changed[~expected], base[~expected]))

    def test_decoy_condition_clamps_g5_and_none_clamps_nothing(self):
        # Found in review 2026-10-16: the hook looked up the condition name, so "other" clamped nothing.
        model = tiny_model(seed=101, layers=2)
        groups = {"G1": [(0, 1)], "G2": [(1, 2)], "G3": [(0, 3)], "G4": [(1, 4)], "G5": [(0, 5)]}
        highs = {key: np.array([100.0 + i], dtype=np.float32) for i, key in enumerate(groups)}
        ids = torch.tensor([[1, 3, 5, 7, 9]])
        target_module = model.model.layers[0].mlp.down_proj
        seen = {}
        for condition in ("none", "other"):
            hooks = MultiNeuronHooks(model, groups, highs)
            got = []
            h = target_module.register_forward_pre_hook(lambda _m, ins: got.append(ins[0].detach().clone()))
            hooks.prepare([[1, 3]], [condition])
            with torch.inference_mode(): model(input_ids=ids, use_cache=False)
            h.remove(); hooks.close(); seen[condition] = got[0]
        expected = torch.zeros_like(seen["none"], dtype=torch.bool)
        expected[0, torch.tensor([1, 3]), 5] = True
        self.assertTrue(torch.equal(seen["other"][expected], torch.tensor([104.0, 104.0])))
        self.assertTrue(torch.equal(seen["other"][~expected], seen["none"][~expected]))
        self.assertFalse(torch.equal(seen["none"][expected], torch.tensor([104.0, 104.0])))

    def test_eval_examples_cover_every_condition_with_identical_prompt_ids(self):
        tokenizer = FakeTokenizer()
        sources = build_examples(tokenizer, tiny_spec(frame_count=2))
        held = [e for e in sources if e.frame_index == 1]
        rows = build_evaluation_examples(tokenizer, held)
        by_source = {}
        for row in rows: by_source.setdefault((row.source.frame_index, row.source.family), []).append(row)
        self.assertEqual(len(rows), len(held) * 6)
        for copies in by_source.values():
            self.assertEqual({x.condition for x in copies}, set(CONDITIONS))
            self.assertTrue(all(x.prompt_ids == copies[0].prompt_ids for x in copies))

    def test_lora_training_steps_lower_loss(self):
        model = tiny_model(seed=103, layers=2)
        install_lora(model, rank=8)
        groups = {"G1": [(0, 1)], "G2": [(0, 2)], "G3": [(0, 3)], "G4": [(0, 4)], "G5": [(0, 5)]}
        highs = {key: np.array([0.0], dtype=np.float32) for key in groups}
        hooks = MultiNeuronHooks(model, groups, highs)
        rows = []
        for i in range(4):
            rows.append(MultiExample(None, "none", "A", [1, 4, 8, 12], [1], train_label="A"))
        ids, mask, positions = padded_multi_batch(rows, 0, [17] * len(rows))
        with torch.no_grad(): initial = F.cross_entropy(model(input_ids=ids, attention_mask=mask).logits[:, -2], torch.tensor([17] * 4))
        train_multi(model, rows, hooks, 0, {"A": 17, "B": 18, "C": 19, "D": 20, "none": 21}, epochs=8, batch_size=4, learning_rate=0.03)
        with torch.no_grad(): final = F.cross_entropy(model(input_ids=ids, attention_mask=mask).logits[:, -2], torch.tensor([17] * 4))
        hooks.close()
        self.assertLess(float(final), float(initial))

    def test_shuffled_training_labels_do_not_change_eval_labels(self):
        tokenizer = FakeTokenizer()
        sources = build_examples(tokenizer, tiny_spec())
        train = [MultiExample(sources[0], c, condition_label(c), [1, 3], [1], train_label=condition_label(c)) for c in CONDITIONS]
        evaluation = [MultiExample(sources[0], c, condition_label(c), [1, 3], [1]) for c in CONDITIONS]
        eval_before = [e.label for e in evaluation]
        train_before = [e.train_label for e in train]
        shuffle_training_labels(train, seed=42)
        self.assertCountEqual([e.train_label for e in train], train_before)
        self.assertNotEqual([e.train_label for e in train], train_before)
        self.assertEqual([e.label for e in evaluation], eval_before)
        self.assertEqual(eval_before, ["A", "B", "C", "D", "none", "none"])


if __name__ == "__main__":
    unittest.main()
