"""Offline checks for Part 12g/12i multiclass self-report mechanics."""
import argparse
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from selfreport import install_lora, load_lora_parameters, save_lora_parameters
from test_selfreport import FakeTokenizer, tiny_model, tiny_spec
from selfreport_multi import (
    ANSWERS, CONDITIONS, MultiExample, MultiNeuronHooks, assign_conditions, assign_training_scales,
    build_evaluation_examples, build_extra_decoy_examples, build_result, condition_label,
    evaluate_extra_decoys, first_token_ids, make_extra_decoys, make_groups,
    evaluate_natural_readout, natural_strength_summary, padded_multi_batch, parse_eval_scales,
    parse_train_scales,
    scaled_group_highs, sha256_file, shuffle_training_labels, train_multi,
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

    def test_extra_decoys_are_fresh_layer_matched_disjoint_and_seeded(self):
        g1 = [(layer, i) for layer, count in ((0, 2), (1, 3), (2, 2), (3, 2), (5, 2)) for i in range(count)]
        groups = make_groups(g1, 128, seed=19)
        first = make_extra_decoys(groups, 128, 5, seed=2026)
        second = make_extra_decoys(groups, 128, 5, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"X1", "X2", "X3", "X4", "X5"})
        base_units = {unit for group in groups.values() for unit in group}
        extra_units = [unit for group in first.values() for unit in group]
        self.assertFalse(base_units & set(extra_units))
        self.assertEqual(len(extra_units), len(set(extra_units)))
        for group in first.values():
            self.assertEqual(Counter(layer for layer, _ in group), Counter(layer for layer, _ in g1))

    def test_eval_scales_parse_and_reject_invalid_or_duplicate_values(self):
        self.assertEqual(parse_eval_scales("0.25, 0.5,1,2,4,8"), (0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
        for invalid in ("", "1,nope", "0,1", "1,inf", "1,1"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_eval_scales(invalid)

    def test_train_scales_parse_and_seeded_draws_touch_only_pushed_examples(self):
        scales = parse_train_scales("0.5, 1,2")
        self.assertEqual(scales, (0.5, 1.0, 2.0))
        for invalid in ("", "1,nope", "0,1", "1,inf"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_train_scales(invalid)
        conditions = ["G1", "G2", "G3", "G4", "none", "other"] * 20
        rows_a = [MultiExample(None, condition, "A", [1, 2], [0]) for condition in conditions]
        rows_b = [MultiExample(None, condition, "A", [1, 2], [0]) for condition in conditions]
        assign_training_scales(rows_a, scales, seed=804)
        assign_training_scales(rows_b, scales, seed=804)
        self.assertEqual([row.train_scale for row in rows_a], [row.train_scale for row in rows_b])
        for row in rows_a:
            if row.condition in ("G1", "G2", "G3", "G4"):
                self.assertIn(row.train_scale, scales)
            else:
                self.assertIsNone(row.train_scale)

    def test_each_eval_scale_is_recomputed_from_the_same_quantile_pair(self):
        bounds = {"G1": (np.array([10.0], dtype=np.float32), np.array([2.0], dtype=np.float32))}
        highs = {scale: scaled_group_highs(bounds, scale)["G1"][0] for scale in (1.0, 2.0, 4.0, 8.0)}
        self.assertEqual(highs, {1.0: 10.0, 2.0: 14.0, 4.0: 22.0, 8.0: 38.0})

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

    def test_training_clamp_scale_is_selected_per_row_and_leaves_none_and_decoy_unchanged(self):
        model = tiny_model(seed=102, layers=2)
        groups = {
            "G1": [(0, 1)], "G2": [(0, 2)], "G3": [(0, 3)],
            "G4": [(0, 4)], "G5": [(0, 5)],
        }
        default_highs = {name: np.array([10.0 + i], dtype=np.float32) for i, name in enumerate(groups)}
        highs_by_scale = {
            0.5: {name: np.array([50.0 + i], dtype=np.float32) for i, name in enumerate(groups)},
            2.0: {name: np.array([20.0 + i], dtype=np.float32) for i, name in enumerate(groups)},
        }
        ids = torch.tensor([[1, 3, 5, 7, 9]] * 4)
        positions = [[1]] * 4
        module = model.model.layers[0].mlp.down_proj
        baseline = []
        baseline_handle = module.register_forward_pre_hook(lambda _m, ins: baseline.append(ins[0].detach().clone()))
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        baseline_handle.remove()

        hooks = MultiNeuronHooks(model, groups, default_highs, highs_by_scale)
        changed = []
        changed_handle = module.register_forward_pre_hook(lambda _m, ins: changed.append(ins[0].detach().clone()))
        hooks.prepare(positions, ["G1", "G2", "none", "other"], [0.5, 2.0, None, None])
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        changed_handle.remove()
        hooks.close()

        expected = baseline[0].clone()
        expected[0, 1, 1] = 50.0
        expected[1, 1, 2] = 21.0
        expected[3, 1, 5] = 14.0
        self.assertTrue(torch.equal(changed[0], expected))

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

    def test_extra_decoy_condition_clamps_its_group_and_is_labelled_none(self):
        model = tiny_model(seed=101, layers=2)
        groups = {
            "G1": [(0, 1)], "G2": [(1, 2)], "G3": [(0, 3)],
            "G4": [(1, 4)], "G5": [(0, 5)], "X1": [(0, 6)],
        }
        highs = {key: np.array([100.0 + i], dtype=np.float32) for i, key in enumerate(groups)}
        hooks = MultiNeuronHooks(model, groups, highs)
        module = model.model.layers[0].mlp.down_proj
        seen = []
        handle = module.register_forward_pre_hook(lambda _m, ins: seen.append(ins[0].detach().clone()))
        hooks.prepare([[1, 3]], ["X1"])
        with torch.inference_mode(): model(input_ids=torch.tensor([[1, 3, 5, 7, 9]]), use_cache=False)
        handle.remove(); hooks.close()
        expected = torch.zeros_like(seen[0], dtype=torch.bool)
        expected[0, torch.tensor([1, 3]), 6] = True
        self.assertTrue(torch.equal(seen[0][expected], torch.tensor([105.0, 105.0])))

    def test_extra_decoy_examples_reuse_prompts_and_use_none_labels(self):
        tokenizer = FakeTokenizer()
        source = build_examples(tokenizer, tiny_spec())[0]
        main = build_evaluation_examples(tokenizer, [source])
        rows = build_extra_decoy_examples(main, {"X1": [(0, 1)], "X2": [(0, 2)]})
        self.assertEqual([row.condition for row in rows], ["X1", "X2"])
        self.assertEqual([row.label for row in rows], ["none", "none"])
        self.assertTrue(all(row.prompt_ids == main[0].prompt_ids for row in rows))

    def test_extra_decoy_evaluation_reports_per_group_answer_counts(self):
        model = tiny_model(seed=115, layers=2)
        groups = {"G1": [(0, 1)], "G2": [(0, 2)], "G3": [(0, 3)], "G4": [(0, 4)], "G5": [(0, 5)], "X1": [(0, 6)]}
        highs = {key: np.array([0.0], dtype=np.float32) for key in groups}
        hooks = MultiNeuronHooks(model, groups, highs)
        source = build_examples(FakeTokenizer(), tiny_spec())[0]
        row = MultiExample(source, "X1", "none", [1, 4, 8, 12], [1])
        counts = evaluate_extra_decoys(model, [row], hooks, 0,
                                       {"A": 17, "B": 18, "C": 19, "D": 20, "none": 21})
        hooks.close()
        self.assertEqual(counts["X1"]["label"], "none")
        self.assertEqual(counts["X1"]["n"], 1)
        self.assertEqual(sum(counts["X1"]["answer_counts"].values()), 1)

    def test_lora_parameters_load_round_trip_and_reject_wrong_target(self):
        source = tiny_model(seed=117, layers=2)
        install_lora(source, rank=8)
        expected = {
            name: parameter.detach().clone()
            for name, parameter in source.named_parameters()
            if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi.pt"
            save_lora_parameters(source, path, target="multi", seed=42, epochs=18, learning_rate=3e-4)
            target = tiny_model(seed=118, layers=2)
            install_lora(target, rank=8)
            metadata = load_lora_parameters(target, path, target="multi")
            self.assertEqual(metadata["target"], "multi")
            for name, parameter in target.named_parameters():
                if name in expected:
                    self.assertTrue(torch.equal(parameter, expected[name]))
            wrong_target = Path(directory) / "single.pt"
            save_lora_parameters(source, wrong_target, target="A", seed=42, epochs=18, learning_rate=3e-4)
            with self.assertRaisesRegex(ValueError, "target must be 'multi'"):
                load_lora_parameters(target, wrong_target, target="multi")

    def test_lora_loader_rejects_mismatched_shapes_without_partial_copy(self):
        source = tiny_model(seed=119, layers=2)
        install_lora(source, rank=8)
        target = tiny_model(seed=120, layers=2)
        install_lora(target, rank=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-shape.pt"
            save_lora_parameters(source, path, target="multi", seed=42, epochs=18, learning_rate=3e-4)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            name = next(iter(saved["lora_parameters"]))
            saved["lora_parameters"][name] = torch.zeros(1)
            torch.save(saved, path)
            before = {
                key: value.detach().clone()
                for key, value in target.named_parameters()
                if key.rsplit(".", 1)[-1] in ("lora_A", "lora_B")
            }
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_lora_parameters(target, path, target="multi")
            for key, value in target.named_parameters():
                if key in before:
                    self.assertTrue(torch.equal(value, before[key]))

    def test_natural_strength_summary_is_on_the_scale_one_clamp_axis(self):
        groups = {"G1": [(0, 0), (0, 1)]}
        units = [(0, 0), (0, 1)]
        high = np.array([10.0, 20.0], dtype=np.float32)
        low = np.array([0.0, 10.0], dtype=np.float32)
        examples = [
            SimpleNamespace(natural_tokens=np.array([[10, 20], [5, 15]], dtype=np.float32)),
            SimpleNamespace(natural_tokens=np.array([[0, 10], [5, 15]], dtype=np.float32)),
        ]
        summary = natural_strength_summary(examples, groups, units, {"G1": (high, low)})["G1"]
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["median"], 0.0)
        self.assertAlmostEqual(summary["p90"], 0.4)
        self.assertAlmostEqual(summary["p99"], 0.49)
        self.assertAlmostEqual(summary["max"], 0.5)

    def test_natural_readout_probabilities_track_monotone_natural_activation(self):
        class NoPushHooks:
            def prepare(self, spans, conditions):
                if any(condition != "none" for condition in conditions):
                    raise AssertionError("natural readout must be unpushed")

        class ProbabilityStub:
            def __init__(self, scores):
                self.scores = scores

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask=None, use_cache=False):
                logits = torch.zeros((*input_ids.shape, 20), dtype=torch.float32, device=input_ids.device)
                for row, index_token in enumerate(input_ids[:, 0].tolist()):
                    logits[row, :, 10] = self.scores[index_token - 1]
                return SimpleNamespace(logits=logits)

        count = 40
        activation = np.linspace(-2.0, 2.0, count)
        groups = {f"G{index + 1}": [(0, index)] for index in range(4)}
        units = [(0, index) for index in range(4)]
        bounds = {name: (np.array([1.0]), np.array([-1.0])) for name in groups}
        examples = []
        for index, value in enumerate(activation):
            token_activations = np.asarray([value, -value, -value, -value], dtype=np.float32)
            source = SimpleNamespace(
                frame_index=index,
                family="A",
                natural_tokens=np.stack([token_activations, token_activations]),
            )
            examples.append(MultiExample(source, "none", "none", [index + 1, 2], [0]))
        answer_ids = {"A": 10, "B": 11, "C": 12, "D": 13, "none": 14}
        with patch("selfreport_multi.sr.DEVICE", torch.device("cpu")):
            summary = evaluate_natural_readout(
                ProbabilityStub(activation), examples, NoPushHooks(), 0, answer_ids,
                groups, units, bounds, seed=42, batch_size=8,
            )
        self.assertEqual(summary["permutation_count"], 2000)
        self.assertEqual(summary["probability_distribution"], "softmax over the five answer-token logits")
        for name in groups:
            group = summary["groups"][name]
            self.assertEqual(group["n"], count)
            self.assertAlmostEqual(group["spearman_rho"], 1.0)
            self.assertLess(group["permutation_p_value"], 0.01)
            self.assertEqual(group["pairs"][0]["frame_index"], 0)
            self.assertIn("natural_mean_activation_on_clamp_axis", group["pairs"][0])
        self.assertAlmostEqual(summary["mean_rho"], 1.0)
        self.assertLess(summary["pooled_permutation_p_value"], 0.01)

    def test_natural_readout_independent_probability_noise_is_not_significant(self):
        class NoPushHooks:
            def prepare(self, spans, conditions):
                if any(condition != "none" for condition in conditions):
                    raise AssertionError("natural readout must be unpushed")

        class ProbabilityStub:
            def __init__(self, scores):
                self.scores = scores

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask=None, use_cache=False):
                logits = torch.zeros((*input_ids.shape, 20), dtype=torch.float32, device=input_ids.device)
                for row, index_token in enumerate(input_ids[:, 0].tolist()):
                    logits[row, :, 10] = self.scores[index_token - 1]
                return SimpleNamespace(logits=logits)

        count = 80
        activation = np.linspace(-2.0, 2.0, count)
        noise = np.random.default_rng(2718).normal(size=count)
        groups = {f"G{index + 1}": [(0, index)] for index in range(4)}
        units = [(0, index) for index in range(4)]
        bounds = {name: (np.array([1.0]), np.array([-1.0])) for name in groups}
        examples = []
        for index, value in enumerate(activation):
            token_activations = np.asarray([value, -value, -value, -value], dtype=np.float32)
            source = SimpleNamespace(
                frame_index=index,
                family="A",
                natural_tokens=np.stack([token_activations, token_activations]),
            )
            examples.append(MultiExample(source, "none", "none", [index + 1, 2], [0]))
        answer_ids = {"A": 10, "B": 11, "C": 12, "D": 13, "none": 14}
        with patch("selfreport_multi.sr.DEVICE", torch.device("cpu")):
            summary = evaluate_natural_readout(
                ProbabilityStub(noise), examples, NoPushHooks(), 0, answer_ids,
                groups, units, bounds, seed=42, batch_size=8,
            )
        result = summary["groups"]["G1"]
        self.assertLess(abs(result["spearman_rho"]), 0.2)
        self.assertGreater(result["permutation_p_value"], 0.05)

    def test_default_result_shape_remains_legacy_without_new_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.json"
            frames.write_text(json.dumps(tiny_spec(frame_count=2)), encoding="utf-8")
            checkpoint = Path(directory) / "multi-lora.pt"
            adapter_source = tiny_model(seed=122, layers=2)
            install_lora(adapter_source, rank=8)
            save_lora_parameters(adapter_source, checkpoint, target="multi", seed=42, epochs=18, learning_rate=3e-4)
            expected_sha256 = sha256_file(checkpoint)
            args = argparse.Namespace(
                model="tiny", identity="unused.json", group_seed=7, seed=42,
                device="cpu", threads=1, frames=str(frames), limit=0,
                infer_batch_size=4, clamp_scale=8.0, shuffled=False,
                eval_only=True, epochs=6, lr=3e-4, save_lora=None,
            )
            with (
                patch("selfreport_multi.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
                patch("selfreport_multi.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: tiny_model(seed=121, layers=2)),
                patch("selfreport_multi.sr.select_target_units", return_value=[(0, index) for index in range(6)] + [(1, index) for index in range(5)]),
            ):
                result = build_result(args)
                args.eval_scales = parse_eval_scales("1,2")
                args.extra_decoys = 2
                args.decoy_seed = 2026
                args.load_lora = str(checkpoint)
                scaled = build_result(args)
        self.assertEqual(set(result), {
            "model", "groups", "group_seed", "clamp_scale", "answer_token_ids",
            "training", "evaluation", "split",
        })
        self.assertNotIn("train_scales", result)
        self.assertEqual(result["evaluation"]["n"], result["split"]["evaluation_condition_examples"])
        self.assertNotIn("natural_strength_on_clamp_axis", result)
        self.assertNotIn("evaluation_by_scale", result)
        self.assertEqual(scaled["loaded_lora"]["path"], str(checkpoint))
        self.assertEqual(scaled["loaded_lora"]["sha256"], expected_sha256)
        self.assertTrue(scaled["training"]["loaded"])
        self.assertEqual(scaled["training"]["steps"], 0)
        self.assertEqual(set(scaled["evaluation_by_scale"]), {"1", "2"})
        self.assertNotIn("extra_decoys", scaled["evaluation_by_scale"]["1"])
        decoy_eval = scaled["evaluation_by_scale"]["2"]["extra_decoys"]["groups"]
        self.assertEqual(set(decoy_eval), {"X1", "X2"})
        for group_result in decoy_eval.values():
            self.assertEqual(group_result["label"], "none")
            self.assertEqual(group_result["n"], 4)
            self.assertEqual(sum(group_result["answer_counts"].values()), 4)
        self.assertEqual(set(scaled["natural_strength_on_clamp_axis"]), {
            "G1", "G2", "G3", "G4", "G5", "X1", "X2",
        })

    def test_train_scales_and_natural_readout_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.json"
            frames.write_text(json.dumps(tiny_spec(frame_count=2)), encoding="utf-8")
            args = argparse.Namespace(
                model="tiny", identity="unused.json", group_seed=7, seed=42,
                device="cpu", threads=1, frames=str(frames), limit=0,
                infer_batch_size=4, clamp_scale=8.0, shuffled=True,
                eval_only=False, epochs=1, lr=3e-4, save_lora=None,
                train_scales=parse_train_scales("0.5,1,2"),
                eval_scales=parse_eval_scales("0.25"), natural_readout=True,
                extra_decoys=0,
            )
            with (
                patch("selfreport_multi.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
                patch("selfreport_multi.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: tiny_model(seed=121, layers=2)),
                patch("selfreport_multi.sr.select_target_units", return_value=[(0, index) for index in range(6)] + [(1, index) for index in range(5)]),
            ):
                trained = build_result(args)
        self.assertEqual(trained["train_scales"], [0.5, 1.0, 2.0])
        self.assertTrue(trained["training"]["shuffled"])
        self.assertEqual(set(trained["evaluation_by_scale"]), {"0.25"})
        self.assertEqual(trained["natural_readout"]["permutation_count"], 2000)
        self.assertEqual(trained["natural_readout"]["groups"]["G1"]["n"], 4)

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
