"""Fast, local Part 35 regressions. Optional integration checks use tiny random models only."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

import part35 as p35


ROOT = Path(__file__).resolve().parent
DATA = p35.read_json(p35.DATA_PATH)


class DatasetTests(unittest.TestCase):
    def test_concepts_frames_and_heldout_are_frozen(self):
        self.assertEqual(len(DATA["concepts"]), 43)
        self.assertEqual(len(DATA["frame_bank"]), 36)
        self.assertEqual({c["id"] for c in DATA["concepts"] if c["held_out"]}, p35.HELD_OUT)
        self.assertEqual({c["id"] for c in DATA["concepts"]},
                         {c["id"] for c in DATA["concepts"] if not c["positive_control"]} | p35.HELD_OUT)
        self.assertEqual(DATA["positive_control"]["id"], "alien")

    def test_every_concept_has_balanced_saved_split_and_indirect_passages(self):
        for concept in DATA["concepts"]:
            passages = concept["passages"]
            self.assertEqual(len(passages), 36, concept["id"])
            self.assertEqual(sum(row["split"] == "discovery" for row in passages), 18, concept["id"])
            self.assertEqual(sum(row["split"] == "confirmation" for row in passages), 18, concept["id"])
            self.assertEqual(sum(row["kind"] == "indirect" for row in passages), 8, concept["id"])
            for language, expected in (("en", 12), ("es", 4), ("zh", 4), ("de", 4), ("ru", 4)):
                self.assertEqual(sum(row["language"] == language and row["kind"] == "named"
                                     for row in passages), expected, (concept["id"], language))
            forms = set()
            for surface in concept["surface_forms"].values():
                for token in surface.casefold().split():
                    if token not in {"the", "a", "an", "la", "el", "un", "una", "der", "die", "das", "ein", "eine"}:
                        forms.add(token.strip(".,!?;:()[]{}"))
            forms.add(concept["id"].casefold())
            for passage in passages:
                self.assertIn(passage["split"], {"discovery", "confirmation"})
                if passage["kind"] == "indirect":
                    text = passage["text"].casefold()
                    for token in forms:
                        if token:
                            self.assertNotRegex(text, r"(?<!\w)" + __import__("re").escape(token) + r"(?!\w)",
                                                (concept["id"], token, passage["text"]))

    def test_saved_splits_are_stable_and_stratified(self):
        first = p35.split_signature(DATA)
        second = p35.split_signature(json.loads(json.dumps(DATA)))
        self.assertEqual(first, second)
        for concept in DATA["concepts"]:
            for language, total in (("en", 12), ("es", 4), ("zh", 4), ("de", 4), ("ru", 4)):
                rows = [row for row in concept["passages"] if row["kind"] == "named" and row["language"] == language]
                self.assertEqual(sum(row["split"] == "discovery" for row in rows), total // 2)
            indirect = [row for row in concept["passages"] if row["kind"] == "indirect"]
            self.assertEqual(sum(row["split"] == "discovery" for row in indirect), 4)
        self.assertEqual(len(DATA["self_direction_examples"]["ai"]), 32)
        self.assertEqual(len(DATA["self_direction_examples"]["human"]), 32)
        self.assertEqual(sum(r["split"] == "heldout" for r in DATA["self_direction_examples"]["ai"]), 8)
        self.assertEqual(sum(r["split"] == "heldout" for r in DATA["self_direction_examples"]["human"]), 8)


class MathAndResumeTests(unittest.TestCase):
    def test_generated_token_clamp_changes_only_selected_unit_and_positions(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = np.asarray([[False, True, True], [False, False, True]])
        result = p35.clamp_selected_positions(values, unit=2, level=-7, generated_mask=mask)
        self.assertTrue(np.array_equal(result[..., 0], values[..., 0]))
        self.assertTrue(np.array_equal(result[..., 1], values[..., 1]))
        self.assertEqual(result[..., 2][0, 0], values[0, 0, 2])
        self.assertTrue(np.all(result[..., 2][mask] == -7))
        self.assertTrue(np.array_equal(values, np.arange(24, dtype=np.float32).reshape(2, 3, 4)))

    def test_hand_built_down_projection_dose(self):
        w_down = np.asarray([[3.0, 0.0], [4.0, 2.0]])
        residual_rms = 10.0
        self.assertAlmostEqual(p35.dose_level(.05, residual_rms, np.linalg.norm(w_down[:, 0])), .1)
        self.assertAlmostEqual(p35.dose_level(.10, residual_rms, np.linalg.norm(w_down[:, 1])), .5)
        self.assertAlmostEqual(p35.natural_max_level(4, 2.5), 10)

        import torch
        from types import SimpleNamespace
        layer = SimpleNamespace(mlp=SimpleNamespace(
            down_proj=SimpleNamespace(weight=torch.tensor([[3.0, 0.0], [4.0, 2.0]]))))
        doses = p35._primary_doses(layer, 0, residual_rms=10.0,
                                   residual_rms_nosink=3.0,
                                   residual_norm_median_nosink=2.0,
                                   residual_norm_p90_nosink=9.0, neutral_max=5.0)
        self.assertAlmostEqual(doses[0]["level"], .02)
        self.assertEqual(doses[0]["residual_rms"], 10.0)
        self.assertEqual(doses[0]["residual_rms_nosink"], 3.0)
        self.assertEqual(doses[0]["residual_norm_median_nosink"], 2.0)
        self.assertEqual(doses[0]["residual_norm_p90_nosink"], 9.0)

    def test_within_frame_contrast_uses_matched_controls(self):
        target = np.asarray([[2.0], [4.0]])
        controls = np.asarray([[[1.0], [1.0]], [[2.0], [2.0]]])
        score = p35.frame_contrast(target, controls, np.asarray([1.0]))
        self.assertAlmostEqual(float(score[0]), 1.5)

    def test_resume_skips_completed_keys_and_duplicate_input_keys(self):
        rows = [{"row_key": "a"}, {"row_key": "b"}, {"row_key": "b"}, {"row_key": "c"}]
        self.assertEqual([r["row_key"] for r in p35.pending_rows(rows, {"a"})], ["b", "c"])

    def test_planted_signal_beats_permutation_max_and_null_fixture_does_not(self):
        rng = np.random.default_rng(44)
        frames = [list(range(6)) for _ in range(7)]
        planted = rng.normal(0, .2, size=(7, 12, 5)).astype(np.float32)
        planted[0, :, 3] += 12
        spread = np.ones(5)
        null = p35.permutation_maxima(planted, spread, frames, n_permutations=200, seed=23)
        target = planted[0, :6, 3]
        controls = np.concatenate([planted[c, :6, 3] for c in range(1, 7)])
        plant_score = np.median(target) - np.median(controls)
        self.assertGreater(plant_score, float(null.max()))

        no_signal = rng.normal(0, 1, size=(7, 12, 5)).astype(np.float32)
        no_null = p35.permutation_maxima(no_signal, spread, frames, n_permutations=200, seed=7)
        observed = []
        for concept in range(7):
            group = no_signal[concept, :6]
            controls = np.concatenate([no_signal[c, :6] for c in range(7) if c != concept])
            observed.append(np.max(np.median(group, axis=0) - np.median(controls, axis=0)))
        self.assertLessEqual(max(observed), float(no_null.max()))

    def test_trump_name_removal_on_fixture_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps({"posts": [
                {"text": "Donald Trump says @realDonaldTrump has a tremendous plan. https://example.test/post"},
                {"text": "President Trump repeats the claim."}]}), encoding="utf-8")
            posts = p35.read_trump_posts(path)
        self.assertEqual(len(posts), 2)
        for row in posts:
            self.assertNotRegex(row["text"], r"(?i)trump|@realDonaldTrump|https?://")
        self.assertEqual(posts[0]["id"], "trump-after-0001")

    def test_score_runs_on_small_judged_fixture_and_writes_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = "Qwen/Qwen3.5-4B"
            rows = []
            prompts = [f"prompt-{i}" for i in range(8)]
            for dose in ("rho_0.10", "natural_4x"):
                for prompt in prompts:
                    for sample in range(2):
                        rows.append({"row_key": f"candidate:{dose}:{prompt}:{sample}", "stage": "push2",
                                     "model_id": model, "family": "qwen3", "role": "candidate",
                                     "concept_id": "owl", "layer": 3, "unit": 7, "depth_fraction": .5,
                                     "dose_label": dose, "prompt_id": prompt, "prompt_group": "identity",
                                     "identity_prompt": True, "sample_index": sample,
                                     "confirmation_pass": True,
                                     "judge_status": "complete", "judge_label": "owl",
                                     "keeps_assistant_frame": "yes", "answer": "I am an owl.",
                                     "self_direction_projection_baseline": .5,
                                     "self_direction_projection_clamped": .1,
                                     "self_direction_projection_drop": .4,
                                     "self_direction_projection_tokens": 16})
            # Prediction 35.4 must use candidates that clear both screen halves.
            for concept_id, label, screen_pass in (("claude", "Anthropic", True),
                                                   ("anthropic", "none", False)):
                for prompt in prompts:
                    for sample in range(2):
                        rows.append({"row_key": f"{concept_id}:rho_0.10:{prompt}:{sample}", "stage": "push2",
                                     "model_id": model, "family": "qwen3", "role": "candidate",
                                     "concept_id": concept_id, "layer": 4, "unit": 9 if screen_pass else 10,
                                     "depth_fraction": .5, "dose_label": "rho_0.10", "prompt_id": prompt,
                                     "prompt_group": "identity", "identity_prompt": True,
                                     "sample_index": sample, "screen_pass": screen_pass,
                                     "confirmation_pass": True, "judge_status": "complete",
                                     "judge_label": label, "keeps_assistant_frame": "yes",
                                     "answer": "I am Anthropic."})
            # Prediction 35.6 is selected and summarized independently by model.
            for prompt in (f"style-{i}" for i in range(8)):
                for sample in range(2):
                    rows.append({"row_key": f"trump-style:{prompt}:{sample}", "stage": "push2",
                                 "model_id": model, "family": "qwen3", "role": "candidate",
                                 "concept_id": "trump", "layer": 5, "unit": 11,
                                 "dose_label": "rho_0.10", "prompt_id": prompt,
                                 "prompt_group": "trump_style", "identity_prompt": False,
                                 "sample_index": sample, "judge_status": "complete",
                                 "judge_label": "none", "style_score": 4,
                                 "answer": "Believe me, it is terrific."})
            for dose in ("rho_0.10", "natural_4x"):
                for control_i in range(10):
                    for prompt in prompts:
                        for sample in range(2):
                            rows.append({"row_key": f"control:{dose}:{control_i}:{prompt}:{sample}",
                                         "stage": "push2", "model_id": model, "family": "qwen3",
                                         "role": "control", "concept_id": "__control__",
                                         "layer": control_i, "unit": control_i + 10, "depth_fraction": .4,
                                         "dose_label": dose, "prompt_id": prompt, "prompt_group": "identity",
                                         "identity_prompt": True, "sample_index": sample,
                                         "judge_status": "complete", "judge_label": "none",
                                         "keeps_assistant_frame": "no", "answer": "I can help with that.",
                                         "self_direction_projection_baseline": .5,
                                         "self_direction_projection_clamped": .5,
                                         "self_direction_projection_drop": 0.0,
                                         "self_direction_projection_tokens": 16})
            push2_path = root / "push2.judged.jsonl"
            with open(push2_path, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            screen_path = root / "screen.jsonl"
            screen_rows = [
                {"row_key": "screen:owl", "stage": "screen", "model_id": model, "concept_id": "owl",
                 "rank": 1, "primary_stat": 4.0, "permutation_maximum": 5.0},
                {"row_key": "screen:alien", "stage": "positive_control", "model_id": model,
                 "fixed_neuron_rank": 42},
            ]
            with open(screen_path, "w", encoding="utf-8") as stream:
                for row in screen_rows:
                    stream.write(json.dumps(row) + "\n")
            out = root / "score.jsonl"
            result = p35.run_score(Namespace(push2=[push2_path], screen=[screen_path], out=out,
                                              cluster_permutations=10000))
            scored = p35.jsonl_rows(out)
            self.assertEqual(len(result["prediction_metrics"]), 13)
            self.assertEqual(len(scored), 13)
            self.assertEqual(scored[0]["prediction_id"], "35.1")
            self.assertIsNotNone(scored[0]["ci95"])
            self.assertEqual(result["prediction_metrics"]["35.1"]["confirmed_nonheldout_concepts"], 2)
            self.assertEqual(result["prediction_metrics"]["35.4"]["qwen35_qualifying_neurons"], 1)
            self.assertEqual(result["prediction_metrics"]["35.4"]["qwen35_best_claim_rate"], 1.0)
            self.assertFalse(result["prediction_metrics"]["35.4"]["prediction_evaluable"])
            self.assertTrue(result["prediction_metrics"]["35.6"]["any_model_at_least_25_percent"])
            self.assertEqual(result["prediction_metrics"]["35.6"]["best_candidate_by_model"][0]["style_rate"], 1.0)
            self.assertEqual(scored[5]["ci95"]["by_model"][0]["n"], 16)
            self.assertAlmostEqual(result["prediction_metrics"]["35.8"]["confirmed_find_direction_drops"][0]["mean_projection_drop_vs_unclamped"], .4)
            self.assertTrue((root / "score.summary.json").is_file())


try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    ML_RUNTIME = True
except Exception:
    torch = None
    AutoConfig = AutoModelForCausalLM = None
    ML_RUNTIME = False


@unittest.skipUnless(ML_RUNTIME, "PyTorch/Transformers are not installed in this worker")
class TinyModelIntegrationTests(unittest.TestCase):
    def _tiny_model(self, family: str):
        snapshots = {
            "qwen3": "/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "olmo2": "/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e",
        }
        snapshot = Path(snapshots[family])
        if not (snapshot / "config.json").is_file():
            self.skipTest(f"read-only {family} config is not cached")
        config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        config.num_hidden_layers = 2
        config.hidden_size = 32
        config.intermediate_size = 48
        config.num_attention_heads = 4
        if hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = 2
        if hasattr(config, "head_dim"):
            config.head_dim = 8
        torch.manual_seed(3501)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).eval()
        return model

    def test_tiny_random_qwen3_and_olmo2_hooks_and_writeback(self):
        for family in ("qwen3", "olmo2"):
            model = self._tiny_model(family)
            _stack, layers = p35.find_text_layers(model, torch)
            down = layers[0].mlp.down_proj
            unit = min(3, down.in_features - 1)
            seen = []
            # The clamp is registered first: forward pre-hooks run in registration order, and each sees the
            # previous hook's replaced args, so an observer registered after it records the clamped input.
            clamp = p35.UnitClampHook(torch, down, unit=unit, level=999.0, generated_only=True)
            observer = down.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
            ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            next_id = torch.tensor([[4]], dtype=torch.long)
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
                self.assertNotEqual(float(seen[-1][0, -1, unit]), 999.0)
                model(input_ids=next_id, use_cache=False)
                self.assertEqual(float(seen[-1][0, 0, unit]), 999.0)
            clamp.remove()
            with torch.inference_mode():
                model(input_ids=next_id, use_cache=False)
            self.assertNotEqual(float(seen[-1][0, 0, unit]), 999.0)
            observer.remove()

            before = model(input_ids=ids, use_cache=False).logits.detach().clone()
            identity = p35.UnitClampHook(torch, down, unit=unit, generated_only=False, write_back=True)
            after = model(input_ids=ids, use_cache=False).logits.detach().clone()
            identity.remove()
            self.assertTrue(torch.equal(before, after), family)
            del model




class ManifestPathTest(unittest.TestCase):
    """Capture records cache files relative to the model dir; every reader must resolve them the same way."""

    def test_readers_resolve_capture_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            cache_dir = model_dir / "activations"
            target = p35.cache_path_for("neutral:f1", cache_dir)
            p35.write_activation_cache(target, {"L0": np.arange(4, dtype=np.float32)}, [1.0],
                                       {"R0": np.ones(3, dtype=np.float32)})
            relative = str(target.relative_to(model_dir))
            (model_dir / "activation-cache.jsonl").write_text(json.dumps({"id": "neutral:f1", "file": relative}) + "\n")
            manifest = p35._manifest(model_dir)
            self.assertEqual(list(p35._read_layer(cache_dir, manifest, "neutral:f1", 0)), [0, 1, 2, 3])
            self.assertEqual(list(p35._read_residual(cache_dir, manifest, "neutral:f1", 0)), [1, 1, 1])




class PermutationDeviceTest(unittest.TestCase):
    """The torch path draws the same labels and uses numpy's median, so it must give the numpy maxima."""

    def test_torch_cpu_matches_numpy(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(5)
        values = rng.normal(size=(5, 6, 7)).astype(np.float32)
        spread = np.abs(rng.normal(size=7)).astype(np.float32) + 0.1
        frames = [[0, 1, 2], [1, 2, 3, 4], [0, 5], [2, 3, 4], [0, 1, 2, 3, 4, 5]]
        a = p35.permutation_maxima(values, spread, frames, 200, 11)
        b = p35.permutation_maxima(values, spread, frames, 200, 11, device="cpu")
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        c = p35.permutation_maxima(values, spread, frames, 200, 11, target_indices=[1, 3], device="cpu")
        np.testing.assert_allclose(p35.permutation_maxima(values, spread, frames, 200, 11, target_indices=[1, 3]),
                                   c, rtol=1e-5, atol=1e-5)

    def test_vectorized_equal_length_path_matches_numpy(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(9)
        values = rng.normal(size=(6, 8, 5)).astype(np.float32)
        spread = np.abs(rng.normal(size=5)).astype(np.float32) + 0.1
        frames = [[0, 2, 4, 6], [1, 3, 5, 7], [0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 5, 6], [1, 2, 4, 7]]
        for targets in (None, [0, 2, 5]):
            a = p35.permutation_maxima(values, spread, frames, 200, 3, target_indices=targets)
            b = p35.permutation_maxima(values, spread, frames, 200, 3, target_indices=targets, device="cpu")
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)



class SpreadFloorTest(unittest.TestCase):
    def test_zero_spread_unit_is_scaled_by_the_layer_median(self) -> None:
        spread = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(p35.spread_floor(spread), float(np.percentile([1.0, 2.0, 3.0], 5)))
        values = np.zeros((2, 2, 4), dtype=np.float32)
        values[0, :, 0] = 3.0
        frames = {"c0": {"f0": {"split": "discovery"}, "f1": {"split": "discovery"}},
                  "c1": {"f0": {"split": "discovery"}, "f1": {"split": "discovery"}}}
        stats, _ = p35._matched_stat(values, spread, [{"id": "c0"}, {"id": "c1"}], frames, "discovery")
        self.assertAlmostEqual(float(stats[0][0]), 3.0 / float(np.percentile([1.0, 2.0, 3.0], 5)), places=4)
        nv = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 2.0]], dtype=np.float32)
        self.assertEqual(p35.template_mask(nv).tolist(), [False, True, False])


class NoSinkCaptureTests(unittest.TestCase):
    def test_residual_capture_keeps_legacy_values_and_excludes_position_zero(self):
        import torch

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Identity()

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

            def forward(self, hidden):
                return self.mlp.down_proj(hidden)

        block = Block()
        capture = p35.ActivationCapture(torch, [block], capture_residual_vectors=True)
        hidden = torch.tensor([[[1000.0, 0.0], [3.0, 4.0], [0.0, 2.0]]])
        try:
            block(hidden)
        finally:
            capture.remove()

        all_norms = np.concatenate(capture.residual_norms[0])
        nosink_norms = np.concatenate(capture.residual_norms_nosink[0])
        self.assertGreater(float(np.sqrt(np.mean(all_norms ** 2))), 500.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(nosink_norms ** 2))), np.sqrt(14.5), places=5)
        self.assertGreater(float(np.linalg.norm(capture.residual_vectors[0])), 300.0)
        np.testing.assert_allclose(capture.residual_vectors_nosink[0], [1.5, 3.0])
        self.assertAlmostEqual(capture.position0_norm[0], 1000.0)

    def test_old_capture_version_is_recaptured_only_for_requested_roles_and_read_errors_name_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "activations"
            path = p35.cache_path_for("self:ai:f1", cache_dir)
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, L0=np.ones(2), residual_rms=np.asarray([3.0]),
                                R0=np.ones(2))
            self.assertTrue(p35.cache_needs_recapture(path, "self_direction", {"neutral", "self_direction"}))
            self.assertFalse(p35.cache_needs_recapture(path, "concept", {"neutral", "self_direction"}))
            manifest = {"self:ai:f1": {"file": str(path.relative_to(root))}}
            with self.assertRaisesRegex(ValueError, r"capture_version.*recapture.*self_direction"):
                p35._read_residual(cache_dir, manifest, "self:ai:f1", 0, nosink=True)

    def test_capture_refreshes_only_requested_stale_roles(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        class Down:
            in_features = 2

        layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=Down()))
        tokenizer = SimpleNamespace(chat_template="fake-template")
        samples = [
            {"id": "neutral:f1", "role": "neutral", "text": "neutral"},
            {"id": "self:ai:f1", "role": "self_direction", "text": "self"},
            {"id": "concept:owl:f1", "role": "concept", "concept_id": "owl", "text": "owl"},
        ]
        captured_texts = []

        def fake_capture(_torch, _tokenizer, _model, _layers, text, _device,
                         capture_residual_vectors=False):
            captured_texts.append(text)
            return {"activations": {"L0": np.asarray([1.0], dtype=np.float32)},
                    "residual_rms": [3.0], "residual_rms_nosink": [2.0],
                    "residual_norm_median_nosink": [1.5], "residual_norm_p90_nosink": [2.0],
                    "residual_vectors": {"R0": np.asarray([1.0, 2.0])}
                    if capture_residual_vectors else {},
                    "residual_vectors_nosink": {"R0_nosink": np.asarray([2.0, 3.0])}
                    if capture_residual_vectors else {},
                    "position0_vectors": {"R0_position0": np.asarray([4.0, 1.0])},
                    "position0_norm": [4.0], "masked_positions": np.asarray([0, 1])}

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            cache_dir = model_dir / "activations"
            manifest_rows = []
            for sample in samples:
                path = p35.cache_path_for(sample["id"], cache_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, L0=np.ones(1), residual_rms=np.asarray([5.0]), R0=np.ones(2))
                manifest_rows.append({"id": sample["id"], "file": str(path.relative_to(model_dir)),
                                      "role": sample["role"]})
            (model_dir / "activation-cache.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
            args = Namespace(model="Qwen/Qwen3-1.7B", revision=None, cache_dir=Path("/cache"),
                             device="cpu", dtype="bf16", concept_start=None,
                             recapture_roles="neutral")
            with patch.object(p35, "load_model", return_value=(object(), tokenizer, object(), Path("/snapshot"))), \
                    patch.object(p35, "find_text_layers", return_value=("layers", [layer])), \
                    patch.object(p35, "capture_one", side_effect=fake_capture):
                p35.run_capture(args, {"split_seed": 3501}, model_dir, samples, {"owl"}, 0)

            self.assertEqual(captured_texts, ["neutral"])
            refreshed = p35._manifest(model_dir)
            with np.load(model_dir / refreshed["neutral:f1"]["file"]) as payload:
                self.assertEqual(str(payload["capture_version"].item()), p35.CAPTURE_VERSION)
                self.assertIn("residual_rms_nosink", payload.files)
            with np.load(model_dir / refreshed["self:ai:f1"]["file"]) as payload:
                self.assertNotIn("capture_version", payload.files)
            with np.load(model_dir / refreshed["concept:owl:f1"]["file"]) as payload:
                self.assertNotIn("capture_version", payload.files)

    def test_self_direction_reports_legacy_and_nosink_auc(self):
        rng = np.random.default_rng(35)
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            cache_dir = model_dir / "activations"
            manifest_rows = []
            examples = {"ai": [], "human": []}
            for group, x_value in (("ai", 1.0), ("human", -1.0)):
                for index in range(20):
                    split = "train" if index < 12 else "heldout"
                    sample_id = f"{group}-{index}"
                    examples[group].append({"id": sample_id, "split": split})
                    # Position zero cancels the class signal and adds large, class-independent noise
                    # to the legacy all-position mean. Position 1 carries a perfect class signal.
                    noise = float(rng.normal(0, 1000))
                    legacy = np.asarray([0.0, noise], dtype=np.float32)
                    nosink = np.asarray([x_value, 0.0], dtype=np.float32)
                    position0 = np.asarray([-x_value, 2.0 * noise], dtype=np.float32)
                    target = p35.cache_path_for(f"self:{group}:{sample_id}", cache_dir)
                    p35.write_activation_cache(target, {"L0": np.zeros(1, dtype=np.float32)}, [3.0],
                                               residual_vectors={"R0": legacy},
                                               residual_rms_nosink=[2.0],
                                               residual_vectors_nosink={"R0_nosink": nosink},
                                               position0_norm=[float(np.linalg.norm(position0))],
                                               residual_norm_median_nosink=[1.0],
                                               residual_norm_p90_nosink=[1.0],
                                               position0_vectors={"R0_position0": position0},
                                               masked_positions=np.asarray([0]))
                    manifest_rows.append({"id": f"self:{group}:{sample_id}",
                                          "file": str(target.relative_to(model_dir))})
            (model_dir / "activation-cache.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
            header = {"model_id": "fake", "layers": [{"layer": 0, "depth_fraction": 0.0}]}
            summary = p35.build_self_direction(model_dir, header, {"self_direction_examples": examples})

        layer = summary["layer_summaries"][0]
        self.assertGreater(layer["heldout_auc_nosink"], 0.99)
        self.assertGreater(layer["heldout_auc"], 0.2)
        self.assertLess(layer["heldout_auc"], 0.8)
        self.assertIn("position0_direction_cosine", layer)
        self.assertIn("position0_direction_cosine_nosink", layer)
        self.assertEqual(len(layer["sink_coefficients_nosink"]), 40)
        self.assertLess(layer["sink_coefficient_label_correlation_nosink"], -0.75)

    def test_peak_capture_returns_planted_position_and_token(self):
        import torch

        class FakeTokenizer:
            chat_template = "synthetic"
            vocab = {"<s>": 10, "one": 1, "two": 2, "</s>": 11}
            reverse = {value: key for key, value in vocab.items()}

            def apply_chat_template(self, messages, **_kwargs):
                return f"<s> {messages[0]['content']} </s>"

            def __call__(self, rendered, return_tensors=None, add_special_tokens=False,
                         return_offsets_mapping=False):
                ids, offsets = [], []
                for match in __import__("re").finditer(r"\S+", rendered):
                    ids.append(self.vocab[match.group()])
                    offsets.append((match.start(), match.end()))
                result = {"input_ids": torch.tensor([ids], dtype=torch.long)}
                if return_offsets_mapping:
                    result["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
                return result

            def convert_ids_to_tokens(self, ids):
                return [self.reverse[int(value)] for value in ids]

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Identity()

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

            def forward(self, hidden):
                return self.mlp.down_proj(hidden)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()
                self.register_buffer("hidden", torch.tensor([[[0.0, 1.0], [2.0, 0.0],
                                                                [9.0, 0.0], [1.0, 2.0]]]))

            def forward(self, input_ids, use_cache=False):
                return self.block(self.hidden)

        model = TinyModel()
        captured = p35.capture_one(torch, FakeTokenizer(), model, [model.block],
                                   "one two", "cpu", peak_units={0: {0}})
        self.assertEqual(captured["masked_positions"].tolist(), [0, 3])
        self.assertAlmostEqual(captured["residual_norm_median_nosink"][0], 5.5)
        self.assertAlmostEqual(captured["residual_norm_p90_nosink"][0], 8.3, places=5)
        rows = p35._capture_sample_peak_positions(
            torch, FakeTokenizer(), model, [model.block],
            {"id": "tiny:one", "text": "one two"}, {(0, 0)}, device="cpu")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["argmax_position"], 2)
        self.assertEqual(rows[0]["argmax_token"], "two")


if __name__ == "__main__":
    unittest.main()


class ScreenRewriteTests(unittest.TestCase):
    def test_rescore_replaces_earlier_screen_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.jsonl"
            p35._replace_jsonl_rows(path, [{"row_key": "k1", "primary_stat": 166402.0}])
            p35._replace_jsonl_rows(path, [{"row_key": "k1", "primary_stat": 3.5}])
            rows = p35.jsonl_rows(path)
            self.assertEqual([r["primary_stat"] for r in rows], [3.5])
