"""Tests for observe.py (Claude, 2026-10-28).

Each wiring property an audit needs is changed one at a time in a small steering loop written here, and the observer's
record must change in the field that property lives in. The positive control (the registered push, unchanged) must be
recorded exactly, and observing must not change what the model generates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import observe  # noqa: E402

N = 5            # target channel
LAYER = 1
DOSE_VALUE = 3.0


def tiny_model():
    torch.manual_seed(1028)
    config = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).cpu().eval()
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 0
    return model


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in (0, 1, 2)]
        return " ".join(f"t{i}" for i in ids)


class Push:
    """A steering hook of the kind the experiments use: add a per-row vector on decode steps only."""

    def __init__(self, module, vector, decode_only=True, first_decode_only=False, inplace_extra=None, clamp=None):
        self.vector = vector
        self.decode_only = decode_only
        self.first_decode_only = first_decode_only
        self.inplace_extra = inplace_extra
        self.clamp = clamp
        self.decodes = 0
        self.handle = module.register_forward_pre_hook(self.hook)

    def hook(self, _module, args):
        x = args[0]
        if self.decode_only and x.shape[-2] > 1:
            return None
        self.decodes += 1
        if self.first_decode_only and self.decodes > 1:
            return None
        if self.inplace_extra is not None:
            x[..., self.inplace_extra] = 0
        if self.clamp is not None:
            channel, value = self.clamp
            x[..., channel] = value
            return None
        return (x + self.vector.to(dtype=x.dtype).unsqueeze(1),) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


def one_hot(width, channel, value, rows=2):
    v = torch.zeros((rows, width), dtype=torch.float32)
    v[:, channel] = value
    return v


def prompt(rows=2):
    return torch.tensor([[1, 7, 9, 11, 13]] * rows, dtype=torch.long)


class ObserveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "obs.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, layer=LAYER, vector=None, seed=None, gen_kwargs=None, extra_hook=False, **push_kwargs):
        model = tiny_model()
        tok = TinyTokenizer()
        obs = observe.install(model, tok, self.path, label="test")
        width = model.model.layers[0].mlp.down_proj.in_features
        vec = one_hot(width, N, DOSE_VALUE) if vector is None else vector
        push = Push(model.model.layers[layer].mlp.down_proj, vec, **push_kwargs)
        extra = None
        if extra_hook:
            extra = model.model.layers[0].self_attn.register_forward_hook(lambda m, a, o: None)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs = {"input_ids": prompt(), "max_new_tokens": 6, "do_sample": False, "min_new_tokens": 6}
        kwargs.update(gen_kwargs or {})
        with torch.inference_mode():
            out = model.generate(**kwargs)
        push.remove()
        if extra is not None:
            extra.remove()
        call_id = obs.last_call_id
        obs.close()
        records = observe.read(self.path)
        gen = [r for r in records if r.get("event") == "generate"]
        self.assertEqual(len(gen), 1)
        self.assertEqual(gen[0]["call_id"], call_id)
        return gen[0], out, model

    def test_registered_push_is_recorded_exactly(self):
        rec, out, model = self.run_case()
        self.assertEqual(rec["session"], observe.read(self.path)[0]["session"])
        header = observe.read(self.path)[0]
        self.assertEqual(header["observer_sha256"], observe.self_sha256())
        self.assertEqual(sorted(header["down_proj_modules"]), ["0", "1", "2"])
        for row in rec["rows"]:
            self.assertEqual(list(row["layers"]), [str(LAYER)])
            phases = row["layers"][str(LAYER)]
            self.assertEqual(list(phases), ["decode"], "prefill must be unchanged")
            d = phases["decode"]
            self.assertTrue(d["sparse"])
            self.assertEqual(list(d["channels"]), [str(N)])
            dmin, dmax, _, _ = d["channels"][str(N)]
            self.assertGreater(dmin, 0)
            self.assertLess(abs(dmax - DOSE_VALUE) / DOSE_VALUE, 0.05)
            self.assertEqual(d["changed_calls"], d["calls"])
            self.assertEqual(d["calls"], 5, "six generated tokens = one prefill + five decode calls")
            w = model.model.layers[LAYER].mlp.down_proj.weight.detach().float()[:, N]
            expected = DOSE_VALUE * float(torch.linalg.vector_norm(w))
            self.assertLess(abs(d["write_norm"][0] - expected) / expected, 0.05)
        self.assertEqual(rec["effective"]["do_sample"], False)
        self.assertEqual(rec["effective"]["max_new_tokens"], 6)
        self.assertEqual(rec["hooks_at_entry"]["modules"], {f"model.layers.{LAYER}.mlp.down_proj": {"pre": 1, "post": 0}})

    def test_answer_and_tokens_are_bound_to_the_generated_ids(self):
        rec, out, _ = self.run_case()
        tok = TinyTokenizer()
        for row, entry in enumerate(rec["rows"]):
            ids = out[row, 5:].tolist()
            if 2 in ids:
                ids = ids[: ids.index(2) + 1]
            text = tok.decode(ids, skip_special_tokens=True).strip()
            self.assertEqual(entry["answer_sha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(entry["token_ids_sha256"],
                             hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(list(rec["prompt_texts"].values()), ["t1 t7 t9 t11 t13"])

    def test_observing_does_not_change_generation(self):
        model = tiny_model()
        width = model.model.layers[0].mlp.down_proj.in_features
        for sample in (False, True):
            kwargs = {"input_ids": prompt(), "max_new_tokens": 6, "do_sample": sample, "min_new_tokens": 6}
            if sample:
                kwargs.update(temperature=1.0, top_p=1.0)
            results = []
            for use_obs in (False, True):
                obs = observe.install(model, TinyTokenizer(), self.path) if use_obs else None
                push = Push(model.model.layers[LAYER].mlp.down_proj, one_hot(width, N, DOSE_VALUE))
                torch.manual_seed(7)
                with torch.inference_mode():
                    results.append(model.generate(**kwargs))
                push.remove()
                if obs:
                    obs.close()
            self.assertTrue(torch.equal(results[0], results[1]), f"observer changed output (do_sample={sample})")

    def test_negative_push_is_visible(self):
        rec, _, _ = self.run_case(vector=one_hot(64, N, -DOSE_VALUE))
        dmin, dmax, _, _ = rec["rows"][0]["layers"][str(LAYER)]["decode"]["channels"][str(N)]
        self.assertLess(dmax, 0)

    def test_wrong_layer_is_visible(self):
        rec, _, _ = self.run_case(layer=0)
        self.assertEqual(list(rec["rows"][0]["layers"]), ["0"])
        self.assertIn("model.layers.0.mlp.down_proj", rec["hooks_at_entry"]["modules"])

    def test_first_decode_only_is_visible(self):
        rec, _, _ = self.run_case(first_decode_only=True)
        d = rec["rows"][0]["layers"][str(LAYER)]["decode"]
        self.assertEqual(d["changed_calls"], 1)
        self.assertEqual(d["calls"], 5)

    def test_push_on_prompt_is_visible(self):
        rec, _, _ = self.run_case(decode_only=False)
        self.assertIn("prefill", rec["rows"][0]["layers"][str(LAYER)])

    def test_inplace_collateral_change_is_visible(self):
        rec, _, _ = self.run_case(inplace_extra=0)
        self.assertEqual(sorted(rec["rows"][0]["layers"][str(LAYER)]["decode"]["channels"]), ["0", str(N)])

    def test_inplace_clamp_records_the_post_value(self):
        rec, _, _ = self.run_case(clamp=(N, 9.75))
        _, _, pmin, pmax = rec["rows"][0]["layers"][str(LAYER)]["decode"]["channels"][str(N)]
        self.assertEqual((pmin, pmax), (9.75, 9.75))

    def test_sampling_settings_that_arrive_are_recorded(self):
        rec, _, _ = self.run_case(gen_kwargs={"do_sample": True, "temperature": 0.7, "top_p": 0.9})
        self.assertEqual(rec["effective"]["do_sample"], True)
        self.assertAlmostEqual(rec["effective"]["temperature"], 0.7)
        self.assertAlmostEqual(rec["effective"]["top_p"], 0.9)

    def test_seed_at_entry_is_recorded(self):
        a, _, _ = self.run_case(seed=86500)
        self.tearDown(); self.setUp()
        b, _, _ = self.run_case(seed=86501)
        self.assertEqual(a["rng_at_entry"]["cpu_initial_seed"], 86500)
        self.assertEqual(b["rng_at_entry"]["cpu_initial_seed"], 86501)
        torch.manual_seed(86500)
        fresh = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        self.assertEqual(a["rng_at_entry"]["cpu_state_sha256"], fresh, "no draws between seeding and generate")

    def test_draw_between_seed_and_generate_is_visible(self):
        model = tiny_model()
        obs = observe.install(model, TinyTokenizer(), self.path)
        torch.manual_seed(86500)
        torch.rand(1)
        with torch.inference_mode():
            model.generate(input_ids=prompt(), max_new_tokens=2, do_sample=False)
        obs.close()
        rec = [r for r in observe.read(self.path) if r.get("event") == "generate"][0]
        torch.manual_seed(86500)
        fresh = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        self.assertEqual(rec["rng_at_entry"]["cpu_initial_seed"], 86500)
        self.assertNotEqual(rec["rng_at_entry"]["cpu_state_sha256"], fresh)

    def test_extra_hook_anywhere_is_visible(self):
        rec, _, _ = self.run_case(extra_hook=True)
        self.assertIn("model.layers.0.self_attn", rec["hooks_at_entry"]["modules"])

    def test_dense_direction_is_identified_by_its_probes(self):
        width = 64
        g = torch.Generator().manual_seed(86022)
        raw = torch.randn((2, width), generator=g)
        raw = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        rec, _, _ = self.run_case(vector=raw * 2.0)
        d = rec["rows"][0]["layers"][str(LAYER)]["decode"]
        self.assertFalse(d["sparse"])
        probes = observe.probe_matrix(width)

        def cos(a, b):
            a, b = torch.tensor(a), torch.tensor(b)
            return float(a @ b / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))

        right = (raw[0] @ probes.T).tolist()
        g2 = torch.Generator().manual_seed(86023)
        wrong_raw = torch.randn((2, width), generator=g2)
        wrong = (wrong_raw[0] @ probes.T).tolist()
        self.assertGreater(cos(d["probe_first"], right), 0.98)
        self.assertGreater(cos(d["probe_last"], right), 0.98)
        self.assertLess(cos(d["probe_first"], wrong), 0.6)

    def test_rows_are_distinguished(self):
        width = 64
        vec = torch.zeros((2, width))
        vec[0, N] = DOSE_VALUE
        vec[1, N + 1] = DOSE_VALUE
        rec, _, _ = self.run_case(vector=vec)
        self.assertEqual(list(rec["rows"][0]["layers"][str(LAYER)]["decode"]["channels"]), [str(N)])
        self.assertEqual(list(rec["rows"][1]["layers"][str(LAYER)]["decode"]["channels"]), [str(N + 1)])

    def test_hook_registered_before_the_observer_is_still_measured(self):
        model = tiny_model()
        width = model.model.layers[0].mlp.down_proj.in_features
        push = Push(model.model.layers[LAYER].mlp.down_proj, one_hot(width, N, DOSE_VALUE))
        obs = observe.install(model, TinyTokenizer(), self.path)
        with torch.inference_mode():
            model.generate(input_ids=prompt(), max_new_tokens=4, min_new_tokens=4, do_sample=False)
        push.remove()
        obs.close()
        rec = [r for r in observe.read(self.path) if r.get("event") == "generate"][0]
        self.assertEqual(list(rec["rows"][0]["layers"][str(LAYER)]["decode"]["channels"]), [str(N)])

    def test_answer_is_trimmed_at_the_first_eos(self):
        model = tiny_model()
        obs = observe.install(model, TinyTokenizer(), self.path)
        obs.active = {"acc": {}, "phase_counts": {}, "pre_calls": {}, "forward_calls": {}, "shape_changes": []}
        record = {"call_id": "x:1", "batch": 2, "prompt_len": 2, "effective": {"eos_token_id": 2}}
        sequences = torch.tensor([[1, 7, 9, 2, 0, 0], [1, 7, 9, 11, 13, 15]])
        obs._finish(record, ["a", "b"], sequences)
        obs.close()
        rec = [r for r in observe.read(self.path) if r.get("call_id") == "x:1"][0]
        self.assertEqual(rec["rows"][0]["token_count"], 2)
        self.assertEqual(rec["rows"][0]["generated_steps"], 4)
        self.assertEqual(rec["rows"][0]["answer_sha256"], hashlib.sha256(b"t9").hexdigest())
        self.assertEqual(rec["rows"][1]["token_count"], 4)

    def test_direction_changing_mid_answer_shows_in_first_and_last_probes(self):
        model = tiny_model()
        width = model.model.layers[0].mlp.down_proj.in_features
        a = one_hot(width, N, 0.0); a[:, :] = 0
        g = torch.Generator().manual_seed(1)
        da = torch.randn((2, width), generator=g); db = torch.randn((2, width), generator=g)
        push = Push(model.model.layers[LAYER].mlp.down_proj, da)
        original = push.hook

        def switching(module, args):
            if push.decodes >= 1:
                push.vector = db
            return original(module, args)

        push.handle.remove()
        push.handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(switching)
        obs = observe.install(model, TinyTokenizer(), self.path)
        with torch.inference_mode():
            model.generate(input_ids=prompt(), max_new_tokens=4, min_new_tokens=4, do_sample=False)
        push.remove()
        obs.close()
        rec = [r for r in observe.read(self.path) if r.get("event") == "generate"][0]
        d = rec["rows"][0]["layers"][str(LAYER)]["decode"]
        probes = observe.probe_matrix(width)

        def cos(x, y):
            x, y = torch.tensor(x), torch.tensor(y)
            return float(x @ y / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)))

        self.assertGreater(cos(d["probe_first"], (da[0] @ probes.T).tolist()), 0.98)
        self.assertGreater(cos(d["probe_last"], (db[0] @ probes.T).tolist()), 0.98)

    def test_channel_range_is_merged_across_calls(self):
        model = tiny_model()
        width = model.model.layers[0].mlp.down_proj.in_features
        push = Push(model.model.layers[LAYER].mlp.down_proj, one_hot(width, N, 1.0))
        original = push.hook

        def growing(module, args):
            push.vector = one_hot(width, N, float(push.decodes + 1))
            return original(module, args)

        push.handle.remove()
        push.handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(growing)
        obs = observe.install(model, TinyTokenizer(), self.path)
        with torch.inference_mode():
            model.generate(input_ids=prompt(), max_new_tokens=4, min_new_tokens=4, do_sample=False)
        push.remove()
        obs.close()
        rec = [r for r in observe.read(self.path) if r.get("event") == "generate"][0]
        dmin, dmax, _, _ = rec["rows"][0]["layers"][str(LAYER)]["decode"]["channels"][str(N)]
        self.assertLess(abs(dmin - 1.0), 0.05)
        self.assertLess(abs(dmax - 3.0), 0.1)

    def test_a_replaced_forward_is_reported_unobserved(self):
        model = tiny_model()
        width = model.model.layers[0].mlp.down_proj.in_features
        obs = observe.install(model, TinyTokenizer(), self.path)
        down = model.model.layers[LAYER].mlp.down_proj
        down.forward = torch.nn.Linear.forward.__get__(down)
        push = Push(down, one_hot(width, N, DOSE_VALUE))
        with torch.inference_mode():
            model.generate(input_ids=prompt(), max_new_tokens=3, min_new_tokens=3, do_sample=False)
        push.remove()
        del down.forward
        obs.close()
        rec = [r for r in observe.read(self.path) if r.get("event") == "generate"][0]
        self.assertEqual(rec["unobserved_forwards"], {str(LAYER): 3})
        self.assertEqual(rec["rows"][0]["layers"], {})

    def test_unhooked_layers_report_calls_but_no_change(self):
        rec, _, _ = self.run_case()
        self.assertEqual(rec["unobserved_forwards"], {})
        self.assertEqual(sorted(rec["phase_counts"]), ["0", "1", "2"])
        self.assertEqual(rec["phase_counts"]["0"]["decode"][0], 5)

    def test_close_restores_the_model(self):
        model = tiny_model()
        before = model.generate
        obs = observe.install(model, TinyTokenizer(), self.path)
        obs.close()
        self.assertEqual(model.generate, before)
        for layer in model.model.layers:
            self.assertEqual(len(layer.mlp.down_proj._forward_pre_hooks), 0)
            self.assertNotIn("forward", vars(layer.mlp.down_proj))


if __name__ == "__main__":
    unittest.main(verbosity=1)
