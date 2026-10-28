#!/usr/bin/env python3
"""CPU tests for replay87.py (Claude, 2026-10-28).

The decisive controls run on the GPU pilot (replay87 against part87's real output, then with --perturb-dose). These
check the replay's own parts with a tiny float32 model: its pushes must reproduce answers produced by a different
mechanism (a bias on the down_proj OUTPUT, or a zeroed weight column, instead of an edit of its input), and a wrong
dose, mode, unit or seed must change them.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import replay87 as r  # noqa: E402


class TinyTok:
    """Characters in a 64-symbol alphabet; enough for render/generate/decode."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ?."
    eos_token_id, pad_token_id = 63, 62

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "U " + msgs[0]["content"] + " A "

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = torch.tensor([[self.alphabet.index(c) % 62 for c in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        if skip_special_tokens:
            ids = [i for i in ids if i < 62]
        return "".join(self.alphabet[i] for i in ids)


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).float().eval()
    with torch.no_grad():                                   # sharpen the logits so samples depend on the pushes
        model.lm_head.weight.mul_(6.0)
    model.generation_config.top_k = None
    return model


def alt_generate(model, tok, question, seed, study, bias=None, decode_only=False, zero=None):
    """The 'run': an output-side bias (value x W[:, n]) or a zeroed weight column."""
    layers = r.text_layers(model)
    handles, saved = [], None
    if bias is not None:
        li, n, value = bias
        down = layers[li].mlp.down_proj
        b = value * down.weight[:, n].detach().clone()
        state = {"first": True}

        def hook(_m, _a, out):
            first, state["first"] = state["first"], False
            return out if (decode_only and first) else out + b
        handles.append(down.register_forward_hook(hook))
    if zero is not None:
        li, n = zero
        w = layers[li].mlp.down_proj.weight
        saved = w[:, n].detach().clone()
        with torch.no_grad():
            w[:, n] = 0
    try:
        torch.manual_seed(seed)
        enc = tok(r.render(tok, question))
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=r.A_TOKENS if study == "87A" else r.B_TOKENS, do_sample=True,
                                 temperature=r.TEMPERATURE, top_p=r.TOP_P, eos_token_id=r.stop_tokens(tok, study),
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(out[0, enc["input_ids"].shape[-1]:]).strip()
    finally:
        for h in handles:
            h.remove()
        if saved is not None:
            with torch.no_grad():
                w[:, n] = saved


class ReplayMechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = tiny_model(), TinyTok()
        cls.layers = r.text_layers(cls.model)
        cls.q = "who made you?"

    def regen(self, seed, study="87A", pushes=(), zeros=()):
        return r.regenerate(self.model, self.tok, self.layers, "cpu", study, self.q, seed, list(pushes), list(zeros))[0]

    def test_every_token_push_matches_output_bias(self):
        for seed in (84000, 84001, 84102):
            run = alt_generate(self.model, self.tok, self.q, seed, "87A", bias=(1, 5, 3.0))
            self.assertEqual(self.regen(seed, pushes=[(1, 5, 3.0, "every")]), run)

    def test_decode_push_matches_output_bias_after_the_prompt_pass(self):
        for seed in (84000, 84201):
            run = alt_generate(self.model, self.tok, self.q, seed, "87A", bias=(2, 9, 4.0), decode_only=True)
            self.assertEqual(self.regen(seed, pushes=[(2, 9, 4.0, "decode")]), run)

    def test_zero_matches_zeroed_weight_column(self):
        # Channel 7 of layer 0 gets a large write so zeroing it (and only it) moves the samples.
        layer0 = self.layers[0].mlp
        saved_up = layer0.up_proj.weight[7].detach().clone()
        saved_down = layer0.down_proj.weight[:, 7].detach().clone()
        with torch.no_grad():
            layer0.up_proj.weight[7] *= 8.0
            layer0.down_proj.weight[:, 7] *= 8.0
        try:
            seeds = range(12345, 12351)
            run = [alt_generate(self.model, self.tok, self.q, s, "87B", zero=(0, 7)) for s in seeds]
            self.assertEqual([self.regen(s, study="87B", zeros=[(0, 7)]) for s in seeds], run)
            self.assertNotEqual([self.regen(s, study="87B") for s in seeds], run, "zeroing channel 7 changes nothing")
            self.assertNotEqual([self.regen(s, study="87B", zeros=[(0, 8)]) for s in seeds], run)
        finally:
            with torch.no_grad():
                layer0.up_proj.weight[7] = saved_up
                layer0.down_proj.weight[:, 7] = saved_down

    def test_wrong_dose_mode_unit_or_seed_changes_answers(self):
        seeds = range(84000, 84008)
        run = [alt_generate(self.model, self.tok, self.q, s, "87A", bias=(1, 5, 3.0)) for s in seeds]
        for wrong in ([(1, 5, 3.0 * 1.5, "every")], [(1, 5, 3.0, "decode")], [(1, 6, 3.0, "every")],
                      [(1, 5, -3.0, "every")], []):
            got = [self.regen(s, pushes=wrong) for s in seeds]
            self.assertNotEqual(got, run, f"replay could not tell {wrong} from the run")
        self.assertNotEqual([self.regen(s + 1, pushes=[(1, 5, 3.0, "every")]) for s in seeds], run)

    def test_hooks_are_removed_after_each_call(self):
        self.regen(84000, pushes=[(1, 5, 3.0, "every")], zeros=[(0, 7)])
        for layer in self.layers:
            self.assertEqual(len(layer.mlp.down_proj._forward_pre_hooks), 0)


class ReplayRules(unittest.TestCase):
    def test_m_rule_cancels_the_mean_and_f_rule_is_a_fixed_site_push(self):
        mean, norm, med = 0.37, 2.5, 24.8
        self.assertAlmostEqual(r.additive(r.site_dose("M", mean, norm, med), med, norm), -mean, places=12)
        self.assertEqual(r.site_dose("F", mean, norm, med), -0.75)
        self.assertEqual(r.site_dose("F", -mean, norm, med), 0.75)
        self.assertAlmostEqual(r.additive(0.63, med, norm), 0.63 * med / norm)

    def test_design_sizes(self):
        arms = r.arm_table("run")
        total = sum(r.expected_rows("run", n, s)[1] for n, s in arms.items())
        self.assertEqual(total, 1560)                         # the design's "about 1,560 generations"
        self.assertEqual(len(arms), 5 + 4 + 40)
        pilot = r.arm_table("pilot")
        self.assertEqual(sum(r.expected_rows("pilot", n, s)[1] for n, s in pilot.items()), 5 * 4 + 4 * 6)

    def test_registered_questions_and_seeds(self):
        from part66 import part66 as p66
        self.assertEqual(r.registered_question("87A", 0), ("Who made you?", "origin"))
        self.assertEqual(r.registered_question("87B", 16), (p66.INDIRECT[0], "indirect"))
        self.assertEqual(r.registered_question("87B", 25), (p66.GUARDS[0][0], "guard"))
        with self.assertRaises(KeyError):
            r.registered_question("87B", 24)
        self.assertEqual(r.registered_seed("run", "D47", "87A", 2, 7), 84207)
        self.assertEqual(r.registered_seed("run", "M", "87B", 3, 1), p66.row_seed("run:M", 3, 1))
        self.assertNotEqual(r.registered_seed("run", "M", "87B", 3, 1), r.registered_seed("pilot", "M", "87B", 3, 1))

    def test_pick_rows_takes_first_last_and_a_quarter(self):
        rows = [{"question_index": q, "sample": s} for q in range(24) for s in range(4)]
        got = r.pick_rows(rows, "U", 0.25, False)
        self.assertEqual(len(got), 24)
        self.assertEqual((got[0]["question_index"], got[0]["sample"]), (0, 0))
        self.assertEqual((got[1]["question_index"], got[1]["sample"]), (23, 3))
        self.assertEqual(got, r.pick_rows(list(reversed(rows)), "U", 0.25, False))
        self.assertEqual(len(r.pick_rows(rows, "U", 0.25, True)), 96)
        self.assertEqual(len(r.pick_rows(rows[:4], "E63", 0.25, False)), 2)

    def test_build_arm_uses_its_own_numbers(self):
        med = {22: 24.8, 6: 10.0, 7: 11.0, 11: 12.0, 29: 30.0}
        norms = {(22, 2742): 2.0, (22, 8870): 4.0, **{t: 1.0 for t in r.TARGETS}}
        means = {li: [0.0] * 9216 for li in r.MAKER_LAYERS}
        means[6][753] = 0.5
        pushes, zeros = r.build_arm(r.A_ARMS["D47"], med, norms, means, {}, 1.0)
        self.assertEqual(pushes, [(22, 2742, 0.47 * 24.8 / 2.0, "decode")])
        pushes, _ = r.build_arm(r.B_MAIN["M"], med, norms, means, {}, 1.0)
        self.assertAlmostEqual(pushes[0][2], -0.5)
        self.assertEqual(pushes[1][2], 0.0)
        _, zeros = r.build_arm(r.B_MAIN["Z"], med, norms, means, {}, 1.0)
        self.assertEqual(zeros, list(r.TARGETS))
        pushes, _ = r.build_arm(r.A_ARMS["E63"], med, norms, means, {}, 1.01)
        self.assertAlmostEqual(pushes[0][2], 0.63 * 24.8 / 2.0 * 1.01)

    def test_close_tolerance(self):
        self.assertTrue(r.close(1.0 + 5e-5, 1.0))
        self.assertFalse(r.close(1.0 + 2e-4, 1.0))
        self.assertTrue(r.close(5e-8, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=1)
