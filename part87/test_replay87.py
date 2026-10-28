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


def forced_pad_then_eos(model, tok):
    """Logits that emit pad then eos: both decode to nothing, so text cannot tell the two stop rules apart."""
    import functools
    import types
    original = model.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        out = original(*args, **kwargs)
        ids = kwargs.get("input_ids", args[0] if args else None)
        target = tok.eos_token_id if int(ids[0, -1]) == tok.pad_token_id else tok.pad_token_id
        out.logits.fill_(-100.0)
        out.logits[..., target] = 100.0
        return out
    model.forward = types.MethodType(forward, model)
    return model


class Round3(unittest.TestCase):
    """Sol's round-3 counterexamples (part87-after-r3-2026-10-28)."""

    def observed_run(self, eos):
        import json
        import tempfile
        import observe
        tok = TinyTok()
        model = forced_pad_then_eos(tiny_model(), tok)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            obs = observe.install(model, tok, path, label="round 3")
            enc = tok(r.render(tok, "who made you?"))
            torch.manual_seed(4242)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=r.B_TOKENS, do_sample=True, temperature=r.TEMPERATURE,
                                     top_p=r.TOP_P, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)
            obs.close()
            calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        call = [c for c in calls if c.get("event") == "generate"][-1]
        text = tok.decode(out[0, enc["input_ids"].shape[-1]:]).strip()
        return call, text

    def replayed(self):
        tok = TinyTok()
        model = forced_pad_then_eos(tiny_model(), tok)
        return r.regenerate(model, tok, r.text_layers(model), "cpu", "87B", "who made you?", 4242, [], [])

    def test_a_changed_stop_rule_with_the_same_text_is_caught(self):
        tok = TinyTok()
        good_call, good_text = self.observed_run(tok.eos_token_id)
        bad_call, bad_text = self.observed_run([tok.eos_token_id, tok.pad_token_id])
        mine_text, _, mine_ids = self.replayed()
        self.assertEqual(good_text, bad_text)                       # the old text-only comparison passes both
        self.assertEqual(mine_text, bad_text)
        self.assertEqual((good_call["rows"][0]["token_ids_sha256"], good_call["rows"][0]["token_count"]),
                         (r.ids_sha(mine_ids), len(mine_ids)))
        self.assertNotEqual(bad_call["rows"][0]["token_ids_sha256"], r.ids_sha(mine_ids))
        self.assertNotEqual(bad_call["rows"][0]["token_count"], len(mine_ids))
        good_row = {"answer": good_text, "observed": {"call_id": good_call["call_id"], "row": 0}}
        bad_row = {"answer": bad_text, "observed": {"call_id": bad_call["call_id"], "row": 0}}
        self.assertTrue(r.row_matches(good_row, good_call, mine_text, mine_ids)[0])
        same, detail = r.row_matches(bad_row, bad_call, mine_text, mine_ids)
        self.assertFalse(same)
        self.assertEqual((detail["text_match"], detail["tokens_match"]), (True, False))
        self.assertFalse(r.row_matches(dict(good_row, answer=good_text + "x"), good_call, mine_text, mine_ids)[0])
        self.assertEqual(r.stop_setting_problems(good_call["effective"], "87B", tok), [])
        self.assertTrue(r.stop_setting_problems(bad_call["effective"], "87B", tok))

    def test_87a_stop_set_and_pad_are_checked(self):
        tok = TinyTok()
        want = r.stop_tokens(tok, "87A")
        self.assertEqual(r.stop_setting_problems({"eos_token_id": list(want), "pad_token_id": tok.pad_token_id},
                                                 "87A", tok), [])
        self.assertTrue(r.stop_setting_problems({"eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id},
                                                "87A", tok))
        self.assertTrue(r.stop_setting_problems({"eos_token_id": list(want), "pad_token_id": 0}, "87A", tok))

    def test_a_substituted_source_file_is_refused(self):
        import json
        path = HERE.parent / "results" / "part66" / "full" / "answers.jsonl"
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        held = [x for x in rows if x.get("condition") == "unchanged" and x.get("question_kind") in ("direct", "indirect")]
        self.assertEqual(r.source_problems(raw, held), [])
        swapped = [dict(x, answer="I was made by Google.") for x in held]
        fake = "".join(json.dumps(x) + "\n" for x in swapped).encode("utf-8")
        self.assertTrue(any("sha256" in p for p in r.source_problems(fake, swapped)))
        self.assertTrue(any("8 samples" in p for p in r.source_problems(raw, held[:-1])))
        relabelled = [dict(held[0], question="Who made you?")] + held[1:]
        self.assertTrue(any("registered phrasing" in p for p in r.source_problems(raw, relabelled)))

    def test_fraction_below_a_quarter_or_above_one_is_refused(self):
        for bad in ("0", "0.2", "1.5"):
            with self.assertRaises(SystemExit):
                r.main(["--run-dir", "x", "--phase", "pilot", "--calib", "x", "--p66-answers", "x", "--out", "x",
                        "--fraction", bad])

    def test_trim_matches_the_observer_rule(self):
        self.assertEqual(r.trim_like_observer([5, 62, 63, 9], {63}), [5, 62, 63])
        self.assertEqual(r.trim_like_observer([5, 62, 63, 9], {62, 63}), [5, 62])
        self.assertEqual(r.trim_like_observer([5, 6], {63}), [5, 6])


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
