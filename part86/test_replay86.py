#!/usr/bin/env python3
"""CPU tests for replay86.py's verification mode (Claude, 2026-10-28, Amendment 4).

The decisive controls run on the A770 against real verification output. These check the replay's own parts with a
tiny float32 model: its generated-token push must reproduce sampled answers made by a different mechanism (an output
bias W_down @ v on decode steps), and a wrong direction, size or seed must change them.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part87"))

import torch  # noqa: E402
import replay86 as r  # noqa: E402
from test_replay87 import TinyTok, tiny_model  # noqa: E402


def alt_answer(model, tok, layer_no, push, seed, question):
    """The 'run': add W_down @ push to the down_proj output on every decode step, after the prompt pass."""
    down = r.text_layers(model)[layer_no].mlp.down_proj
    bias = push @ down.weight.detach().T
    state = {"first": True}

    def hook(_m, _a, out):
        first, state["first"] = state["first"], False
        return out if first else out + bias
    handle = down.register_forward_hook(hook)
    try:
        prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt)["input_ids"]
        torch.manual_seed(seed)
        stops = [tok.eos_token_id]
        with torch.inference_mode():
            out = model.generate(input_ids=ids, max_new_tokens=r.VERIFY_TOKENS, do_sample=True,
                                 temperature=r.TEMPERATURE, top_p=r.TOP_P, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
    finally:
        handle.remove()
    gen = out[0, ids.shape[-1]:].tolist()
    if tok.eos_token_id in gen:
        gen = gen[:gen.index(tok.eos_token_id) + 1]
    return tok.decode(gen, skip_special_tokens=True).strip()


class VerifyReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = tiny_model(), TinyTok()
        cls.layer_no = 1
        cls.down = r.text_layers(cls.model)[cls.layer_no].mlp.down_proj
        cls.w = cls.down.weight.detach().float()
        cls.dirs = r.scaled_directions(cls.w, r.VERIFY_DIRECTIONS, r.VERIFY_RANDOM_SEED + 10000 * 1 + 5, 6.0)

    def mine(self, push, seed, q=0):
        return r.regenerate_verify(self.model, self.tok, self.down, push, seed, r.VERIFY_QUESTIONS[q], "cpu",
                                   [self.tok.eos_token_id])

    def test_directions_follow_the_registered_rule(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(r.VERIFY_RANDOM_SEED + 10000 * 1 + 5)
        raw = torch.randn((10, int(self.w.shape[1])), generator=g, dtype=torch.float32)
        raw = raw / raw.norm(dim=1, keepdim=True)
        for v, d in zip(raw, self.dirs):
            self.assertLess(float((d / d.norm() - v).abs().max()), 1e-6)
            self.assertAlmostEqual(float((d @ self.w.T).norm()), 6.0, places=4)

    def test_random_and_neuron_pushes_match_the_output_bias(self):
        neuron = torch.zeros(int(self.w.shape[1]))
        neuron[5] = 6.0 / float(self.w[:, 5].norm())
        for push in (self.dirs[0], self.dirs[7], neuron):
            for q, s in ((0, 0), (1, 3), (2, 7)):
                seed = r.VERIFY_SEED + 100 * q + s
                self.assertEqual(self.mine(push, seed, q), alt_answer(self.model, self.tok, 1, push, seed,
                                                                      r.VERIFY_QUESTIONS[q]))

    def test_wrong_direction_size_or_seed_changes_answers(self):
        seeds = [r.VERIFY_SEED + 100 * q + s for q in range(3) for s in range(3)]
        run = [alt_answer(self.model, self.tok, 1, self.dirs[2], sd, r.VERIFY_QUESTIONS[0]) for sd in seeds]
        self.assertEqual([self.mine(self.dirs[2], sd) for sd in seeds], run)
        self.assertNotEqual([self.mine(self.dirs[3], sd) for sd in seeds], run)
        self.assertNotEqual([self.mine(self.dirs[2] * 1.5, sd) for sd in seeds], run)
        self.assertNotEqual([self.mine(self.dirs[2], sd + 1) for sd in seeds], run)
        self.assertNotEqual([self.mine(torch.zeros_like(self.dirs[2]), sd) for sd in seeds], run)

    def test_replay_verify_end_to_end(self):
        import json
        import tempfile
        from types import SimpleNamespace
        calib = "the quick brown fox jumps over the lazy dog. " * 4
        layers = r.text_layers(self.model)
        median = r.my_median(self.model, self.tok, layers[1], calib, "cpu")
        target = r.DOSE * median
        dirs = r.scaled_directions(self.w, r.VERIFY_DIRECTIONS, r.VERIFY_RANDOM_SEED + 10000 * 1 + 5, target)
        neuron = torch.zeros(int(self.w.shape[1]))
        neuron[5] = target / float(self.w[:, 5].norm())
        records = []
        for cond, d, q, s in sorted(r.verify_cases(), key=str):
            push = neuron if cond == "neuron" else dirs[d]
            seed = r.VERIFY_SEED + 100 * q + s
            ans = alt_answer(self.model, self.tok, 1, push, seed, r.VERIFY_QUESTIONS[q])
            records.append({"condition": cond, "direction": d, "question_index": q, "sample": s, "seed": seed,
                            "question": r.VERIFY_QUESTIONS[q], "answer": ans, "median": median, "dose": r.DOSE})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "L1_N5.json").write_text(json.dumps({"status": "complete", "records": records}),
                                                  encoding="utf-8")
            args = SimpleNamespace(verify_dir=tmp, verify_picks=1, device="cpu", fraction=0.25, perturb_median=1.0)
            res = r.replay_verify(self.model, self.tok, layers, args, calib)
            self.assertEqual((res["replayed"], res["mismatches"], res["problems"]), (66, 0, []))
            args.perturb_median = 1.01
            self.assertGreater(r.replay_verify(self.model, self.tok, layers, args, calib)["mismatches"], 0)
            args.perturb_median = 1.0
            records[3]["seed"] += 1
            (Path(tmp) / "L1_N5.json").write_text(json.dumps({"status": "complete", "records": records}),
                                                  encoding="utf-8")
            self.assertTrue(any("seed" in p for p in r.replay_verify(self.model, self.tok, layers, args,
                                                                      calib)["problems"]))

    def test_cases_and_sample_size(self):
        cases = r.verify_cases()
        self.assertEqual(len(cases), 24 + 10 * 24)
        records = [{"condition": c, "direction": d, "question_index": q, "sample": s} for c, d, q, s in cases]
        picked = r.pick_verify(records, 22, 2742, 0.25)
        self.assertEqual(len(picked), 66)
        self.assertEqual((picked[0]["condition"], picked[0]["question_index"], picked[0]["sample"]), ("neuron", 0, 0))
        self.assertEqual((picked[1]["direction"], picked[1]["question_index"], picked[1]["sample"]), (9, 2, 7))
        self.assertEqual(picked, r.pick_verify(list(reversed(records)), 22, 2742, 0.25))


if __name__ == "__main__":
    unittest.main(verbosity=1)
