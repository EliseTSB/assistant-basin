#!/usr/bin/env python3
"""CPU tests for replay86.py's verification mode (Claude, 2026-10-28; Amendments 4 and 5).

The decisive controls run on the A770 against real verification output. These build a real verification folder on a
tiny float32 model under ../observe.py, with the answers made by a different mechanism (an output bias W_down @ v on
decode steps), and check that the replay passes it, regenerating every answer, and refuses each of round 6's survivors
(scratch/review-part86-r6-sol in Journal): a one-answer generate processor outside the old sample, a persona relabel,
malformed records with lying fields, duplicate pick files, tampered token counts, and a changed stop rule.
"""
from __future__ import annotations

import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import observe  # noqa: E402
import replay86 as r  # noqa: E402

CALIB = "the quick brown fox jumps over the lazy dog. " * 4
PICKS = [(1, 5, "a calm traveler"), (2, 9, "a stormy sailor")]


class TinyTok:
    """Characters in a small alphabet; enough for render/generate/decode and the observer."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    eos_token_id, pad_token_id = 63, 62

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "U " + msgs[0]["content"] + " A "

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = torch.tensor([[self.alphabet.find(ch) % 62 if ch in self.alphabet else 61 for ch in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        if skip_special_tokens:
            ids = [i for i in ids if i < 62]
        return "".join(self.alphabet[i] if i < 62 else "<s>" for i in ids)


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(6.0)
    model.generation_config.top_k = None
    return model


def run_case(model, tok, layer_no, push, seed, question, stops, extra=None):
    """The 'run': add W_down @ push to the down_proj output on every decode step, after the prompt pass."""
    down = r.text_layers(model)[layer_no].mlp.down_proj
    bias = push @ down.weight.detach().T
    state = {"first": True}

    def hook(_m, _a, out):
        first, state["first"] = state["first"], False
        return out if first else out + bias
    handle = down.register_forward_hook(hook)
    try:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False))["input_ids"]
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, max_new_tokens=r.VERIFY_TOKENS, do_sample=True,
                                 temperature=r.TEMPERATURE, top_p=r.TOP_P, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id, use_cache=True, **(extra or {}))
    finally:
        handle.remove()
    gen = out[0, ids.shape[-1]:].tolist()
    for pos, t in enumerate(gen):
        if t in stops:
            gen = gen[:pos + 1]
            break
    return tok.decode(gen, skip_special_tokens=True).strip(), gen


def build_verify_dir(root: Path, model, tok, special=None):
    """A verification folder: picked file, one L<layer>_N<neuron>.json per pick, observer records under observe/.
    special(case, pick) may return (extra generate kwargs, stops) for one case."""
    layers = r.text_layers(model)
    picked = root / "picked.json"
    picked.write_text(json.dumps([{"layer": l, "neuron": n, "persona": p} for l, n, p in PICKS]), encoding="utf-8")
    picked_sha = r.load_picked(picked)[1]
    stops = r.registered_stops(model, tok)
    obs = observe.install(model, tok, root / "observe" / "run.jsonl", label="test 86 verify")
    try:
        for layer_no, neuron, persona in PICKS:
            median = r.my_median(model, tok, layers[layer_no], CALIB, "cpu")
            w = layers[layer_no].mlp.down_proj.weight.detach().float()
            target = r.DOSE * median
            dirs = r.scaled_directions(w, r.VERIFY_DIRECTIONS, r.VERIFY_RANDOM_SEED + 10000 * layer_no + neuron, target)
            npush = torch.zeros(int(w.shape[1]))
            npush[neuron] = target / float(w[:, neuron].norm())
            records = []
            for cond, d, q, s in sorted(r.verify_cases(), key=str):
                seed = r.VERIFY_SEED + 100 * q + s
                extra, case_stops = (special((cond, d, q, s), (layer_no, neuron)) if special else None) or ({}, stops)
                answer, ids = run_case(model, tok, layer_no, npush if cond == "neuron" else dirs[d], seed,
                                       r.VERIFY_QUESTIONS[q], case_stops, extra)
                records.append({"layer": layer_no, "condition": cond, "neuron": neuron, "direction": d,
                                "persona": persona, "question": r.VERIFY_QUESTIONS[q], "question_index": q,
                                "sample": s, "seed": seed, "answer": answer, "token_count": len(ids),
                                "observed": {"call_id": obs.last_call_id, "row": 0}, "median": median, "dose": r.DOSE})
            config = {"layer": layer_no, "neuron": neuron, "persona": persona, "selected_set_sha256": picked_sha}
            (root / f"L{layer_no}_N{neuron}.json").write_text(
                json.dumps({"status": "complete", "config": config, "records": records}), encoding="utf-8")
    finally:
        obs.close()
    return picked


class VerifyReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = tiny_model(), TinyTok()
        cls.layers = r.text_layers(cls.model)
        cls.base = Path(tempfile.mkdtemp())
        cls.picked = build_verify_dir(cls.base, cls.model, cls.tok)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def replay(self, root: Path, fraction=1.0, perturb=1.0, picks=2, picked=None):
        args = SimpleNamespace(verify_dir=str(root), picked=str(picked or root / "picked.json"), verify_picks=picks,
                               device="cpu", fraction=fraction, perturb_median=perturb)
        return r.replay_verify(self.model, self.tok, self.layers, args, CALIB)

    def copy(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(self.base, tmp, dirs_exist_ok=True)
        return tmp

    def edit(self, root: Path, name: str, fn):
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        fn(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_the_registered_folder_passes_with_every_answer_replayed(self):
        res = self.replay(self.base)
        self.assertEqual((res["replayed"], res["mismatches"], res["problems"]), (528, 0, []))

    def test_negative_control_changes_answers(self):
        self.assertGreater(self.replay(self.base, fraction=0.1, perturb=1.01)["mismatches"], 0)

    def test_one_answer_processor_outside_the_old_sample_is_caught(self):
        records = json.loads((self.base / "L1_N5.json").read_text(encoding="utf-8"))["records"]
        old_sample = {(x["condition"], x["direction"], x["question_index"], x["sample"])
                      for x in r.pick_verify(records, 1, 5, 0.25)}
        target = next(x for x in records if x["condition"] == "neuron" and
                      ("neuron", None, x["question_index"], x["sample"]) not in old_sample)
        case = ("neuron", None, target["question_index"], target["sample"])
        seed = r.VERIFY_SEED + 100 * case[2] + case[3]
        layer_no, neuron = 1, 5
        w = self.layers[1].mlp.down_proj.weight.detach().float()
        median = r.my_median(self.model, self.tok, self.layers[1], CALIB, "cpu")
        npush = torch.zeros(int(w.shape[1]))
        npush[neuron] = r.DOSE * median / float(w[:, neuron].norm())
        _, ids = run_case(self.model, self.tok, 1, npush, seed, r.VERIFY_QUESTIONS[case[2]],
                          r.registered_stops(self.model, self.tok))
        banned = {"bad_words_ids": [[ids[0]]]}
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        build_verify_dir(root, self.model, self.tok,
                         special=lambda c, p: (banned, r.registered_stops(self.model, self.tok))
                         if c == case and p == (layer_no, neuron) else None)
        self.assertEqual(self.replay(root, fraction=0.25)["mismatches"], 0, "the old quarter sample sees nothing")
        res = self.replay(root)
        self.assertEqual(res["mismatches"], 1)
        whys = [w for pick in res["picks"] for m in pick["mismatched"] for w in m["why"]]
        self.assertIn("tokens", whys)

    def test_persona_relabel_is_refused(self):
        root = self.copy()

        def relabel(payload):
            payload["config"]["persona"] = "a false replacement"
            for rec in payload["records"]:
                rec["persona"] = "a false replacement"
        self.edit(root, "L1_N5.json", relabel)
        res = self.replay(root, fraction=0.01)
        self.assertEqual(res["mismatches"], 0)
        self.assertTrue(any("picked file" in p and "persona" in p for p in res["problems"]), res["problems"])

    def test_malformed_record_with_lying_fields_is_refused(self):
        root = self.copy()

        def corrupt(payload):
            rec = next(x for x in payload["records"] if x["condition"] == "neuron" and x["question_index"] == 0
                       and x["sample"] == 0)
            rec.update({"question_index": False, "median": -10, "dose": -0.8, "seed": 123, "question": "Who made you?"})
        self.edit(root, "L1_N5.json", corrupt)
        problems = self.replay(root, fraction=0.01)["problems"]
        self.assertTrue(any("malformed case" in p for p in problems), problems)
        self.assertTrue(any("median -10" in p for p in problems), problems)

    def test_duplicate_or_misnamed_pick_files_are_refused(self):
        root = self.copy()
        shutil.copy(root / "L1_N5.json", root / "L01_N5.json")
        (root / "L2_N9.json").unlink()
        problems = self.replay(root, fraction=0.01)["problems"]
        self.assertTrue(any("canonical" in p for p in problems), problems)
        self.assertTrue(any("no verification file" in p for p in problems), problems)

    def test_tampered_token_counts_are_caught(self):
        root = self.copy()
        self.edit(root, "L2_N9.json", lambda payload: [rec.update({"token_count": 1}) for rec in payload["records"]])
        res = self.replay(root, fraction=0.05)
        self.assertGreater(res["mismatches"], 10)

    def test_a_changed_stop_rule_is_caught(self):
        stops = r.registered_stops(self.model, self.tok)
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        build_verify_dir(root, self.model, self.tok,
                         special=lambda c, p: ({}, stops + [self.tok.pad_token_id]) if c[2:] == (1, 2) else None)
        res = self.replay(root)
        whys = [w for pick in res["picks"] for m in pick["mismatched"] for w in m["why"]]
        self.assertTrue(any(w.startswith("stop tokens") for w in whys), whys[:5])

    def test_wrong_selected_digest_is_refused(self):
        root = self.copy()
        self.edit(root, "L2_N9.json", lambda payload: payload["config"].update({"selected_set_sha256": "c" * 64}))
        self.assertTrue(any("selected_set_sha256" in p for p in self.replay(root, fraction=0.01)["problems"]))

    def test_main_prints_pass_only_for_the_whole_population(self):
        model = self.model
        model.config._commit_hash = r.REVISION
        out = self.base / "replay.json"
        argv = ["--verify-dir", str(self.base), "--picked", str(self.picked), "--calib", str(self.base / "calib.txt"),
                "--device", "cpu", "--out", str(out), "--verify-picks", "2"]
        (self.base / "calib.txt").write_text(CALIB, encoding="utf-8")
        with patch("neuron_report.load_model", return_value=(self.tok, model)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = r.main(argv)
            self.assertEqual(code, 0, buf.getvalue())
            self.assertIn("REPLAY86 verify: PASS (all 528 answers match", buf.getvalue())
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = r.main(argv + ["--fraction", "0.25"])
            self.assertEqual(code, 3, buf.getvalue())
            self.assertNotIn(": PASS", buf.getvalue())
            self.assertIn("NON-PRODUCTION SAMPLE", buf.getvalue())

    def test_directions_follow_the_registered_rule(self):
        w = self.layers[1].mlp.down_proj.weight.detach().float()
        dirs = r.scaled_directions(w, r.VERIFY_DIRECTIONS, r.VERIFY_RANDOM_SEED + 10000 + 5, 6.0)
        g = torch.Generator(device="cpu")
        g.manual_seed(r.VERIFY_RANDOM_SEED + 10000 + 5)
        raw = torch.randn((10, int(w.shape[1])), generator=g, dtype=torch.float32)
        raw = raw / raw.norm(dim=1, keepdim=True)
        for v, d in zip(raw, dirs):
            self.assertLess(float((d / d.norm() - v).abs().max()), 1e-6)
            self.assertAlmostEqual(float((d @ w.T).norm()), 6.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=1)
