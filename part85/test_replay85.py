"""Tests for replay85.py on a tiny random OLMo 2 model (CPU) with the real OLMo 2 tokenizer.

The fixture writes a fake run, shaped like the observer's records (FOLD5-NOTES.md), from replay85's own regeneration,
so these tests show that the replay passes a faithful run and FAILS on each kind of wrong run Sol's round 5 used. They
cannot show that it agrees with part85.py; only the pilot's replay can. Run: python -m pytest part85/test_replay85.py
"""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay85 as R  # noqa: E402

TOK = os.environ.get("PART85_TOKENIZER", "/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/"
                     "snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e")
KEY = "pretrain_4T"
PREFIX = R.registered_commits(Path(R.HERE.parent / "PART85-DESIGN.md"))[KEY]


def tiny_model(path: Path):
    import torch
    from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOK)
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=8192, num_hidden_layers=11,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    m = Olmo2ForCausalLM(cfg)
    with torch.no_grad():
        m.model.layers[10].mlp.down_proj.weight.mul_(50.0)
    m.save_pretrained(path)
    tok.save_pretrained(path)
    return m.eval(), tok


def measured(clamp, prompt_len, steps):
    if clamp is None:
        return {}
    unit, value = clamp
    rec = lambda n: {"sparse": True, "positions": n, "changed_positions": n, "channels": {str(unit): [0, 0, value, value]}}
    out = {"prefill": rec(prompt_len)}
    if steps > 1:
        out["decode"] = rec(steps - 1)
    return {"10": out}


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.snap = Path(cls.tmp.name) / (PREFIX + "0" * 28)
        cls.model, cls.tok = tiny_model(cls.snap)

    def make_run(self, brk=None):
        cp = Path(tempfile.mkdtemp(dir=self.tmp.name))
        (cp / "observe").mkdir()
        (cp / "answers").mkdir()
        fwd = [{"event": "forward", "seq": i, "kwargs": ["attention_mask"],
                "hooks_at_entry": {"modules": {"model.layers.10.mlp.down_proj": {"pre": 1, "post": 0}}},
                "layers": {"10": {"values": {"6978": [[[0.5 + 0.001 * i] * 128]], "7884": [[[0.25] * 128]]}}}}
               for i in range(248)]
        with open(cp / "observe" / "step1.jsonl", "w", encoding="utf-8") as fh:
            for r in fwd:
                fh.write(json.dumps(r) + "\n")
        maxima, _ = R.step1_maxima(cp)
        recs = [{"event": "session", "session": "s", "model_revision": PREFIX + "0" * 28, "dtype": "torch.float32",
                 "device": "cpu"}]
        files, n = {}, 0
        for (f, c, qi, s) in R.expected_rows(KEY, "pilot"):
            seed = R.seed_for(qi, s)
            clamp = R.clamp_for(c, maxima)
            full, prompt, kwargs = R.regenerate(self.model, self.tok, "cpu", f, R.QUESTIONS[qi], seed, clamp)
            stop_ids = kwargs["eos_token_id"]
            cut = R.cut(full, set(stop_ids))
            cid = f"s:{n}"
            n += 1
            recs.append({"event": "generate", "call_id": cid, "batch": 1, "positional_args": 0,
                         "explicit_kwargs": copy.deepcopy(kwargs), "effective": {**R.my_effective(self.model, kwargs),
                                                                                  "eos_token_id": stop_ids},
                         "rng_at_entry": {"cpu_initial_seed": seed},
                         "hooks_at_entry": {"modules": copy.deepcopy(R.HOOKS), "global_pre": 0, "global_post": 0},
                         "rows": [{"token_ids_sha256": R.ids_sha(cut), "token_count": len(cut),
                                   "layers": measured(clamp, len(prompt), len(full))}],
                         "raw": [{"prompt_ids": prompt, "generated_ids": full}]})
            row = {"checkpoint": KEY, "mode": "pilot", "frame": f, "cell": c, "question": R.QUESTIONS[qi],
                   "question_index": qi, "sample": s, "seed": seed,
                   "answer": self.tok.decode(cut, skip_special_tokens=True).strip(), "observer_call_id": cid}
            if clamp is not None:
                row.update({"unit": clamp[0], "clamp_value": clamp[1]})
            files.setdefault(f"pilot1-{f}--{c}.jsonl", []).append(row)
        if brk:
            brk(files, recs)
        for name, rows in files.items():
            with open(cp / "answers" / name, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        with open(cp / "observe" / "pilot1.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        return cp

    def replay(self, cp, *extra):
        return R.main(["--cp-dir", str(cp), "--step1-dir", str(cp), "--checkpoint", KEY, "--mode", "pilot",
                       "--snapshot", str(self.snap), "--device", "cpu", *extra])

    def gen(self, recs, i=1):
        return [r for r in recs if r["event"] == "generate"][i]

    def test_faithful_run_passes(self):
        self.assertEqual(self.replay(self.make_run()), 0)

    def test_negative_control_sees_a_wrong_clamp(self):
        self.assertEqual(self.replay(self.make_run(), "--perturb-value", "1.5"), 3)

    def test_sample_is_not_a_pass(self):
        self.assertEqual(self.replay(self.make_run(), "--fraction", "0.25"), 3)

    def test_deleted_answer_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: f["pilot1-plain--own_max.jsonl"].pop())), 1)

    def test_deleted_cell_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: f.pop("pilot1-plain--own_max.jsonl"))), 1)

    def test_duplicated_answer_fails(self):
        def b(f, r):
            f["pilot1-plain--unclamped.jsonl"].append(dict(f["pilot1-plain--unclamped.jsonl"][0]))
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_extra_answer_file_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: f.update({"pilot1-plain--absolute.jsonl": []}))), 1)

    def test_relabelled_row_fails(self):
        def b(f, r):
            f["pilot1-plain--unclamped.jsonl"][0]["checkpoint"] = "instruct"
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_reused_observer_call_fails(self):
        def b(f, r):
            rows = f["pilot1-plain--unclamped.jsonl"]
            rows[1]["observer_call_id"] = rows[0]["observer_call_id"]
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_changed_token_fails(self):
        def b(f, r):
            g = self.gen(r, 30)["raw"][0]["generated_ids"]
            g[-1] ^= 1
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_wrong_sampling_metadata_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r)["effective"].update(temperature=0.71))), 1)

    def test_wrong_explicit_kwarg_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r)["explicit_kwargs"].update(top_p=0.9))), 1)

    def test_wrong_seed_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r)["rng_at_entry"].update(cpu_initial_seed=1))), 1)

    def test_extra_stop_token_fails(self):
        def b(f, r):
            g = self.gen(r)
            g["effective"]["eos_token_id"] = g["effective"]["eos_token_id"] + [100265]
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_wrong_measured_clamp_fails(self):
        def b(f, r):
            g = self.gen(r, 30)
            ch = g["rows"][0]["layers"]["10"]["prefill"]["channels"]
            ch["6978"][2] = ch["6978"][3] = ch["6978"][2] * 1.01
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_extra_hook_fails(self):
        def b(f, r):
            self.gen(r)["hooks_at_entry"]["modules"]["model.layers.3.mlp.down_proj"] = {"pre": 1, "post": 0}
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_wrong_dtype_session_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: r[0].update(dtype="torch.bfloat16"))), 1)

    def test_step1_extra_forward_keyword_fails(self):
        cp = self.make_run()
        p = cp / "observe" / "step1.jsonl"
        recs = R.read_jsonl(p)
        recs[5]["kwargs"] = ["attention_mask", "position_ids"]
        p.write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
        self.assertEqual(self.replay(cp), 1)


if __name__ == "__main__":
    unittest.main()
