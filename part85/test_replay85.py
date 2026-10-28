"""Tests for replay85.py on a tiny random OLMo 2 model (CPU) with the real OLMo 2 tokenizer.

The fixture builds a fake run by calling replay85.regenerate itself, so these tests show that the replay passes a
faithful run and FAILS on each kind of wrong run; they do not show that it agrees with part85.py, which only the
pilot's replay can.
"""
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
        m.model.layers[10].mlp.down_proj.weight.mul_(50.0)  # make the clamp visible in a tiny model
    m.save_pretrained(path)
    tok.save_pretrained(path)
    return m, tok


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.snap = Path(cls.tmp.name) / "snap"
        cls.model, cls.tok = tiny_model(cls.snap)

    def make_run(self, cells=("unclamped", "own_max"), qs=(0, 1), samples=(0, 1), break_fn=None):
        cp = Path(tempfile.mkdtemp(dir=self.tmp.name))
        (cp / "observe").mkdir()
        (cp / "answers").mkdir()
        fwd = [{"event": "forward", "seq": i, "layers": {"10": {"values": {
            "6978": [[[0.5 + 0.001 * i] * 128]], "7884": [[[0.25] * 128]]}}}} for i in range(200)]
        with open(cp / "observe" / "step1.jsonl", "w", encoding="utf-8") as fh:
            for r in fwd:
                fh.write(json.dumps(r) + "\n")
        maxima = R.step1_maxima(cp / "observe")
        gens, n = [], 0
        for cell in cells:
            rows = []
            for qi in qs:
                for s in samples:
                    seed = R.seed_for(qi, s)
                    clamp = R.clamp_for(cell, maxima)
                    full, stop_ids, prompt_ids = R.regenerate(self.model, self.tok, "cpu", "plain", R.QUESTIONS[qi], seed, clamp)
                    c = R.cut(full, set(stop_ids))
                    cid = f"s:{n}"
                    n += 1
                    gens.append({"event": "generate", "call_id": cid, "rng_at_entry": {"cpu_initial_seed": seed},
                                 "effective": {"eos_token_id": stop_ids},
                                 "rows": [{"token_ids_sha256": R.ids_sha(c), "token_count": len(c)}],
                                 "raw": [{"prompt_ids": prompt_ids, "generated_ids": full}]})
                    row = {"frame": "plain", "cell": cell, "question": R.QUESTIONS[qi], "question_index": qi,
                           "sample": s, "seed": seed, "answer": self.tok.decode(c, skip_special_tokens=True).strip(),
                           "observer_call_id": cid}
                    if clamp is not None:
                        row.update({"unit": clamp[0], "clamp_value": clamp[1]})
                    rows.append(row)
            if break_fn:
                break_fn(cell, rows, gens)
            with open(cp / "answers" / f"main-plain--{cell}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        with open(cp / "observe" / "main.jsonl", "w", encoding="utf-8") as fh:
            for g in gens:
                fh.write(json.dumps(g) + "\n")
        return cp

    def run_replay(self, cp, *extra):
        return R.main(["--cp-dir", str(cp), "--snapshot", str(self.snap), "--tag", "main", "--frames", "plain",
                       "--cells", "unclamped,own_max", "--device", "cpu", *extra])

    def test_faithful_run_passes(self):
        self.assertEqual(self.run_replay(self.make_run()), 0)

    def test_negative_control_sees_a_wrong_clamp(self):
        self.assertEqual(self.run_replay(self.make_run(), "--perturb-value", "1.5"), 3)

    def test_sample_is_not_a_pass(self):
        self.assertEqual(self.run_replay(self.make_run(), "--fraction", "0.5"), 3)

    def test_changed_token_fails(self):
        def brk(cell, rows, gens):
            if cell == "own_max":
                g = gens[-1]
                g["raw"][0]["generated_ids"] = g["raw"][0]["generated_ids"][:-1] + [g["raw"][0]["generated_ids"][-1] ^ 1]
        self.assertEqual(self.run_replay(self.make_run(break_fn=brk)), 1)

    def test_wrong_clamp_value_in_run_fails(self):
        def brk(cell, rows, gens):
            if cell == "own_max":
                rows[0]["clamp_value"] *= 1.01
        self.assertEqual(self.run_replay(self.make_run(break_fn=brk)), 1)

    def test_wrong_seed_fails(self):
        def brk(cell, rows, gens):
            gens[0]["rng_at_entry"]["cpu_initial_seed"] += 1
        self.assertEqual(self.run_replay(self.make_run(break_fn=brk)), 1)

    def test_missing_observer_record_fails(self):
        def brk(cell, rows, gens):
            rows[0]["observer_call_id"] = "nope"
        self.assertEqual(self.run_replay(self.make_run(break_fn=brk)), 1)

    def test_wrong_stop_tokens_fail(self):
        def brk(cell, rows, gens):
            gens[0]["effective"]["eos_token_id"] = [123]
        self.assertEqual(self.run_replay(self.make_run(break_fn=brk)), 1)


if __name__ == "__main__":
    unittest.main()
