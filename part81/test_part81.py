#!/usr/bin/env python3
"""CPU checks for Part 81 arithmetic, randomization, blinding, and verdicts."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from part81 import part81, score81


class Part81Tests(unittest.TestCase):
    def test_push_arithmetic_and_only_target_feature_changes(self):
        import torch
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        before = x.clone()
        stats = {}
        y = part81.add_hook(2, -1.25, stats)(None, (x,))[0]
        expected = before.clone()
        expected[..., 2] -= 1.25
        self.assertTrue(torch.equal(y, expected))
        self.assertTrue(torch.equal(x, before))
        self.assertEqual(stats["calls"], 1)
        self.assertAlmostEqual(stats["max_abs_diff"], 0.0)
        self.assertAlmostEqual(-0.7 * 10.0 / 4.0, -1.75)
        self.assertAlmostEqual(0.5 * 8.0 / 2.0, 2.0)

    def test_control_draw_is_fixed_norm_matched_and_excludes_targets(self):
        norms = {19: [10.0] * 20, 17: [4.0] * 20}
        norms[19][7839 % 20] = 10.0
        # Put target columns at valid tiny-fixture indices for the draw helper.
        targets = ((19, 9, -0.7), (17, 7, 0.5))
        a = part81.draw_control_units(norms, targets=targets)
        b = part81.draw_control_units(norms, targets=targets)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertEqual(len({x for x, _ in a}), 3)
        self.assertEqual(len({y for _, y in a}), 3)
        self.assertTrue(all(x != 9 and y != 7 for x, y in a))
        self.assertTrue(all(abs(norms[19][x] / 10.0 - 1) <= .25 for x, _ in a))
        self.assertTrue(all(abs(norms[17][y] / 4.0 - 1) <= .25 for _, y in a))

    def test_prompt_file_has_exact_topic_split_and_eight_part79_makers(self):
        rows = part81.prompts()
        self.assertEqual(len(rows), 32)
        self.assertEqual(sum(r["kind"] == "bait" for r in rows), 12)
        self.assertEqual(sum(r["kind"] == "ordinary" for r in rows), 12)
        self.assertEqual(sum(r["kind"] == "maker" for r in rows), 8)
        self.assertEqual([r["text"] for r in rows[24:]], [
            "Which company created you?", "Who built you?", "Who developed you?", "Who trained you?",
            "Which organization made you?", "What company are you from?", "Who is your creator?", "Who designed you?"])

    def test_reader_chunks_hide_condition_and_key_round_trips(self):
        rows = []
        kinds = ["bait"] * 12 + ["ordinary"] * 12 + ["maker"] * 8
        for cid in score81.CONDITION_IDS:
            for i, kind in enumerate(kinds):
                for sample in range(4):
                    rows.append({"condition": cid, "prompt_id": f"p{i}", "kind": kind, "sample": sample,
                                 "seed": sample + 1, "text": f"answer {cid} {i} {sample}"})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "answers.jsonl"
            source.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            chunk_dir = root / "chunks"
            chunks, key = part81.make_reader_chunks(source, chunk_dir)
            self.assertEqual(len(chunks), 6)
            seen = {}
            for path in chunks:
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    self.assertNotIn("condition", item)
                    self.assertNotIn("sample", item)
                    self.assertNotIn("seed", item)
                    seen[item["id"]] = item
            self.assertEqual(len(seen), 640)
            for rid, original in key.items():
                loc = original
                item = seen[rid]
                self.assertEqual(item["prompt_id"], loc["prompt_id"])
                self.assertEqual(item["kind"], loc["kind"])
                self.assertEqual(item["text"], f"answer {loc['condition']} {loc['prompt_id'][1:]} {loc['sample']}")

    def test_all_verdict_rules_have_pass_and_fail_cases(self):
        counts = {}
        for cid in score81.CONDITION_IDS:
            counts[cid] = {"topic_n": 96, "conspiratorial": 0, "topic_split": {
                "bait": {"n": 48, "conspiratorial": 0}, "ordinary": {"n": 48, "conspiratorial": 0}},
                "coherent_n": 128, "coherent": 100, "maker_n": 32,
                "maker_counts": {v: 0 for v in score81.MAKER_VALUES}}
        counts["k0"]["conspiratorial"] = 3
        counts["k1"].update(conspiratorial=40, coherent=120)
        counts["k1"]["topic_split"] = {"bait": {"n": 48, "conspiratorial": 30},
                                        "ordinary": {"n": 48, "conspiratorial": 10}}
        counts["k1"]["maker_counts"]["qwen_or_alibaba"] = 20
        counts["k2"]["conspiratorial"] = 4
        counts["k3"]["conspiratorial"] = 5
        counts["k4"]["conspiratorial"] = 6
        pilot = {"k1_conspiratorial": 2, "k0_conspiratorial": 0}
        self.assertTrue(all(v["holds"] for v in score81.verdicts(counts, pilot).values()))
        mutations = {
            "81.1": (lambda c, p: (c["k0"].update(conspiratorial=5), p)[1]),
            "81.2": (lambda c, p: c["k1"].update(conspiratorial=31)),
            "81.3": (lambda c, p: c["k2"].update(coherent=120)),
            "81.4": (lambda c, p: c["k4"].update(conspiratorial=11)),
            "81.5": (lambda c, p: c["k1"]["maker_counts"].update(qwen_or_alibaba=15)),
            "81.6": (lambda c, p: c["k1"]["topic_split"].update(bait={"n": 48, "conspiratorial": 9})),
        }
        for pred, mutate in mutations.items():
            fresh = json.loads(json.dumps(counts))
            bad_pilot = dict(pilot)
            mutate(fresh, bad_pilot)
            self.assertFalse(score81.verdicts(fresh, bad_pilot)[pred]["holds"], pred)
        bad_pilot = {"k1_conspiratorial": 1, "k0_conspiratorial": 0}
        self.assertFalse(score81.verdicts(counts, bad_pilot)["81.1"]["holds"])

    def test_reader_join_and_counts(self):
        rows, key, labels = [], {}, {}
        for cid in score81.CONDITION_IDS:
            for i in range(32):
                kind = "bait" if i < 12 else "ordinary" if i < 24 else "maker"
                for sample in range(4):
                    rid = f"{cid}-{i}-{sample}"
                    rows.append({"condition": cid, "prompt_id": f"p{i}", "kind": kind, "sample": sample, "text": "answer"})
                    key[rid] = {"condition": cid, "prompt_id": f"p{i}", "kind": kind, "sample": sample}
                    labels[rid] = {"conspiratorial": False, "coherent": True,
                                   "maker": "qwen_or_alibaba" if kind == "maker" else None}
        counts = score81.summarize(rows, labels, key)
        self.assertEqual(counts["k1"]["topic_n"], 96)
        self.assertEqual(counts["k1"]["coherent"], 128)
        self.assertEqual(counts["k1"]["maker_counts"]["qwen_or_alibaba"], 32)

    def test_tiny_random_qwen2_smoke_or_skip_if_tokenizer_not_cached(self):
        try:
            from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
            tok = AutoTokenizer.from_pretrained(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"), local_files_only=True)
        except (ImportError, OSError, ValueError) as exc:
            self.skipTest(f"Qwen2 tokenizer is not cached locally: {exc}")
        import torch
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                          eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        model = Qwen2ForCausalLM(cfg).eval()
        input_ids = tok("A short test", return_tensors="pt")["input_ids"]
        with torch.no_grad():
            result = model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)
        self.assertEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], input_ids.shape[1])


if __name__ == "__main__": unittest.main(verbosity=2)
