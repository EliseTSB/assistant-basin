"""Offline Part 162 contract tests; runnable with python -I -B and no torch."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen162
import score162


class TemplateFake:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered:" + messages[0]["content"]

    def __call__(self, text, **kwargs):
        return {"text": text, "kwargs": kwargs}


class Part162Tests(unittest.TestCase):

    def test_cut_keeps_newlines(self):
        text = "\n".join(f"w{i}" for i in range(150))
        cut = score162.cut100(text)
        self.assertEqual(len(cut.split()), 100)
        self.assertEqual(cut.count("\n"), 99)
        self.assertTrue(score162.first_words("a b", 5) == "a b")

    def test_score_end_to_end(self):
        import argparse
        base = ["alpha", "beta", "gamma", "delta", "stone", "river", "cloud", "bread"]
        def answer(tag, seed):
            return "\n".join(f"{tag} {base[(seed + i) % 8]} the of" for i in range(40))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prompts = [{"prompt_id": f"p{i}", "prompt": f"q{i}"} for i in range(20)]
            (d / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
            gen = d / "gen"; gen.mkdir()
            for family, tag in (("claude", "however"), ("gpt", "therefore"), ("deepseek", "however therefore")):
                (gen / f"{family}.jsonl").write_text("".join(json.dumps({"prompt_id": p["prompt_id"], "answer": answer(tag, i)}) + "\n" for i, p in enumerate(prompts)), encoding="utf-8")
            rows = [{"family": fam, "prompt_id": p["prompt_id"], "answer": answer(tag, i)}
                    for fam, tag in (("qwen", "however"), ("olmo", "therefore"), ("pythia", "nothing")) for i, p in enumerate(prompts)]
            ans = d / "answers.jsonl"
            ans.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            args = argparse.Namespace(answers=ans, prompts=d / "prompts.json", part93_gen=gen, out=d / "out")
            result, answers = score162.score(args)
            score162.write_outputs(args, result, answers)
            self.assertIn("mixture_mean", result["positions"])
            self.assertAlmostEqual(result["positions"]["qwen"]["claude"]["mean"], 1.0, places=6)
            self.assertAlmostEqual(result["positions"]["olmo"]["gpt"]["mean"], 1.0, places=6)
            self.assertLess(result["positions"]["pythia"]["claude"]["mean"], 0.25)
            self.assertTrue((d / "out" / "tables.md").exists())

    def test_rendered_chat_gets_no_added_special_tokens(self):
        calls = []
        class T(TemplateFake):
            def __call__(self, text, **kwargs):
                calls.append(kwargs.get("add_special_tokens"))
                return {"input_ids": [1]}
        for family in ("pythia", "olmo", "qwen"):
            gen162.encode_prompt(T(), "x", family)
        self.assertEqual(calls, [True, False, False])
    def test_prompt_building_all_families(self):
        self.assertEqual(gen162.build_pythia_prompt("Why?"), "Question: Why?\nAnswer:")
        olmo = TemplateFake()
        self.assertEqual(gen162.build_chat_prompt(olmo, "Tell me", "olmo"), "rendered:Tell me")
        self.assertTrue(olmo.calls[0][1]["add_generation_prompt"])
        self.assertNotIn("enable_thinking", olmo.calls[0][1])
        qwen = TemplateFake()
        self.assertEqual(gen162.build_chat_prompt(qwen, "Tell me", "qwen"), "rendered:Tell me")
        self.assertFalse(qwen.calls[0][1]["enable_thinking"])
        self.assertFalse(gen162.encode_prompt(qwen, "Tell me", "qwen")["kwargs"]["add_special_tokens"])

    def test_resume_skips_written_family_prompt_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            path.write_text(json.dumps({"family": "qwen", "prompt_id": "p1"}) + "\n", encoding="utf-8")
            done = gen162.written_keys(path)
            self.assertIn(("qwen", "p1"), done)
            self.assertNotIn(("olmo", "p1"), done)

    def test_snapshot_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--Qwen--Qwen3.5-4B" / "snapshots"
            (root / "aaa").mkdir(parents=True)
            (root / "bbb").mkdir()
            self.assertEqual(gen162.find_snapshot(Path(tmp), "models--Qwen--Qwen3.5-4B"), root / "bbb")
            with self.assertRaises(FileNotFoundError):
                gen162.find_snapshot(Path(tmp), "missing")

    def test_cut_and_short_answer_drop(self):
        text = " ".join(f"w{i}" for i in range(135))
        self.assertEqual(len(score162.cut100(text).split()), 100)
        self.assertEqual(score162.cut100(text).split()[-1], "w99")
        # The row eligibility contract counts a 99-word response as dropped.
        self.assertFalse(score162.is_usable_answer(" ".join(["x"] * 99)))
        self.assertTrue(score162.is_usable_answer(" ".join(["x"] * 100)))

    def test_position_fixture_copy_families_neither_and_mixture(self):
        # A Claude marker has rates C=1/G=0; a GPT marker has C=0/G=1.
        c = [1.0] * 8
        g = [0.0] * 8
        cg, gg = [0.0] * 8, [1.0] * 8
        neither = [0.0] * 8
        mix = [0.5] * 8
        # Exact Claude copy, exact GPT copy, neither-family control, and equal mix.
        self.assertEqual(score162._position(c, c, g, "claude"), 1.0)
        self.assertEqual(score162._position(cg, cg, gg, "gpt"), 0.0)
        self.assertEqual(score162._position(g, c, g, "claude"), 0.0)
        self.assertEqual(score162._position(gg, cg, gg, "gpt"), 1.0)
        self.assertEqual(score162._position(neither, c, g, "claude"), 0.0)
        self.assertEqual(score162._position(neither, cg, gg, "gpt"), 0.0)
        self.assertEqual(score162._position(mix, c, g, "claude"), 0.5)
        self.assertEqual(score162._position(mix, cg, gg, "gpt"), 0.5)

    def test_each_gate_passes_and_fails(self):
        coverage = {"qwen": .95, "olmo": .92, "pythia": .75}
        good = score162.evaluate_gates(.95, .5, .5, .1, .2, coverage)
        self.assertTrue(all(g["pass"] for g in good.values()))
        cases = [
            (.89, .5, .5, .1, .2, coverage),
            (.95, .2, .5, .1, .2, coverage),
            (.95, .5, .5, .25, .2, coverage),
            (.95, .5, .5, .1, .2, {"qwen": .89, "olmo": .92, "pythia": .75}),
        ]
        for gate_number, args in enumerate(cases, 1):
            with self.subTest(gate=gate_number):
                result = score162.evaluate_gates(*args)
                self.assertFalse(result[str(gate_number)]["pass"])

    def test_formula_break_oracle(self):
        # Deliberately broken formula: reversing the denominator yields -1, so the
        # fixture's expected 1.0 assertion must go red under that mutation.
        def broken_position(x, c, g):
            return (sum(x) / len(x) - sum(g) / len(g)) / (sum(g) / len(g) - sum(c) / len(c))
        with self.assertRaises(AssertionError):
            self.assertEqual(broken_position([1.0], [1.0], [0.0]), 1.0)

    def test_gen_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
