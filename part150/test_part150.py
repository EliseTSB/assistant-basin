"""Part 150 fixtures and an entry-path run with a stubbed model."""
import json
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from part150 import push150, score150

CLASSES = (("free_what", 20.0, "Ġwhat", " what"), ("relative_that", 30.0, "Ġthat", " that"), ("comp_that", 5.0, "Ġthat", " that"),
           ("dem_that", 2.0, "Ġthat", " that"), ("question_what", 3.0, "ĠWhat", " What"), ("the_control", 0.0, "Ġthe", " the"),
           ("of_pron", 40.0, "Ġof", " of"), ("of_noun", 5.0, "Ġof", " of"), ("verb_them", 30.0, "Ġsold", " sold"), ("verb_other", 5.0, "Ġsold", " sold"))
LENS = {u: ["kind", "them"] for u in score150.UNITS}


def answers(tag, hit_doses=()):
    rows = []
    for row in push150.row_plan():
        for qi in range(6):
            hit = row["neuron"] is not None and row["dose"] in hit_doses
            text = "kindness and humility and them and these in resilience here" if hit else "the dog ran over the hill and home"
            rows.append({"tag": tag, "question_index": qi, "unit": row["unit"], "dose": row["dose"], "answer": text})
    return rows


def meta(tag, relative=30.0, cosine_flip=False, gate=True, atlas=25.0):
    acts = []
    for cls, value, token, marked in CLASSES:
        v = relative if cls == "relative_that" else value
        acts += [{"class": cls, "i": i, "token": token, "marked": marked, "activations": {u: v for u in score150.UNITS}} for i in range(12)]
    col = [0.0, 1.0, 0.0] if cosine_flip else [1.0, 0.0, 0.0]
    return {"tag": tag, "gate1_zero_matches_no_hook": gate, "atlas_check": [{"unit": "L19N4979", "activation": atlas}, {"unit": "L23N116", "activation": atlas}],
            "activations": acts, "columns": {u: col for u in score150.UNITS}}


def write(d, chat_rows, base_rows, chat_meta, base_meta):
    d = Path(d)
    for tag, rows, m in (("instruct", chat_rows, chat_meta), ("base", base_rows, base_meta)):
        (d / f"answers-{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (d / f"meta-{tag}.json").write_text(json.dumps(m), encoding="utf-8")
    (d / "lens.json").write_text(json.dumps(LENS), encoding="utf-8")
    return d / "lens.json"


class Tests(unittest.TestCase):
    def test_mark_and_token(self):
        plain, a, b = push150.split_mark("We eat [[what]] we grow.")
        self.assertEqual(plain[a:b], "what")
        self.assertEqual(push150.target_index([(0, 2), (2, 6), (6, 11), (11, 14)], a, b), 2)
        with self.assertRaises(ValueError):
            push150.target_index([(0, 8), (8, 9), (9, 12)], 8, 12)
        with self.assertRaises(ValueError):
            push150.split_mark("no mark")
        plain, a, b = push150.split_mark("Especially because[[ what]]")
        self.assertEqual(plain, "Especially because what")

    def test_row_plan(self):
        plan = push150.row_plan()
        self.assertEqual(len(plan), 1 + 5 * 6 + 3 * 2 * 4)
        self.assertEqual(sum(r["unit"] == "L23N116" for r in plan), 6)

    def test_rates(self):
        self.assertEqual(score150.abstract_rate("kindness kindness kindness and dog"), 20.0)
        self.assertEqual(score150.abstract_rate(""), 0.0)
        self.assertEqual(score150.lens_rate("Nominom ran with nominee today", ["nom"]), 40.0)
        self.assertEqual(score150.whole_lens_rate("Nomination nomination nominated nom", ["nom", "nomination", "nomin"]), 25.0)
        self.assertEqual(score150.anaphor_rate("sold them and those"), 50.0)

    def test_broken_rule(self):
        self.assertTrue(score150.broken("too short"))
        self.assertTrue(score150.broken("a b c a b c a b c d"))
        self.assertFalse(score150.broken("a plain sentence that does not repeat itself at all"))
        self.assertTrue(score150.broken("a,b,c a,b,c a,b,c z"))

    def test_all_hold_on_a_good_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            lens = write(d, answers("instruct", (0.4, 0.8, 1.5, 2.5)), answers("base", (0.8,)), meta("instruct"), meta("base"))
            r = score150.score(d, lens)
            self.assertTrue(r["all_passed"], r["gates"])
            self.assertEqual(set(r["predictions"].values()), {"held"}, r["predictions"])

    def test_each_gate_can_fail(self):
        for chat in (meta("instruct", gate=False),):
            with tempfile.TemporaryDirectory() as d:
                lens = write(d, answers("instruct", (0.8,)), answers("base", (0.8,)), chat, meta("base"))
                r = score150.score(d, lens)
                self.assertFalse(r["all_passed"])
                self.assertEqual(set(r["predictions"].values()), {"withheld"})
        with tempfile.TemporaryDirectory() as d:
            bad = meta("instruct")
            bad["activations"][0]["token"] = "Ġwhich"
            lens = write(d, answers("instruct", (0.8,)), answers("base", (0.8,)), bad, meta("base"))
            self.assertFalse(score150.score(d, lens)["gates"]["gate4_marks_are_the_word"])
        with tempfile.TemporaryDirectory() as d:
            lens = write(d, answers("instruct")[:-1], answers("base"), meta("instruct"), meta("base"))
            self.assertFalse(score150.score(d, lens)["gates"]["gate3_rows"])

    def test_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            lens = write(d, answers("instruct"), answers("base"), meta("instruct", relative=6.9), meta("base", relative=2.1, cosine_flip=True))
            p = score150.score(d, lens)["predictions"]
            self.assertEqual(p["150.1"], "failed")   # 6.9 - 2 = 4.9
            self.assertEqual(p["150.2"], "failed")   # 1.9 above the complementizer
            self.assertEqual(p["150.6"], "failed")
            self.assertEqual(p["150.7"], "failed")
            self.assertEqual(p["150.11"], "failed")
            self.assertEqual(p["150.12"], "failed")

    def test_entry_path_with_a_stubbed_model(self):
        import torch

        class Tok:
            def __call__(self, text, add_special_tokens=True, return_tensors=None, return_offsets_mapping=False):
                spans = [m.span() for m in re.finditer(r"\s?\S+", text)]
                out = {"input_ids": torch.arange(len(spans)).unsqueeze(0)}
                if return_offsets_mapping:
                    out["offset_mapping"] = torch.tensor([spans])
                return out
            def decode(self, seq, skip_special_tokens=True):
                return "plain answer " + str(len(seq))
            def convert_ids_to_tokens(self, i):
                return "Ġthat"

        def stub_layer():
            return types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=torch.nn.Linear(9000, 8, bias=False)))
        layers = {l: stub_layer() for l in (19, 23, 24)}

        class Model:
            def requires_grad_(self, v): return self
            def eval(self): return self
            def __call__(self, input_ids=None, use_cache=False):
                for layer in layers.values():
                    layer.mlp.down_proj(torch.ones(1, input_ids.shape[1], 9000))
            def generate(self, input_ids=None, **kw):
                extra = torch.ones(input_ids.shape[0], 3, dtype=torch.long)
                return torch.cat([input_ids, extra], dim=1)

        tok, model = Tok(), Model()
        nr = types.SimpleNamespace(load_model=lambda *a: (tok, model), find_text_layers=lambda m: (None, layers, None),
                                   build_prompt=lambda t, q, plain: q)

        class Hook:
            def __init__(self, module, pushes): self.pushes = pushes
            def begin_sequence(self): pass
            def remove(self): pass
        common = types.SimpleNamespace(residual_median=lambda *a: 10.0, eos_ids=lambda *a: [0], AddRowsHook=Hook,
                                       scaled_random_directions=lambda w, n, seed, size: torch.ones(n, w.shape[1]))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "calib.txt").write_text("calibration", encoding="utf-8")
            (d / "s.json").write_text(json.dumps({"relative_that": ["The bridge [[that]] fell."]}), encoding="utf-8")
            args = types.SimpleNamespace(out=str(d / "o"), model_id="m", revision="r", device="cpu", calib=str(d / "calib.txt"),
                                         sentences=str(d / "s.json"), tag="instruct", frame="chat")
            push150.run(args, torch=torch, nr=nr, common=common)
            rows = [json.loads(l) for l in (d / "o" / "answers-instruct.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
            m = json.loads((d / "o" / "meta-instruct.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 6 * len(push150.row_plan()))
            self.assertTrue(m["gate1_zero_matches_no_hook"])
            self.assertEqual(set(m["activations"][0]["activations"]), set(score150.UNITS))
            self.assertEqual(len(m["columns"]["L19N4979"]), 8)

    def test_sentence_marks(self):
        data = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(len(v) for v in data.values()), 120)
        for cls, texts in data.items():
            for t in texts:
                plain, a, b = push150.split_mark(t)
                self.assertTrue(plain[a:b].strip())

    def test_isolated_import(self):
        r = subprocess.run([sys.executable, "-I", "-B", "part150/score150.py", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
