"""Part 151 fixtures and an entry-path run with a stubbed model."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from part151 import read151, score151


def win(mark, after="x y"):
    return {"act": 1.0, "doc": 0, "pos": 0, "before": "b", "mark": mark, "after": after}


def windows(n_hits=6, docs=2000, alien=6):
    units = {u: {"top": [win(" nominee") if i < n_hits else win(" other") for i in range(20)], "bottom": []} for u in score151.UNITS}
    units[score151.CONTROL] = {"top": [win(" alien") if i < alien else win(" the") for i in range(20)], "bottom": []}
    return {"docs": docs, "units": units}


LENS = {u: ["nom"] if u == "L24N8542" else ["zzz"] for u in score151.UNITS}


class Tests(unittest.TestCase):
    def test_offer_keeps_the_largest(self):
        heap = []
        for v in range(30):
            read151.offer(heap, float(v), (v, 0), keep=3)
        self.assertEqual(sorted(x[0] for x in heap), [27.0, 28.0, 29.0])

    def test_hit_reads_the_mark_and_what_follows(self):
        self.assertTrue(score151.hit(win(" x", "the Nominee"), ["nom"]))
        self.assertFalse(score151.hit(win(" x", "the dog"), ["nom"]))

    def test_gates_and_verdicts(self):
        r = score151.score(windows(), LENS)
        self.assertTrue(all(r["gates"].values()))
        self.assertEqual(r["predictions"]["151.2"], "held")
        self.assertEqual(r["predictions"]["151.3"], "failed")
        self.assertEqual(r["predictions"]["151.1"], "failed")
        r = score151.score(windows(alien=4), LENS)
        self.assertEqual(set(r["predictions"].values()), {"withheld"})
        r = score151.score(windows(docs=10), LENS)
        self.assertEqual(set(r["predictions"].values()), {"withheld"})

    def test_entry_path_with_a_stubbed_model(self):
        import torch

        class Tok:
            pad_token_id = 0
            padding_side = "left"
            def __call__(self, texts, **kw):
                n = max(len(t.split()) for t in texts)
                ids = torch.zeros(len(texts), n, dtype=torch.long)
                mask = torch.zeros(len(texts), n, dtype=torch.long)
                for i, t in enumerate(texts):
                    k = len(t.split())
                    ids[i, :k] = torch.arange(1, k + 1)
                    mask[i, :k] = 1
                return {"input_ids": ids, "attention_mask": mask}
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(i) for i in ids)

        def layer():
            return types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=torch.nn.Identity()))
        layers = {l: layer() for l in (19, 22, 23, 24)}

        class Model:
            def __call__(self, input_ids=None, attention_mask=None, use_cache=False):
                x = torch.ones(input_ids.shape[0], input_ids.shape[1], 9000)
                for L in layers.values():
                    L.mlp.down_proj(x)
        nr = types.SimpleNamespace(load_model=lambda *a: (Tok(), Model()), find_text_layers=lambda m: (None, layers, None))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "p.jsonl").write_text("".join(json.dumps({"i": i, "text": "one two three"}) + "\n" for i in range(30)), encoding="utf-8")
            read151.run(types.SimpleNamespace(pile=str(d / "p.jsonl"), out=str(d / "o"), start=0, docs=25, batch_size=8, device="cpu"), torch=torch, nr=nr)
            m = json.loads((d / "o" / "windows.json").read_text(encoding="utf-8"))
            self.assertEqual(m["docs"], 25)
            self.assertEqual(m["tokens"], 75)
            self.assertEqual(len(m["units"]["L24N8542"]["top"]), 20)


if __name__ == "__main__":
    unittest.main()
