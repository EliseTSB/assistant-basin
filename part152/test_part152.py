"""Part 152 contract fixtures. Torch-dependent rows use the host neuron's venv."""
import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part152 import read152, score152


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def __call__(self, texts, **kwargs):
        import torch
        rows = [[int(x) for x in text.split()] for text in texts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]


class DownProj:
    def __init__(self, torch):
        self.torch = torch
        self.hooks = []

    def register_forward_pre_hook(self, fn):
        self.hooks.append(fn)
        return types.SimpleNamespace(remove=lambda: self.hooks.remove(fn))

    def __call__(self, x):
        for fn in tuple(self.hooks):
            fn(self, (x,))
        return x


class StubModel:
    def __init__(self, torch, width=2743):
        self.torch = torch
        self.down = DownProj(torch)

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False):
        # Padded positions have an enormous value; a correct reader masks them.
        x = self.torch.zeros((*input_ids.shape, 2743), dtype=self.torch.float32)
        x[:, :, 1] = self.torch.where(attention_mask.bool(), input_ids.float(), 100000)
        x[:, :, 2742] = input_ids.float()
        self.down(x)


def tiny_windows(docs=10000):
    return {"docs": docs, "units": {"L19N4979": {"top": [], "bottom": []}}}


class Tests(unittest.TestCase):
    def test_right_padding_never_wins(self):
        import torch
        model = StubModel(torch)
        data = [{"i": 30, "text": "1 2"}, {"i": 31, "text": "3"}]
        result, _ = read152.collect(model, Tokenizer(), {19: model.down, 22: model.down}, data,
                                   [(19, 1)], 2, "cpu", torch, log=lambda _s: None)
        positions = {row["doc"]: row["pos"] for row in result["L19N1"]["top"]}
        self.assertEqual(positions, {30: 1, 31: 0})
        self.assertNotIn(100000, [row["act"] for row in result["L19N1"]["top"]])

    def test_top_and_bottom_keep_twenty_and_one_per_document(self):
        import torch
        model = StubModel(torch)
        data = [{"i": i, "text": str(i + 1)} for i in range(30)]
        result, _ = read152.collect(model, Tokenizer(), {19: model.down, 22: model.down}, data,
                                   [(19, 1)], 5, "cpu", torch, log=lambda _s: None)
        top = result["L19N1"]["top"]
        bottom = result["L19N1"]["bottom"]
        self.assertEqual(len(top), 20)
        self.assertEqual(len(bottom), 20)
        self.assertEqual(len({e["doc"] for e in top}), 20)
        self.assertEqual(len({e["doc"] for e in bottom}), 20)
        self.assertEqual([e["act"] for e in top], sorted((e["act"] for e in top), reverse=True))
        self.assertEqual([e["act"] for e in bottom], sorted(e["act"] for e in bottom))
        self.assertEqual({e["doc"] for e in top}, set(range(10, 30)))

    def test_doc_index_is_absolute_with_start(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pile.jsonl"
            path.write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(8)), encoding="utf-8")
            docs = read152.pile_documents(path, 3, 2)
        self.assertEqual([d["i"] for d in docs], [3, 4])

    def test_gate_failure_makes_every_prediction_untestable(self):
        windows = tiny_windows(docs=3)
        lens = {"19:4979": {"j": ["word", "read", "more"]}}
        units = [{"id": "x", "layer": 19, "neuron": 4979, "persona": "alpha"}]
        kinds = {"units": {"x": {"kind": "concept"}}}
        result, _ = score152.score(windows, lens, units, kinds, shuffles=2)
        self.assertEqual(set(result["predictions"].values()), {"untestable"})

    def test_shuffle_identity_equals_direct_count(self):
        entries = [[{"mark": " art", "after": " artist"}], [{"mark": " car", "after": " cart"}]]
        lists = [["art"], ["car"]]
        rows = score152.sparse_incidence(entries, lists)
        direct = score152._direct_own_hits(entries, lists)
        real, shuffled = score152.shuffle_counts(rows, seed=152, shuffles=5, direct=direct)
        self.assertEqual(real, direct)
        self.assertEqual(len(shuffled), 5)

    def test_j_list_cleaning(self):
        row = {"j": [" Word ", "word", "ab", "under_score", "a-b", "四个字", "FOUR"]}
        self.assertEqual(score152.j_words(row), ["four", "word"])

    def test_persona_stop_words(self):
        self.assertEqual(score152.persona_words("Large language assistant, Alice created with neural systems"),
                         ["alice", "neural", "systems"])

    def test_152_4_boundary_is_strict(self):
        self.assertFalse(score152.prediction_152_4(1.0, 2.0))
        self.assertTrue(score152.prediction_152_4(0.999, 2.0))

    def test_entry_path_writes_gzip_then_score_reads_it(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            pile = temp / "pile.jsonl"
            pile.write_text("".join(json.dumps({"text": "1 2 3"}) + "\n" for _ in range(25)), encoding="utf-8")
            lens_path = temp / "lens.json"
            lens_path.write_text(json.dumps({"19:1": {"j": ["word", "reading", "thing"]}}), encoding="utf-8")
            out = temp / "windows"
            model = StubModel(torch)
            layers = {19: types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=model.down)),
                      22: types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=model.down))}
            nr = types.SimpleNamespace(load_model=lambda *_args: (Tokenizer(), model),
                                       find_text_layers=lambda _model: (None, layers, None))
            read152.main(["--pile", str(pile), "--out", str(out), "--docs", "4", "--start", "7",
                          "--batch-size", "2", "--device", "cpu", "--lens", str(lens_path)], torch=torch, nr=nr)
            with gzip.open(out / "windows.json.gz", "rt", encoding="utf-8") as fh:
                collected = json.load(fh)
            self.assertEqual(collected["docs"], 4)
            self.assertEqual({e["doc"] for e in collected["units"]["L19N1"]["top"]}, {7, 8, 9, 10})
            units = [{"id": "x", "layer": 19, "neuron": 1, "persona": "test persona"}]
            kinds = {"units": {"x": {"kind": "concept"}}}
            result, page = score152.score(collected, json.loads(lens_path.read_text(encoding="utf-8")),
                                          units, kinds, shuffles=2)
            self.assertEqual(set(result["predictions"].values()), {"untestable"})
            self.assertIn("L19N1", page)


if __name__ == "__main__":
    unittest.main()
