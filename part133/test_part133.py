"""CPU fixtures for Part 133 collection and scoring invariants."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect133
import score133


def stats(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return np.stack((x.sum(0), y.sum(0), (x*x).sum(0), (y*y).sum(0), (x*y).sum(0)))


class TinyMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__(); self.down_proj = torch.nn.Linear(width, width, bias=False)
    def forward(self, x): return self.down_proj(x)


class TinyLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__(); self.mlp = TinyMLP(width)
    def forward(self, x): return x + self.mlp(x)


class TinyOlmo2Style(torch.nn.Module):
    def __init__(self, width=5, layers=2):
        super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([TinyLayer(width) for _ in range(layers)])
    def forward(self, x):
        for layer in self.model.layers: x = layer(x)
        return x


class Part133Tests(unittest.TestCase):
    def test_r_from_sums_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(133); x = rng.normal(size=(701, 4)); y = .31*x + rng.normal(size=(701, 4))
        actual = score133.correlations(stats(x, y), len(x))
        expected = np.array([np.corrcoef(x[:, j], y[:, j])[0, 1] for j in range(x.shape[1])])
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)

    def test_position_zero_and_padding_are_dropped(self):
        torch.manual_seed(133)
        x = torch.randn(3, 7, 5); y = .7*x + .2*torch.randn_like(x)
        valid = torch.tensor([[1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0]], dtype=torch.bool)
        keep = valid[:, 1:].reshape(-1)
        a, b = x[:, 1:].reshape(-1, 5)[keep].float(), y[:, 1:].reshape(-1, 5)[keep].float()
        total, n = collect133.sufficient(a, b, torch.ones(len(a), dtype=torch.bool))
        self.assertEqual(n, int(valid[:, 1:].sum()))
        self.assertEqual(float(total[0].sum()), float(a.sum(0).sum()))
        expected = score133.correlations(total.cpu().numpy(), n)
        np.testing.assert_allclose(expected, [np.corrcoef(a[:, k].numpy(), b[:, k].numpy())[0,1] for k in range(5)], atol=1e-6)

    def test_role_masks_from_real_local_rendered_example(self):
        snapshot = "/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e"
        tok = collect133.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        roles, rendered = collect133.role_markers(tok)
        ids = torch.tensor([tok(rendered, add_special_tokens=False)["input_ids"]])
        masks = collect133.role_masks(ids, torch.ones_like(ids, dtype=torch.bool), roles)
        self.assertGreater(int(masks["chat-user"].sum()), 0)
        self.assertGreater(int(masks["chat-assistant"].sum()), 0)
        self.assertFalse(bool((masks["chat-user"] & masks["chat-assistant"]).any()))
        self.assertTrue(all(roles[k] for k in ("user", "assistant")))

    def test_float32_batch_reductions_added_to_float64_totals(self):
        torch.manual_seed(4); x = torch.randn(250, 7); y = torch.randn(250, 7)
        total = torch.zeros((5, 7), dtype=torch.float64)
        for lo, hi in ((0, 80), (80, 180), (180, 250)):
            batch, _ = collect133.sufficient(x[lo:hi], y[lo:hi], torch.ones(hi-lo, dtype=torch.bool))
            self.assertEqual(batch.dtype, torch.float32)
            total += batch.cpu().to(torch.float64)
        self.assertEqual(total.dtype, torch.float64)
        expected = torch.stack((x.sum(0), y.sum(0), x.square().sum(0), y.square().sum(0), (x*y).sum(0))).double()
        torch.testing.assert_close(total, expected, atol=2e-5, rtol=2e-6)

    def test_resume_payload_restores_identical_totals(self):
        torch.manual_seed(12); x = torch.randn(10, 5); y = torch.randn(10, 5)
        uninterrupted = torch.zeros((5, 5), dtype=torch.float64)
        for lo, hi in ((0, 4), (4, 8), (8, 10)):
            red, _ = collect133.sufficient(x[lo:hi], y[lo:hi], torch.ones(hi-lo, dtype=torch.bool))
            uninterrupted += red.cpu().double()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "resume.pt"
            first, _ = collect133.sufficient(x[:4], y[:4], torch.ones(4, dtype=torch.bool))
            state = torch.zeros((5, 5), dtype=torch.float64); state += first.cpu().double()
            collect133.atomic_save(path, {"meta": {"fixture": True}, "totals": state, "next_doc": 4})
            restored = torch.load(path, map_location="cpu", weights_only=False)["totals"]
            for lo, hi in ((4, 8), (8, 10)):
                red, _ = collect133.sufficient(x[lo:hi], y[lo:hi], torch.ones(hi-lo, dtype=torch.bool))
                restored += red.cpu().double()
            torch.testing.assert_close(restored, uninterrupted, rtol=0, atol=0)

    def test_tiny_olmo_style_down_proj_input_hook(self):
        torch.manual_seed(3); model = TinyOlmo2Style(); seen = []
        handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: seen.append(inp[0].detach().clone()))
        x = torch.randn(2, 4, 5); model(x); handle.remove()
        self.assertEqual(len(seen), 1); torch.testing.assert_close(seen[0], x)

    def test_scorer_fixture_and_100_position_floor(self):
        rng = np.random.default_rng(7); layers, width = 2, 4
        totals = torch.zeros((3,4,layers,5,width), dtype=torch.float64)
        counts = torch.full((3,4,layers), 180, dtype=torch.int64)
        active = torch.full((3,4,layers,2,width), 150, dtype=torch.int64)
        for gi in range(3):
            for pi in range(4):
                for li in range(layers):
                    x = rng.normal(size=(180,width)); y=x*.95+rng.normal(scale=.1,size=(180,width))
                    totals[gi,pi,li] = torch.from_numpy(stats(x,y))
        active[0,0,0,:,0] = 99
        nstats = torch.stack([torch.from_numpy(stats((x:=rng.normal(size=(200,width))), x+rng.normal(scale=.01,size=(200,width)))) for _ in range(layers)])
        data = {"totals": totals, "counts": counts, "active": active,
                "null_totals": nstats, "null_counts": torch.full((layers,),200),
                "self_totals": nstats, "self_counts": torch.full((layers,),200), "meta": {"fixture": True},
                "part92_neurons": [{"id":"concept-0","layer":0,"neuron":1},{"id":"slot-0","layer":1,"neuron":2}]}
        classes = {"kinds": {"concept-0":"concept", "slot-0":"slot"}}
        result = score133.score_data(data, classes)
        self.assertEqual(result["excluded_below_100_nonzero_positions"]["web"]["parent-sft:layer-0"], 1)
        self.assertTrue(result["gates"]["gate1_instruct_self"]["pass"])
        self.assertIsNotNone(result["predictions"]["133.1"]["parent-sft"]["web_median_r"])

    def test_deliberate_break_on_temporary_copy_turns_a_test_red(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); module = td / "score133.py"
            src = (HERE / "score133.py").read_text(encoding="utf-8")
            marker = "return np.divide(xy - sx*sy/n, denom, out=np.full_like(sx, np.nan), where=(vx > 0) & (vy > 0))"
            self.assertIn(marker, src)
            module.write_text(src.replace(marker, "return np.full_like(sx, np.nan)"), encoding="utf-8")
            test = td / "test_broken.py"
            test.write_text("""import importlib.util, numpy as np\ns=importlib.util.spec_from_file_location('broken', r'%s')\nm=importlib.util.module_from_spec(s); s.loader.exec_module(m)\nx=np.array([[0.],[1.],[2.]])\nd=np.stack((x.sum(0),x.sum(0), (x*x).sum(0),(x*x).sum(0),(x*x).sum(0)))\nassert m.correlations(d,3)[0] == 1.0\n""" % module, encoding="utf-8")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test)],
                                  cwd=td, env=env, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0, "deliberate scorer break unexpectedly passed")


if __name__ == "__main__": unittest.main()
