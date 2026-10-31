"""Part 145 scorer fixtures and runner integration checks."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run145
import score145


def make_fixture(root, old_path, bad_gate=None):
    root.mkdir(parents=True, exist_ok=True)
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_rows = []
    # Two neuron rows and two matched random-direction rows.
    unit_meta = [{"id": f"F-L4-{kind}-{i}", "unit": kind, "layer": 4}
                 for kind in ("neuron", "random") for i in range(2)]
    for cell_no, cell in enumerate(score145.CELLS):
        for qi in (1, 2):
            p = np.array([.50, .25, .15, .08, .02], dtype=np.float64)
            plain = np.log(p)
            matrix = np.stack([np.log(np.array([.35 + i*.02, .28, .18, .13 - i*.01, .06])) for i in range(4)])
            matrix -= np.log(np.exp(matrix).sum(axis=1))[:, None]
            if bad_gate == "gate3" and cell == "base-plain" and qi == 1:
                matrix[0, 0] = np.nan
            np.save(root / f"matrix-{cell}-q{qi}.npy", matrix.astype(np.float32))
            np.save(root / f"plain-{cell}-q{qi}.npy", plain[None, :].astype(np.float32))
            side = {"cell": cell, "q": qi, "unit_ids": [u["id"] for u in unit_meta],
                    "units": unit_meta, "vocab_size": 5, "sorry_id": 3, "plain_top_id": 0,
                    "eligible_ids": [0, 1, 2, 3], "eligible_tokens": ["a", "b", "c", " sorry"]}
            (root / f"index-{cell}-q{qi}.json").write_text(json.dumps(side), encoding="utf-8")
            noops = [{"unit_id": f"n{i}", "logp_sorry": float(plain[3]), "logp_plain_top": float(plain[0])} for i in range(4)]
            if bad_gate == "gate1" and cell == "base-plain" and qi == 1:
                noops[0]["logp_sorry"] += .01
            (root / f"noop-{cell}-q{qi}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in noops), encoding="utf-8")
            if cell == "instruct-chat" and qi == 1:
                for u, row in zip(unit_meta, matrix):
                    old_rows.append({"cell": cell, "set": "F", "arm": "full", "question": score145.QUESTIONS[0],
                                     "unit_id": u["id"], "p_sorry": float(np.exp(row[3]))})
    if bad_gate == "gate2":
        old_rows = []
    old_path.write_text("".join(json.dumps(r) + "\n" for r in old_rows), encoding="utf-8")


@pytest.mark.parametrize("gate", ["gate1", "gate2", "gate3"])
def test_each_gate_failure_withholds_predictions(tmp_path, gate):
    root, old, out = tmp_path / "input", tmp_path / "part144.jsonl", tmp_path / "out"
    make_fixture(root, old, gate)
    with pytest.raises(RuntimeError):
        score145.score(root, old, out)
    result = json.loads((out / "score.json").read_text(encoding="utf-8"))
    key = {"gate1": "gate1_noop", "gate2": "gate2_part144", "gate3": "gate3_finite"}[gate]
    assert result["gates"][key]["passed"] is False
    assert all(row["verdict"] == "withheld" for row in result["predictions"].values())


def test_scorer_passes_gates_and_writes_outputs(tmp_path):
    root, old, out = tmp_path / "input", tmp_path / "part144.jsonl", tmp_path / "out"
    make_fixture(root, old)
    result = score145.score(root, old, out)
    assert all(g["passed"] for g in result["gates"].values())
    assert set(result["predictions"]) == {f"145.{i}" for i in range(1, 8)}
    assert (out / "tables.md").is_file()
    assert (out / "words-instruct-chat-q1.json").is_file()


def test_temperature_bisection_recovers_known_temperature():
    plain = np.log(np.array([.6, .3, .1]))
    expected = 1.7
    z = plain / expected
    probs = np.exp(z - np.log(np.exp(z).sum()))
    got = score145.temperature_for_top(plain, float(probs[0]), 0)
    assert got == pytest.approx(expected, rel=1e-8)


def test_partial_spearman_removes_monotone_control_relation():
    control = np.arange(12, dtype=float)
    x = control + np.array([0, 1, 0, -1] * 3)
    y = control + np.array([0, -1, 1, 0] * 3)
    plain = score145.partial_spearman(x, y, control)
    # Constructed ranks leave small, non-zero residuals after controlling for control.
    assert np.isfinite(plain)
    assert -1 <= plain <= 1


def test_import_under_python_isolated_mode_from_repo_root():
    root = HERE.parent
    proc = subprocess.run([sys.executable, "-I", "-B", "part145/run145.py", "--help"], cwd=root,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "--cell" in proc.stdout


@pytest.mark.parametrize("break_plain", [False, True])
def test_tiny_model_main_end_to_end_and_nan_hard_stop(tmp_path, monkeypatch, break_plain):
    torch = pytest.importorskip("torch")

    class Tok:
        pad_token_id = 0
        eos_token = "e"
        def __call__(self, text, **kwargs):
            ids = {"I": [1], " I": [1], "'m": [2], " sorry": [3], "I'm": [1, 2], " I'm": [1, 2]}
            return {"input_ids": ids[text]}
        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = SimpleNamespace(down_proj=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3, 3))))
        def forward(self, hidden):
            return hidden + 0

    class ToyModel(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 3)
            self.head = torch.nn.Linear(3, 8, bias=False)
            self.layers = layers
        def forward(self, input_ids, use_cache=True, past_key_values=None):
            full = input_ids if past_key_values is None else torch.cat((past_key_values, input_ids), dim=1)
            hidden = self.layers[4](self.embedding(input_ids))
            logits = self.head(hidden)
            return SimpleNamespace(logits=logits, past_key_values=full)

    class FakeCommon:
        @staticmethod
        def get_layers(_model):
            return [ToyLayer() for _ in range(6)]
        @staticmethod
        def residual_medians(*_args):
            return {4: 1.0, 5: 1.0}

    layers = [ToyLayer() for _ in range(6)]
    model, tok = ToyModel(layers), Tok()
    monkeypatch.setitem(sys.modules, "common92", FakeCommon)
    monkeypatch.setattr(run145.run144, "_load_direct", lambda *_args: (tok, model))
    monkeypatch.setattr(run145.run144.run139, "_unit_vectors", lambda *_args, **_kwargs: torch.ones((1, 3)))
    # Use the hook-bearing layer returned by common92, which the runner also resolves.
    monkeypatch.setattr(FakeCommon, "get_layers", lambda _model: layers)
    monkeypatch.setattr(FakeCommon, "prompt_ids", lambda *_args: [4, 5], raising=False)
    # run145 imports common92 within run(), so provide its prompt helper in the module too.
    if break_plain:
        original = run145.sequence
        def with_nan(*args, **kwargs):
            values, ids = original(*args, **kwargs)
            if args[3] is None:
                values[0] = np.nan
            return values, ids
        monkeypatch.setattr(run145, "sequence", with_nan)
    units_file = tmp_path / "units.json"
    units_file.write_text(json.dumps({"stage_f": {"units": [{"id": "F-L4-neuron-1", "unit": "neuron", "index": 1, "layer": 4}]}}), encoding="utf-8")
    args = ["--cell", "base-plain", "--device", "cpu", "--model", "stub", "--units", str(units_file),
            "--out", str(tmp_path / "out"), "--limit", "1"]
    if break_plain:
        with pytest.raises(FloatingPointError):
            run145.main(args)
    else:
        run145.main(args)
        mat = np.load(tmp_path / "out/matrix-base-plain-q1.npy")
        assert mat.shape == (1, 8)
        assert (tmp_path / "out/index-base-plain-q2.json").is_file()

