"""Part 144 scorer fixtures and runner phase checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run144
import score144


def fixture():
    data = {}
    old = []
    for ci, cell in enumerate(score144.CELLS):
        rows = []
        base = .02 + ci * .001
        rows.append({"key": cell + "|plain", "cell": cell, "arm": "plain", "set": "plain", "unit": "plain", "unit_id": "plain",
                     "p_first": .1, "p_m": .2, "p_sorry": base,
                     "token_checks": {"im_ids": [1, 2], "first": "I" if cell.endswith("chat") else " I", "m": "'m", "sorry": " sorry"}})
        for set_name, units in (("R", [("r1", "neuron"), ("r2", "random")]),
                                ("S", [("s1", "neuron")]), ("F", [("f1", "neuron"), ("f2", "random")])):
            for uid, typ in units:
                for arm in ("full", "prompt-only", "answer-only", "noop"):
                    # Noop exactly matches baseline; answer-only preserves the first read.
                    p_first = .1 if arm != "prompt-only" else .2
                    row = {"key": f"{cell}|{uid}|{arm}", "cell": cell, "arm": arm, "set": set_name,
                           "unit": typ, "unit_id": uid, "p_first": p_first,
                           "p_m": .2 if arm == "noop" else .25, "p_sorry": base if arm == "noop" else base + .03,
                           "token_checks": {"im_ids": [1, 2], "first": "I" if cell.endswith("chat") else " I", "m": "'m", "sorry": " sorry"}}
                    rows.append(row)
                    if cell == "instruct-chat" and set_name == "R" and arm in ("full", "prompt-only"):
                        old.append({"readout": "im_after_i", "arm": arm, "unit_id": uid, "set": "R", "probability": .25})
                        old.append({"readout": "sorry_after_im", "arm": arm, "unit_id": uid, "set": "R", "probability": base + .03})
        data[cell] = rows
    return data, old


def test_score_passes_gates_and_emits_all_predictions():
    rows, old = fixture()
    result = score144.score(rows, old)
    assert all(g["passed"] for g in result["gates"].values())
    assert set(result["predictions"]) == {f"144.{i}" for i in range(1, 10)}
    assert "stats" in result


@pytest.mark.parametrize("gate", ["gate1_noop_and_token_checks", "gate2_matches_part143", "gate3_answer_only_first_and_tokens"])
def test_each_gate_failure_withholds_predictions(gate):
    rows, old = fixture()
    if gate.startswith("gate1"):
        next(r for r in rows["base-plain"] if r["arm"] == "noop")["p_sorry"] += .01
    elif gate.startswith("gate2"):
        old.clear()
    else:
        next(r for r in rows["base-plain"] if r["arm"] == "answer-only")["p_first"] += 1e-4
    result = score144.score(rows, old)
    assert not result["gates"][gate]["passed"]
    assert "stats" not in result
    assert all(p["verdict"] == "withheld" for p in result["predictions"].values())


def test_runner_main_end_to_end_stub_loader_and_resume(tmp_path, monkeypatch):
    class FakeVector:
        def reshape(self, *_):
            return self
    class FakeTorch:
        @staticmethod
        def zeros_like(value):
            return value
    class Tok:
        pad_token_id = 0
        eos_token = "e"
        def __call__(self, text, **_):
            return {"input_ids": {"I": [1], " I": [1], "'m": [2], " sorry": [3], "I'm": [1, 2], " I'm": [1, 2]}[text]}
    class Layer:
        class MLP:
            class Down:
                weight = object()
            down_proj = Down()
        mlp = MLP()
    class Model:
        class Inner:
            layers = [Layer() for _ in range(6)]
        model = Inner()

    torch_mod = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(run144, "_load_direct", lambda *_: (Tok(), Model()))
    monkeypatch.setattr(run144.c, "residual_medians", lambda *_: {4: 2.0, 5: 2.0})
    monkeypatch.setattr(run144.run139, "_unit_vectors", lambda *_args, **_kwargs: [FakeVector()])
    monkeypatch.setattr(run144.c, "get_layers", lambda m: m.model.layers)
    monkeypatch.setattr(run144.c, "prompt_ids", lambda *_: [8, 9])
    monkeypatch.setattr(run144, "sequence", lambda *args: {"p_first": .1, "p_m": .2, "p_sorry": .3,
                                                             "top5_first": [], "top5_m": [], "top5_sorry": [],
                                                             "token_checks": {"im_ids": [1, 2], "first": "I", "m": "'m", "sorry": " sorry"}})
    units_path = tmp_path / "units.json"
    units_path.write_text(json.dumps({"sets": {"R": [{"id": "r0", "set": "R", "unit": "neuron", "index": 0, "layer": 4}], "S": []}, "stage_f": {"units": []}}))
    argv = ["--cell", "base-chat", "--model", "stub", "--units", str(units_path), "--out", str(tmp_path)]
    run144.main(argv)
    path = tmp_path / "base-chat.jsonl"
    first = path.read_text().splitlines()
    assert len(first) == 5  # plain plus four arms
    run144.main(argv)
    assert len(path.read_text().splitlines()) == len(first)


def test_imports_under_python_isolated_mode():
    import subprocess
    root = HERE.parent
    code = "import sys; sys.path.insert(0, 'part144'); import run144, score144"
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_tiny_model_phase_effects():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Block(torch.nn.Module):
        def forward(self, x):
            return x
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Block()
        def forward(self, input_ids, past_key_values=None, use_cache=True):
            call = 0 if past_key_values is None else past_key_values
            hidden = self.layer(torch.zeros((1, input_ids.shape[1], 3)))
            logits = torch.zeros((1, input_ids.shape[1], 5))
            logits[0, -1, call + 1] = hidden[0, -1].sum()
            return SimpleNamespace(logits=logits, past_key_values=call + 1)
    class TinyTok:
        def __call__(self, text, **_):
            ids = {"I": [1], " I": [1], "'m": [2], " sorry": [3], "I'm": [1, 2], " I'm": [1, 2]}[text]
            return {"input_ids": ids}
        def decode(self, ids, **_):
            return str(ids[0])
    vector = torch.ones(3)
    original = run144.c.prompt_ids
    run144.c.prompt_ids = lambda *_: [4, 4]
    try:
        # Each sequence must use the layer owned by the model.
        def measure(arm, pushed):
            model = Tiny()
            return run144.sequence(model, TinyTok(), model.layer, pushed, arm, "chat", "cpu")
        plain = measure("full", None)
        prompt = measure("prompt-only", vector)
        answer = measure("answer-only", vector)
        noop = measure("noop", torch.zeros_like(vector))
    finally:
        run144.c.prompt_ids = original
    assert prompt["p_first"] != plain["p_first"]
    assert answer["p_first"] == plain["p_first"]
    assert answer["p_sorry"] != plain["p_sorry"]
    assert noop["p_first"] == plain["p_first"] and noop["p_m"] == plain["p_m"] and noop["p_sorry"] == plain["p_sorry"]
