"""Part 192 wrapper tests: eager loading, fail-closed records, and append-only run entries."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part192 import run192  # noqa: E402


def test_loader_passes_and_records_eager(monkeypatch):
    import torch
    import neuron_report as nr

    seen = {}

    class Tok:
        pad_token_id = 0
        eos_token = "<eos>"

    class Model:
        config = types.SimpleNamespace(_attn_implementation="eager")

        def eval(self):
            return self

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["tokenizer"] = (model_id, kwargs)
            return Tok()

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["model"] = (model_id, kwargs)
            return Model()

    modeling_utils = types.SimpleNamespace(caching_allocator_warmup=lambda *a, **k: None)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.modeling_utils = modeling_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch, "set_grad_enabled", lambda enabled: None)
    nr.LAST_LOAD = None

    tok, model = run192.eager_load_model_direct("model-id", "rev", "cpu")

    assert isinstance(tok, Tok)
    assert isinstance(model, Model)
    assert seen["model"][1]["attn_implementation"] == "eager"
    assert nr.LAST_LOAD["requested_attn_implementation"] == "eager"
    assert nr.LAST_LOAD["attn_implementation"] == "eager"


@pytest.mark.parametrize("kernel, expected", [("eager", (True, "")),
                                                ("sdpa", (False, "non-eager")),
                                                (None, (False, "missing attention"))])
def test_gate_refuses_non_eager_or_missing_kernel(kernel, expected):
    ok, why = run192.check_kernels([kernel])
    assert ok is expected[0]
    if expected[1]:
        assert expected[1] in why


def test_gate_refuses_missing_record(tmp_path):
    assert run192.check_record(tmp_path / "missing.json") == 3
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"loads": []}), encoding="utf-8")
    assert run192.check_record(path) == 3


def test_record_has_one_entry_per_invocation_that_wrote_rows(tmp_path, monkeypatch):
    import part39
    import neuron_report as nr

    def fake_loader(model_id, revision, device):
        nr.LAST_LOAD = {"model": model_id, "revision": revision,
                        "attn_implementation": "eager"}
        return object(), object()

    calls = {"n": 0}

    def fake_main(argv):
        args = part39.build_parser().parse_args(argv)
        part39.load_model_direct("model-id", "rev", "xpu")
        calls["n"] += 1
        if calls["n"] <= 2:
            args.out.mkdir(parents=True, exist_ok=True)
            with (args.out / "zero.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"row_key": f"row-{calls['n']}"}) + "\n")

    monkeypatch.setattr(run192, "eager_load_model_direct", fake_loader)
    monkeypatch.setattr(part39, "main", fake_main)
    out = tmp_path / "part192"
    command = ["zero", "--out", str(out)]

    assert run192.main(command, part39_module=part39) == 0
    assert run192.main(command, part39_module=part39) == 0
    assert run192.main(command, part39_module=part39) == 0  # loads but writes no new rows
    record = json.loads((out / "load.json").read_text(encoding="utf-8"))
    assert len(record["loads"]) == 2
    assert [entry["rows_added"] for entry in record["loads"]] == [1, 1]
    assert all(entry["model_loads"][0]["attn_implementation"] == "eager"
               for entry in record["loads"])
