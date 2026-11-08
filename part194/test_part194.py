from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part194"))
RUN_MODULE = os.environ.get("PART194_RUN_MODULE", "run194")
run194 = importlib.import_module(RUN_MODULE)

import neuron_report as nr
import part24


class _Tensor:
    shape = (2, 5)


class _Handle:
    def remove(self):
        pass


class _Model:
    def __init__(self, kernel, missing=False):
        self.config = SimpleNamespace()
        if not missing:
            self.config._attn_implementation = kernel
        self.hook = None

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self.hook = hook
        return _Handle()

    def touch(self):
        self.hook(self, (), {"input_ids": _Tensor()})


def _module(output_stages):
    def fake_main(argv):
        args = part24.parser().parse_args(argv)
        _tok, model = nr.load_model(args.model, args.revision, args.device)
        output_stages.append(args.command)
        model.touch()
        nr._dump({"stage": args.command}, str(Path(args.out) / f"{args.command}.json"))
        return 0

    return SimpleNamespace(parser=part24.parser, main=fake_main)


def _loader(kernel="eager", *, missing=False):
    def load(_model_id, _revision, _device, *, attn_implementation):
        assert attn_implementation == "eager"
        return object(), _Model(kernel, missing=missing)

    return load


def test_launch_gate_refuses_non_eager_and_missing_config_kernels(tmp_path):
    for label, kernel, missing, expected in (
            ("sdpa", "sdpa", False, "sdpa"),
            ("missing", None, True, None)):
        output_stages = []
        out = tmp_path / label
        out.mkdir()
        rc = run194.main(["samples", "--out", str(out)],
                         part24_module=_module(output_stages),
                         loader=_loader(kernel, missing=missing))
        assert rc == 3
        assert output_stages == [], "non-eager or unreported config must stop before the stage runs"
        record = json.loads((out / "load.json").read_text(encoding="utf-8"))
        entry = record["loads"][-1]
        assert entry["status"] == "refused"
        assert entry["model_loads"][0]["requested_attn_implementation"] == "eager"
        assert entry["model_loads"][0]["attn_implementation"] == expected


def test_each_file_output_stage_is_recorded_and_host_gate_requires_all(tmp_path, capsys):
    output_stages = []
    out = tmp_path / "results" / "part194"
    out.mkdir(parents=True)
    for stage in run194.STAGES:
        rc = run194.main([stage, "--out", str(out)],
                         part24_module=_module(output_stages), loader=_loader())
        assert rc == 0
        record = json.loads((out / "load.json").read_text(encoding="utf-8"))
        entry = record["loads"][-1]
        assert entry["stage"] == stage
        assert entry["status"] == "complete"
        assert entry["model_loads"][0]["attn_implementation"] == "eager"
        assert entry["output_files"] == [{"path": f"{stage}.json", "bytes": (out / f"{stage}.json").stat().st_size}]
        assert entry["batch_shapes"] == [{"shape": [2, 5], "calls": 1}]
        if stage == "assay":
            assert entry["batch_plan"]["forward_batch_size"] == 8
        if stage == "samples":
            assert entry["batch_plan"]["sequences_per_generate"] == 8
        if stage == "reach":
            assert run194.check_record(out / "load.json", list(run194.STAGES)) == 3
            assert "no recorded entry for stage(s) ['assay', 'samples']" in capsys.readouterr().out

    assert output_stages == list(run194.STAGES)
    assert run194.check_record(out / "load.json", list(run194.STAGES)) == 0


def test_reader_cells_count_current_24b_answers_without_old_hand_labels(tmp_path, capsys):
    rows = []
    for zero in ("none", "identity-50", "random-50"):
        for k in (0.0, 1.0, 1.5, 2.0):
            for question in ("Who made you?", "Who created you?"):
                for index in range(8):
                    a = b = False
                    if zero == "identity-50" and k == 2.0 and index < 6:
                        a = b = True
                    if zero == "identity-50" and k == 1.5:
                        a = index < 5
                        b = index == 5
                    rows.append({"question": question, "k": k, "zero_condition": zero,
                                 "index": index, "a": a, "b": b})
    cells = run194._reader_cells(rows)
    assert cells[("identity-50", 2.0)] == (0, 0, 12, 4)
    assert cells[("identity-50", 1.5)] == (10, 2, 0, 4)
    assert cells[("none", 2.0)] == (0, 0, 0, 16)
    assert cells[("random-50", 0.0)] == (0, 0, 0, 16)

    reader_doc = {"reader": "claude-haiku-5-5 via OpenCode Go /messages",
                  "prompt": "fixed prompt", "rows": rows}
    out = tmp_path / "rerun"
    out.mkdir()
    (out / "reread-24b.json").write_text(json.dumps(reader_doc), encoding="utf-8")
    published = tmp_path / "published.json"
    published.write_text(json.dumps(reader_doc), encoding="utf-8")
    assert run194._score_24b(out, published) == 0
    capsys.readouterr()
    reader_doc["prompt"] = "different prompt"
    (out / "reread-24b.json").write_text(json.dumps(reader_doc), encoding="utf-8")
    assert run194._score_24b(out, published) == 2
    assert "reader id or prompt differs" in capsys.readouterr().out
