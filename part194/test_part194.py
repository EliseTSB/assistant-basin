from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part194"))
RUN_MODULE = os.environ.get("PART194_RUN_MODULE", "run194")
run194 = importlib.import_module(RUN_MODULE)

import neuron_report as nr
import part24


class _Tensor:
    def __init__(self, shape=(2, 5)):
        self.shape = shape


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

    def touch(self, shape=(2, 5)):
        self.hook(self, (), {"input_ids": _Tensor(shape)})

    def generate(self, **kwargs):
        sequences = kwargs["num_return_sequences"]
        self.touch((sequences, 30))
        self.touch((sequences, 1))
        return kwargs


def _module(output_stages):
    def fake_main(argv):
        args = part24.parser().parse_args(argv)
        _tok, model = nr.load_model(args.model, args.revision, args.device)
        output_stages.append(args.command)
        if args.command == "samples":
            model.generate(num_return_sequences=args.samples, max_new_tokens=args.max_new)
            model.generate(num_return_sequences=3, max_new_tokens=17)
        else:
            model.touch((2, 5))
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
        assert entry["run_id"]
        assert entry["model_loads"][0]["attn_implementation"] == "eager"
        output_path = out / f"{stage}.json"
        output_bytes = output_path.read_bytes()
        assert entry["output_files"] == [{"path": f"{stage}.json", "bytes": len(output_bytes),
                                           "sha256": hashlib.sha256(output_bytes).hexdigest()}]
        if stage == "samples":
            assert entry["batch_shapes"] == [
                {"shape": [3, 1], "calls": 1},
                {"shape": [3, 30], "calls": 1},
                {"shape": [8, 1], "calls": 1},
                {"shape": [8, 30], "calls": 1},
            ]
            assert entry["generate_calls"] == [
                {"num_return_sequences": 8, "max_new_tokens": 64},
                {"num_return_sequences": 3, "max_new_tokens": 17},
            ]
        else:
            assert entry["batch_shapes"] == [{"shape": [2, 5], "calls": 1}]
            assert entry["generate_calls"] == []
        if stage == "assay":
            assert entry["batch_plan"]["forward_batch_size"] == 8
        if stage == "samples":
            assert entry["batch_plan"]["sequences_per_generate"] == 8
        if stage == "reach":
            assert run194.check_record(out / "load.json", list(run194.STAGES)) == 3
            assert "no recorded entry for stage(s) ['assay', 'samples']" in capsys.readouterr().out

    assert output_stages == list(run194.STAGES)
    assert run194.check_record(out / "load.json", list(run194.STAGES)) == 0

    record_path = out / "load.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    run_ids = {entry["run_id"] for entry in record["loads"]}
    assert len(run_ids) == 1

    for field, replacement in (("run_id", "different-run"), ("model", "different-model"),
                               ("revision", "different-revision")):
        changed_record = json.loads(json.dumps(record))
        last_load = changed_record["loads"][-1]
        if field == "run_id":
            last_load[field] = replacement
        else:
            last_load["model_loads"][0][field] = replacement
        record_path.write_text(json.dumps(changed_record), encoding="utf-8")
        assert run194.check_record(record_path, list(run194.STAGES)) == 3
        assert "do not share one model, revision, and run id" in capsys.readouterr().out

    changed_metadata = json.loads(json.dumps(record))
    changed_metadata["part194"]["run_id"] = "different-record-run"
    record_path.write_text(json.dumps(changed_metadata), encoding="utf-8")
    assert run194.check_record(record_path, list(run194.STAGES)) == 3
    assert "do not share one model, revision, and run id" in capsys.readouterr().out

    record_path.write_text(json.dumps(record), encoding="utf-8")
    reach_path = out / "reach.json"
    original_reach = reach_path.read_bytes()
    swapped_reach = original_reach.replace(b"reach", b"Reach", 1)
    assert len(swapped_reach) == len(original_reach)
    reach_path.write_bytes(swapped_reach)
    assert run194.check_record(record_path, list(run194.STAGES)) == 3
    assert "reach has no recorded reach.json" in capsys.readouterr().out
    reach_path.write_bytes(original_reach)
    assert run194.check_record(record_path, list(run194.STAGES)) == 0


def test_reader_cells_count_current_24b_answers_without_old_hand_labels(tmp_path, capsys):
    rows = []
    b_counts = {
        "none": {0.0: 1, 1.0: 2, 1.5: 3, 2.0: 9},
        "random-50": {0.0: 4, 1.0: 5, 1.5: 6, 2.0: 10},
    }
    for zero in ("none", "identity-50", "random-50"):
        for k in (0.0, 1.0, 1.5, 2.0):
            for question_index, question in enumerate(("Who made you?", "Who created you?")):
                for index in range(8):
                    answer_index = question_index * 8 + index
                    a = b = False
                    if zero in b_counts:
                        b = answer_index < b_counts[zero][k]
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
    assert cells[("none", 2.0)] == (0, 9, 0, 7)
    assert cells[("random-50", 0.0)] == (0, 4, 0, 12)

    wrong_questions = [dict(row) for row in rows]
    for row in wrong_questions:
        if row["zero_condition"] == "none" and row["k"] == 0.0 and row["question"] == "Who made you?":
            row["question"] = "Who created you?"
    with pytest.raises(ValueError, match="8 rows from each question"):
        run194._reader_cells(wrong_questions)

    repeated_indices = [dict(row) for row in rows]
    for row in repeated_indices:
        if row["zero_condition"] == "none" and row["k"] == 0.0 and row["question"] == "Who made you?":
            row["index"] = 0
    with pytest.raises(ValueError, match="indices 0..7 once per question"):
        run194._reader_cells(repeated_indices)

    reader_doc = {"reader": "claude-haiku-5-5 via OpenCode Go /messages",
                  "prompt": "fixed prompt", "rows": rows}
    out = tmp_path / "rerun"
    out.mkdir()
    (out / "reread-24b.json").write_text(json.dumps(reader_doc), encoding="utf-8")
    published = tmp_path / "published.json"
    published.write_text(json.dumps(reader_doc), encoding="utf-8")
    assert run194._score_24b(out, published) == 0
    scoring_output = capsys.readouterr().out
    assert "194.7 24b.2 identity k=1.5 (a)-only share among alien: 0.833 (10/12; published 10/12)" in scoring_output
    assert "194.9 24b.4 (b) counts for none/random at k=0,1,1.5,2:" in scoring_output
    assert ("none=['1/16', '2/16', '3/16', '9/16']; random=['4/16', '5/16', '6/16', '10/16']; "
            "published none=['1/16', '2/16', '3/16', '9/16']; "
            "random=['4/16', '5/16', '6/16', '10/16']") in scoring_output
    assert "24b.4 (b) at most 2/16 in none/random at every k  -> failed" in scoring_output
    assert ("194.9 (b): k=2 none/random in 6-12/16: 9/16 and 10/16 (held); "
            "k=0,1,1.5 none/random at most 2/16: 1/16,2/16,3/16; 4/16,5/16,6/16 (failed) -> failed") in scoring_output

    def score_with_b_counts(target_counts):
        changed_rows = [dict(row) for row in rows]
        for zero in ("none", "random-50"):
            for k, target in target_counts[zero].items():
                selected = [row for row in changed_rows
                            if row["zero_condition"] == zero and row["k"] == k]
                for index, row in enumerate(selected):
                    row["b"] = index < target
        changed_doc = {**reader_doc, "rows": changed_rows}
        (out / "reread-24b.json").write_text(json.dumps(changed_doc), encoding="utf-8")
        assert run194._score_24b(out, published) == 0
        return capsys.readouterr().out

    held_1949 = score_with_b_counts({
        "none": {0.0: 0, 1.0: 1, 1.5: 2, 2.0: 9},
        "random-50": {0.0: 1, 1.0: 0, 1.5: 2, 2.0: 10},
    })
    assert "k=0,1,1.5 none/random at most 2/16: 0/16,1/16,2/16; 1/16,0/16,2/16 (held) -> held" in held_1949

    bad_k2_1949 = score_with_b_counts({
        "none": {0.0: 0, 1.0: 1, 1.5: 2, 2.0: 5},
        "random-50": {0.0: 1, 1.0: 0, 1.5: 2, 2.0: 10},
    })
    assert "k=2 none/random in 6-12/16: 5/16 and 10/16 (failed)" in bad_k2_1949
    assert "(held) -> failed" in bad_k2_1949

    reader_doc["prompt"] = "different prompt"
    (out / "reread-24b.json").write_text(json.dumps(reader_doc), encoding="utf-8")
    assert run194._score_24b(out, published) == 2
    assert "reader id or prompt differs" in capsys.readouterr().out


def test_score_core_reports_null_metric_as_failure(tmp_path, capsys):
    out = tmp_path / "part24"
    out.mkdir()
    reach = json.loads((ROOT / "results/part24/reach.json").read_text(encoding="utf-8"))
    assay = json.loads((ROOT / "results/part24/assay.json").read_text(encoding="utf-8"))
    reach["summary"]["identity-50"]["2.0"]["median_relative_change_layers_above_target"] = None
    (out / "reach.json").write_text(json.dumps(reach), encoding="utf-8")
    (out / "assay.json").write_text(json.dumps(assay), encoding="utf-8")

    assert run194._score_core(out) == 2
    output = capsys.readouterr().out
    assert output.startswith("PART194 SCORE FAILED:")
    assert "24.2 identity median is null" in output
    assert "24.1 max relative change" not in output


def test_score_core_prints_metrics_for_predictions_194_3_and_194_4(capsys):
    assert run194._score_core(ROOT / "results/part24") == 0
    output = capsys.readouterr().out
    assert "24.3 identity median name retention at k=2 0.583 (random 0.647)" in output
    assert ("24.4 Alibaba Cloud per token: identity -3.136, random -2.960, none -2.885; "
            "extra drop 0.176") in output
