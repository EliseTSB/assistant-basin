"""CPU-only tests for Part 195's load gate and its recorded file stages."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part195 import run195  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _record_entry(stage: str, *, gpu: bool = False) -> dict:
    return {"stage": stage, "status": "complete", "requires_model_load": gpu,
            "model_loads": ([{"attn_implementation": "eager"}] if gpu else []),
            "batch_shapes": {"planned": {}, "observed": {}}, "outputs": []}


def _seed_record(directory: Path, expected: list[str]) -> None:
    path = directory.parent / "load.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"part195": {}, "stages": [
        _record_entry(stage, gpu=stage in run195.GPU_STAGES) for stage in expected
    ]}), encoding="utf-8")


@pytest.mark.parametrize("configured,expected_rc", [
    ("eager", 0), ("sdpa", 3), (None, 3), ("missing", 3),
])
def test_wrapper_persists_config_kernel_before_battery_and_gates(configured, expected_rc, tmp_path, monkeypatch):
    """The launched wrapper stops before the runner can generate if config is non-eager or omits its setting."""
    from part81 import part81
    import neuron_report as nr

    class Model:
        config = types.SimpleNamespace() if configured == "missing" else \
            types.SimpleNamespace(_attn_implementation=configured)

        def forward(self, input_ids=None):
            return input_ids

    def fake_loader(*_args, **_kwargs):
        # Simulate neuron_report's tempting requested-value fallback. Part 195 must replace it from config.
        nr.LAST_LOAD = {"model": "fixture", "revision": "r1", "device": "xpu",
                        "attn_implementation": "eager", "attn_implementation_requested": "eager"}
        return object(), Model()

    started = {"generation": False}

    def fake_part81_main():
        import json as json_module
        import sys as sys_module
        out = Path(sys_module.argv[sys_module.argv.index("--out") + 1])
        _tok, model = nr.load_model("fixture", "r1", "xpu", attn_implementation="eager")
        saved = json_module.loads((out.parent / "load.json").read_text(encoding="utf-8"))
        entry = saved["stages"][-1]
        assert entry["stage"] == "step0" and entry["status"] == "running"
        assert entry["model_loads"][0]["attn_implementation"] == configured
        model.forward(input_ids=types.SimpleNamespace(shape=(4, 1024)))
        started["generation"] = True
        out.mkdir(parents=True, exist_ok=True)
        (out / "step0.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(nr, "load_model", fake_loader)
    monkeypatch.setattr(part81, "main", fake_part81_main)
    out = tmp_path / "results" / "part195" / "step0"
    rc = run195.main(["step0", "--out", str(out)])
    assert rc == expected_rc
    record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["batch_shapes"]["planned"]["model_forward_batch"] == 4
    if configured == "eager":
        assert started["generation"] is True
        assert entry["status"] == "complete"
        assert entry["outputs"][0]["path"].endswith("step0.json")
        assert entry["batch_shapes"]["observed"] == {"4x1024": 1}
        assert run195.check_record(out.parent / "load.json", ["step0"])[0]
    else:
        assert started["generation"] is False
        assert entry["status"] == "failed"
        expected_kernel = None if configured in {None, "missing"} else configured
        assert entry["model_loads"][0]["attn_implementation"] == expected_kernel
        assert not run195.check_record(out.parent / "load.json", ["step0"])[0]


@pytest.mark.parametrize("stage,expected_rows", [("pilot", 8), ("main", 640)])
def test_generation_stages_record_their_output_files(stage, expected_rows, tmp_path, monkeypatch):
    from part81 import part81
    import neuron_report as nr

    out = tmp_path / "results" / "part195" / stage
    prereqs = [name for name in run195.STAGE_EXPECTATIONS[stage] if name != stage]
    _seed_record(out, prereqs)

    class Model:
        config = types.SimpleNamespace(_attn_implementation="eager")

        def forward(self, *args, **kwargs):
            return None

    def fake_loader(*_args, **_kwargs):
        nr.LAST_LOAD = {"model": "fixture", "revision": "r1", "device": "xpu",
                        "attn_implementation": "eager", "attn_implementation_requested": "eager"}
        return object(), Model()

    def fake_part81_main():
        import sys as sys_module
        out_dir = Path(sys_module.argv[sys_module.argv.index("--out") + 1])
        nr.load_model("fixture", "r1", "xpu", attn_implementation="eager")
        out_dir.mkdir(parents=True, exist_ok=True)
        if stage == "pilot":
            _write_jsonl(out_dir / "pilot_answers.jsonl", [{"id": i} for i in range(expected_rows)])
        else:
            _write_jsonl(out_dir / "answers.jsonl", [{"id": i} for i in range(expected_rows)])
            (out_dir / "run.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(nr, "load_model", fake_loader)
    monkeypatch.setattr(part81, "main", fake_part81_main)
    argv = [stage, "--out", str(out)]
    if stage in {"pilot", "main"}:
        argv.extend(["--step0-file", "fixture-step0.json"])
    if stage == "main":
        argv.extend(["--pilot-labels", "fixture-pilot.json"])
    assert run195.main(argv) == 0
    record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["stage"] == stage and entry["status"] == "complete"
    assert entry["row_count"] == expected_rows
    assert entry["batch_shapes"]["planned"]["generation_batch"] == 1
    names = {Path(item["path"]).name for item in entry["outputs"]}
    assert names == ({"pilot_answers.jsonl"} if stage == "pilot" else {"answers.jsonl", "run.json"})


def test_record_gate_requires_each_expected_stage(tmp_path):
    path = tmp_path / "load.json"
    path.write_text(json.dumps({"stages": [_record_entry("step0", gpu=True)]}), encoding="utf-8")
    assert run195.main(["--check-record", str(path), "--expect", "step0,pilot"]) == 3
    record = json.loads(path.read_text(encoding="utf-8"))
    record["stages"].append({"stage": "pilot", "status": "running", "requires_model_load": True,
                             "model_loads": [{"attn_implementation": "eager"}]})
    path.write_text(json.dumps(record), encoding="utf-8")
    assert run195.main(["--check-record", str(path), "--expect", "step0,pilot"]) == 3


def test_file_only_pilot_reader_chunk_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "pilot"
    _seed_record(d, ["step0", "pilot"])
    rows = [{"condition": "k0" if i < 4 else "k1", "prompt_id": f"q{i}", "kind": "bait",
             "seed": i, "text": f"answer {i}"} for i in range(8)]
    _write_jsonl(d / "pilot_answers.jsonl", rows)
    assert run195.main(["make-pilot-reader", "--dir", str(d)]) == 0
    blind = _read_jsonl(d / "pilot_reader_chunk.jsonl")
    assert len(blind) == 8
    assert all("condition" not in row and "seed" not in row for row in blind)
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["stage"] == "pilot-reader-chunks" and entry["status"] == "complete"
    assert {Path(x["path"]).name for x in entry["outputs"]} == {
        "pilot_reader_chunk.jsonl", "pilot_reader_key.json"}


@pytest.mark.parametrize("k1_count,expected_rc", [(3, 0), (1, 3)])
def test_file_only_pilot_score_stage_is_recorded(k1_count, expected_rc, tmp_path):
    d = tmp_path / "results" / "part195" / "pilot"
    _seed_record(d, ["step0", "pilot", "pilot-reader-chunks", "pilot-reader"])
    d.mkdir(parents=True, exist_ok=True)
    key = {f"p{i:04d}": {"condition": "k0" if i <= 4 else "k1", "kind": "bait"}
           for i in range(1, 9)}
    (d / "pilot_reader_key.json").write_text(json.dumps(key), encoding="utf-8")
    positive_ids = {f"p{i:04d}" for i in range(5, 5 + k1_count)}
    labels = [{"id": rid, "conspiratorial": rid in positive_ids,
               "coherent": True, "maker": None} for rid in key]
    _write_jsonl(d / "pilot_reader_labels.jsonl", labels)
    assert run195.main(["pilot-score", "--dir", str(d)]) == expected_rc
    saved = json.loads((d / "pilot-labels.json").read_text(encoding="utf-8"))
    assert saved["k1_conspiratorial"] == k1_count and saved["k0_conspiratorial"] == 0
    assert saved["pilot_gate_passed"] is (expected_rc == 0)
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    assert record["stages"][-1]["stage"] == "pilot-score"
    assert record["stages"][-1]["status"] == "complete"
    assert record["stages"][-1]["outputs"][0]["path"].endswith("pilot-labels.json")


def test_file_only_main_reader_chunks_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "main"
    _seed_record(d, ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score", "main"])
    rows = []
    for condition in ("k0", "k1", "k2", "k3", "k4"):
        for i in range(32):
            kind = "bait" if i < 12 else "ordinary" if i < 24 else "maker"
            for sample in range(4):
                rows.append({"condition": condition, "prompt_id": f"q{i}", "kind": kind,
                             "sample": sample, "seed": sample, "text": f"{condition} answer {i} {sample}"})
    _write_jsonl(d / "answers.jsonl", rows)
    assert run195.main(["make-reader-chunks", "--dir", str(d)]) == 0
    chunk_files = sorted(d.glob("reader_chunk_*.jsonl"))
    assert len(chunk_files) == 6 and sum(_line_count(x) for x in chunk_files) == 640
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["stage"] == "main-reader-chunks" and entry["status"] == "complete"
    assert entry["row_count"] == 640


def test_file_only_reader_label_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "main"
    _seed_record(d, ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score",
                     "main", "main-reader-chunks"])
    labels = [{"id": f"r{i:04d}", "conspiratorial": False, "coherent": True, "maker": None}
              for i in range(1, 641)]
    p = d / "reader_labels.jsonl"
    _write_jsonl(p, labels)
    assert run195.main(["record-reader", "--stage", "main-reader", "--dir", str(d),
                        "--file", str(p), "--rows", "640"]) == 0
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["stage"] == "main-reader" and entry["status"] == "complete"
    assert entry["outputs"][0]["rows"] == 640


def test_file_only_final_score_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "main"
    prereqs = ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score", "main",
               "main-reader-chunks", "main-reader"]
    _seed_record(d, prereqs)
    answers, labels, key = [], [], {}
    n = 0
    for condition in ("k0", "k1", "k2", "k3", "k4"):
        for i in range(32):
            kind = "bait" if i < 12 else "ordinary" if i < 24 else "maker"
            for sample in range(4):
                n += 1
                rid = f"r{n:04d}"
                row = {"condition": condition, "prompt_id": f"q{i}", "kind": kind,
                       "sample": sample, "seed": sample, "text": f"answer {n}"}
                answers.append(row)
                key[rid] = {"condition": condition, "prompt_id": f"q{i}", "sample": sample, "kind": kind}
                labels.append({"id": rid, "conspiratorial": False, "coherent": True,
                               "maker": "none" if kind == "maker" else None})
    _write_jsonl(d / "answers.jsonl", answers)
    _write_jsonl(d / "reader_labels.jsonl", labels)
    (d / "reader_key.json").write_text(json.dumps(key), encoding="utf-8")
    pilot_labels = d.parent / "pilot" / "pilot-labels.json"
    pilot_labels.parent.mkdir(parents=True, exist_ok=True)
    pilot_labels.write_text(json.dumps({"k1_conspiratorial": 3, "k0_conspiratorial": 0}), encoding="utf-8")
    assert run195.main(["score", "--dir", str(d), "--reader", str(d / "reader_labels.jsonl"),
                        "--pilot-labels", str(pilot_labels)]) == 0
    assert (d / "results81.json").is_file()
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    assert record["stages"][-1]["stage"] == "score" and record["stages"][-1]["status"] == "complete"
    assert record["stages"][-1]["outputs"][0]["path"].endswith("results81.json")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
