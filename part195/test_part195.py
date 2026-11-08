"""CPU-only tests for Part 195's load gate and its recorded file stages."""
from __future__ import annotations

import json
import hashlib
import sys
import time
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
            "model_loads": ([{"model": run195.EXPECTED_MODEL, "revision": run195.EXPECTED_REVISION,
                              "dtype": run195.EXPECTED_DTYPE, "attn_implementation": "eager"}] if gpu else []),
            "batch_shapes": {"planned": {}, "observed": {}, "generation_calls": []}, "outputs": []}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_answers(count: int) -> list[dict]:
    rows = []
    conditions = ("k0", "k1", "k2", "k3", "k4")
    for condition in conditions:
        for prompt in range(32):
            for sample in range(4):
                rows.append({"condition": condition, "prompt_id": f"q{prompt:02d}",
                             "kind": "bait" if prompt < 12 else "ordinary" if prompt < 24 else "maker",
                             "sample": sample, "seed": sample, "text": f"{condition} answer {prompt} {sample}"})
                if len(rows) == count:
                    return rows
    return rows


def _write_seed_stage(root: Path, stage: str) -> list[Path]:
    stage_dir = root / ("step0" if stage == "step0" else "pilot" if stage.startswith("pilot") else "main")
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == "step0":
        path = stage_dir / "step0.json"
        _write_json(path, {"medians": {"L17": {"median": 11.4}, "L19": {"median": 15.9}}})
        return [path]
    if stage == "pilot":
        path = stage_dir / "pilot_answers.jsonl"
        rows = [{"condition": "k0" if i < 4 else "k1", "prompt_id": f"t{i % 4}",
                 "kind": "bait", "text": f"pilot answer {i}"} for i in range(8)]
        _write_jsonl(path, rows)
        return [path]
    if stage == "pilot-reader-chunks":
        chunk = stage_dir / "pilot_reader_chunk.jsonl"
        key_path = stage_dir / "pilot_reader_key.json"
        key = {f"p{i:04d}": {"condition": "k0" if i <= 4 else "k1", "prompt_id": f"t{i % 4}",
                              "kind": "bait"} for i in range(1, 9)}
        _write_jsonl(chunk, [{"id": rid, "text": "pilot answer"} for rid in key])
        _write_json(key_path, key)
        return [chunk, key_path]
    if stage == "pilot-reader":
        path = stage_dir / "pilot_reader_labels.jsonl"
        rows = [{"id": f"p{i:04d}", "conspiratorial": 5 <= i <= 7,
                 "coherent": True, "maker": None} for i in range(1, 9)]
        _write_jsonl(path, rows)
        return [path]
    if stage == "pilot-score":
        path = stage_dir / "pilot-labels.json"
        _write_json(path, {"k0_conspiratorial": 0, "k1_conspiratorial": 3,
                           "k0_coherent": 4, "k1_coherent": 4, "pilot_gate_passed": True})
        return [path]
    if stage == "main":
        answers = stage_dir / "answers.jsonl"
        run = stage_dir / "run.json"
        _write_jsonl(answers, _default_answers(640))
        _write_json(run, {"requested_rows": 640})
        return [answers, run]
    if stage == "main-reader-chunks":
        key_path = stage_dir / "reader_key.json"
        answers = _default_answers(640)
        key = {f"r{i:04d}": {"condition": row["condition"], "prompt_id": row["prompt_id"],
                              "sample": row["sample"], "kind": row["kind"]}
               for i, row in enumerate(answers, 1)}
        _write_json(key_path, key)
        output_paths = [key_path]
        offset = 1
        for index, count in enumerate([107, 107, 107, 107, 107, 105], 1):
            path = stage_dir / f"reader_chunk_{index:02d}.jsonl"
            _write_jsonl(path, [{"id": f"r{i:04d}", "text": "answer"}
                                for i in range(offset, offset + count)])
            output_paths.append(path)
            offset += count
        return output_paths
    if stage == "main-reader":
        path = stage_dir / "reader_labels.jsonl"
        key = json.loads((stage_dir / "reader_key.json").read_text(encoding="utf-8"))
        _write_jsonl(path, [{"id": rid, "conspiratorial": False, "coherent": True,
                             "maker": "none" if info["kind"] == "maker" else None}
                            for rid, info in key.items()])
        return [path]
    if stage == "score":
        path = stage_dir / "results81.json"
        conditions = {cid: {"topic_n": 96, "coherent_n": 128, "maker_n": 32,
                            "topic_split": {"bait": {"n": 48}, "ordinary": {"n": 48}}}
                      for cid in ("k0", "k1", "k2", "k3", "k4")}
        _write_json(path, {"conditions": conditions,
                           "predictions": {f"81.{i}": {"status": "HOLDS"} for i in range(1, 7)}})
        return [path]
    raise ValueError(stage)


def _output_record(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": run195._line_count(path) if path.suffix == ".jsonl" else None}


def _seed_record(directory: Path, expected: list[str]) -> None:
    path = directory.parent / "load.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for stage in expected:
        outputs = _write_seed_stage(path.parent, stage)
        entry = _record_entry(stage, gpu=stage in run195.GPU_STAGES)
        entry.update(run_id="fixture-run", outputs=[_output_record(path.parent, p) for p in outputs])
        if stage == "step0":
            entry["row_count"] = None
        else:
            entry["row_count"] = run195.EXPECTED_STAGE_ROWS[stage]
        entries.append(entry)
    path.write_text(json.dumps({"part195": {}, "run_id": "fixture-run", "stages": entries}), encoding="utf-8")


def _refresh_record_stage(directory: Path, stage: str) -> None:
    path = directory.parent / "load.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in record["stages"] if entry.get("stage") == stage]
    assert entries, f"no seeded {stage} stage"
    entry = entries[-1]
    output_paths = [run195._output_path_from_record(path, output) for output in entry["outputs"]]
    entry["outputs"] = [_output_record(path.parent, output) for output in output_paths]
    entry["row_count"] = sum(run195._line_count(output) for output in output_paths if output.suffix == ".jsonl") \
        if stage in run195.EXPECTED_STAGE_ROWS else None
    path.write_text(json.dumps(record), encoding="utf-8")


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
        nr.LAST_LOAD = {"model": run195.EXPECTED_MODEL, "revision": run195.EXPECTED_REVISION,
                        "device": "xpu", "dtype": run195.EXPECTED_DTYPE,
                        "attn_implementation": "eager", "attn_implementation_requested": "sdpa"}
        return object(), Model()

    started = {"generation": False}

    def fake_part81_main():
        import json as json_module
        import sys as sys_module
        out = Path(sys_module.argv[sys_module.argv.index("--out") + 1])
        _tok, model = nr.load_model(run195.EXPECTED_MODEL, run195.EXPECTED_REVISION,
                                    "xpu", attn_implementation="eager")
        saved = json_module.loads((out.parent / "load.json").read_text(encoding="utf-8"))
        entry = saved["stages"][-1]
        assert entry["stage"] == "step0" and entry["status"] == "running"
        assert entry["model_loads"][0]["attn_implementation"] == configured
        model.forward(input_ids=types.SimpleNamespace(shape=(4, 1024)))
        started["generation"] = True
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "step0.json", {"medians": {"L17": {"median": 11.4},
                                                      "L19": {"median": 15.9}}})

    monkeypatch.setattr(nr, "load_model", fake_loader)
    monkeypatch.setattr(part81, "main", fake_part81_main)
    out = tmp_path / "results" / "part195" / "step0"
    rc = run195.main(["step0", "--out", str(out)])
    assert rc == expected_rc
    record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["batch_shapes"]["planned"]["model_forward_batch"] == 4
    assert entry["model_loads"][0]["requested_attn_implementation"] == "eager"
    assert entry["model_loads"][0]["dtype"] == run195.EXPECTED_DTYPE
    if configured == "eager":
        assert started["generation"] is True
        assert entry["status"] == "complete"
        assert entry["outputs"][0]["path"].endswith("step0/step0.json")
        assert "\\" not in entry["outputs"][0]["path"]
        assert not Path(entry["outputs"][0]["path"]).is_absolute()
        assert entry["run_id"] == record["run_id"]
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
    pilot_file = out.parent / "pilot" / "pilot-labels.json"
    if stage == "main":
        pilot_dir = pilot_file.parent
        pilot_dir.mkdir(parents=True, exist_ok=True)
        key = {f"p{i:04d}": {"condition": "k0" if i <= 4 else "k1", "kind": "bait"}
               for i in range(1, 9)}
        (pilot_dir / "pilot_reader_key.json").write_text(json.dumps(key), encoding="utf-8")
        labels = [{"id": rid, "conspiratorial": rid in {"p0005", "p0006", "p0007"},
                   "coherent": True, "maker": None} for rid in key]
        _write_jsonl(pilot_dir / "pilot_reader_labels.jsonl", labels)
        pilot_file.write_text(json.dumps({"k0_conspiratorial": 0, "k1_conspiratorial": 3,
                                          "k0_coherent": 4, "k1_coherent": 4,
                                          "pilot_gate_passed": True}), encoding="utf-8")
        _refresh_record_stage(out, "pilot-reader-chunks")
        _refresh_record_stage(out, "pilot-reader")
        _refresh_record_stage(out, "pilot-score")

    class Model:
        config = types.SimpleNamespace(_attn_implementation="eager")

        def forward(self, *args, **kwargs):
            return None

        def generate(self, input_ids=None):
            return input_ids

    def fake_loader(*_args, **_kwargs):
        nr.LAST_LOAD = {"model": run195.EXPECTED_MODEL, "revision": run195.EXPECTED_REVISION,
                        "device": "xpu", "dtype": run195.EXPECTED_DTYPE,
                        "attn_implementation": "eager", "attn_implementation_requested": "sdpa"}
        return object(), Model()

    def fake_part81_main():
        import sys as sys_module
        out_dir = Path(sys_module.argv[sys_module.argv.index("--out") + 1])
        _tok, model = nr.load_model(run195.EXPECTED_MODEL, run195.EXPECTED_REVISION,
                                    "xpu", attn_implementation="eager")
        model.generate(input_ids=types.SimpleNamespace(shape=(1, 5)))
        out_dir.mkdir(parents=True, exist_ok=True)
        if stage == "pilot":
            _write_jsonl(out_dir / "pilot_answers.jsonl", [{"id": i} for i in range(expected_rows)])
        else:
            _write_jsonl(out_dir / "answers.jsonl", _default_answers(expected_rows))
            _write_json(out_dir / "run.json", {"requested_rows": 640})

    monkeypatch.setattr(nr, "load_model", fake_loader)
    monkeypatch.setattr(part81, "main", fake_part81_main)
    argv = [stage, "--out", str(out)]
    if stage in {"pilot", "main"}:
        argv.extend(["--step0-file", "fixture-step0.json"])
    if stage == "main":
        argv.extend(["--pilot-labels", str(pilot_file)])
    assert run195.main(argv) == 0
    record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    assert entry["stage"] == stage and entry["status"] == "complete"
    assert entry["row_count"] == expected_rows
    assert entry["batch_shapes"]["planned"]["generation_batch"] == 1
    assert entry["batch_shapes"]["generation_calls"] == [
        {"call": 1, "batch": 1, "sequence_length": 5}
    ]
    names = {Path(item["path"]).name for item in entry["outputs"]}
    assert names == ({"pilot_answers.jsonl"} if stage == "pilot" else {"answers.jsonl", "run.json"})


def test_record_gate_requires_each_expected_stage(tmp_path):
    directory = tmp_path / "results" / "part195" / "step0"
    _seed_record(directory, ["step0"])
    path = directory.parent / "load.json"
    assert run195.main(["--check-record", str(path), "--expect", "step0,pilot"]) == 3
    record = json.loads(path.read_text(encoding="utf-8"))
    running = _record_entry("pilot", gpu=True)
    running.update(run_id=record["run_id"], status="running")
    record["stages"].append(running)
    path.write_text(json.dumps(record), encoding="utf-8")
    assert run195.main(["--check-record", str(path), "--expect", "step0,pilot"]) == 3


@pytest.mark.parametrize("tamper", ["same-size-swap", "wrong-size", "deleted", "rewritten-path"])
def test_record_gate_certifies_each_expected_output(tamper, tmp_path):
    directory = tmp_path / "results" / "part195" / "step0"
    _seed_record(directory, ["step0"])
    record_path = directory.parent / "load.json"
    assert run195.check_record(record_path, ["step0"])[0]
    artifact = directory / "step0.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    output = record["stages"][0]["outputs"][0]
    if tamper == "same-size-swap":
        before = artifact.read_bytes()
        after = before.replace(b"11.4", b"11.5")
        assert len(after) == len(before) and after != before
        artifact.write_bytes(after)
    elif tamper == "wrong-size":
        output["bytes"] += 1
    elif tamper == "rewritten-path":
        output["path"] = "step0/missing.json"
    else:
        artifact.unlink()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    ok, why = run195.check_record(record_path, ["step0"])
    assert not ok
    assert "sha256" in why if tamper == "same-size-swap" else \
        ("size" in why if tamper == "wrong-size" else "missing" in why)


@pytest.mark.parametrize("stage", ["pilot", "main", "pilot-reader-chunks", "pilot-reader",
                                    "main-reader-chunks", "main-reader"])
def test_record_gate_rejects_short_stage_rows(stage, tmp_path):
    directory = tmp_path / "results" / "part195" / ("pilot" if stage.startswith("pilot") else "main")
    prerequisites = {
        "pilot": ["step0", "pilot"],
        "main": ["main"],
        "pilot-reader-chunks": ["step0", "pilot", "pilot-reader-chunks"],
        "pilot-reader": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader"],
        "main-reader-chunks": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader",
                               "pilot-score", "main", "main-reader-chunks"],
        "main-reader": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score",
                        "main", "main-reader-chunks", "main-reader"],
    }[stage]
    _seed_record(directory, prerequisites)
    record_path = directory.parent / "load.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    entry = record["stages"][-1]
    if stage == "main-reader-chunks":
        chunk = directory / "reader_chunk_06.jsonl"
        _write_jsonl(chunk, _read_jsonl(chunk)[:-1])
    else:
        jsonl = next(run195._output_path_from_record(record_path, o)
                     for o in entry["outputs"] if o["path"].endswith(".jsonl"))
        _write_jsonl(jsonl, _read_jsonl(jsonl)[:-1])
    _refresh_record_stage(directory, stage)
    ok, why = run195.check_record(record_path, [stage])
    assert not ok and "expected" in why


def test_record_gate_requires_step0_median_keys(tmp_path):
    directory = tmp_path / "results" / "part195" / "step0"
    _seed_record(directory, ["step0"])
    artifact = directory / "step0.json"
    _write_json(artifact, {})
    _refresh_record_stage(directory, "step0")
    ok, why = run195.check_record(directory.parent / "load.json", ["step0"])
    assert not ok and "medians.L17.median" in why


def test_record_gate_rejects_empty_main_run_metadata(tmp_path):
    directory = tmp_path / "results" / "part195" / "main"
    _seed_record(directory, ["step0", "main"])
    _write_json(directory / "run.json", {})
    _refresh_record_stage(directory, "main")
    ok, why = run195.check_record(directory.parent / "load.json", ["main"])
    assert not ok and "requested_rows" in why


def test_record_gate_refuses_different_stage_run_ids(tmp_path):
    directory = tmp_path / "results" / "part195" / "pilot"
    _seed_record(directory, ["step0", "pilot"])
    record_path = directory.parent / "load.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["stages"][1]["run_id"] = "other-run"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    ok, why = run195.check_record(record_path, ["step0", "pilot"])
    assert not ok and "run ids" in why


def test_stage_refuses_existing_record_lock_with_recovery_details(tmp_path, capsys):
    directory = tmp_path / "results" / "part195" / "pilot"
    _seed_record(directory, ["step0", "pilot"])
    lock_path = directory.parent / "load.json.lock"
    lock_path.write_text(f"987654321 {time.time() - 60}\n", encoding="ascii")
    called = {"value": False}

    def stage_fn():
        called["value"] = True
        chunk = directory / "pilot_reader_chunk.jsonl"
        key_path = directory / "pilot_reader_key.json"
        key = {f"p{i:04d}": {"condition": "k0" if i <= 4 else "k1"} for i in range(1, 9)}
        _write_jsonl(chunk, [{"id": rid, "text": "answer"} for rid in key])
        _write_json(key_path, key)
        return [chunk, key_path]

    rc = run195.run_host_stage("pilot-reader-chunks", directory, stage_fn)
    assert rc == 3
    diagnostic = capsys.readouterr().out
    assert str(lock_path) in diagnostic
    assert "age " in diagnostic
    assert "recorded pid 987654321" in diagnostic
    assert "ps -p 987654321" in diagnostic
    assert "only then remove" in diagnostic
    assert called["value"] is False
    assert lock_path.is_file()
    record = json.loads((directory.parent / "load.json").read_text(encoding="utf-8"))
    assert [entry["stage"] for entry in record["stages"]] == ["step0", "pilot"]


@pytest.mark.parametrize(("field", "bad_value"), [
    ("model", "Qwen/Other-Model"),
    ("revision", "deadbeef"),
    ("dtype", "torch.float32"),
])
def test_record_gate_requires_pinned_model_revision_and_dtype(field, bad_value, tmp_path):
    directory = tmp_path / "results" / "part195" / "step0"
    _seed_record(directory, ["step0"])
    path = directory.parent / "load.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["stages"][0]["model_loads"][0][field] = bad_value
    path.write_text(json.dumps(record), encoding="utf-8")
    ok, why = run195.check_record(path, ["step0"])
    assert not ok and field in why


def test_file_only_pilot_reader_chunk_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "pilot"
    _seed_record(d, ["step0", "pilot"])
    assert run195.main(["make-pilot-reader", "--dir", str(d)]) == 0
    assert not (d.parent / "load.json.lock").exists()
    blind = _read_jsonl(d / "pilot_reader_chunk.jsonl")
    assert len(blind) == 8
    assert all(set(row) == {"id", "text"} for row in blind)
    key = json.loads((d / "pilot_reader_key.json").read_text(encoding="utf-8"))
    assert all({"condition", "prompt_id", "kind"} <= set(row) for row in key.values())
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
    _refresh_record_stage(d, "pilot-reader-chunks")
    _refresh_record_stage(d, "pilot-reader")
    assert run195.main(["pilot-score", "--dir", str(d)]) == expected_rc
    saved = json.loads((d / "pilot-labels.json").read_text(encoding="utf-8"))
    assert saved["k1_conspiratorial"] == k1_count and saved["k0_conspiratorial"] == 0
    assert saved["pilot_gate_passed"] is (expected_rc == 0)
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    assert record["stages"][-1]["stage"] == "pilot-score"
    assert record["stages"][-1]["status"] == "complete"
    assert record["stages"][-1]["outputs"][0]["path"].endswith("pilot-labels.json")


def test_main_pilot_gate_is_recomputed_from_raw_reader_labels(tmp_path, monkeypatch):
    from part81 import part81

    out = tmp_path / "results" / "part195" / "main"
    prereqs = [name for name in run195.STAGE_EXPECTATIONS["main"] if name != "main"]
    _seed_record(out, prereqs)
    pilot = out.parent / "pilot"
    pilot.mkdir(parents=True, exist_ok=True)
    key = {f"p{i:04d}": {"condition": "k0" if i <= 4 else "k1", "kind": "bait"}
           for i in range(1, 9)}
    (pilot / "pilot_reader_key.json").write_text(json.dumps(key), encoding="utf-8")
    labels = [{"id": rid, "conspiratorial": rid == "p0005", "coherent": True, "maker": None}
              for rid in key]
    _write_jsonl(pilot / "pilot_reader_labels.jsonl", labels)
    pilot_file = pilot / "pilot-labels.json"
    pilot_file.write_text(json.dumps({"k0_conspiratorial": 0, "k1_conspiratorial": 1,
                                      "k0_coherent": 4, "k1_coherent": 4,
                                      "pilot_gate_passed": False}), encoding="utf-8")
    _refresh_record_stage(out, "pilot-reader-chunks")
    _refresh_record_stage(out, "pilot-reader")
    _refresh_record_stage(out, "pilot-score")
    runner_started = {"value": False}

    def fake_part81_main():
        import sys as sys_module

        runner_started["value"] = True
        out_dir = Path(sys_module.argv[sys_module.argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "answers.jsonl", [{"id": 1}])
        (out_dir / "run.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(part81, "main", fake_part81_main)
    assert run195.main(["main", "--out", str(out), "--step0-file", "fixture-step0.json",
                        "--pilot-labels", str(pilot_file)]) == 3
    assert runner_started["value"] is False
    record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
    assert record["stages"][-1]["stage"] == "main"
    assert record["stages"][-1]["status"] == "failed"
    assert "raw pilot labels fail gate" in record["stages"][-1]["error"]


def test_main_pilot_gate_refuses_labels_outside_recorded_pilot(tmp_path, monkeypatch):
    from part81 import part81
    import neuron_report as nr

    out = tmp_path / "results" / "part195" / "main"
    prereqs = [name for name in run195.STAGE_EXPECTATIONS["main"] if name != "main"]
    _seed_record(out, prereqs)
    unrelated = tmp_path / "elsewhere" / "pilot-labels.json"
    _write_json(unrelated, {"k0_conspiratorial": 0, "k1_conspiratorial": 3,
                            "k0_coherent": 4, "k1_coherent": 4, "pilot_gate_passed": True})
    runner_started = {"value": False}

    class Model:
        config = types.SimpleNamespace(_attn_implementation="eager")

        def forward(self, *args, **kwargs):
            return None

        def generate(self, input_ids=None):
            return input_ids

    def fake_loader(*_args, **_kwargs):
        nr.LAST_LOAD = {"model": run195.EXPECTED_MODEL, "revision": run195.EXPECTED_REVISION,
                        "device": "xpu", "dtype": run195.EXPECTED_DTYPE,
                        "attn_implementation": "eager"}
        return object(), Model()

    def fake_part81_main():
        import sys as sys_module

        runner_started["value"] = True
        out_dir = Path(sys_module.argv[sys_module.argv.index("--out") + 1])
        _tok, model = nr.load_model(run195.EXPECTED_MODEL, run195.EXPECTED_REVISION,
                                    "xpu", attn_implementation="eager")
        model.generate(input_ids=types.SimpleNamespace(shape=(1, 5)))
        _write_jsonl(out_dir / "answers.jsonl", _default_answers(640))
        _write_json(out_dir / "run.json", {"requested_rows": 640})

    monkeypatch.setattr(nr, "load_model", fake_loader)
    monkeypatch.setattr(part81, "main", fake_part81_main)
    assert run195.main(["main", "--out", str(out), "--step0-file", "fixture-step0.json",
                        "--pilot-labels", str(unrelated)]) == 3
    assert runner_started["value"] is False
    record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
    assert "does not name the recorded pilot-score output" in record["stages"][-1]["error"]


def test_file_only_main_reader_chunks_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "main"
    _seed_record(d, ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score", "main"])
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


def test_record_reader_retry_supersedes_old_attempt_at_same_path(tmp_path):
    d = tmp_path / "results" / "part195" / "pilot"
    _seed_record(d, ["step0", "pilot", "pilot-reader-chunks"])
    labels_path = d / "pilot_reader_labels.jsonl"
    first_labels = [{"id": f"p{i:04d}", "conspiratorial": False,
                     "coherent": True, "maker": None} for i in range(1, 9)]
    _write_jsonl(labels_path, first_labels)
    args = ["record-reader", "--stage", "pilot-reader", "--dir", str(d),
            "--file", str(labels_path), "--rows", "8"]
    assert run195.main(args) == 0
    record_path = d.parent / "load.json"
    first_hash = json.loads(record_path.read_text(encoding="utf-8"))["stages"][-1]["outputs"][0]["sha256"]

    second_labels = [dict(row, coherent=False) for row in first_labels]
    _write_jsonl(labels_path, second_labels)
    assert run195.main(args) == 0

    record = json.loads(record_path.read_text(encoding="utf-8"))
    attempts = [entry for entry in record["stages"] if entry["stage"] == "pilot-reader"]
    assert [entry["status"] for entry in attempts] == ["superseded", "complete"]
    assert [entry["attempt"] for entry in attempts] == [1, 2]
    assert attempts[0]["superseded_by_attempt"] == 2
    assert attempts[0]["outputs"][0]["sha256"] == first_hash
    assert attempts[1]["outputs"][0]["sha256"] != first_hash
    assert run195.check_record(record_path, ["step0", "pilot", "pilot-reader-chunks", "pilot-reader"])[0]


def test_pilot_fail_then_corrected_read_passes_steps_record_gate(tmp_path):
    d = tmp_path / "results" / "part195" / "pilot"
    _seed_record(d, ["step0", "pilot"])
    record_path = d.parent / "load.json"
    expected = "step0,pilot,pilot-reader-chunks,pilot-reader,pilot-score"
    assert run195.main(["make-pilot-reader", "--dir", str(d)]) == 0
    key = json.loads((d / "pilot_reader_key.json").read_text(encoding="utf-8"))
    labels_path = d / "pilot_reader_labels.jsonl"

    def write_labels(k1_conspiratorial: int) -> None:
        k1_ids = [rid for rid, location in key.items() if location["condition"] == "k1"]
        positive_ids = set(k1_ids[:k1_conspiratorial])
        _write_jsonl(labels_path, [{"id": rid, "conspiratorial": rid in positive_ids,
                                   "coherent": True, "maker": None} for rid in key])

    args = ["record-reader", "--stage", "pilot-reader", "--dir", str(d),
            "--file", str(labels_path), "--rows", "8"]
    write_labels(1)
    assert run195.main(args) == 0
    assert run195.main(["pilot-score", "--dir", str(d)]) == 3
    assert run195.main(["--check-record", str(record_path), "--expect", expected]) == 0

    write_labels(3)
    assert run195.main(args) == 0
    assert run195.main(["pilot-score", "--dir", str(d)]) == 0
    assert run195.main(["--check-record", str(record_path), "--expect", expected]) == 0

    record = json.loads(record_path.read_text(encoding="utf-8"))
    for stage in ("pilot-reader", "pilot-score"):
        attempts = [entry for entry in record["stages"] if entry["stage"] == stage]
        assert [entry["status"] for entry in attempts] == ["superseded", "complete"]
        assert [entry["attempt"] for entry in attempts] == [1, 2]


def test_record_gate_uses_code_row_count_constant_not_record_value(tmp_path, monkeypatch):
    directory = tmp_path / "results" / "part195" / "pilot"
    _seed_record(directory, ["step0", "pilot"])
    record_path = directory.parent / "load.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["stages"][1]["row_count"] = 7
    record_path.write_text(json.dumps(record), encoding="utf-8")

    observed = []
    validate = run195._validate_stage_artifacts

    def capture_row_count(stage, paths, row_count):
        observed.append((stage, row_count))
        return validate(stage, paths, row_count)

    monkeypatch.setattr(run195, "_validate_stage_artifacts", capture_row_count)
    ok, why = run195.check_record(record_path, ["pilot"])
    assert not ok and "recorded row_count" in why
    assert observed == [("pilot", run195.EXPECTED_STAGE_ROWS["pilot"])]


def test_file_only_final_score_stage_is_recorded(tmp_path):
    d = tmp_path / "results" / "part195" / "main"
    prereqs = ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score", "main",
               "main-reader-chunks", "main-reader"]
    _seed_record(d, prereqs)
    reader_file = d / "reader_labels.jsonl"
    pilot_labels = d.parent / "pilot" / "pilot-labels.json"
    assert run195.main(["score", "--dir", str(d), "--reader", str(reader_file),
                        "--key", str(d / "reader_key.json"),
                        "--pilot-labels", str(pilot_labels)]) == 0
    assert (d / "results81.json").is_file()
    record = json.loads((d.parent / "load.json").read_text(encoding="utf-8"))
    assert record["stages"][-1]["stage"] == "score" and record["stages"][-1]["status"] == "complete"
    assert record["stages"][-1]["outputs"][0]["path"].endswith("results81.json")


def test_score_refuses_a_different_key_beside_the_labels(tmp_path):
    """Part 81's scorer reads the key from beside the label file; a stale different key there must stop scoring."""
    d = tmp_path / "results" / "part195" / "main"
    _seed_record(d, ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score", "main",
                     "main-reader-chunks", "main-reader"])
    reader_file = d / "labels" / "reader_labels.jsonl"
    _write_jsonl(reader_file, [{"id": "r0001", "conspiratorial": False, "coherent": True, "maker": None}])
    (d / "reader_key.json").write_text(json.dumps({"r0001": {"condition": "k0"}}), encoding="utf-8")
    (d / "labels" / "reader_key.json").write_text(json.dumps({"r0001": {"condition": "k1"}}), encoding="utf-8")
    try:
        rc = run195.main(["score", "--dir", str(d), "--reader", str(reader_file),
                          "--key", str(d / "reader_key.json"), "--pilot-labels", str(d / "none.json")])
    except ValueError as exc:
        assert "differs from --key" in str(exc)
    else:
        assert rc != 0
    assert not (d / "results81.json").exists()
    assert json.loads((d / "labels" / "reader_key.json").read_text(encoding="utf-8"))["r0001"]["condition"] == "k1"

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
