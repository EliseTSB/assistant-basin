"""End-to-end and deliberate-break tests for Part 126."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PART_DIR = Path(__file__).resolve().parent
ROOT = PART_DIR.parent
PYTHON = "/home/elisepc/neuron-probe/.venv/bin/python"
LAUNCH_COPY_LIST = (
    "part126/reads126.py", "part126/lens126.py", "part126/score126.py",
    "part82/part82.py", "part83/part83.py", "part80/part80.py", "part80b/part80b.py",
    "part111/jlens111.py", "neuron_report.py", "part37/results/full/smol-gens.jsonl",
)
sys.path.insert(0, str(PART_DIR))


def _temporary_directory():
    return tempfile.TemporaryDirectory(prefix=".part126-test-", dir=PART_DIR)


def _run(command, check=True):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    if check and result.returncode != 0:
        raise AssertionError(f"command failed ({result.returncode}): {' '.join(map(str, command))}\n{result.stdout}\n{result.stderr}")
    return result


def _copy_launch_tree(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for relative in LAUNCH_COPY_LIST:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    assert actual == set(LAUNCH_COPY_LIST)


def _part83_fixture(path: Path):
    singles = {}
    for layer in (11, 15):
        for head in range(16):
            index = (0 if layer == 11 else 16) + head
            if index < 20:
                drop = 0.08 + index * 0.002
                ci = [drop - 0.02, drop + 0.02]
            else:
                drop = 0.01
                ci = [-0.01, 0.03]
            singles[f"L{layer}h{head}"] = {"drop": drop, "drop_ci95": ci}
    path.write_text(json.dumps({"singles": singles}, ensure_ascii=False, indent=1), encoding="utf-8")


def _scoring_fixture(root: Path):
    reads = root / "reads"
    lens = root / "lens"
    out = root / "score"
    reads.mkdir(parents=True, exist_ok=True)
    lens.mkdir(parents=True, exist_ok=True)
    n, hidden = 12, 8
    rows = []
    for i in range(n):
        row = {"row_index": i, "body_id": f"q{i // 6}", "rendering": "A" if i % 2 == 0 else "U",
               "order": i % 3 + 1, "n_tokens": 20}
        rows.append(row)
    (reads / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    role_values = np.linspace(0.08, 0.38, 32, dtype=np.float32).reshape(2, 16)
    region_mass = np.zeros((n, 2, 16, 6), dtype=np.float32)
    for i in range(n):
        for layer in range(2):
            for head in range(16):
                role = role_values[layer, head]
                region_mass[i, layer, head] = [role, 0.1, 0.2, 0.15, 0.15, 0.4 - role]
    rng = np.random.default_rng(126)
    region_values = rng.normal(size=(n, 2, 16, 6, hidden)).astype(np.float16)
    body_values = rng.normal(size=(n, 2, 16, hidden)).astype(np.float16)
    np.savez_compressed(reads / "reads126.npz", region_mass=region_mass.astype(np.float16),
                        region_values=region_values, random_body_values=body_values,
                        row_indexes=np.arange(n, dtype=np.int32), layer_ids=np.asarray([11, 15], dtype=np.int16),
                        region_names=np.asarray(["role", "template", "body", "question", "options", "other"], dtype="U16"))
    run = {"gates": {
        "recomputation": {"max_abs_difference": 0.0, "tolerance": 0.05, "region_mass_tolerance": 0.01},
        "zero_effect": {"bitwise_equal": True, "max_abs_difference": 0.0},
        "positive_control": {"max_rate_over_uniform": 4.0, "head": "L11h5"},
    }}
    (reads / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    readouts = {}
    for layer in (11, 15):
        for head in range(16):
            readouts[f"L{layer}h{head}"] = {}
            for copy in ("A", "U"):
                readouts[f"L{layer}h{head}"][copy] = {
                    "role_token": {"top20_written_tokens": [" I", " assistant"], "top20_suppressed_tokens": [" no"]},
                    "most_attended_region": {"name": "body", "mean_attention_mass": 0.4,
                                              "top20_written_tokens": [" copy"], "top20_suppressed_tokens": [" no"]},
                    "random_body_token_control": {"top20_written_tokens": [" I", " body"], "top20_suppressed_tokens": [" no"]},
                }
    lens_json = {"gates": {"lens_shape": {"passed": True, "lens_shape": [31, hidden, hidden],
                                           "expected_shape": [31, hidden, hidden]}},
                 "vectors_file": "lens126-vectors.npz", "readouts": readouts}
    (lens / "lens126.json").write_text(json.dumps(lens_json, ensure_ascii=False, indent=1), encoding="utf-8")
    vectors = rng.normal(size=(2, 16, 2, hidden)).astype(np.float32)
    np.savez_compressed(lens / "lens126-vectors.npz", role_writes=vectors,
                        region_writes=vectors, body_control_writes=vectors,
                        layer_ids=np.asarray([11, 15], dtype=np.int16), copy_names=np.asarray(["A", "U"], dtype="U1"))
    part83 = root / "part83.json"
    _part83_fixture(part83)
    return reads, lens, part83, out


def _score_command(reads, lens, part83, out, script=None, stub=True):
    target = Path(script) if script else PART_DIR / "score126.py"
    command = [PYTHON, "-I", "-B", str(target), "--reads-dir", str(reads), "--lens-dir", str(lens),
               "--part83", str(part83), "--out", str(out)]
    if stub:
        command.append("--stub")
    return command


def test_all_three_mains_run_in_tree_holding_only_launch_copy_list():
    with _temporary_directory() as temp:
        temp = Path(temp)
        isolated = temp / "isolation"
        _copy_launch_tree(isolated)
        bodies = temp / "bodies.jsonl"
        fixtures = [
            {"question_id": "q001", "kind": "greedy", "text": "The cue word is marble. Repeat marble."},
            {"question_id": "q002", "kind": "greedy", "text": "A second body says cedar, then cedar again."},
        ]
        bodies.write_text("".join(json.dumps(row) + "\n" for row in fixtures), encoding="utf-8")
        reads = temp / "reads"
        command = [PYTHON, "-I", "-B", str(isolated / "part126/reads126.py"), "--stub", "--bodies", str(bodies),
                   "--limit-bodies", "2", "--batch-size", "2", "--out", str(reads), "--progress-every", "99"]
        read_result = _run(command)
        assert "12 prompt rows" in read_result.stdout
        with np.load(reads / "reads126.npz", allow_pickle=False) as data:
            assert data["attention"].shape[:3] == (12, 2, 16)
            assert data["region_values"].shape == (12, 2, 16, 6, 2)
            assert data["region_mass"].shape == (12, 2, 16, 6)
        run = json.loads((reads / "run.json").read_text(encoding="utf-8"))
        assert run["gates"]["zero_effect"]["passed"]
        assert run["gates"]["positive_control"]["passed"]

        lens = temp / "lens"
        lens_result = _run([PYTHON, "-I", "-B", str(isolated / "part126/lens126.py"),
                            "--activations", str(reads / "reads126.npz"), "--rows", str(reads / "rows.jsonl"),
                            "--run-json", str(reads / "run.json"), "--out", str(lens), "--stub"])
        assert "lens gate True" in lens_result.stdout
        lens_json = json.loads((lens / "lens126.json").read_text(encoding="utf-8"))
        assert len(lens_json["readouts"]) == 32
        assert (lens / "lens126-vectors.npz").is_file()

        part83 = temp / "part83.json"
        _part83_fixture(part83)
        scored = temp / "scored"
        score_result = _run(_score_command(reads, lens, part83, scored, isolated / "part126/score126.py"))
        assert "gates passed: 4/4" in score_result.stdout
        score_json = json.loads((scored / "score126.json").read_text(encoding="utf-8"))
        assert all(gate["passed"] for gate in score_json["gates"].values())
        assert set(score_json["predictions"]) == {"126.1", "126.2", "126.3", "126.4"}
        assert (scored / "SCORE.md").is_file()


@pytest.mark.parametrize("gate_name", ("recomputation", "zero_effect", "positive_control", "lens_shape"))
def test_each_registered_gate_can_fail_in_score_from_raw_files(gate_name):
    with _temporary_directory() as temp:
        temp = Path(temp)
        reads, lens, part83, out = _scoring_fixture(temp)
        run_path = reads / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        lens_path = lens / "lens126.json"
        lens_json = json.loads(lens_path.read_text(encoding="utf-8"))
        if gate_name == "recomputation":
            run["gates"][gate_name]["max_abs_difference"] = 0.2
            run_path.write_text(json.dumps(run), encoding="utf-8")
        elif gate_name == "zero_effect":
            run["gates"][gate_name]["bitwise_equal"] = False
            run["gates"][gate_name]["max_abs_difference"] = 0.1
            run_path.write_text(json.dumps(run), encoding="utf-8")
        elif gate_name == "positive_control":
            run["gates"][gate_name]["max_rate_over_uniform"] = 2.5
            run_path.write_text(json.dumps(run), encoding="utf-8")
        else:
            lens_json["gates"][gate_name]["passed"] = False
            lens_json["gates"][gate_name]["lens_shape"] = [30, 8, 8]
            lens_path.write_text(json.dumps(lens_json), encoding="utf-8")
        result = _run(_score_command(reads, lens, part83, out))
        assert "gates passed: 3/4" in result.stdout
        score = json.loads((out / "score126.json").read_text(encoding="utf-8"))
        assert not score["gates"][gate_name]["passed"]
        assert all(row["status"] == "not read" for row in score["predictions"].values())


def test_torn_final_jsonl_is_logged_and_torn_middle_line_raises():
    from reads126 import _jsonl_rows
    from score126 import _read_jsonl

    with _temporary_directory() as temp:
        temp = Path(temp)
        final = temp / "final.jsonl"
        final.write_text('{"ok":1}\n{"broken":', encoding="utf-8")
        from contextlib import redirect_stderr
        from io import StringIO
        log = StringIO()
        with redirect_stderr(log):
            assert _jsonl_rows(final) == [{"ok": 1}]
            assert _read_jsonl(final) == [{"ok": 1}]
        assert log.getvalue().count("skipping torn final JSONL line") == 2
        middle = temp / "middle.jsonl"
        middle.write_text('{"ok":1}\n{"broken":}\n{"ok":2}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSONL"):
            _jsonl_rows(middle)
        with pytest.raises(ValueError, match="invalid JSONL"):
            _read_jsonl(middle)


def test_score_skips_a_torn_final_capture_row_and_withholds_predictions():
    with _temporary_directory() as temp:
        temp = Path(temp)
        reads, lens, part83, out = _scoring_fixture(temp)
        row_lines = (reads / "rows.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        (reads / "rows.jsonl").write_text("".join(row_lines[:-1]) + '{"torn":', encoding="utf-8")
        result = _run(_score_command(reads, lens, part83, out, stub=False))
        assert "skipping torn final JSONL line" in result.stderr
        assert "skipped the activation row for the torn final JSONL line" in result.stderr
        score = json.loads((out / "score126.json").read_text(encoding="utf-8"))
        assert score["row_count"] == 11
        assert not score["capture_complete"]
        assert all(row["status"] == "not read" for row in score["predictions"].values())


def test_two_deliberate_gate_break_copies_make_the_good_gate_test_red():
    replacements = (
        ("recompute_error <= recompute_tolerance", "recompute_error < 0.0", "recomputation"),
        ('zero.get("bitwise_equal") is True', 'zero.get("bitwise_equal") is False', "zero_effect"),
    )
    for original, replacement, gate_name in replacements:
        with _temporary_directory() as temp:
            temp = Path(temp)
            reads, lens, part83, out = _scoring_fixture(temp)
            source = (PART_DIR / "score126.py").read_text(encoding="utf-8")
            assert original in source
            broken_script = temp / f"score126-broken-{gate_name}.py"
            broken_script.write_text(source.replace(original, replacement, 1), encoding="utf-8")
            broken_out = temp / "broken-out"
            test_file = temp / "test_expected_green.py"
            test_file.write_text(
                "import json, subprocess, sys\n"
                f"cmd = {repr(_score_command(reads, lens, part83, broken_out, broken_script))}\n"
                "def test_good_gate_remains_passed_on_unbroken_fixture():\n"
                "    subprocess.run(cmd, check=True, capture_output=True, text=True)\n"
                "    score = json.loads(open(" + repr(str(broken_out / "score126.json")) + ", encoding='utf-8').read())\n"
                f"    assert score['gates'][{gate_name!r}]['passed']\n",
                encoding="utf-8",
            )
            failed = _run([PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)], check=False)
            assert failed.returncode != 0
            assert "1 failed" in failed.stdout


def test_capture_resumes_from_two_body_prefix_and_repairs_torn_tail():
    with _temporary_directory() as temp:
        temp = Path(temp)
        isolated = temp / "isolation"
        _copy_launch_tree(isolated)
        bodies = temp / "bodies.jsonl"
        fixtures = [
            {"question_id": "q001", "kind": "greedy", "text": "The cue word is marble. Repeat marble."},
            {"question_id": "q002", "kind": "greedy", "text": "A second body says cedar, then cedar again."},
        ]
        bodies.write_text("".join(json.dumps(row) + "\n" for row in fixtures), encoding="utf-8")
        reads = temp / "reads"
        script = isolated / "part126/reads126.py"
        pilot = [PYTHON, "-I", "-B", str(script), "--stub", "--bodies", str(bodies), "--limit-bodies", "1",
                 "--batch-size", "1", "--out", str(reads), "--progress-every", "99"]
        _run(pilot)
        assert len((reads / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 6
        with (reads / "rows.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"torn":')
        full = [PYTHON, "-I", "-B", str(script), "--stub", "--bodies", str(bodies), "--batch-size", "2",
                "--out", str(reads), "--progress-every", "99"]
        result = _run(full)
        assert "skipping torn final JSONL line" in result.stderr
        assert len((reads / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 12
        with np.load(reads / "reads126.npz", allow_pickle=False) as data:
            assert data["attention"].shape[0] == 12

