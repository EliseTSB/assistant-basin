"""Tests for part104/score104.py on synthetic labels (no model): the verdicts follow the registered rules."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score104  # noqa: E402


def write_fixture(tmp, outcomes, coherent_ext=True, n_early=10, n_later=8):
    """outcomes: run -> unit -> shows (True/False); every unit is slot; units e* sit in layer 2, l* in layer 6."""
    units = [f"e{i}" for i in range(n_early)] + [f"l{i}" for i in range(n_later)]
    (tmp / "kinds.json").write_text(json.dumps({"kinds": {u: "slot" for u in units}}))
    (tmp / "neurons.json").write_text(json.dumps([{"id": u, "layer": 2 if u[0] == "e" else 6} for u in units]))
    for run in score104.RUNS:
        rows = []
        for u in units:
            shows = outcomes.get(run, {}).get(u, True)
            coh = coherent_ext if run == "attnEXT_all" else True
            rows.append({"id": u, "unit": "neuron", "question_index": 0, "shows_persona": shows, "coherent": coh})
            rows.append({"id": u, "unit": "baseline", "question_index": 0, "shows_persona": False, "coherent": True})
        (tmp / f"labels-{run}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (tmp / f"{run}.meta.json").write_text(json.dumps({"tensors_copied": 0, "mode": "copy", "distance_ratio": None}))
    return units


def run_score(tmp, monkeypatch):
    monkeypatch.setattr(score104, "R", tmp)
    score104.cmd_score()
    return json.loads((tmp / "score104.json").read_text())["predictions"]


def test_dependence_outcome_holds_104_1_and_104_2(tmp_path, monkeypatch):
    lose = {f"e{i}": False for i in range(7)}
    write_fixture(tmp_path, {"attnSFT_all": lose})
    p = run_score(tmp_path, monkeypatch)
    assert p["104.1"]["verdict"] == "held"           # 7 of 10 against 0 of 8
    assert p["104.3"]["verdict"] == "held"
    assert p["104.2"]["verdict"] == "held"           # 7 discordant one way: p = 1/128
    assert p["104.4"]["verdict"] == "held"


def test_fragility_outcome_fails_104_2(tmp_path, monkeypatch):
    lose = {f"e{i}": False for i in range(7)}
    write_fixture(tmp_path, {"attnSFT_all": lose, "attnEXT_all": dict(lose)})
    p = run_score(tmp_path, monkeypatch)
    assert p["104.1"]["verdict"] == "held"
    assert p["104.2"]["verdict"] == "failed"


def test_no_split_fails_104_1_and_104_2_is_not_read(tmp_path, monkeypatch):
    lose = {"e0": False, "e1": False, "l0": False, "l1": False}
    write_fixture(tmp_path, {"attnSFT_all": lose})
    p = run_score(tmp_path, monkeypatch)
    assert p["104.1"]["verdict"] == "failed"
    assert p["104.2"]["verdict"].startswith("not read")


def test_incoherent_extrapolation_blocks_104_2(tmp_path, monkeypatch):
    write_fixture(tmp_path, {"attnSFT_all": {f"e{i}": False for i in range(7)}}, coherent_ext=False)
    p = run_score(tmp_path, monkeypatch)
    assert p["104.3"]["verdict"] == "failed"
    assert p["104.2"]["verdict"] == "not read (104.3 did not hold)"


def test_unit_not_shown_in_inst_none_is_not_counted(tmp_path, monkeypatch):
    write_fixture(tmp_path, {"inst_none": {"e0": False}, "attnSFT_all": {"e0": False}})
    out_p = run_score(tmp_path, monkeypatch)
    data = json.loads((tmp_path / "score104.json").read_text())
    assert "e0" not in data["shown_early"] and out_p["104.1"]["early"] == "0 of 9"


def test_dpo_losing_two_later_units_fails_104_4(tmp_path, monkeypatch):
    write_fixture(tmp_path, {"attnDPO_all": {"l0": False, "l1": False}})
    assert run_score(tmp_path, monkeypatch)["104.4"]["verdict"] == "failed"


def test_exact_tests():
    assert abs(score104.mcnemar_one_sided(5, 0) - 1 / 32) < 1e-12
    assert abs(score104.mcnemar_one_sided(4, 0) - 0.0625) < 1e-12
    # Part 103's own split: 7 of 10 against 0 of 9 gives 0.0024 one-sided
    assert abs(score104.fisher_one_sided(7, 10, 0, 9) - 0.0024) < 0.0001
