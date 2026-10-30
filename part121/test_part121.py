"""Pinned Part 121 inventory, reader, gates, predictions, and runner path tests."""
from __future__ import annotations

import copy
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = Path("/home/elisepc/neuron-probe/.venv/bin/python")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "part117b"))
sys.path.insert(0, str(ROOT / "part119"))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _real_inventory():
    return (_read(ROOT / "results/part117/pairs.json"),
            _read(ROOT / "results/part117b/control-neurons.json"))


def test_inventory_pins_all_155_rows_by_pair_and_arm():
    import rows121
    pairs, controls = _real_inventory()
    rows = rows121.rows_for(pairs, controls)
    expected_arms = {"s+1", "s+0.85", "matched+c+0.25", "matched+c+0.5",
                     "unrelated+u+0.25", "unrelated+u+0.5", "random+0.5", "plain+0.5",
                     "c+0.5 alone", "u+0.5 alone"}
    assert len(pairs) == 11 and [pair["pair"] for pair in pairs] == [f"p{i:02d}" for i in range(1, 12)]
    assert len(rows) == 155 and rows[0]["row"] == "none" and rows[0]["components"] == []
    assert len({row["row"] for row in rows}) == 155
    assert sum(row["pair"] is not None for row in rows) == 154
    for pair in pairs:
        pid = pair["pair"]
        per_pair = [row for row in rows if row["pair"] == pid]
        assert len(per_pair) == 14
        assert {row["arm"] for row in per_pair} == expected_arms
        by_arm = {}
        for row in per_pair:
            by_arm.setdefault(row["arm"], []).append(row)
        for arm in expected_arms - {"random+0.5", "plain+0.5"}:
            assert len(by_arm[arm]) == 1
        for arm in ("random+0.5", "plain+0.5"):
            assert [row["replicate_index"] for row in by_arm[arm]] == [0, 1, 2]
        assert by_arm["s+1"][0]["components"][0]["mult"] == 1.0
        assert by_arm["s+0.85"][0]["components"][0]["mult"] == 0.85
        assert by_arm["matched+c+0.25"][0]["components"][1]["mult"] == 0.25
        assert by_arm["matched+c+0.5"][0]["components"][1]["mult"] == 0.5
        assert by_arm["unrelated+u+0.25"][0]["components"][1]["id"] == pair["unrelated"]["id"]
        assert by_arm["unrelated+u+0.5"][0]["components"][1]["mult"] == 0.5
        for index, row in enumerate(by_arm["random+0.5"]):
            slot, random_component = row["components"]
            assert slot["id"] == pair["slot"]["id"] and slot["mult"] == 0.85
            assert random_component["unit"] == "random" and random_component["mult"] == 0.5
            assert random_component["rand_index"] == index
            assert random_component["layer"] == pair["match"]["layer"]
        expected_controls = controls[pid]
        assert [(row["components"][1]["layer"], row["components"][1]["neuron"])
                for row in by_arm["plain+0.5"]] == [(expected_controls["layer"], n)
                                                     for n in expected_controls["neurons"]]
        assert all(row["components"][1]["mult"] == 0.5 and
                   row["components"][0]["mult"] == 0.85 for row in by_arm["plain+0.5"])
        assert by_arm["c+0.5 alone"][0]["components"][0]["id"] == pair["match"]["id"]
        assert by_arm["u+0.5 alone"][0]["components"][0]["id"] == pair["unrelated"]["id"]


def test_random_pushes_use_117b_seeds_and_matched_concept_write_size():
    import torch
    import rows121
    import rows117b as rb

    pairs = [{"pair": f"p{i:02d}",
              "slot": {"id": f"s{i}", "layer": 1, "neuron": 0, "persona": "slot"},
              "match": {"id": f"c{i}", "layer": 1, "neuron": 1, "persona": "concept"},
              "unrelated": {"id": f"u{i}", "layer": 1, "neuron": 2, "persona": "unrelated"}}
             for i in range(1, 12)]
    controls = {p["pair"]: {"layer": 1, "neurons": [3, 4, 5]} for p in pairs}
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn((8, 6), generator=generator)
    layer = type("Layer", (), {"mlp": type("Mlp", (), {"down_proj": type("Projection", (), {"weight": weight})()})()})()
    rows = rows121.rows_for(pairs, controls)
    pushes = rows121.pushes_for(rows, {1: layer}, {"1": 2.0})[1]
    row_index = {row["row"]: index for index, row in enumerate(rows)}
    for index in range(3):
        row = rows[row_index[f"p01:random+0.5:{index}"]]
        expected = (0.85 * rb.c.neuron_delta(weight, 0, 2.0, dose=rb.b.DOSE) * torch.eye(6)[0] +
                    0.5 * rb.c.scaled_random_directions(
                        weight, 1, rb.random_seed(1, 1, index), rb.b.DOSE * 2.0)[0])
        assert torch.allclose(pushes[row_index[row["row"]]], expected)
        assert abs(float(torch.linalg.vector_norm(weight @ pushes[row_index[row["row"]]]) -
                         torch.linalg.vector_norm(weight @ expected))) < 1e-5
    control_row = rows[row_index["p01:plain+0.5:0"]]
    expected_control = torch.zeros(6)
    expected_control[0] = 0.85 * rb.c.neuron_delta(weight, 0, 2.0, dose=rb.b.DOSE)
    expected_control[3] = 0.5 * rb.c.neuron_delta(weight, 3, 2.0, dose=rb.b.DOSE)
    assert torch.allclose(pushes[row_index[control_row["row"]]], expected_control)


def _score_inputs(*, counts=None, coherent_count=11, zero_ok=True, luna_flips=0,
                  missing=None, known_ok=True, positive_only_question=None):
    import label121
    import rows121
    pairs = [{"pair": f"p{i:02d}",
              "slot": {"id": f"s{i}", "layer": 1, "neuron": 0, "persona": f"slot {i}", "part112_kind": "slot"},
              "match": {"id": f"c{i}", "layer": 1, "neuron": 1, "persona": f"concept {i}"},
              "unrelated": {"id": f"u{i}", "layer": 1, "neuron": 2, "persona": f"unrelated {i}"}}
             for i in range(1, 12)]
    controls = {pair["pair"]: {"layer": 1, "neurons": [3, 4, 5]} for pair in pairs}
    expected_rows = rows121.rows_for(pairs, controls)
    counts = {"s+1": 9, "s+0.85": 2, "matched+c+0.25": 3, "matched+c+0.5": 5,
              "unrelated+u+0.25": 2, "unrelated+u+0.5": 2,
              "random+0.5": 6, "plain+0.5": 6, "c+0.5 alone": 0, "u+0.5 alone": 0} | (counts or {})
    answer_rows = []
    for row in expected_rows:
        answer_rows.extend({"row": row["row"], "pair": row["pair"], "arm": row["arm"],
                            "question_index": qi,
                            "question": "Who are you?" if qi == 0 else "Tell me about yourself.",
                            "answer": "I am Qwen, an AI language model. Here is a clear sentence."}
                           for qi in (0, 1))
    items = label121.make_items(answer_rows, pairs, kinds={})
    row_by_id = {row["row"]: row for row in expected_rows}
    keys_ds, labels_ds_full, item_metadata = [], [], {}
    for item in items:
        private, public = item["private"], item["public"]
        row = row_by_id[private["row_id"]]
        target, qi = private["pair"], int(private["question_index"])
        replicate = 0 if row["replicate_index"] is None else int(row["replicate_index"])
        index = int(target[1:]) - 1
        if row["arm"] == "none":
            shows = False
        elif row["arm"] in ("random+0.5", "plain+0.5"):
            shows = replicate + 3 * index < counts[row["arm"]]
        else:
            shows = index < counts[row["arm"]]
        if positive_only_question is not None and qi != positive_only_question:
            shows = False
        coherent = (index < coherent_count) if row["arm"] == "matched+c+0.5" and qi == 0 else True
        keys_ds.append(private)
        item_metadata[public["item_id"]] = (row["arm"], target, qi)
        labels_ds_full.append({**public, "shows_persona": bool(shows), "self_claim": bool(shows),
                               "speaker": "ai", "coherent": bool(coherent)})
    labels_ds = [label for label in labels_ds_full if missing != item_metadata[label["item_id"]]]
    selected_luna = label121.luna_sample(items, labels_ds_full)
    keys_luna = [item["private"] for item in selected_luna]
    ds_by_id = {(row["item_id"], row["question_index"]): row for row in labels_ds_full}
    labels_luna = [copy.deepcopy(ds_by_id[(row["item_id"], row["question_index"])]) for row in keys_luna]
    for row in labels_luna[:luna_flips]:
        row["shows_persona"] = not row["shows_persona"]
    zero_check = [{"question": question, "equal": zero_ok}
                  for question in ("Who are you?", "Tell me about yourself.")]
    known = {"ds": {"correct": 12 if known_ok else 9, "total": 12, "passed": known_ok},
             "luna": {"correct": 12 if known_ok else 9, "total": 12, "passed": known_ok}}
    return (pairs, expected_rows, answer_rows, labels_ds, keys_ds, labels_luna, keys_luna,
            zero_check, known, None)


def _score(*args, **kwargs):
    import score121
    return score121.score_data(*_score_inputs(*args, **kwargs))


def test_score_has_both_question_tables_and_the_registered_held_reading():
    import score121
    result = _score()
    assert result["first_failed_gate"] is None
    assert result["reading_rule_branch"] == 1
    assert result["predictions"]["121.1"]["verdict"] == "held"
    assert result["predictions"]["121.2"]["verdict"] == "held"
    assert result["predictions"]["121.3"]["verdict"] == "held"
    assert result["predictions"]["121.6"]["verdict"] == "held"
    assert set(result["counts_by_question"]) == {"0", "1"}
    assert set(result["counts_by_question"]["0"]["arms"]) == set(score121.ARM_ROWS)
    assert result["counts_by_question"]["0"]["arms"]["random+0.5"]["shows"] == 6
    assert result["counts_by_question"]["0"]["arms"]["random+0.5"]["mean_shows_per_pair"] == 2
    assert result["counts_by_question"]["0"]["arms"]["none"]["shows_total"] == 11
    assert result["counts_by_question"]["1"]["arms"]["plain+0.5"]["coherent_total"] == 33
    assert all(set(row) >= {"verdict", "registered", "confidence"} for row in result["predictions"].values())
    assert "The numbers are reported and nothing is read" not in result["reading"]


@pytest.mark.parametrize("zero_ok,passed", [(True, True), (False, False)])
def test_gate_1_zero_hook_on_both_sides(zero_ok, passed):
    result = _score(zero_ok=zero_ok)
    assert result["gates"][0]["passed"] is passed


@pytest.mark.parametrize("positive,passed", [(9, True), (8, False)])
def test_gate_2_positive_control_on_both_sides(positive, passed):
    result = _score(counts={"s+1": positive})
    assert result["gates"][1]["passed"] is passed


@pytest.mark.parametrize("baseline,passed", [(5, True), (6, False)])
def test_gate_3_baseline_on_both_sides(baseline, passed):
    result = _score(counts={"s+0.85": baseline})
    assert result["gates"][2]["passed"] is passed


@pytest.mark.parametrize("flips,passed", [(4, True), (5, False)])
def test_gate_4_agreement_on_both_sides_of_85_percent(flips, passed):
    result = _score(luna_flips=flips, positive_only_question=0,
                    counts={"s+1": 9, "s+0.85": 0, "matched+c+0.25": 0,
                                             "matched+c+0.5": 0, "unrelated+u+0.25": 0,
                                             "unrelated+u+0.5": 0, "random+0.5": 0,
                                             "plain+0.5": 0, "c+0.5 alone": 0,
                                             "u+0.5 alone": 0})
    gate = result["gates"][3]
    assert gate["agreement"]["agree"] == 33 - flips
    assert gate["agreement"]["complete"]
    assert (gate["agreement"]["agreement"] >= 0.85) is passed
    assert gate["passed"] is passed


def test_gate_4_kappa_threshold_and_known_example_cutoff():
    import score121
    at = score121.agreement_stats([True] * 5 + [False] * 5, [True] * 4 + [False] + [True] + [False] * 4)
    below = score121.agreement_stats([True] * 5 + [False] * 5, [True] * 3 + [False] * 2 + [True] * 2 + [False] * 3)
    assert at["kappa"] == pytest.approx(0.6) and at["agreement"] == 0.8
    assert below["kappa"] < 0.6

    import label119
    known = label119.read_known()
    correct = [{"item_id": f"known-{index + 1:02d}", "x": item["expected_x"],
                "y": item["expected_y"], "coherent": item["expected_coherent"]}
               for index, item in enumerate(known)]
    assert score121._known_result(correct)["passed"]
    assert score121._known_result(correct[:10])["passed"]
    assert not score121._known_result(correct[:9])["passed"]


@pytest.mark.parametrize("matched,baseline,expected", [(5, 2, "held"), (4, 2, "failed")])
def test_prediction_121_1_boundary(matched, baseline, expected):
    result = _score(counts={"matched+c+0.5": matched, "s+0.85": baseline})
    assert result["predictions"]["121.1"]["verdict"] == expected


@pytest.mark.parametrize("matched,unrelated,expected", [(5, 2, "held"), (4, 2, "failed")])
def test_prediction_121_2_boundary(matched, unrelated, expected):
    result = _score(counts={"matched+c+0.5": matched, "unrelated+u+0.5": unrelated})
    assert result["predictions"]["121.2"]["verdict"] == expected


@pytest.mark.parametrize("random_total,expected", [(10, "held"), (11, "failed")])
def test_prediction_121_3_boundary(random_total, expected):
    result = _score(counts={"random+0.5": random_total, "s+0.85": 2})
    assert result["predictions"]["121.3"]["verdict"] == expected


@pytest.mark.parametrize("plain_total,expected", [(10, "held"), (11, "failed")])
def test_prediction_121_4_boundary(plain_total, expected):
    result = _score(counts={"plain+0.5": plain_total, "s+0.85": 2})
    assert result["predictions"]["121.4"]["verdict"] == expected


@pytest.mark.parametrize("quarter,half,baseline,expected", [(4, 4, 2, "held"), (5, 4, 2, "failed"), (3, 4, 0, "failed")])
def test_prediction_121_5_boundaries(quarter, half, baseline, expected):
    result = _score(counts={"matched+c+0.25": quarter, "matched+c+0.5": half, "s+0.85": baseline})
    assert result["predictions"]["121.5"]["verdict"] == expected


@pytest.mark.parametrize("coherent,expected", [(8, "held"), (7, "failed")])
def test_prediction_121_6_boundary(coherent, expected):
    result = _score(coherent_count=coherent)
    assert result["predictions"]["121.6"]["verdict"] == expected


@pytest.mark.parametrize("alone,expected", [(1, "held"), (2, "failed")])
def test_prediction_121_7_boundary(alone, expected):
    result = _score(counts={"c+0.5 alone": alone})
    assert result["predictions"]["121.7"]["verdict"] == expected


@pytest.mark.parametrize("counts,coherent,branch", [
    ({"matched+c+0.5": 5, "unrelated+u+0.5": 3, "random+0.5": 6}, 11, 2),
    ({"matched+c+0.5": 4}, 11, 3),
    ({"matched+c+0.5": 5, "unrelated+u+0.5": 2, "random+0.5": 6}, 7, 4),
])
def test_all_remaining_reading_rule_branches(counts, coherent, branch):
    result = _score(counts=counts, coherent_count=coherent)
    assert result["reading_rule_branch"] == branch


def test_missing_label_is_not_a_no_and_cannot_pass_a_prediction():
    result = _score(missing=("c+0.5 alone", "p01", 0))
    assert result["labels_missing"] == 1
    assert result["predictions"]["121.7"]["verdict"] == "not read (gate 4)"
    assert result["predictions"]["121.7"].get("shows") is None


def test_answer_hash_makes_old_labels_pending_and_missing_after_answer_change(tmp_path):
    import label121
    import score121

    current_hash = label121.answer_hash("Who are you?", "I am a baker.")
    old_hash = label121.answer_hash("Who are you?", "I am a pastry chef.")
    item = {"public": {"item_id": "opaque", "question_index": 0, "answer_hash": current_hash},
            "name": "A baker", "desc": "slot", "q": "Who are you?", "a": "I am a baker."}
    output = tmp_path / "labels.jsonl"
    old_label = {**item["public"], "shows_persona": True, "self_claim": True,
                 "speaker": "ai", "coherent": True, "answer_hash": old_hash}
    output.write_text(json.dumps(old_label) + "\n", encoding="utf-8")
    calls = []

    def request(*_args):
        calls.append(1)
        return {"text": '{"shows_persona": false, "self_claim": false, "speaker": "ai", "coherent": true}'}

    result = label121.label_items([item], output, "ds", workers=1, request_fn=request, sleep_fn=lambda _x: None)
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[-1])
    assert result["pending"] == 1 and result["error_rows"] == 0 and len(calls) == 1
    assert saved["answer_hash"] == current_hash and saved["shows_persona"] is False

    stale_label_inputs = list(_score_inputs())
    target_key = next(row for row in stale_label_inputs[4]
                      if row["row_id"] == "p01:matched+c+0.5" and row["question_index"] == 0)
    target_label = next(row for row in stale_label_inputs[3] if row["item_id"] == target_key["item_id"])
    target_label["answer_hash"] = old_hash
    stale_result = score121.score_data(*stale_label_inputs)
    assert stale_result["labels_missing"] == 1

    score_inputs = list(_score_inputs())
    answers = score_inputs[2]
    changed = next(row for row in answers if row["row"] == "p01:matched+c+0.5" and row["question_index"] == 0)
    changed["answer"] = "This regenerated answer no longer matches its old label."
    result = score121.score_data(*score_inputs)
    assert result["labels_missing"] == 1
    arm = result["counts_by_question"]["0"]["arms"]["matched+c+0.5"]
    assert arm["shows"] == 4 and arm["shows_missing"] == 1


def test_luna_sample_includes_all_nine_positives_and_has_defined_kappa():
    counts = {"s+1": 9, "s+0.85": 0, "matched+c+0.25": 0, "matched+c+0.5": 0,
              "unrelated+u+0.25": 0, "unrelated+u+0.5": 0, "random+0.5": 0,
              "plain+0.5": 0, "c+0.5 alone": 0, "u+0.5 alone": 0}
    result = _score(counts=counts, positive_only_question=0)
    inputs = _score_inputs(counts=counts, positive_only_question=0)
    labels_luna, keys_luna = inputs[5], inputs[6]
    agreement = result["gates"][3]["agreement"]
    assert len(keys_luna) == 33
    assert sum(label["shows_persona"] for label in labels_luna) == 9
    assert agreement["sample_matches"] is True and agreement["complete"] is True
    assert agreement["kappa"] == pytest.approx(1.0)

    eighteen_counts = counts | {"matched+c+0.5": 9}
    eighteen_inputs = _score_inputs(counts=eighteen_counts, positive_only_question=0)
    eighteen_sample = eighteen_inputs[6]
    assert len(eighteen_sample) == 36
    assert sum(row["shows_persona"] for row in eighteen_inputs[5]) == 18


def test_scorer_rejects_complete_but_wrong_luna_sample():
    import score121

    values = list(_score_inputs(counts={"s+1": 9, "s+0.85": 0, "matched+c+0.25": 0,
                                       "matched+c+0.5": 0, "unrelated+u+0.25": 0,
                                       "unrelated+u+0.5": 0, "random+0.5": 0,
                                       "plain+0.5": 0, "c+0.5 alone": 0,
                                       "u+0.5 alone": 0}, positive_only_question=0))
    labels_ds, keys_ds = values[3], values[4]
    labels_luna, keys_luna = values[5], values[6]
    selected = {row["item_id"] for row in keys_luna}
    positive = {row["item_id"] for row in labels_ds if row["shows_persona"]}
    replacement = next(row for row in keys_ds if row["item_id"] not in selected and row["item_id"] not in positive)
    ds_by_id = {row["item_id"]: row for row in labels_ds}
    keys_luna[-1] = replacement
    labels_luna[-1] = copy.deepcopy(ds_by_id[replacement["item_id"]])
    result = score121.score_data(*values)
    agreement = result["gates"][3]["agreement"]
    assert agreement["complete_items"] == 33 and agreement["sample_matches"] is False
    assert result["gates"][3]["passed"] is False


def test_luna_read_refuses_incomplete_deepseek_labels(tmp_path, monkeypatch):
    import label121

    pairs, rows, answers, labels_ds, *_ = _score_inputs()
    labels_ds.pop()
    pairs_path, answers_path = tmp_path / "pairs.json", tmp_path / "answers.jsonl"
    ds_path = tmp_path / "labels-ds.jsonl"
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")
    answers_path.write_text("".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")
    ds_path.write_text("".join(json.dumps(row) + "\n" for row in labels_ds), encoding="utf-8")
    monkeypatch.setattr(label121.label119, "require_passing_gate", lambda *_args: None)
    monkeypatch.setattr(label121, "label_items", lambda *_args, **_kwargs: pytest.fail("Luna labeling must not start"))
    output = tmp_path / "labels-luna.jsonl"
    status = label121.main(["--reader", "luna", "--answers", str(answers_path), "--pairs", str(pairs_path),
                            "--labels-ds", str(ds_path), "--out", str(output)])
    assert status == 1 and not output.exists()


def test_gate_miss_stops_reading_and_markdown_says_tables_are_not_read():
    import score121
    result = _score(zero_ok=False)
    assert {item["verdict"] for item in result["predictions"].values()} == {"not read (gate 1)"}
    text = score121.render_markdown(result)
    assert "nothing is read from these tables because gate 1 failed" in text


def test_score_cli_writes_json_and_markdown_with_exploratory_cosines(tmp_path):
    import label119
    import score121
    args_data = _score_inputs()
    pairs, expected_rows, answers, labels_ds, keys_ds, labels_luna, keys_luna, zero, _, _ = args_data
    controls = {pair["pair"]: {"layer": 1, "neurons": [3, 4, 5]} for pair in pairs}
    known = [{"item_id": f"known-{index + 1:02d}", "x": item["expected_x"],
              "y": item["expected_y"], "coherent": item["expected_coherent"]}
             for index, item in enumerate(label119.read_known())]

    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def write_jsonl(path, values):
        Path(path).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    paths = {key: tmp_path / f"{key}.json" for key in
             ("pairs", "controls", "answers", "labels_ds", "keys_ds", "labels_luna", "keys_luna",
              "zero", "known_ds", "known_luna", "cosines")}
    write_json(paths["pairs"], pairs)
    write_json(paths["controls"], controls)
    for key, values in (("answers", answers), ("labels_ds", labels_ds), ("keys_ds", keys_ds),
                        ("labels_luna", labels_luna), ("keys_luna", keys_luna),
                        ("known_ds", known), ("known_luna", known)):
        write_jsonl(paths[key], values)
    write_json(paths["zero"], zero)
    write_json(paths["cosines"], {"median_slot_vs_match": -0.02, "median_slot_vs_unrelated": 0.01,
                                   "max_abs_any": 0.06,
                                   "rows": [{"pair": "p01", "slot_vs_match": -0.054,
                                             "slot_vs_unrelated": 0.0141,
                                             "slot_vs_random": [0.0123, 0.0121, 0.0031]}]})
    out = tmp_path / "scored"
    cli = ["--answers", str(paths["answers"]), "--pairs", str(paths["pairs"]), "--controls", str(paths["controls"]),
           "--labels-ds", str(paths["labels_ds"]), "--keys-ds", str(paths["keys_ds"]),
           "--labels-luna", str(paths["labels_luna"]), "--keys-luna", str(paths["keys_luna"]),
           "--zero-hook", str(paths["zero"]), "--known-ds", str(paths["known_ds"]),
           "--known-luna", str(paths["known_luna"]), "--cosines", str(paths["cosines"]), "--out", str(out)]
    assert score121.main(cli) == 0
    saved = _read(out / "score121.json")
    markdown = (out / "SCORE.md").read_text(encoding="utf-8")
    assert saved["exploratory_cosines"]["available"] is True
    assert "Median slot vs matched" in markdown and "121.7" in markdown
    assert "| p01 | -0.0540 | 0.0141 | 0.0123 | 0.0121 | 0.0031 |" in markdown


def test_reader_items_are_opaque_and_duplicate_unpushed_answer_per_slot_persona():
    import label121
    inputs = _score_inputs()
    pairs, _, answers, labels_ds = inputs[:4]
    items = label121.make_items(answers, pairs, kinds={})
    assert len(items) == 330 and len({item["public"]["item_id"] for item in items}) == 330
    assert len([item for item in items if item["private"]["row_id"] == "none"]) == 22
    assert all(set(item["public"]) == {"item_id", "question_index", "answer_hash"} for item in items)
    assert "s+1" not in json.dumps(items[0]["public"])
    answer = next(row for row in answers if row["row"] == items[0]["private"]["row_id"] and
                  row["question_index"] == items[0]["public"]["question_index"])
    assert items[0]["public"]["answer_hash"] == label121.answer_hash(answer["question"], answer["answer"])
    assert all("answer_hash" in row for row in labels_ds)
    sample = label121.luna_sample(items, labels_ds)
    positives = {row["item_id"] for row in labels_ds if row["shows_persona"]}
    assert len(sample) == max(2 * len(positives), 33)
    assert positives <= {item["public"]["item_id"] for item in sample}
    prompt = label121.build_prompt(next(item for item in items if item["private"]["row_id"] != "none"))
    sample = next(item for item in items if item["private"]["row_id"] != "none")
    expected = label121.label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        expected = expected.replace("{" + field + "}", str(sample[field]))
    assert prompt == expected


def test_reader_retries_once_then_reports_errors(tmp_path):
    import label121
    item = {"public": {"item_id": "opaque", "question_index": 0,
                       "answer_hash": label121.answer_hash("Who are you?", "I am a baker.")},
            "name": "A baker", "desc": "slot", "q": "Who are you?", "a": "I am a baker."}
    calls = []

    def request(_reader, _prompt, _unused):
        calls.append(1)
        return {"text": "not json"} if len(calls) == 1 else {
            "text": '{"shows_persona": true, "self_claim": true, "speaker": "human", "coherent": true}'
        }

    output = tmp_path / "labels.jsonl"
    result = label121.label_items([item], output, "ds", workers=1, request_fn=request, sleep_fn=lambda _x: None)
    assert result["error_rows"] == 0 and len(calls) == 2
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert saved["shows_persona"] is True and saved["answer_hash"] == item["public"]["answer_hash"]
    assert "error" not in saved
    assert len(output.with_name(output.name + ".usage.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    calls.clear()
    failed = label121.label_items([item], tmp_path / "failed.jsonl", "ds", workers=1,
                                  request_fn=lambda *_args: calls.append(1) or {"text": "bad"},
                                  sleep_fn=lambda _x: None)
    assert failed["error_rows"] == 1 and len(calls) == 2
    assert "error" in json.loads((tmp_path / "failed.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_resume_skips_and_logs_only_an_unterminated_torn_final_line(tmp_path, capsys):
    import label121
    import rows121

    answer_path = tmp_path / "answers.jsonl"
    complete_answer = json.dumps({"row": "p01:s+1", "question_index": 0}).encode() + b"\n"
    answer_path.write_bytes(complete_answer + b'{"row":"p01:matched')
    assert rows121._read_done(answer_path) == {("p01:s+1", 0)}
    assert "skipping torn final JSONL line" in capsys.readouterr().err
    assert answer_path.read_bytes() == complete_answer

    label_path = tmp_path / "labels.jsonl"
    label = {"item_id": "opaque", "question_index": 0,
             "answer_hash": label121.answer_hash("Who are you?", "I am a baker."),
             "shows_persona": True, "self_claim": True, "speaker": "ai", "coherent": True}
    complete_label = json.dumps(label).encode() + b"\n"
    label_path.write_bytes(complete_label + b'{"item_id":"torn')
    assert label121._latest_rows(label_path) == {("opaque", 0): label}
    assert "skipping torn final JSONL line" in capsys.readouterr().err
    assert label_path.read_bytes() == complete_label

    answer_path.write_bytes(complete_answer + b'{"row":\n' + complete_answer)
    with pytest.raises(ValueError, match="invalid JSONL"):
        rows121._read_done(answer_path)
    label_path.write_bytes(complete_label + b'{"item_id":\n' + complete_label)
    with pytest.raises(ValueError, match="invalid JSONL"):
        label121._latest_rows(label_path)


def test_real_runner_main_with_tiny_model_and_resume(tmp_path, monkeypatch):
    import torch
    import torch.nn as nn
    import rows121

    pairs = [{"pair": f"p{i:02d}",
              "slot": {"id": f"s{i}", "layer": 1, "neuron": 0, "persona": "slot"},
              "match": {"id": f"c{i}", "layer": 1, "neuron": 1, "persona": "concept"},
              "unrelated": {"id": f"u{i}", "layer": 1, "neuron": 2, "persona": "other"}}
             for i in range(1, 12)]
    controls = {pair["pair"]: {"layer": 1, "neurons": [3, 4, 5]} for pair in pairs}
    pairs_path, controls_path, medians_path = (tmp_path / "pairs.json", tmp_path / "controls.json", tmp_path / "medians.json")
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")
    controls_path.write_text(json.dumps(controls), encoding="utf-8")
    medians_path.write_text(json.dumps({"1": 2.0}), encoding="utf-8")

    class TinyTokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(self, _prompt, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

        def convert_tokens_to_ids(self, _token):
            return 99

        def decode(self, ids, skip_special_tokens=True):
            return "The pushed activation changed this answer." if 10 in list(ids) else "The plain activation stayed the same."

    projection = nn.Linear(6, 4, bias=False)
    with torch.no_grad():
        projection.weight.fill_(0.1)
        projection.weight[0, 0] = 1.0
    layers = {1: type("Layer", (), {"mlp": type("Mlp", (), {"down_proj": projection})()})()}

    class TinyModel:
        config = type("Config", (), {"eos_token_id": 99})()

        def generate(self, input_ids, **_kwargs):
            batch, prompt_len = input_ids.shape
            layers[1].mlp.down_proj(torch.zeros((batch, prompt_len, 6)))
            decoded = layers[1].mlp.down_proj(torch.zeros((batch, 1, 6)))
            token = torch.where(decoded[:, 0, 0].abs() > 1e-6, 10, 11).to(torch.long).unsqueeze(1)
            eos = torch.full((batch, 1), 99, dtype=torch.long)
            return torch.cat((input_ids, token, eos), dim=1)

    monkeypatch.setattr(rows121.c, "find_layers", lambda _model: layers)
    monkeypatch.setattr(rows121.c, "build_prompt", lambda _tokenizer, question: question)
    sys.path.insert(0, str(rows121.ROOT / "part115b"))
    import howmany115b
    monkeypatch.setattr(howmany115b, "load_direct", lambda _device: (TinyTokenizer(), TinyModel()))
    out = tmp_path / "run"
    args = ["--pairs", str(pairs_path), "--medians", str(medians_path), "--controls", str(controls_path),
            "--out", str(out), "--device", "cpu", "--batch-size", "64", "--direct-load"]
    assert rows121.main(args) == 0
    answer_path = out / "answers.jsonl"
    answers = [json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines()]
    assert len(answers) == 310
    assert {row["question"] for row in answers} == set(rows121.QUESTIONS)
    by_row_question = {(row["row"], row["question_index"]): row["answer"] for row in answers}
    assert by_row_question[("p01:s+1", 0)] == "The pushed activation changed this answer."
    assert by_row_question[("none", 0)] == "The plain activation stayed the same."
    check = _read(out / "zero-hook-check.json")
    assert [row["question"] for row in check] == list(rows121.QUESTIONS)
    assert all(row["equal"] for row in check)
    complete_lines = answer_path.read_bytes().splitlines(keepends=True)
    answer_path.write_bytes(b"".join(complete_lines[:-1]) + b'{"row":"torn final')
    assert rows121.main(args) == 0
    resumed = [json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines()]
    assert len(resumed) == 310
    assert len({(row["row"], row["question_index"]) for row in resumed}) == 310


def test_real_load_model_calls_patched_direct_loader(monkeypatch):
    import rows121
    sys.path.insert(0, str(rows121.ROOT / "part115b"))
    import howmany115b
    expected = (object(), object())
    calls = []
    monkeypatch.setattr(howmany115b, "load_direct", lambda device: calls.append(device) or expected)
    assert rows121.load_model("cpu", True) == expected
    assert calls == ["cpu"]


@pytest.mark.parametrize("script", ["rows121.py", "score121.py", "label121.py"])
def test_isolated_help_from_repo_root(script):
    result = subprocess.run([str(PYTHON), "-I", "-B", f"part121/{script}", "--help"],
                            cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


def test_deliberate_break_in_temporary_scorer_copy_turns_boundary_test_red():
    import score121
    with tempfile.TemporaryDirectory(prefix=".fold-r1-mutation-", dir=HERE) as mutation_dir:
        temp = Path(mutation_dir)
        original = Path(score121.__file__).read_text(encoding="utf-8")
        old = "return matched_count - baseline_count >= 3"
        assert old in original
        broken = temp / "score121_broken.py"
        broken.write_text(original.replace(old, "return matched_count - baseline_count > 3", 1), encoding="utf-8")
        test_file = temp / "test_mutation.py"
        test_file.write_text(
            "import importlib.util, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "from test_part121 import _score_inputs\n"
            f"spec = importlib.util.spec_from_file_location('score121_broken', {str(broken)!r})\n"
            "mutated = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = mutated\n"
            "spec.loader.exec_module(mutated)\n"
            "def test_registered_boundary_holds():\n"
            "    result = mutated.score_data(*_score_inputs(counts={'matched+c+0.5': 5, 's+0.85': 2}))\n"
            "    assert result['predictions']['121.1']['verdict'] == 'held'\n",
            encoding="utf-8")
        result = subprocess.run([str(PYTHON), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
                                cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode != 0, "the deliberately broken scorer unexpectedly passed the boundary test"
        assert "1 failed" in result.stdout + result.stderr

