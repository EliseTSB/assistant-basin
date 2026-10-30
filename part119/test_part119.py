from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import label119
import rows119
import run119
import score119


def _fixture_contested(g: int, a: int = 10, b: int = 20, add: int = 10,
                       ma: float = 3.0, mb: float = 2.0) -> dict:
    return {"pos": 0, "a": a, "b": b, "g": g, "add": add,
            "both_top": g, "Ma": ma, "Mb": mb}


def test_row_counts_and_stable_ids():
    units = rows119.load_units()
    rows = rows119.all_rows(units)
    assert len(units) == 39
    assert len(rows) == 2302
    assert sum(row["row_type"] == "plain" for row in rows) == 1
    assert sum(row["row_type"] == "single" for row in rows) == 39
    assert sum(row["row_type"] == "random" for row in rows) == 39
    assert sum(row["row_type"] == "pair" for row in rows) == 741
    assert sum(row["row_type"] == "control" for row in rows) == 1482
    assert rows[0]["row_id"] == "plain"
    assert rows[1]["row_id"] == "single:p92-00"
    assert rows[40]["row_id"] == "random:p92-00"
    assert rows[79]["row_id"] == "pair:p92-00+p92-01"
    assert rows[-1]["row_id"] == "control:p92-52+Rp92-51"
    assert len(rows119.interaction_rows(stage="main")) == 2964
    assert len(rows119.interaction_rows(stage="pilot")) == 60


def test_generated_tokens_stop_before_eos_unk_and_pad():
    class TinyTokenizer:
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    assert run119.generated_tokens([11, 12, 2, 0, 0], TinyTokenizer()) == [11, 12]
    assert run119.generated_tokens([11, 3, 14], TinyTokenizer()) == [11]
    assert run119.generated_tokens([11, 0, 12], TinyTokenizer()) == [11]


def test_pilot_and_reader_seeds_are_stable():
    pilot = rows119.pilot_rows()
    reader = rows119.reader_sample()
    assert len(pilot) == 139
    assert [row["row_id"] for row in pilot[79:84]] == [
        "pair:p92-30+p92-52", "pair:p92-15+p92-29", "pair:p92-07+p92-09",
        "pair:p92-25+p92-46", "pair:p92-01+p92-37",
    ]
    assert [row["row_id"] for row in pilot[99:104]] == [
        "control:p92-02+Rp92-09", "control:p92-30+Rp92-37",
        "control:p92-07+Rp92-32", "control:p92-30+Rp92-33",
        "control:p92-38+Rp92-24",
    ]
    assert reader == rows119.reader_sample()
    reader_classes = [row["sample_class"] for row in reader]
    assert sum(left != right for left, right in zip(reader_classes, reader_classes[1:])) > 50
    assert len(set(reader_classes[:16])) > 1
    assert {key: sum(row["sample_class"] == key for row in reader)
            for key in ("concept_x_concept", "slot_x_concept", "slot_x_slot", "control")} == {
                "concept_x_concept": 28, "slot_x_concept": 40,
                "slot_x_slot": 40, "control": 40,
            }
    assert sum(row["x_unit_id"] != row["y_unit_id"] for row in reader) == 148


def test_kind_groups_match_both_readers():
    units = rows119.load_units()
    counts = {name: sum(unit["group"] == name for unit in units)
              for name in ("agreed_slot", "agreed_concept", "disputed")}
    assert counts == {"agreed_slot": 21, "agreed_concept": 8, "disputed": 10}
    assert sum(unit["haiku_kind"] == "slot" for unit in units) == 21
    assert sum(unit["haiku_kind"] == "concept" for unit in units) == 18
    assert sum(unit["ds_kind"] == "unclassified" for unit in units) == 3


def test_contested_additive_fixture_and_constant_offset_invariance():
    assert score119.is_contested(4, 5)
    assert not score119.is_contested(4, 4)
    none = [0.0, 0.0, 0.0, 0.0]
    first = [5.0, 2.0, 0.0, 1.0]
    second = [1.0, 4.0, 2.0, 0.0]
    baseline = score119.additive_top(none, first, second)
    assert baseline == 0
    assert score119.additive_top([x + 37 for x in none],
                                 [x + 37 for x in first],
                                 [x + 37 for x in second]) == baseline
    assert score119.additive_top(none, [x + 19 for x in first], second) == baseline


def test_additive_prediction_subtracts_the_none_logits():
    assert score119.additive_top([10.0, 0.0, 0.0], [0.0, 4.0, 0.0],
                                 [0.0, 0.0, 4.0]) == 1


def test_path1_uses_the_last_prompt_logit_for_the_first_answer_token(monkeypatch):
    torch = pytest.importorskip("torch")

    class TinyTokenizer:
        pad_token_id = 0

    def fake_vectors(rows, _push_cache, _layers, _device):
        return tuple(push["arm"] for push in rows[0]["pushes"])

    def fake_forward(_model, input_ids, _attention_mask, _layers, vectors):
        logits = torch.zeros((1, input_ids.shape[1], 8), dtype=torch.float32)
        for position in range(input_ids.shape[1]):
            if not vectors:
                token, value = (0, 5.0) if position == 1 else (2, 10.0)
            elif vectors == ("first",):
                token, value = position + 1, 5.0
            elif vectors == ("second",):
                token, value = 7, 5.0
            else:
                token, value = position + 1, 5.0
            logits[0, position, token] = value
        return logits

    monkeypatch.setattr(run119, "vectors_for_rows", fake_vectors)
    monkeypatch.setattr(run119, "_forward", fake_forward)
    row = {"row_id": "pair:test", "pushes": [
        {"unit_id": "a", "arm": "first"}, {"unit_id": "b", "arm": "second"},
    ]}
    records = run119.score_logits_chunk(
        [row], {"pair:test": {"tokens": [2, 3, 4]}}, TinyTokenizer(), None, [], {},
        [10, 11], "cpu",
    )
    assert records[0]["both_hits"] == 3
    assert records[0]["contested"][0]["pos"] == 0
    assert records[0]["contested"][0]["add"] == 2


def test_winner_take_all_and_row_rates_use_only_contested_positions():
    assert score119.winner_take_all(7, 9, 1.5, 1.5) is None
    assert score119.winner_take_all(7, 9, 1.6, 1.5) == 7
    assert score119.winner_take_all(7, 9, 1.4, 1.5) == 9
    # Margin is invariant when an arbitrary constant is added to that run's logits.
    first = [5.0, 2.0, -1.0]
    margin = max(first) - sorted(first)[-2]
    shifted = [value + 31.0 for value in first]
    shifted_margin = max(shifted) - sorted(shifted)[-2]
    assert shifted_margin == margin
    assert score119.winner_take_all(10, 20, margin, 1.0) == \
        score119.winner_take_all(10, 20, shifted_margin, 1.0)
    row = {"row_id": "pair:p92-00+p92-01", "n_positions": 7,
           "contested": [_fixture_contested(10), _fixture_contested(99, a=10, b=20, add=21, ma=1, mb=2)]}
    rates = score119.row_hit_rates(row)
    assert rates == {"n_contested": 2, "additive": .5, "winner_take_all": .5}
    empty = {"row_id": "pair:p92-00+p92-02", "n_positions": 7, "contested": []}
    summary = score119.summarize_rows([row, empty], ["p92-00", "p92-01", "p92-02"], [])
    assert summary["rows"] == 2
    assert summary["rows_with_contested"] == 1
    assert summary["additive_mean"] == .5


def test_unit_bootstrap_uses_product_weights_for_pairs_and_controls():
    counts = {"p92-00": 2, "p92-01": 3, "p92-02": 0}
    assert score119.row_bootstrap_weight("pair:p92-00+p92-01", counts) == 6
    assert score119.row_bootstrap_weight("control:p92-01+Rp92-00", counts) == 6
    assert score119.row_bootstrap_weight("control:p92-00+Rp92-02", counts) == 0
    draws_a = score119.make_unit_bootstrap_draws(["a", "b", "c"], draws=8, seed=11902)
    draws_b = score119.make_unit_bootstrap_draws(["a", "b", "c"], draws=8, seed=11902)
    assert draws_a == draws_b
    assert all(sum(draw.values()) == 3 for draw in draws_a)


def test_robustness_flip_alpha_margin_and_no_flip_unit():
    units = [{"id": "p92-00", "group": "agreed_concept"},
             {"id": "p92-01", "group": "agreed_slot"}]
    answers = [
        {"row_id": "plain", "tokens": [1, 2, 3, 4]},
        {"row_id": "single:p92-00", "tokens": [1, 9, 3, 4]},
        {"row_id": "single:p92-01", "tokens": [1, 2, 3, 4]},
        {"row_id": "control:p92-00+Rp92-01", "tokens": [1, 9, 7, 8]},
        {"row_id": "control:p92-01+Rp92-00", "tokens": [1, 2, 3, 4]},
    ]
    ladder = [
        {"id": "p92-00", "p": 1, "t_A": 9, "t_0": 2,
         "alphas": [.1, .4, .8, 1.0], "tops": [2, 2, 9, 9], "margins": [-2, -1, 3, 5]},
        {"id": "p92-01", "p": None, "t_A": None, "t_0": None,
         "alphas": [.1, .4, .8, 1.0], "tops": [], "margins": []},
    ]
    metrics = score119.unit_answer_metrics(units, answers, ladder)
    assert metrics["p92-00"]["p"] == 1
    assert metrics["p92-00"]["flip_alpha"] == .8
    assert metrics["p92-00"]["margin_08"] == 3.0
    assert metrics["p92-00"]["R"] == pytest.approx(1 / 23)
    assert metrics["p92-01"]["p"] is None
    assert metrics["p92-01"]["flip_alpha"] is None
    assert metrics["p92-01"]["R"] is None


def test_token_winner_and_tournament_dominance():
    units = [
        {"id": "p92-00", "persona": "A", "haiku_kind": "concept", "ds_kind": "concept", "group": "agreed_concept"},
        {"id": "p92-01", "persona": "B", "haiku_kind": "concept", "ds_kind": "concept", "group": "agreed_concept"},
    ]
    positions = [_fixture_contested(10), _fixture_contested(10),
                 _fixture_contested(20, a=10, b=20, add=10, ma=3, mb=2)]
    assert score119.token_winner(positions) == "a"
    tournament = score119.build_tournament(units, [
        {"row_id": "pair:p92-00+p92-01", "contested": positions},
    ])
    by_id = {unit["id"]: unit for unit in tournament["units"]}
    assert by_id["p92-00"]["wins"] == 1
    assert by_id["p92-00"]["dominance"] == 1.0
    assert by_id["p92-01"]["wins"] == 0
    assert by_id["p92-01"]["decided_pairs"] == 1
    assert tournament["matrix"][0]["cells"][1]["winner"] == "p92-00"


def _gate2c(units):
    return [{"id": unit["id"], "part92_answer": "same", "row_hook_answer": "same"} for unit in units]


def _fabricated_dataset(gate3_pass: bool = True):
    units = rows119.load_units()
    rows = rows119.all_rows(units)
    units_by_id = {unit["id"]: unit for unit in units}
    plain = [0] + list(range(2, 25))
    answers = [{"row_id": "plain", "tokens": plain, "text": "I am a clear example answer."}]
    for index, unit in enumerate(units):
        token = 100 + index
        tokens = [token] + plain[1:]
        text = "I am a clear example answer."
        answers.extend([
            {"row_id": f"single:{unit['id']}", "tokens": tokens, "text": text},
            {"row_id": f"random:{unit['id']}", "tokens": tokens, "text": text},
        ])
    for row in rows:
        if row["row_type"] in ("pair", "control"):
            first_id = row["unit_ids"][0]
            first_index = next(i for i, unit in enumerate(units) if unit["id"] == first_id)
            tokens = [100 + first_index] + plain[1:]
            answers.append({"row_id": row["row_id"], "tokens": tokens,
                            "text": "I am a clear example answer."})
    part118 = [{"id": unit["id"], "arm": "neuron", "alpha": .8,
                "question_index": 0, "question": "Who are you?",
                "answer": "I am a clear example answer."} for unit in units]
    class_pair = next(row for row in rows if row["row_type"] == "pair"
                      and rows119.pair_class(row, units_by_id) == "slot_x_concept")
    control = next(row for row in rows if row["row_type"] == "control")
    logits = []
    for row in (class_pair, control):
        logits.append({"row_id": row["row_id"], "n_positions": 1,
                       "both_hits": int(gate3_pass),
                       "contested": [_fixture_contested(10) if gate3_pass else
                                     {**_fixture_contested(10), "both_top": 0}]})
    alpha = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    ladder = []
    for index, unit in enumerate(units):
        t_a = 100 + index
        ladder.append({"id": unit["id"], "p": 0, "t_A": t_a, "t_0": 0,
                       "alphas": alpha, "tops": [0] * 7 + [t_a] * 3,
                       "margins": [-1.0] * 7 + [1.0] * 3})
    return units, rows, answers, logits, ladder, part118


def test_gate_three_failure_keeps_answer_only_predictions_readable():
    units, _rows, answers, logits, ladder, part118 = _fabricated_dataset(gate3_pass=False)
    result, _tournament = score119.score_data(
        answers, logits, ladder, units,
        {"no_hook_tokens": [1, 2], "zero_hook_tokens": [1, 2]}, part118, _gate2c(units),
        labels_ds=[], labels_luna=[], known=label119.read_known(),
        reader_files_present={"ds": False, "luna": False, "known_ds": False, "known_luna": False},
    )
    by_id = {row["id"]: row for row in result["predictions"]}
    assert result["gate"]["gate1"]["passed"]
    assert result["gate"]["gate2"]["passed"]
    assert not result["gate"]["gate3"]["passed"]
    for number in ("119.1", "119.2", "119.3", "119.4", "119.5", "119.7", "119.8"):
        assert by_id[number]["verdict"] == "not read"
    assert by_id["119.6"]["verdict"] != "not read"
    assert by_id["119.9"]["verdict"] != "not read"


def test_one_missing_logit_row_fails_gate_three_and_withholds_119_3():
    units, _rows, answers, logits, ladder, part118 = _fabricated_dataset(gate3_pass=True)
    expected_ids = {row["row_id"] for row in logits}
    result, _tournament = score119.score_data(
        answers, logits[:1], ladder, units,
        {"no_hook_tokens": [7, 8], "zero_hook_tokens": [7, 8]}, part118, _gate2c(units),
        expected_logit_ids=expected_ids, interactions=[], expected_interaction_keys=set(),
    )
    by_id = {row["id"]: row for row in result["predictions"]}
    assert result["gate"]["gate3"]["complete"] is False
    assert by_id["119.3"]["verdict"] == "not read"


def test_no_flip_path2_rows_are_counted_and_one_unit_still_reads():
    rows = rows119.interaction_rows(stage="main")
    real_rows = [row for row in rows if row["kind"] == "real"]
    interactions = []
    for row in real_rows:
        no_flip = row["first"] == "p92-00"
        interactions.append({"first": row["first"], "second": row["second"], "kind": "real",
                             "p_A": None if no_flip else 0,
                             "positions": [] if no_flip else [
                                 {"nI": 1.0, "nA": 1.0, "n2": 1.0, "nPair": 1.0, "tv": 0.0}
                             ]})
    summary = score119.interaction_summaries(interactions)["real"]
    effect = score119.effect_matched_comparison(
        interactions, [unit["id"] for unit in rows119.load_units()], expected_real_rows=1482,
    )
    assert summary["rows_with_positions"] == 1444
    assert summary["no_flip_units"] == ["p92-00"]
    assert effect["real_rows_with_positions"] == summary["rows_with_positions"]
    assert effect["real_rows_without_positions"] == 38
    assert score119.interaction_row_coverage_ok(38, 1482)
    assert not score119.interaction_row_coverage_ok(75, 1482)


def test_run_interactions_keeps_rows_for_a_no_flip_unit(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(row) + "\n" for row in (
        {"row_id": "plain", "tokens": [1, 2, 3]},
        {"row_id": "single:u0", "tokens": [1, 2, 3]},
    )), encoding="utf-8")
    monkeypatch.setattr(run119.c, "prompt_ids", lambda *_args: [10])
    rows = [
        {"first": "u0", "second": "u1", "kind": "real", "pushes": []},
        {"first": "u0", "second": "u1", "kind": "control", "pushes": []},
    ]
    count, _seconds, passes = run119.run_interactions(
        [{"id": "u0", "layer": 0}], rows, answers, tmp_path / "interactions.jsonl",
        None, None, None, {}, "cpu",
    )
    saved = label119.read_jsonl(tmp_path / "interactions.jsonl")
    assert (count, passes) == (2, 0)
    assert [(row["kind"], row["p_A"], row["positions"]) for row in saved] == [
        ("real", None, []), ("control", None, []),
    ]


def test_scorer_end_to_end_on_fabricated_files():
    units, rows, answers, logits, ladder, part118 = _fabricated_dataset(gate3_pass=True)
    result, tournament = score119.score_data(
        answers, logits, ladder, units,
        {"no_hook_tokens": [7, 8], "zero_hook_tokens": [7, 8]}, part118, _gate2c(units),
        labels_ds=[], labels_luna=[], known=label119.read_known(),
        reader_files_present={"ds": False, "luna": False, "known_ds": False, "known_luna": False},
    )
    assert result["gate"]["core_passed"]
    assert result["gate"]["gate3"]["passed"]
    assert result["logit_summary"]["real_pairs"]["rows"] == 1
    assert result["logit_summary"]["controls"]["rows"] == 1
    assert result["logit_summary"]["class_splits"]["agreed"]["slot_x_concept"]["expected_rows"] == 168
    assert result["broken_answers"]["all"]["total"] == len(answers)
    assert len(tournament["units"]) == 39
    assert len(tournament["matrix"]) == 39
    assert len(tournament["matrix"][0]["cells"]) == 39
    assert score119.render_markdown(result).startswith("# Part 119 score\n")


def test_path2_interaction_share_fixture():
    row = {"first": "p92-00", "second": "p92-01", "kind": "real",
           "positions": [{"nI": 3.0, "nA": 2.0, "n2": 1.0, "nPair": 4.0, "tv": .2},
                          {"nI": 2.0, "nA": 2.0, "n2": 1.0, "nPair": 3.0, "tv": .1}]}
    assert score119.interaction_share(row["positions"][0]) == pytest.approx(9 / 16)
    values = score119._interaction_row_values(row)
    assert values["share"] == pytest.approx((9 / 16 + 4 / 9) / 2)
    assert values["tv"] == pytest.approx(.15)


def test_effect_matched_fit_and_support_coverage_fixture():
    import math

    controls = [{"first": "p92-00", "second": "p92-01", "kind": "control", "positions": []}]
    for n_a in (1.0, 2.0, 4.0, 8.0):
        for n_2 in (1.0, 2.0, 4.0, 8.0):
            n_i = math.exp(.2) * n_a ** .5 * n_2 ** .25
            controls[0]["positions"].append({"nA": n_a, "n2": n_2, "nI": n_i, "nPair": 1, "tv": .1})
    expected = math.exp(.2) * 2 ** .5 * 2 ** .25
    real = [{"first": "p92-02", "second": "p92-03", "kind": "real", "positions": [
        {"nA": 2.0, "n2": 2.0, "nI": expected * 1.1, "nPair": 2.0, "tv": .2},
        {"nA": 2.0, "n2": 1000.0, "nI": 8.0, "nPair": 3.0, "tv": .3},
    ]}]
    result = score119.effect_matched_comparison(
        controls + real, [unit["id"] for unit in rows119.load_units()],
        score119.make_unit_bootstrap_draws([unit["id"] for unit in rows119.load_units()], 100, 11902),
    )
    assert result["fit_available"]
    assert result["mean_residual"] == pytest.approx(math.log(1.1), abs=1e-8)
    assert result["real_positions_total"] == 2
    assert result["real_positions_kept"] == 1
    assert result["share_real_positions_kept"] == .5


def test_gate2_concept_subgate_and_gate3_contested_coverage():
    units = rows119.load_units()
    by_group = {name: [unit for unit in units if unit["group"] == name]
                for name in ("agreed_slot", "agreed_concept", "disputed")}
    matched_ids = {unit["id"] for unit in by_group["agreed_slot"]}
    matched_ids.update(unit["id"] for unit in by_group["agreed_concept"][:5])
    matched_ids.update(unit["id"] for unit in by_group["disputed"])
    answers = [{"row_id": f"single:{unit['id']}", "text": "MATCH"} for unit in units]
    gate2c = [{"id": unit["id"], "part92_answer": "MATCH",
               "row_hook_answer": "MATCH" if unit["id"] in matched_ids else "MISS"} for unit in units]
    part118 = [{"id": unit["id"], "arm": "neuron", "alpha": .8,
                "question_index": 0, "answer": "MATCH"} for unit in units]
    result = score119.compute_gates({"no_hook_tokens": [1], "zero_hook_tokens": [1]}, answers,
                                    part118, [], units=units, gate2b=gate2c)
    assert not result["gate2"]["passed"]  # 36 of 39: the amended gate needs 38
    assert result["gate2"]["matches"] == 36
    assert result["gate2"]["agreed_concept_matches"] == 5
    assert not result["gate2"]["kind_claims_passed"]
    nogate = score119.compute_gates({"no_hook_tokens": [1], "zero_hook_tokens": [1]}, answers,
                                    part118, [], units=units)
    assert not nogate["gate2"]["passed"] and nogate["gate2"]["compared"] == 0  # no reference file: fails, never passes

    answers = [{"row_id": f"single:{unit['id']}", "text": "MATCH"} for unit in units]
    pair_positions = [{"g": 1, "both_top": 1} for _ in range(9)] + [{"g": 1, "both_top": 2}]
    logits = [
        {"row_id": "pair:p92-00+p92-01", "n_positions": 10, "both_hits": 10,
         "contested": pair_positions},
        {"row_id": "control:p92-02+Rp92-03", "n_positions": 90, "both_hits": 90,
         "contested": []},
    ]
    result = score119.compute_gates({"no_hook_tokens": [1], "zero_hook_tokens": [1]}, answers,
                                    part118, logits, units=units)
    assert result["gate3"]["accuracy"] == 1.0
    assert result["gate3"]["real_contested_accuracy"] == .9
    assert not result["gate3"]["passed"]


def test_label_prompts_do_not_leak_row_or_unit_metadata():
    item = {"public": {"row_id": "control:p92-02+Rp92-08", "unit_id": "p92-02",
                        "arm": "neuron+random", "layer": 4, "kind": "concept"},
            "x_name": "Canal lock tender", "x_desc": "Keeps canal locks.",
            "y_name": "Sourdough baker", "y_desc": "Makes bread.",
            "q": "Who are you?", "a": "I am a canal lock tender."}
    prompt = label119.build_prompt(item)
    for secret in ("control:p92-02+Rp92-08", "p92-02", "neuron+random", "layer=4", "kind=concept"):
        assert secret not in prompt
    assert "A theme nearby does not count." in prompt
    assert '"x": true' in prompt and '"y": false' in prompt
    assert label119.parse_response('{"x":true,"y":false,"coherent":true}') == {
        "x": True, "y": False, "coherent": True,
    }


def test_sample_labels_use_opaque_ids_and_private_keys_are_joined_by_scorer(tmp_path):
    sample = rows119.reader_sample()
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps({"row_id": row["row_id"], "text": "I am a persona."}) + "\n"
                                  for row in sample), encoding="utf-8")
    items = label119.sample_items(answers)
    keys_path = tmp_path / "reader-keys.jsonl"
    label119.write_sample_keys(items, keys_path)
    prompts = [label119.build_prompt(items[0])]
    for secret in (items[0]["private"]["row_id"], items[0]["private"]["sample_class"],
                   items[0]["private"]["x_unit_id"], items[0]["private"]["y_unit_id"]):
        assert secret not in prompts[0]

    labels_path = tmp_path / "labels.jsonl"
    label119.label_items(
        items, labels_path, "ds", workers=4,
        request_fn=lambda *_args: {"text": '{"x":true,"y":false,"coherent":true}'},
        sleep_fn=lambda _seconds: None,
    )
    labels = label119.read_jsonl(labels_path)
    keys = label119.read_jsonl(keys_path)
    assert len(labels) == len(keys) == 148
    assert len({row["item_id"] for row in keys}) == 148
    assert [row["item_id"] for row in labels] == [row["item_id"] for row in keys]
    assert all(set(row) == {"item_id", "x", "y", "coherent"} for row in labels)
    assert all(not row["item_id"].startswith(("pair:", "control:")) for row in labels)
    joined = score119.join_reader_keys(labels, keys)
    assert len(joined) == 148
    assert joined[0]["row_id"] == keys[0]["row_id"]
    assert joined[0]["sample_class"] == keys[0]["sample_class"]


def test_retry_instruction_cap_usage_valid_preference_and_retry_count(tmp_path):
    item = {"public": {"item_id": "opaque-123"}, "x_name": "A", "x_desc": "First.",
            "y_name": "B", "y_desc": "Second.", "q": "Who are you?", "a": "I am A."}
    prompts = []
    responses = iter([
        {"text": "truncated", "stop_reason": "length"},
        {"text": '{"x":true,"y":false,"coherent":true}', "stop_reason": "stop"},
    ])

    def request(_reader, prompt, _system):
        prompts.append(prompt)
        return next(responses)

    output = tmp_path / "labels.jsonl"
    result = label119.label_items([item], output, "ds", workers=1,
                                  request_fn=request, sleep_fn=lambda _seconds: None)
    labels = label119.read_jsonl(output)
    usage = label119.read_jsonl(Path(str(output) + ".usage.jsonl"))
    assert len(prompts) == 2 and prompts[0] != prompts[1]
    assert prompts[1].endswith("On this attempt, reply with the JSON object only.")
    assert usage[0]["stop_reason"] == "length" and usage[0]["cap_hit"] is True
    assert result["retry_items"] == 1
    assert score119.retry_item_count(labels) == 1
    assert score119.preferred_label_by(labels, "item_id")["opaque-123"]["x"] is True
    sample_key = rows119.reader_sample()[0]
    valid_then_error = [
        {"item_id": "opaque-123", **sample_key, "x": True, "y": False, "coherent": True},
        {"item_id": "opaque-123", **sample_key, "error": "late failure"},
    ]
    assert score119.preferred_label_by(valid_then_error, "item_id")["opaque-123"]["x"] is True
    reader = score119.reader_analysis(
        rows119.load_units(), valid_then_error, [], label119.read_known(), [], [],
        {"ds": True, "luna": False, "known_ds": False, "known_luna": False},
    )
    assert reader["sample_coverage"]["ds_retry_items"] == 1
    assert reader["sample_coverage"]["ds_errors"] == 0


def test_sample_read_refuses_missing_reader_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(label119, "sample_items", lambda *_args, **_kwargs: pytest.fail("sample started"))
    result = label119.main([
        "--reader", "ds", "--out", str(tmp_path / "labels.jsonl"),
        "--answers", str(tmp_path / "missing-answers.jsonl"),
        "--gate-labels", str(tmp_path / "missing-gate.jsonl"),
    ])
    assert result == 1
    error = capsys.readouterr().err
    assert "sample read refused" in error
    assert "run the ds gate first" in error


def test_build_notes_document_two_pilot_dirs_and_main_reuse():
    notes = (HERE / "BUILD-NOTES.md").read_text(encoding="utf-8")
    assert "--batch 64 --out results/part119/pilot-64" in notes
    assert "--batch 128 --out results/part119/pilot-128" in notes
    assert "--stage main --batch 64 --out results/part119/pilot-64" in notes
    assert "The other pilot folder is only for the second timing and is not reused." in notes


def test_known_personas_are_unrelated_to_the_part92_persona_list():
    with (HERE / "known119.jsonl").open(encoding="utf-8") as stream:
        known = [json.loads(line) for line in stream if line.strip()]
    with rows119.repo_path(rows119.DEFAULT_NEURONS).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    existing = {row["persona"].casefold() for row in neurons}
    known_names = [item[key].casefold() for item in known for key in ("x_name", "y_name")]
    assert len(known) == 12
    assert not (set(known_names) & existing)
    combos = {(item["expected_x"], item["expected_y"]) for item in known}
    assert combos == {(True, True), (True, False), (False, True), (False, False)}


def _tiny_olmo2():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Olmo2Config(
        vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
        pad_token_id=0, eos_token_id=2,
    )
    model = transformers.Olmo2ForCausalLM(config).eval()
    return torch, model


def test_torch_two_layer_row_hook_matches_two_single_layer_hooks():
    torch, model = _tiny_olmo2()
    layers = model.model.layers
    ids = torch.tensor([[1, 3, 5, 7], [1, 4, 6, 8]], dtype=torch.long)
    first = torch.randn((2, 32), generator=torch.Generator().manual_seed(119))
    second = torch.randn((2, 32), generator=torch.Generator().manual_seed(120))
    with run119.row_hooks(layers, {0: first, 1: second}), torch.inference_mode():
        combined = model(input_ids=ids, use_cache=False).logits

    def reference_hook(layer, vectors):
        def add_push(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pushed = hidden + vectors.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
            if isinstance(output, tuple):
                return (pushed, *output[1:])
            return pushed
        return layer.register_forward_hook(add_push)

    handle0 = reference_hook(layers[0], first)
    handle1 = reference_hook(layers[1], second)
    try:
        with torch.inference_mode():
            separate = model(input_ids=ids, use_cache=False).logits
    finally:
        handle1.remove()
        handle0.remove()
    torch.testing.assert_close(combined, separate, rtol=0, atol=0)


def test_torch_zero_vectors_leave_model_logits_unchanged():
    torch, model = _tiny_olmo2()
    layers = model.model.layers
    ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    with torch.inference_mode():
        plain = model(input_ids=ids, use_cache=False).logits
    zero = {index: torch.zeros((1, 32), dtype=torch.float32) for index in range(2)}
    with run119.row_hooks(layers, zero), torch.inference_mode():
        hooked = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(plain, hooked, rtol=0, atol=0)


def test_torch_teacher_forced_and_cached_greedy_tokens_agree():
    torch, model = _tiny_olmo2()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    prompt = torch.tensor([[1, 3, 5]], dtype=torch.long)
    with torch.inference_mode():
        generated = model.generate(prompt, do_sample=False, max_new_tokens=5,
                                   use_cache=True, pad_token_id=0, eos_token_id=None)
        forced = []
        for end in range(prompt.shape[1], generated.shape[1]):
            prefix = generated[:, :end]
            next_id = model(input_ids=prefix, use_cache=False).logits[0, -1].argmax()
            forced.append(int(next_id.item()))
    assert forced == generated[0, prompt.shape[1]:].tolist()


def test_torch_path2_interaction_metrics_on_additive_and_nonadditive_logits():
    torch = pytest.importorskip("torch")
    base = torch.zeros((1, 32), dtype=torch.float32)
    first = base.clone()
    second = base.clone()
    first[0, 3] = 3.0
    second[0, 7] = 2.0
    additive_pair = first + second
    additive = run119._interaction_position_metrics(base, first, second, additive_pair)[0]
    assert additive["nI"] == pytest.approx(0.0, abs=1e-6)
    assert additive["nPair"] > 0
    assert additive["tv"] == pytest.approx(0.0, abs=1e-7)

    altered_pair = additive_pair.clone()
    altered_pair[0, 12] = 6.0
    nonadditive = run119._interaction_position_metrics(base, first, second, altered_pair)[0]
    assert nonadditive["nI"] > 0
    assert nonadditive["tv"] > 0
