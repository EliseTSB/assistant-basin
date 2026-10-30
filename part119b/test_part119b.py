from __future__ import annotations

import copy
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119", ROOT / "part92", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common92
import rows119
import run119
import rows119b
import run119b
import score119b
import label119b


def _position(share=.04, resolved=True, t_hash="same-support"):
    return {"pos": 0, "nI": math.sqrt(share) * 3.0,
            "nA": 2.0 if resolved else .1,
            "n2": 2.0 if resolved else .1,
            "nPair": 3.0 if resolved else .1,
            "tv": .1, "b_A": .95, "b_2": 1.05,
            "t_size": 4, "t_hash": t_hash,
            "top_none": 1, "top_first": 2, "top_second": 3,
            "top_both": 4, "top_additive": 4,
            "margin_first": 2.0, "margin_second": 1.0}


def _four_dose_rows(*, share=.04, contest=(True, True, True, True), t_hash="same-support"):
    rows = []
    for dose, is_contested in zip(rows119b.DOSES, contest):
        pos = _position(share=share, t_hash=t_hash)
        pos["top_first"] = 1
        pos["top_second"] = 2 if is_contested else 1
        rows.append({"key": rows119b.path2_key("real", dose, "a", "b"),
                     "kind": "real", "first": "a", "second": "b", "dose": dose,
                     "p_A": 0, "positions": [pos]})
    return rows


def test_row_inventory_counts_dose_ids_and_pilot_seed():
    units = rows119.load_units()
    planned = rows119b.all_answer_rows(units)
    pilot_a = rows119b.pilot_answer_rows(units)
    pilot_b = rows119b.pilot_answer_rows(units)
    assert len(planned) == 7878
    assert sum(not row["reused"] for row in planned) == 5577
    assert sum(row["reused"] for row in planned) == 2301
    assert [row["row_id"] for row in pilot_a] == [row["row_id"] for row in pilot_b]
    assert len(pilot_a) == 529
    assert len(rows119b.all_path2_rows(units)) == 2964
    assert len(rows119b.pilot_path2_rows(units)) == 32
    ids = {row["row_id"] for row in planned}
    assert "pair@0.4:p92-02+p92-25" in ids
    assert "randpair@0.8:p92-02+p92-25" in ids
    assert "single@1.13:p92-00" in ids


def test_shared_t_is_checked_at_all_four_doses():
    interactions = _four_dose_rows()
    assert score119b.validate_shared_support(interactions)["passed"]
    broken = copy.deepcopy(interactions)
    broken[1]["positions"][0]["t_hash"] = "dose-specific-T"
    result = score119b.validate_shared_support(broken)
    assert not result["passed"]
    assert result["problem_count"] == 1


def test_synthetic_path2_recovers_known_additive_interaction():
    torch = pytest.importorskip("torch")
    base = torch.zeros(5, dtype=torch.float32)
    d_a = torch.tensor([1., -1., 0., 0., 0.])
    d_2 = torch.tensor([0., 0., 1., -1., 0.])
    first = base + d_a
    second = base + d_2
    both = base + d_a + d_2
    record = run119b._position_record([0, 1, 2, 3, 4], base, first, second, both,
                                      0, 0, 2, 0, 0, 1., 1., True)
    assert record["nI"] == pytest.approx(0.0, abs=1e-7)
    assert record["b_A"] == pytest.approx(1.0)
    assert record["b_2"] == pytest.approx(1.0)
    assert record["tv"] == pytest.approx(0.0, abs=1e-7)
    assert record["t_size"] == 5
    assert record["t_ids"] == [0, 1, 2, 3, 4]


def test_regression_uses_no_intercept():
    d_a = np.array([1., 2., 4., 0.])
    d_2 = np.array([0., 1., 0., 3.])
    target = 1.5 * d_a + .5 * d_2 + 2.0
    actual = run119b.no_intercept_slopes(d_a, d_2, target)
    expected = np.linalg.lstsq(np.column_stack((d_a, d_2)), target, rcond=None)[0]
    with_intercept = np.linalg.lstsq(np.column_stack((np.ones(4), d_a, d_2)),
                                     target, rcond=None)[0][1:]
    assert actual == pytest.approx(tuple(expected))
    assert not np.allclose(actual, with_intercept)


def test_shared_contested_mask_requires_both_doses():
    low = _four_dose_rows(contest=(True, True, False, True))
    high = _four_dose_rows(contest=(True, False, True, True))
    # Put two positions in each row: neither position is contested at both 0.4 and 0.8.
    low04, low08 = low[1]["positions"][0], low[3]["positions"][0]
    high04, high08 = high[1]["positions"][0], high[3]["positions"][0]
    low04["pos"], high04["pos"] = 0, 1
    low08["pos"], high08["pos"] = 0, 1
    # Keep unique row keys by scoring one real path with the two contested patterns.
    interactions = []
    for dose_index, dose in enumerate(rows119b.DOSES):
        positions = [_position(), _position()]
        positions[0]["pos"] = 0
        positions[1]["pos"] = 1
        if dose_index == 1:
            positions[0]["top_first"], positions[0]["top_second"] = 1, 2
            positions[1]["top_first"], positions[1]["top_second"] = 1, 1
        if dose_index == 3:
            positions[0]["top_first"], positions[0]["top_second"] = 1, 1
            positions[1]["top_first"], positions[1]["top_second"] = 1, 2
        interactions.append({"key": rows119b.path2_key("real", dose, "a", "b"),
                             "kind": "real", "first": "a", "second": "b",
                             "dose": dose, "positions": positions})
    summary = score119b.pair_top_agreement(interactions, [{"id": "a"}, {"id": "b"}])
    assert summary["shared_contested_positions"] == 0
    assert summary["rows_with_shared_contested_positions"] == 0


def test_resolved_position_floor_is_one_not_zero():
    assert not score119b.resolved({"nA": 0., "n2": 0., "nPair": 0.})
    assert score119b.resolved({"nA": 1., "n2": 1., "nPair": 1.})


def test_gate3_requires_its_registered_coverage_clause():
    summary = {"by_dose": {"0.2": {"real": {
        "median_row_share_common_resolved_all_doses": .05}}},
        "coverage": {"real": {"all_four_row_coverage": .69,
                               "rows_with_resolved_position_all_four_doses": 69,
                               "expected_rows": 100,
                               "all_four_resolved_positions": 50}}}
    result = score119b.gate3_result(summary, [], {"passed": True})
    assert not result["passed"]
    assert not result["coverage_clause_passed"]
    assert result["share_clause_passed"]


def test_p9_is_withheld_below_seventy_percent_coverage():
    status = score119b.p9_read_status(True, {"share_real_positions_kept_all_at_0.4": .699,
                                             "fit_available": True,
                                             "mean_residual_ci95": [-.01, .01]})
    assert not status["available"]
    assert status["verdict"] == "withheld"


def test_headline_first_branch_requires_119b1():
    predictions = [{"id": "119b.1", "verdict": "failed"},
                   {"id": "119b.4", "verdict": "held"},
                   {"id": "119b.6", "verdict": "held"}]
    result = score119b.headline_by_rule(
        predictions, .1, True, {"passed": True}, {"passed": True})
    assert result == "No registered headline reading; report the numbers."


def _known_reader_labels():
    known = label119b.shared.read_known()
    return [{"item_id": f"known-{index + 1:02d}",
             "x": row["expected_x"], "y": row["expected_y"],
             "coherent": row["expected_coherent"]}
            for index, row in enumerate(known)]


def _reader_key(item_id, row_id=None, sample_class="slot_x_slot"):
    return {"item_id": item_id,
            "row_id": row_id or f"pair@0.4:{item_id}",
            "row_type": "pair", "sample_class": sample_class,
            "x_unit_id": "unit-x", "y_unit_id": "unit-y"}


def test_undefined_constant_reader_kappa_reports_raw_agreement_and_reads_count():
    keys = [_reader_key("sample-a"), _reader_key("sample-b")]
    ds = [{"item_id": key["item_id"], "coherent": True} for key in keys]
    luna = [{"item_id": key["item_id"], "coherent": True} for key in keys]
    result = score119b.reader_coherence_analysis(
        keys, ds, luna, [], [], _known_reader_labels(), _known_reader_labels())
    assert result["cohen_kappa_on_coherent"]["kappa"] is None
    assert result["cohen_kappa_on_coherent"]["observed_agreement"] == 1.0
    assert result["cohen_kappa_on_coherent"]["undefined_reason"] == \
        "one reader gave one value throughout"
    assert result["read"]
    assert not result["withheld"]


def test_defined_half_kappa_withholds_reader_count():
    keys = [_reader_key(f"sample-{index}") for index in range(4)]
    ds_values = [True, True, False, False]
    luna_values = [True, True, False, True]
    ds = [{"item_id": key["item_id"], "coherent": value}
          for key, value in zip(keys, ds_values)]
    luna = [{"item_id": key["item_id"], "coherent": value}
             for key, value in zip(keys, luna_values)]
    result = score119b.reader_coherence_analysis(
        keys, ds, luna, [], [], _known_reader_labels(), _known_reader_labels())
    assert result["cohen_kappa_on_coherent"]["kappa"] == pytest.approx(.5)
    assert not result["read"]
    assert result["withheld"]
    prediction = score119b.prediction_119b1(
        {"broken": 0, "total": 741, "share": 0.0,
         "part86_triage_share": 0.0, "complete": True}, result, True)
    assert prediction["verdict"] == "withheld"
    assert prediction["observed"] is None


def test_known_example_gate_below_ten_withholds_reader_count():
    keys = [_reader_key("sample-a"), _reader_key("sample-b")]
    ds = [{"item_id": key["item_id"], "coherent": True} for key in keys]
    luna = [{"item_id": key["item_id"], "coherent": True} for key in keys]
    bad_ds_gate = _known_reader_labels()
    for row in bad_ds_gate[:3]:
        row["x"] = not row["x"]
    result = score119b.reader_coherence_analysis(
        keys, ds, luna, [], [], bad_ds_gate, _known_reader_labels())
    assert result["known_ds"]["correct"] == 9
    assert not result["known_ds"]["passed"]
    assert result["withheld"] and not result["read"]


def test_reader_dose_and_row_ids_are_absent_from_prompt_and_opaque_ids(tmp_path):
    prior_keys = label119b.shared.read_jsonl(
        "results/part119/labels-ds.jsonl.keys.jsonl")
    answers = []
    for key in prior_keys:
        dose_row_id = label119b._dose_row_id(key)
        answers.append({"row_id": dose_row_id,
                        "text": "A complete answer about who I am."})
    answer_path = tmp_path / "answers.jsonl"
    label119b.shared.write_jsonl(answer_path, answers)
    items = label119b.sample_items(answer_path)
    assert len(items) == 148
    assert [item["public"]["item_id"] for item in items] == [
        item["public"]["item_id"] for item in label119b.sample_items(
            answer_path)]
    original_by_dose_row = {label119b._dose_row_id(key): key for key in prior_keys}
    for item in items:
        item_id = item["public"]["item_id"]
        row_id = item["private"]["row_id"]
        original = original_by_dose_row[row_id]
        prompt = label119b.shared.build_prompt(item)
        assert len(item_id) == 20 and all(char in "0123456789abcdef" for char in item_id)
        assert "p92-" not in item_id and "@0.4" not in item_id
        assert row_id not in item_id
        assert item["private"]["x_unit_id"] == original["x_unit_id"]
        assert item["private"]["y_unit_id"] == original["y_unit_id"]
        assert item["a"] == "A complete answer about who I am."
        assert row_id not in prompt and item_id not in prompt
        assert "0.4" not in prompt
def test_missing_reader_label_is_neither_coherent_nor_incoherent():
    keys = [_reader_key("known-a"), _reader_key("known-b"), _reader_key("missing")]
    ds = [{"item_id": "known-a", "coherent": True},
          {"item_id": "known-b", "coherent": False}]
    luna = [{"item_id": "known-a", "coherent": True},
            {"item_id": "known-b", "coherent": False}]
    result = score119b.reader_coherence_analysis(
        keys, ds, luna, [], [], _known_reader_labels(), _known_reader_labels())
    counts = result["real_pair_coherent_count_0.4"]
    assert counts["coherent"] == 1
    assert counts["incoherent"] == 1
    assert counts["missing"] == 1
    assert counts["labeled"] == 2
    assert counts["expected"] == 3


def test_reader_coherence_pairs_same_rows_with_part119_by_class():
    keys = [_reader_key("new-pair", "pair@0.4:p92-a+p92-b", "slot_x_slot"),
            {"item_id": "new-control", "row_id": "control@0.4:p92-c+Rp92-d",
             "row_type": "control", "sample_class": "control",
             "x_unit_id": "unit-c", "y_unit_id": "unit-d"}]
    ds = [{"item_id": "new-pair", "coherent": True},
          {"item_id": "new-control", "coherent": False}]
    luna = [{"item_id": "new-pair", "coherent": True},
            {"item_id": "new-control", "coherent": False}]
    prior_keys = [{"item_id": "old-pair", "row_id": "pair:p92-a+p92-b"},
                  {"item_id": "old-control", "row_id": "control:p92-c+Rp92-d"}]
    prior_labels = [{"item_id": "old-pair", "coherent": False},
                    {"item_id": "old-control", "coherent": False}]
    result = score119b.reader_coherence_analysis(
        keys, ds, luna, prior_keys, prior_labels,
        _known_reader_labels(), _known_reader_labels())
    pair = result["paired_0.4_vs_part119_0.8_by_class"]["slot_x_slot"]
    control = result["paired_0.4_vs_part119_0.8_by_class"]["control"]
    assert pair["paired"] == 1
    assert pair["coherent_04"] == 1 and pair["coherent_08"] == 0
    assert pair["agreement"] == 0.0
    assert control["paired"] == 1 and control["agreement"] == 1.0


def test_119b1_reader_clause_and_headline_require_sixty_percent():
    reader = {"read": True,
              "withheld": False,
              "real_pair_coherent_count_0.4": {
                  "coherent": 64, "incoherent": 44, "missing": 0, "labeled": 108,
                  "expected": 108, "coherent_share_of_expected": 64 / 108}}
    clause = score119b.reader_119b1_clause(reader)
    assert clause["read"] and clause["holds"] is False
    p1 = score119b.prediction_119b1(
        {"broken": 0, "total": 741, "share": 0.0,
         "part86_triage_share": 0.0, "complete": True}, reader, True)
    assert p1["verdict"] == "failed"
    predictions = [
        p1,
        {"id": "119b.4", "verdict": "held"},
        {"id": "119b.6", "verdict": "held"},
    ]
    result = score119b.headline_by_rule(
        predictions, .1, True, {"passed": True}, {"passed": True})
    assert result == "No registered headline reading; report the numbers."


def test_dose_scaling_matches_part92_neuron_and_random_pushes():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    weight = torch.tensor([[1., 0., 2., 0.], [0., 3., 0., 1.],
                           [2., 1., 0., 2.], [1., 2., 1., 0.]], dtype=torch.float32)
    layer = SimpleNamespace(mlp=SimpleNamespace(
        down_proj=SimpleNamespace(weight=weight)))
    units = [{"id": "a", "layer": 0, "neuron": 1, "control_seed": 11},
             {"id": "b", "layer": 0, "neuron": 3, "control_seed": 12}]
    base = run119b.build_base_push_cache(units, [layer], {0: 2.5})
    scaled = run119b.scale_push_cache(base, .4)
    for unit in units:
        direct = common92.neuron_push(weight, unit["neuron"], .4 * 2.5)
        direct_random = common92.random_direction(weight, unit["control_seed"], .4 * 2.5)
        torch.testing.assert_close(scaled[(unit["id"], "neuron")].float(), direct.float(),
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(scaled[(unit["id"], "random")].float(), direct_random.float(),
                                   rtol=1e-5, atol=1e-6)
    # A second push is scaled through the same per-dose cache as the first.
    torch.testing.assert_close(scaled[("b", "neuron")], base[("b", "neuron")] * .5)


def test_gate_stopped_path2_verdicts_are_not_read(monkeypatch):
    units = rows119.load_units()
    planned = rows119b.all_answer_rows(units)
    source_ids = {row["source_row_id"] for row in planned if row["reused"]}
    prior_answers = [{"row_id": source_id, "tokens": [1, 2, 3],
                      "text": "I am a complete assistant."} for source_id in source_ids]
    prior_answers.append({"row_id": "plain", "tokens": [1, 2, 3],
                          "text": "I am a complete assistant."})
    local = [{"row_id": row["row_id"], "tokens": [1, 2, 3],
              "text": "I am a complete assistant."}
             for row in planned if not row["reused"]]
    monkeypatch.setattr(score119b.score119, "make_unit_bootstrap_draws",
                        lambda unit_ids, **_kwargs: [Counter({unit_id: 1 for unit_id in unit_ids})])
    stopped_interactions = []
    for row in rows119b.all_path2_rows(units):
        for dose in rows119b.DOSES:
            stopped_interactions.append({
                "key": rows119b.path2_key(row["kind"], dose, row["first"], row["second"]),
                "kind": row["kind"], "first": row["first"],
                "second": row["second"], "dose": dose, "p_A": 0,
                "positions": [_position(share=.04)]})
    tournament = {"units": [{"id": unit["id"], "dominance": 0.5} for unit in units]}
    result = score119b.score_data(
        local, stopped_interactions, [], {"passed": True,
            "no_hook_tokens": [1, 2], "zero_hook_tokens": [1, 2]},
        prior_answers, [], {}, tournament, {}, units)
    predictions = {row["id"]: row for row in result["predictions"]}
    assert not result["gates"]["gate2"]["passed"]
    assert result["path2_summary"] is None
    assert predictions["119b.4"]["verdict"] == "not read"
    assert predictions["119b.4"]["observed"] is None
    assert predictions["119b.5"]["verdict"] == "not read"
    assert predictions["119b.7"]["verdict"] == "not read"
    assert predictions["119b.8"]["verdict"] == "not read"
    assert all(predictions[f"119b.{n}"]["verdict"] not in ("held", "failed")
               for n in (4, 5, 6, 7, 8, 9, 10, 11))
    report = score119b.render_markdown(score119b._json_safe(result))
    assert "Not read for Path 2 because Gate 2 failed" in report
    assert "119b.4" in report


def _tiny_olmo2():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Olmo2Config(
        vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
        pad_token_id=0, eos_token_id=2)
    return torch, transformers.Olmo2ForCausalLM(config).eval()


def test_tiny_cpu_model_zero_hook_matches_plain_logits():
    torch, model = _tiny_olmo2()
    ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    layers = run119.c.get_layers(model)
    zero = {index: torch.zeros((1, 32), dtype=torch.float32) for index in range(2)}
    with torch.inference_mode():
        plain = model(input_ids=ids, use_cache=False).logits
    with run119.row_hooks(layers, zero), torch.inference_mode():
        hooked = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(plain, hooked, rtol=0, atol=0)


def test_tiny_cpu_model_path2_push_changes_logits():
    torch, model = _tiny_olmo2()
    ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    layers = run119.c.get_layers(model)
    push = torch.ones((1, 32), dtype=torch.float32) * .1
    with torch.inference_mode():
        plain = model(input_ids=ids, use_cache=False).logits
    with run119.row_hooks(layers, {0: push}), torch.inference_mode():
        altered = model(input_ids=ids, use_cache=False).logits
    assert not torch.equal(plain, altered)

