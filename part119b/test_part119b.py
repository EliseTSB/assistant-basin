from __future__ import annotations

import copy
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

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
import gate119b


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


def test_dose_cache_is_built_directly_and_a_wrong_alpha_is_refused():
    weight = torch.tensor([[1., 0., 2., 0.], [0., 3., 0., 1.],
                           [2., 1., 0., 2.], [1., 2., 1., 0.]], dtype=torch.bfloat16)
    layer = SimpleNamespace(mlp=SimpleNamespace(
        down_proj=SimpleNamespace(weight=weight)))
    units = [{"id": "a", "layer": 0, "neuron": 1, "control_seed": 11},
             {"id": "b", "layer": 0, "neuron": 3, "control_seed": 12}]
    base = run119b.build_base_push_cache(units, [layer], {0: 2.5})
    at_04 = run119b.build_dose_push_cache(units, [layer], {0: 2.5}, .4)
    for unit in units:
        assert torch.equal(at_04[(unit["id"], "neuron")],
                           common92.neuron_push(weight, unit["neuron"], .4 * 2.5))
        assert torch.equal(at_04[(unit["id"], "random")],
                           common92.random_direction(weight, unit["control_seed"], .4 * 2.5))
    run119b.validate_dose_scaling(units, base, at_04, .4)
    with pytest.raises(AssertionError):
        run119b.validate_dose_scaling(units, base, at_04, .6)
    swapped = dict(at_04)
    swapped[("a", "neuron")] = at_04[("b", "neuron")]
    with pytest.raises(AssertionError):
        run119b.validate_dose_scaling(units, base, swapped, .4)


def test_gate2_stopped_path2_verdicts_keep_observed_values(monkeypatch):
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
    assert predictions["119b.4"]["verdict"] == "stopped by Gate 2"
    assert predictions["119b.4"]["observed"] is not None
    assert predictions["119b.5"]["verdict"] == "stopped by Gate 2"
    assert predictions["119b.7"]["verdict"] == "stopped by Gate 2"
    assert predictions["119b.8"]["verdict"] == "stopped by Gate 2"
    assert all(predictions[f"119b.{n}"]["verdict"] == "stopped by Gate 2"
               for n in (4, 5, 6, 7, 8, 9, 10, 11))
    report = score119b.render_markdown(score119b._json_safe(result))
    assert "stopped by Gate 2" in report
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


def _tiny_units():
    return [
        {"id": "tiny-a", "layer": 0, "neuron": 0, "control_seed": 11901,
         "haiku_kind": "slot", "ds_kind": "slot", "group": "agreed_slot"},
        {"id": "tiny-b", "layer": 1, "neuron": 1, "control_seed": 11902,
         "haiku_kind": "concept", "ds_kind": "concept", "group": "agreed_concept"},
        {"id": "tiny-c", "layer": 0, "neuron": 2, "control_seed": 11903,
         "haiku_kind": "slot", "ds_kind": "slot", "group": "agreed_slot"},
    ]


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = None

    def apply_chat_template(self, *args, **kwargs):
        return "tiny prompt"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 3, 5]}

    def decode(self, ids, **kwargs):
        return "tiny answer"


def _tiny_path2_inputs(torch, units):
    model = _tiny_olmo2()[1]
    layers = run119.c.get_layers(model)
    base_cache = {"_units": units}
    for index, unit in enumerate(units):
        for arm in ("neuron", "random"):
            base_cache[(unit["id"], arm)] = torch.full(
                (32,), 0.01 * (index + 1), dtype=torch.float32)
    caches = {dose: run119b.scale_push_cache(base_cache, dose)
              for dose in rows119b.DOSES}
    rows = [
        {"key": "real:tiny-a+tiny-b", "kind": "real", "first": "tiny-a",
         "second": "tiny-b", "pushes": [{"unit_id": "tiny-a", "arm": "neuron"},
                                          {"unit_id": "tiny-b", "arm": "neuron"}]},
        {"key": "control:tiny-a+tiny-c", "kind": "control", "first": "tiny-a",
         "second": "tiny-c", "pushes": [{"unit_id": "tiny-a", "arm": "neuron"},
                                             {"unit_id": "tiny-c", "arm": "random"}]},
    ]
    answers = {"plain": {"tokens": [4, 5, 6, 7]},
               "single@0.8:tiny-a": {"tokens": [4, 9, 6, 7]}}
    return model, layers, base_cache, caches, rows, answers


def test_run_path2_tiny_olmo2_all_doses_and_resume_uses_stored_support(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    units = _tiny_units()
    model, layers, base_cache, caches, rows, answers = _tiny_path2_inputs(torch, units)
    monkeypatch.setattr(run119, "first_flip", lambda _single, _plain: (1, 9, 5))
    output = tmp_path / "interaction.jsonl"
    first = run119b.run_path2(
        units, rows, rows119b.DOSES, output, _TinyTokenizer(), model, layers,
        base_cache, caches, answers, "cpu", 2, quiet=True)
    stored = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(stored) == len(rows) * len(rows119b.DOSES)
    assert {record["dose"] for record in stored} == set(rows119b.DOSES)
    assert first["rows_new"] == len(stored)

    stored_08 = {(record["kind"], record["first"], record["second"]): record
                 for record in stored if record["dose"] == 0.8}
    output.write_text("".join(json.dumps(record) + "\n" for record in stored
                               if record["dose"] == 0.8), encoding="utf-8")
    top10_calls = []
    original_top10 = run119._top10_repeated_argmax

    def count_top10(logits):
        top10_calls.append(tuple(logits.shape))
        return original_top10(logits)

    monkeypatch.setattr(run119, "_top10_repeated_argmax", count_top10)
    resumed = run119b.run_path2(
        units, rows, rows119b.DOSES, output, _TinyTokenizer(), model, layers,
        base_cache, caches, answers, "cpu", 2, quiet=True)
    replayed = [json.loads(line) for line in output.read_text().splitlines()]
    assert resumed["rows_new"] == len(rows) * 3
    assert resumed["logical_passes"] == 1 + 3 + 2 * len(rows) * 3
    assert top10_calls == []
    for record in replayed:
        if record["dose"] == 0.8:
            continue
        stored_record = stored_08[(record["kind"], record["first"], record["second"])]
        assert [pos["t_hash"] for pos in record["positions"]] == [
            pos["t_hash"] for pos in stored_record["positions"]]


def test_path2_raises_when_memory_and_stored_support_disagree():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 1, 4), dtype=torch.float32)
    row = {"key": "real:tiny-a+tiny-b", "kind": "real", "first": "tiny-a",
           "second": "tiny-b", "p_A": 0, "position_start": 0}
    with pytest.raises(ValueError, match="support.*disagree"):
        run119b._extract_batch_metrics(
            "tiny-a", [row], 0.2, logits, logits, logits, logits,
            [0], [0.0], [0], [0.0], [[0]], [[0]],
            {("real:tiny-a+tiny-b", 0): [0, 1]},
            {("real:tiny-a+tiny-b", 0): [0, 2]}, add_support=False)


def test_gate2_uses_selected_reference_medians_and_row_level_npair():
    selected = [
        {"key": "real@0.8:a+b", "kind": "real", "first": "a", "second": "b"},
        {"key": "real@0.8:c+d", "kind": "real", "first": "c", "second": "d"},
        {"key": "control@0.8:a+b", "kind": "control", "first": "a", "second": "b"},
        {"key": "control@0.8:c+d", "kind": "control", "first": "c", "second": "d"},
    ]
    current, reference = [], []
    for row in selected:
        heavy = row["first"] == "a"
        count = 100 if heavy else 1
        old_share = 0.1 if heavy else 0.3
        relative = 0.015 if heavy else 0.0
        old_positions, new_positions = [], []
        for pos in range(count):
            old_npair = 2.0
            old = _position(share=old_share)
            old.update({"pos": pos, "nPair": old_npair,
                        "nI": math.sqrt(old_share) * old_npair})
            new = dict(old)
            new_npair = old_npair * (1 + relative)
            new.update({"nPair": new_npair,
                        "nI": math.sqrt(old_share) * new_npair})
            old_positions.append(old)
            new_positions.append(new)
        reference.append({**row, "positions": old_positions})
        current.append({**row, "dose": 0.8, "positions": new_positions})
    expected = {
        "real": {("real", "a", "b"), ("real", "c", "d")},
        "control": {("control", "a", "b"), ("control", "c", "d")},
    }
    result = score119b.gate2_result(current, reference, expected)
    pilot = run119b.pilot_reproduction_gate(
        {record["key"]: record for record in current}, reference, selected)
    assert result["passed"]
    assert pilot == result
    assert result["row_medians"]["real"]["median_row_share_0.8"] == pytest.approx(0.2)
    assert result["row_medians"]["real"]["reference_median_row_share_0.8"] == pytest.approx(0.2)
    assert result["median_relative_nPair_difference"] == pytest.approx(0.0075)


@pytest.mark.parametrize("kind", ["real", "control"])
def test_gate2_empty_selected_class_is_vacuous(kind):
    key = (kind, "a", "b")
    reference_row = {"kind": kind, "first": "a", "second": "b",
                     "positions": [_position()]}
    pilot_row = {**reference_row, "dose": 0.8}
    expected = {"real": set(), "control": set()}
    expected[kind].add(key)

    result = gate119b.gate2_result([pilot_row], [reference_row], expected)

    assert result["passed"]
    assert result["class_thresholds_passed"][kind]
    assert result["class_thresholds_passed"]["control" if kind == "real" else "real"]


def test_registered_part119_reference_is_pinned_by_path_count_and_hash(tmp_path, monkeypatch):
    path = ROOT / gate119b.PART119_INTERACTIONS_PATH
    rows = run119b.read_jsonl_readonly(path)
    attestation = gate119b.validate_part119_reference(path, rows)

    assert attestation == {
        "path": gate119b.PART119_INTERACTIONS_PATH,
        "row_count": 2964,
        "sha256": "89cee0f0114d464c3405d21982beb753a2eeb977c771562198bd06c370104c56",
    }

    copied = tmp_path / "foreign-interaction.jsonl"
    shutil.copyfile(path, copied)
    with pytest.raises(ValueError, match="registered Part 119 path"):
        gate119b.validate_part119_reference(copied, rows)
    with pytest.raises(ValueError, match="expected 2,964 rows"):
        gate119b.validate_part119_reference(path, rows[:-1])
    monkeypatch.setattr(gate119b, "PART119_INTERACTIONS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gate119b.validate_part119_reference(path, rows)


def test_registered_inventory_is_selected_by_unit_ids(monkeypatch):
    units = copy.deepcopy(rows119.load_units())
    units[0]["id"] = "unregistered-unit"
    original_load_units = rows119.load_units
    calls = []

    def load_registered(*args, **kwargs):
        calls.append(True)
        return original_load_units(*args, **kwargs)

    monkeypatch.setattr(rows119, "load_units", load_registered)
    assert not rows119b._registered_inventory(units)
    planned = rows119b.all_answer_rows(units)
    pilot = rows119b.pilot_answer_rows(units)
    path2 = rows119b.all_path2_rows(units)
    pilot_path2 = rows119b.pilot_path2_rows(units)

    assert len(calls) >= 6
    assert any(row["row_id"].endswith("unregistered-unit") for row in planned)
    assert any(row["row_id"].endswith("unregistered-unit") for row in pilot)
    assert len(path2) == 2964
    assert len(pilot_path2) == 32


def _tiny_pilot_setup(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    units = _tiny_units()
    torch.manual_seed(119)
    _torch, model = _tiny_olmo2()
    tokenizer = _TinyTokenizer()
    monkeypatch.setattr(run119b.c, "load_model", lambda _path, _device: (tokenizer, model))
    monkeypatch.setattr(run119b.c, "residual_medians",
                        lambda *_args, **_kwargs: {0: 1.0, 1: 1.0})
    monkeypatch.setattr(run119, "first_flip", lambda _single, _plain: (1, 9, 5))
    monkeypatch.setattr(run119, "ANSWER_TOKENS", 1)
    monkeypatch.setattr(run119b, "ANSWER_TOKENS", 1)
    part119_answers = tmp_path / "part119-answers.jsonl"
    prior = [{"row_id": "plain", "tokens": [4, 5, 6, 7],
              "text": "plain answer"}]
    for unit in units:
        for row_type in ("single", "random"):
            prior.append({"row_id": f"{row_type}:{unit['id']}",
                          "tokens": [4, 9, 6, 7], "text": "unit answer"})
    part119_answers.write_text("".join(json.dumps(row) + "\n" for row in prior),
                               encoding="utf-8")
    part119_interactions = tmp_path / "part119-interactions.jsonl"
    part119_interactions.write_text("", encoding="utf-8")
    args = SimpleNamespace(
        model="unused", out=str(tmp_path / "pilot-run"), device="cpu", stage="pilot",
        batch=8, path2_batch=1, path1_batch=1, neurons="unused", kinds="unused",
        kinds_ds="unused", part119_answers=str(part119_answers),
        part119_interactions=str(part119_interactions))
    return args, units


def _materialize_tiny_pilot_reference(args, units):
    with pytest.raises(RuntimeError, match="selected rows are missing"):
        run119b.run(args, units=units,
                    test_only_allow_unregistered_part119_reference=True)
    output = Path(args.out)
    pilot_rows = [row for row in run119b.read_jsonl_readonly(
        output / "interaction-pilot.jsonl") if float(row["dose"]) == 0.8]
    reference_path = Path(args.out).parent / "matching-reference.jsonl"
    reference_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pilot_rows), encoding="utf-8")
    args.part119_interactions = str(reference_path)
    return output, pilot_rows


def test_real_pilot_stage_on_tiny_inventory_passes_matching_gate2(tmp_path, monkeypatch):
    args, units = _tiny_pilot_setup(tmp_path, monkeypatch)
    output, pilot_rows = _materialize_tiny_pilot_reference(args, units)

    result = run119b.run(
        args, units=units,
        test_only_allow_unregistered_part119_reference=True)
    timing = json.loads((output / "timing-pilot.json").read_text(encoding="utf-8"))
    assert result["gate2_sample_passed"]
    assert timing["gate2"]["passed"]
    assert timing["part119_reference"]["path"] == args.part119_interactions
    assert timing["part119_reference"]["sha256"] == hashlib.sha256(
        Path(args.part119_interactions).read_bytes()).hexdigest()
    assert len(pilot_rows) == 8
    assert "row_medians" not in timing["gate2"]
    assert (output / "interaction-pilot.jsonl").is_file()
    assert not (output / "interaction.jsonl").exists()


def test_real_pilot_stage_on_tiny_inventory_reports_reference_disagreement(tmp_path, monkeypatch):
    args, units = _tiny_pilot_setup(tmp_path, monkeypatch)
    _output, pilot_rows = _materialize_tiny_pilot_reference(args, units)
    for row in pilot_rows:
        row["positions"] = []
    reference_path = Path(args.part119_interactions)
    reference_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pilot_rows), encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected rows have different position lists"):
        run119b.run(
            args, units=units,
            test_only_allow_unregistered_part119_reference=True)


def test_runner_help_imports_from_exact_build_copy_lists(tmp_path):
    notes = (HERE / "BUILD-NOTES.md").read_text(encoding="utf-8")
    copy_lists = [
        shlex.split(line.split("--copy", 1)[1].strip())
        for line in notes.splitlines() if line.strip().startswith("--copy ")
    ]
    assert len(copy_lists) == 2
    assert copy_lists[0] == copy_lists[1]
    assert "part119b/gate119b.py" in copy_lists[0]

    copied_root = tmp_path / "copy-list-tree"
    for relative in copy_lists[0]:
        source = ROOT / relative
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    assert sorted(str(path.relative_to(copied_root))
                  for path in copied_root.rglob("*") if path.is_file()) == sorted(copy_lists[0])

    result = subprocess.run(
        ["/home/elisepc/neuron-probe/.venv/bin/python", "-I", "-B",
         "part119b/run119b.py", "--help"],
        cwd=copied_root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--stage" in result.stdout


def test_scorer_default_uses_only_canonical_interaction_file(monkeypatch):
    captured = {}
    monkeypatch.setattr(score119b, "score_files",
                        lambda args: captured.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["score119b.py"])
    score119b.main()
    assert captured["interactions"] == "results/part119b/interaction.jsonl"
    assert "pilot" not in captured["interactions"]


def test_gate3_stop_verdicts_keep_observed_values(monkeypatch):
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
    interactions, reference = [], []
    for row in rows119b.all_path2_rows(units):
        for dose, share in ((0.2, 0.4), (0.4, 0.35), (0.6, 0.2), (0.8, 0.04)):
            position = _position(share=share, t_hash="same-t")
            position["pos"] = 0
            interactions.append({
                "key": rows119b.path2_key(row["kind"], dose, row["first"], row["second"]),
                "kind": row["kind"], "first": row["first"],
                "second": row["second"], "dose": dose, "p_A": 0,
                "positions": [position]})
            if dose == 0.8:
                reference.append({**row, "positions": [_position(share=0.04, t_hash="same-t")]})
    tournament = {"units": [{"id": unit["id"], "dominance": 0.5} for unit in units]}
    result = score119b.score_data(
        local, interactions, [], {"passed": True, "no_hook_tokens": [1, 2],
                                  "zero_hook_tokens": [1, 2]},
        prior_answers, reference, {}, tournament, {}, units)
    predictions = {row["id"]: row for row in result["predictions"]}
    assert result["gates"]["gate2"]["passed"]
    assert not result["gates"]["gate3"]["passed"]
    for number in (4, 5, 7, 8):
        assert predictions[f"119b.{number}"]["verdict"] == "stopped by Gate 3"
        assert predictions[f"119b.{number}"]["observed"] is not None
    assert predictions["119b.4"]["observed"]["median_row_share_0.4_common_resolved"] == pytest.approx(.35)

