"""Part 173 rules, gates, reader flow, bootstrap speed, and a stubbed full run."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import names173
import run173
import score173

try:
    import torch
except ImportError:  # pragma: no cover - explicit, readable environment failure
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch is not installed in this interpreter")


def test_entity_alias_paths_are_idempotent_and_keep_unknowns_apart():
    for canonical, aliases in names173.ALIAS_GROUPS.items():
        assert names173.normalize_name(canonical) == canonical
        assert names173.normalize_name(names173.normalize_name(canonical)) == canonical
        for alias in aliases:
            resolved = names173.normalize_name(alias)
            assert resolved == canonical
            assert names173.normalize_name(resolved) == canonical
            raw = names173.normalize_name(alias, "raw")
            assert names173.normalize_name(raw, "raw") == raw
    assert names173.normalize_name("Tongyi Labs") == "qwen"
    assert names173.normalize_name("The Google Research, Inc.") == "google"
    assert names173.normalize_name("[Google AI](https://google.com)") == "google"
    assert names173.normalize_name("Hugging Labs") == "hugging labs"
    assert names173.normalize_name("Hugging Face") == "hugging face"
    assert names173.normalize_name("Hugging Labs", "generic") == "hugging"
    assert names173.normalize_name("Hugging Face", "generic") == "hugging face"
    assert names173.normalize_name("Beijing A") == "beijing a"
    assert names173.normalize_name("Beijing B") == "beijing b"
    assert names173.resolve_entity("Beijing A")["invented"]
    assert not names173.resolve_entity("Tongyi Labs")["invented"]


def test_known_examples_have_registered_shape_and_expected_resolutions():
    examples = label173.known_examples()
    assert len(examples) == 20
    assert sum("several-names" in row["features"] and row["expected_primary"] is None
               for row in examples) >= 3
    assert sum("honorific" in row["features"] for row in examples) >= 2
    assert sum("invented-organization" in row["features"] or "invented-person" in row["features"]
               for row in examples) >= 3
    assert sum("plain-qwen-answer" in row["features"] for row in examples) >= 3
    assert all(names173.normalize_name(row["expected_primary"]) == row["expected_primary"]
               for row in examples)


@pytest.mark.parametrize("count,expected", [(4, "failed"), (5, "held"), (6, "held")])
def test_verdict_173_1_boundary(count, expected):
    assert score173.verdict_173_1(count) == expected


@pytest.mark.parametrize("count,expected", [(1, "failed"), (2, "held"), (3, "held")])
def test_verdict_173_2_boundary(count, expected):
    assert score173.verdict_173_2(count) == expected


def test_verdict_173_3_boundaries_and_not_read():
    assert score173.verdict_173_3(2, 0.50, 8) == "failed"
    assert score173.verdict_173_3(3, 0.51, 8) == "failed"
    assert score173.verdict_173_3(3, 0.50, 8) == "held"
    assert score173.verdict_173_3(3, 0.50, 0) == "not read"
    assert score173.verdict_173_3(3, None, 4) == "not read"


def test_verdict_173_4_boundaries_and_wrong_entity():
    google = [{"modal": "google", "modal_count": 30, "broken": 4},
              {"modal": "google", "modal_count": 31, "broken": 0}]
    assert score173.verdict_173_4(google) == "held"
    assert score173.verdict_173_4([{**google[0], "modal_count": 29}, google[1]]) == "failed"
    assert score173.verdict_173_4([{**google[0], "broken": 5}, google[1]]) == "failed"
    assert score173.verdict_173_4([{**google[0], "modal": "tencent"}, google[1]]) == "failed"
    assert score173.verdict_173_4(google[:1]) == "not read"


def test_verdict_173_5_boundaries():
    cell = {"modal": "tencent", "modal_count": 10, "broken": 8}
    assert score173.verdict_173_5(cell) == "held"
    assert score173.verdict_173_5({**cell, "modal_count": 9}) == "failed"
    assert score173.verdict_173_5({**cell, "broken": 9}) == "failed"
    assert score173.verdict_173_5({**cell, "modal": "google"}) == "failed"


def test_verdict_173_6_signed_direction_boundaries():
    assert score173.verdict_173_6(15, 10, 16) == "held"
    assert score173.verdict_173_6(14, 10, 16) == "failed"
    assert score173.verdict_173_6(15, 10, 15) == "failed"
    assert score173.verdict_173_6(None, 10, 16) == "not read"


def test_verdict_173_7_boundaries():
    assert score173.verdict_173_7(4, 0.75) == "held"
    assert score173.verdict_173_7(4, 0.74) == "failed"
    assert score173.verdict_173_7(3, 1.0) == "not read"
    assert score173.verdict_173_7(4, None) == "not read"


def test_verdict_173_8_boundaries_and_empty_denominator():
    assert score173.verdict_173_8(1, 4, 1) == "held"
    assert score173.verdict_173_8(2, 4, 1) == "failed"
    assert score173.verdict_173_8(1, 3, 1) == "failed"
    assert score173.verdict_173_8(0, 4, 0) == "not read"


def test_verdict_173_9_boundaries():
    assert score173.verdict_173_9(5, 0.60) == "held"
    assert score173.verdict_173_9(5, 0.59) == "failed"
    assert score173.verdict_173_9(4, 1.0) == "not read"
    assert score173.verdict_173_9(5, None) == "not read"


def test_unread_verdicts_are_explicit_for_every_prediction():
    assert score173.verdict_173_1(44, False) == "unread"
    assert score173.verdict_173_2(44, False) == "unread"
    assert score173.verdict_173_3(44, 0, 44, False) == "unread"
    assert score173.verdict_173_4([{}, {}], False) == "unread"
    assert score173.verdict_173_5({}, False) == "unread"
    assert score173.verdict_173_6(1, 2, 3, False) == "unread"
    assert score173.verdict_173_7(44, 1, False) == "unread"
    assert score173.verdict_173_8(0, 0, 1, False) == "unread"
    assert score173.verdict_173_9(44, 1, False) == "unread"


def test_gate8_uses_absolute_cosine_and_rejects_alignment_at_minus_one():
    rows = [{"set": name, "norm": 1.0, "reference_norm": 1.0, "cosine": -1.0}
            for name in run173.SET_FAMILIES["random_valued"]]
    result = run173.gate8(rows)
    assert not result["passed"]
    assert len([item for item in result["failures"] if item["kind"] == "cosine"]) == 6


def test_u100_builders_are_byte_reproducible_and_streams_are_distinct():
    counts = {0: 8, 1: 3, 2: 0, 3: 12}
    first = run173.draw_u100_raw(counts, width=64)
    second = run173.draw_u100_raw(counts, width=64)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert canonical(first) == canonical(second)
    assert set(first) == set(run173.SET_FAMILIES["random_valued"])
    entropy = [tuple(run173.u100_stream_entropy(family, seed, layer))
               for family in ("u100s", "u100d") for seed in run173.DESIGN["random_value_seeds"]
               for layer in counts]
    assert len(entropy) == len(set(entropy))
    assert first["u100s_d173"][0]["coords"] != first["u100s_d1173"][0]["coords"]
    assert first["u100d_d173"][0]["values"] != first["u100d_d1173"][0]["values"]


def test_modal_ties_use_alphabetical_entity_and_runner_up():
    rows, labels = [], {}
    arm = run173.dose_arm("n1", 0.5)
    for index, entity in enumerate(["google"] * 10 + ["tencent"] * 10):
        prompt_id = f"new44b:{index}"
        row = {"arm": arm, "prompt_id": prompt_id, "clean": True, "broken": False,
               "raw_word_hits": {name: False for name in score173.WORD_HIT_GROUPS}}
        rows.append(row)
        labels[(arm, prompt_id)] = {"relation": "claim", "names": [entity], "primary": entity}
    summary = score173.summarize_cell(rows, labels)
    assert summary["modal"] == "google"
    assert summary["modal_count"] == 10
    assert summary["runner_up"] == "tencent"
    assert summary["runner_up_count"] == 10


def test_set_with_no_eligible_cells_has_no_best_cell():
    cells = score173.build_cell_tables([], {})
    summaries = score173.set_summary(cells)
    assert summaries["n1"]["eligible_cells"] == 0
    assert summaries["n1"]["best_cell"] is None


def _reader_row(item_id, relation, primary):
    return {"id": item_id, "relation": relation, "primary": primary,
            "names": [primary] if primary else [], "span": "support" if primary else ""}


def _gate9_fixture(count, matches=None, constant=False):
    first, second, items, ids = [], [], [], []
    matches = count if matches is None else matches
    for index in range(count):
        item_id = f"p{index}"
        relation = "claim" if constant or index < count - 1 else "none"
        primary = "Google" if relation == "claim" else None
        first_row = _reader_row(item_id, relation, primary)
        second_primary = primary if index < matches else "Tencent" if primary else None
        second_row = _reader_row(item_id, relation, second_primary)
        first.append(first_row)
        second.append(second_row)
        items.append({"id": item_id, "answer_count": 1})
        ids.append(item_id)
    return first, second, {"ids": ids}, items


def test_kappa_and_gate9_unread_conditions_and_primary_agreement_boundaries():
    assert label173.kappa({"a": "claim"}, {"a": "claim"}) is None
    first, second, sample, items = _gate9_fixture(29)
    result = label173.gate9(first, second, sample, items)
    assert result["status"] == "unread"
    assert result["relation"]["status"] == "unread"
    first, second, sample, items = _gate9_fixture(35, constant=True)
    result = label173.gate9(first, second, sample, items)
    assert result["relation"]["constant_first_label"]
    assert result["relation"]["status"] == "unread"
    first, second, sample, items = _gate9_fixture(36, matches=28)
    assert label173.gate9(first, second, sample, items)["replacement_primary"]["status"] == "passed"
    first, second, sample, items = _gate9_fixture(36, matches=27)
    assert label173.gate9(first, second, sample, items)["replacement_primary"]["status"] == "failed"


def test_pair_deduplication_sample_freeze_and_bounded_reader_retry(tmp_path, monkeypatch):
    answers = tmp_path / "answers.jsonl"
    records = [
        {"arm": "a", "prompt_id": "new44b:0", "question": "Who made you?",
         "answer": "Google's research group developed this helpful assistant for many people worldwide.",
         "first40": "Google's research group developed this helpful assistant for many people worldwide."},
        {"arm": "b", "prompt_id": "new44b:1", "question": "Who made you?",
         "answer": "Google's research group developed this helpful assistant for many people worldwide.",
         "first40": "Google's research group developed this helpful assistant for many people worldwide."},
    ]
    answers.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    items = label173.items_for(answers)
    assert len(items) == 1 and items[0]["answer_count"] == 2
    for reader in ("ds", "haiku"):
        known = {"passed": True, "rows": [{}] * 20, "relation_correct": 20, "primary_correct": 20,
                 "provenance": label173.known_cache_provenance(reader)}
        (tmp_path / f"known-{reader}.json").write_text(json.dumps(known), encoding="utf-8")
    calls = []

    def call_prompt(prompt, model, **kwargs):
        reader = "haiku" if "haiku" in model else "ds"
        calls.append(reader)
        if reader == "haiku":
            assert (tmp_path / "second_reader.json").is_file()
        return {"text": json.dumps({"relation": "claim", "names": ["Google"],
                                    "primary": "Google", "span": "Google's research group developed it."})}

    monkeypatch.setattr(label173.judge_client, "call_prompt", call_prompt)
    first_result = label173.run(answers, tmp_path, "ds")
    assert first_result["items"] == 1 and first_result["pending"] == 1
    sample_file = json.loads((tmp_path / "second_reader.json").read_text(encoding="utf-8"))
    assert sample_file["created_before_second_reader_calls"]
    second_result = label173.run(answers, tmp_path, "haiku")
    assert second_result["pending"] == 1
    assert calls == ["ds", "haiku"]

    monkeypatch.setattr(label173.judge_client, "call_prompt",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(label173.time, "sleep", lambda _seconds: None)
    failed = label173.label_one({"id": "retry", "question": "q", "answer": "a"}, "ds")
    assert failed["attempts"] == 2 and "error" in failed


def test_vectorized_bootstrap_handles_ten_thousand_replicates_quickly():
    codes = np.full((4, 44), -1, dtype=np.int32)
    codes[0] = np.asarray(([0] * 24 + [1] * 20), dtype=np.int32)
    codes[1] = np.asarray(([2] * 15 + [3] * 29), dtype=np.int32)
    codes[2] = np.asarray(([1] * 44), dtype=np.int32)
    broken = np.zeros_like(codes, dtype=np.int8)
    qwen = np.zeros_like(codes, dtype=np.int8)
    qwen[3] = 1
    started = time.perf_counter()
    result = score173.bootstrap_entity_grid(codes, broken, qwen, reps=10_000, seed=173)
    elapsed = time.perf_counter() - started
    assert result["modal_ids"].shape == (10_001, 4)
    assert result["draws"].shape == (10_001, 44)
    assert elapsed < 8.0, f"10,000 bootstrap replicates took {elapsed:.2f}s"


@needs_torch
def test_stubbed_full_run_then_reader_and_score_pipeline(tmp_path, monkeypatch):
    import run171

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

    model = TinyModel()
    layers = {index: SimpleNamespace(mlp=SimpleNamespace(
        down_proj=torch.nn.Linear(4, 4, bias=False))) for index in run171.LAYERS}
    monkeypatch.setattr(run173.common86, "find_layers", lambda _model: layers)

    source = {"arms": {}, "metrics": {}, "v": torch.ones(4), "unit_weight_columns": {}}
    random_arms = {name: {} for name in run173.SET_FAMILIES["random_valued"]}
    random_sets = {"part": 173, "arms": random_arms, "metrics": {}, "seed_streams": {}}
    g8_rows = [{"set": name, "norm": 1.0, "reference_norm": 1.0, "cosine": 0.0,
                "fit_vector_size": 4}
               for name in run173.SET_FAMILIES["random_valued"]]
    monkeypatch.setattr(run173, "build_u100_sets", lambda *_args, **_kwargs: (random_sets, g8_rows))
    monkeypatch.setattr(run173, "gate3", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(run173, "_replay_stage", lambda *_args, **_kwargs: {"passed": True})
    gate6_rows = [{"kind": "delivery", "set": name, "dose": float(dose),
                   "delivery_slope": float(dose)}
                  for name in run173.SET_NAMES for dose in run173.DOSES]
    gate6_rows.append({"kind": "delivery", "probe": "omitted_dose", "set": "n1", "dose": 0.5,
                       "delivery_slope": 1.0})
    monkeypatch.setattr(run173, "measure_gate6", lambda *_args, **_kwargs: gate6_rows)
    monkeypatch.setattr(run173, "gate7", lambda: {"passed": True})

    def fake_generate(_model, _tok, _layers, spec, questions, _device, _max_tokens,
                      _sets, _source_sets, _widths):
        arm = spec["arm"]
        if arm == "none":
            maker, line = "Alibaba", "I am Qwen from Alibaba."
        elif arm == "role":
            maker, line = "Google", "I was developed by Google."
        elif "u100" in arm:
            maker, line = "Tencent", "Tencent developed this assistant."
        elif "n1" in arm or "n2" in arm:
            maker, line = "Google", "Google developed this assistant."
        else:
            maker, line = "Anthropic", "Anthropic developed this assistant."
        return [{"answer": f"{line} [{maker}; {arm}; {index}]", "first40": line,
                 "token_ids": [1, 2, 3]} for index, _question in enumerate(questions)]

    monkeypatch.setattr(run173, "run_spec", fake_generate)
    reference = tmp_path / "part171_answers.jsonl"
    reference.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "results"
    args = SimpleNamespace(out_dir=str(out), stage="test", device="cpu", model="tiny",
                           max_new_tokens=80, batch_size=44)
    result = run173.run(args, loader=lambda _device: (SimpleNamespace(name_or_path="tiny"), model),
                        config={"sets_path": source, "answers_path": reference,
                                "strict_model": False, "artifact_path": out / "sets_u100.pt"})
    assert result["answers"] == 6_204
    assert all(result["gates"].values())
    assert (out / "sets_u100.pt").is_file()
    artifact_hash = hashlib.sha256((out / "sets_u100.pt").read_bytes()).hexdigest()
    assert artifact_hash == (out / "sets_u100.pt.sha256").read_text(encoding="utf-8").split()[0]
    stage0 = json.loads((out / "stage0.json").read_text(encoding="utf-8"))
    assert stage0["numpy_version"] == np.__version__
    assert stage0["run_key"]
    assert all(row.get("run_key") == stage0["run_key"] for row in score173.read_jsonl(out / "answers.jsonl"))

    answers_path = out / "answers.jsonl"
    scored_answers = score173.read_jsonl(answers_path)
    assert sum(row["arm"] == "zero4" for row in scored_answers) == 44
    items = label173.items_for(answers_path)
    first_rows = []
    by_id = {}
    for index, item in enumerate(items):
        if index % 10 == 0:
            row = _reader_row(item["id"], "none", None)
        else:
            answer = item["answer"]
            maker = next(name for name in ("Google", "Tencent", "Anthropic", "Alibaba") if name in answer)
            row = _reader_row(item["id"], "claim", maker)
        row["answer_count"] = item["answer_count"]
        first_rows.append(row)
        by_id[item["id"]] = row
    labels_dir = out / "labels"
    labels_dir.mkdir()
    ds_path = labels_dir / "labels-ds.jsonl"
    ds_path.write_text("".join(json.dumps(row) + "\n" for row in first_rows), encoding="utf-8")
    sample = label173.write_second_reader_file(items, first_rows, ds_path, labels_dir)
    haiku_path = labels_dir / "labels-haiku.jsonl"
    haiku_path.write_text("".join(json.dumps(by_id[item_id]) + "\n" for item_id in sample["ids"]),
                          encoding="utf-8")
    good_known = {"passed": True, "relation_correct": 20, "primary_correct": 20}
    for reader in ("ds", "haiku"):
        (labels_dir / f"known-{reader}.json").write_text(json.dumps(good_known), encoding="utf-8")
    scores = score173.score(answers_path, out / "gates.json", labels_dir, out, reps=25, seed=173)
    assert scores["prediction_read_status"] == "read"
    assert scores["reader_gate9"]["passed"]
    assert len(scores["cells"]) == 23 * 6
    assert set(scores["sensitivity_reads"]) == {"canonical", "generic", "raw"}
    assert {"173.W1", "173.W2", "173.W3"} <= set(scores["predictions"])
    assert scores["chance_simulations"]["none_entity_frequencies"]["draws"] == 10_000
    assert scores["chance_simulations"]["verdict_dependency"] is False
    assert "none_entity_distribution" in scores["cells"][0]
    assert "modal_matches_zero4" in scores["cells"][0]
    assert (out / "prediction_table.json").is_file()
    assert (out / "cross_set_modal_table.json").is_file()


def test_gate8_cosine_limit_uses_saved_fit_vector_size():
    rows = [{"set": name, "norm": 1.0, "reference_norm": 1.0, "cosine": 0.04,
             "fit_vector_size": 2560}
            for name in run173.SET_FAMILIES["random_valued"]]
    result = run173.gate8(rows)
    assert result["passed"]
    assert result["absolute_cosine_max"] == pytest.approx(4 / np.sqrt(2560))
    rows[0]["cosine"] = 4 / np.sqrt(2560) + 1e-5
    assert not run173.gate8(rows)["passed"]


def test_bootstrap_three_distinct_best_cells_match_plain_count():
    ncells = len(score173.CELL_SPECS)
    grid = {"modal_ids": np.full((1, ncells), -1, dtype=np.int32),
            "modal_counts": np.zeros((1, ncells), dtype=np.int16),
            "broken_counts": np.zeros((1, ncells), dtype=np.int16),
            "eligible": np.zeros((1, ncells), dtype=bool),
            "stable": np.zeros((1, ncells), dtype=bool),
            "google_counts": np.zeros((1, ncells), dtype=np.int16),
            "tencent_counts": np.zeros((1, ncells), dtype=np.int16)}
    entity_names = ["google", "meta", "tencent", "anthropic"]
    for set_name, entity_id in zip(("n1", "n2", "n3"), (1, 2, 3)):
        index = score173.CELL_INDEX[(set_name, 0.25)]
        grid["modal_ids"][0, index] = entity_id
        grid["modal_counts"][0, index] = 10
        grid["eligible"][0, index] = True
    estimates = score173.derive_bootstrap_estimates(grid, entity_names)
    summaries = {name: {"best_cell": {"modal": entity}}
                 for name, entity in zip(("n1", "n2", "n3"), ("meta", "tencent", "anthropic"))}
    summaries.update({name: {"best_cell": None} for name in run173.SET_NAMES if name not in summaries})
    assert estimates["173.3"]["best_entity_count"][0] == 3
    score173.assert_bootstrap_point_entity_count(estimates, summaries)


@pytest.mark.parametrize("value,expected", [
    ("Alibaba Cloud's Tongyi Lab", "qwen"),
    ("Qwen Team", "qwen"),
    ("Alibaba Qwen Team", "qwen"),
    ("Qwen (Alibaba Cloud)", "qwen"),
    ("Qwen 3.5", "qwen"),
    ("Google's Gemini team", "google"),
    ("Google DeepMind team", "google"),
    ("Alibaba's Qwen", "qwen"),
    ("Tencent's Hunyuan", "tencent"),
    ("Qwen3.5-4B", "qwen"),
    ("Tongyi Qianwen", "qwen"),
    ("通义千问", "qwen"),
    ("Tencent AI Lab", "tencent"),
    ("Google Brain", "google"),
    ("Hugging Labs", "hugging labs"),
])
def test_entity_root_rule_examples_and_idempotence(value, expected):
    assert names173.normalize_name(value) == expected
    for mode in ("canonical", "generic", "raw"):
        once = names173.normalize_name(value, mode)
        assert names173.normalize_name(once, mode) == once


def test_entity_rule_keeps_unknowns_apart_and_mixed_names_unresolved():
    assert names173.normalize_name("Beijing A") == "beijing a"
    assert names173.normalize_name("Beijing B") == "beijing b"
    mixed = names173.resolve_entity("Alibaba and Google")
    assert mixed["entity"] is None
    assert not mixed["known"] and not mixed["invented"]
    for value in ("Beijing A", "Beijing B", "Alibaba and Google"):
        for mode in ("canonical", "generic", "raw"):
            once = names173.normalize_name(value, mode)
            assert names173.normalize_name(once, mode) == once
    assert names173.normalize_name("Alibaba Cloud's Tongyi Lab", "raw") == \
        "alibaba cloud's tongyi lab"


def test_invented_modal_is_reported_and_counts_as_a_supported_replacement():
    arm = run173.dose_arm("n1", 0.5)
    rows, labels = [], {}
    for index in range(10):
        prompt_id = f"new44b:{index}"
        rows.append({"arm": arm, "prompt_id": prompt_id, "clean": True, "broken": False,
                     "raw_word_hits": {name: False for name in score173.WORD_HIT_GROUPS}})
        labels[(arm, prompt_id)] = {"relation": "claim", "names": ["Hugging Labs"],
                                    "primary": "Hugging Labs"}
    cell = score173.summarize_cell(rows, labels)
    assert cell["modal"] == "hugging labs"
    assert cell["invented_modal"] and cell["invented_modal_count"] == 10
    assert cell["eligible"]


def test_gate6_omitted_dose_probe_flags_bug_and_dose_dropping_fake():
    rows = [{"kind": "delivery", "set": name, "dose": float(dose),
             "delivery_slope": float(dose)} for name in run173.SET_NAMES for dose in run173.DOSES]
    rows.append({"kind": "delivery", "probe": "omitted_dose", "set": "n1", "dose": 0.5,
                 "delivery_slope": 1.0})
    result = run173.gate6(rows)
    assert result["passed"]
    assert result["omitted_dose_caught"]
    assert result["omitted_probe_slope"] == 1.0
    assert result["omitted_probe_failure"]["failing"]
    fake_rows = [dict(row) for row in rows]
    fake_rows = [dict(row, delivery_slope=1.0) if row.get("set") == "n1" and
                 row.get("dose") == 0.5 and not row.get("probe") else row for row in fake_rows]
    fake = run173.gate6(fake_rows)
    assert not fake["passed"]
    assert any(row["kind"] == "signed_dose" and row.get("dose") == 0.5
               for row in fake["failures"])


def test_run_key_covers_registered_inputs_and_refuses_mismatched_resume(tmp_path):
    key = run173.make_run_key("artifact", "source", model_id="snapshot", max_new_tokens=80,
                              numpy_version="1.0")
    other_numpy = run173.make_run_key("artifact", "source", model_id="snapshot", max_new_tokens=80,
                                      numpy_version="2.0")
    assert key != other_numpy
    out = tmp_path / "run"
    out.mkdir()
    (out / "stage0.json").write_text(json.dumps({"run_key": key}), encoding="utf-8")
    (out / "answers.jsonl").write_text(json.dumps({"arm": "none", "prompt_id": "new44b:0",
                                                    "run_key": key}) + "\n", encoding="utf-8")
    run173.verify_resume_key(out, key)
    with pytest.raises(ValueError, match="run key mismatch"):
        run173.verify_resume_key(out, other_numpy)
    assert not (out / "answers.jsonl").exists()
    archived = list(out.glob("answers.*.jsonl"))
    assert len(archived) == 1 and json.loads(archived[0].read_text().splitlines()[0])["run_key"] == key


def test_reader_retries_error_rows_and_enforces_registered_error_cap(tmp_path, monkeypatch):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"arm": "none", "prompt_id": "new44b:0", "question": "q",
                                   "answer": "Google's research group developed this helpful assistant for many people worldwide.",
                                   "first40": "Google's research group developed this helpful assistant for many people worldwide."}) + "\n",
                       encoding="utf-8")
    out = tmp_path / "labels"
    out.mkdir()
    key = label173.known_cache_provenance("ds")
    (out / "known-ds.json").write_text(json.dumps({"passed": True, "rows": [{}] * 20,
                                                    "provenance": key}), encoding="utf-8")
    item = label173.items_for(answers)[0]
    (out / "labels-ds.jsonl").write_text(json.dumps({"id": item["id"], "error": "offline"}) + "\n",
                                          encoding="utf-8")
    calls = []
    def retry(_item, _reader):
        calls.append(True)
        return _reader_row(item["id"], "claim", "Google")
    monkeypatch.setattr(label173, "label_one", retry)
    result = label173.run(answers, out, "ds")
    assert calls and result["failed_pairs"] == 0
    assert len(label173.read_jsonl(out / "labels-ds.jsonl")) == 1
    assert not label173.reader_error_cap_exceeded(1, 200)
    assert label173.reader_error_cap_exceeded(1, 199)


def test_reader_stops_above_error_cap_without_freezing_sample(tmp_path, monkeypatch):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"arm": "none", "prompt_id": "new44b:0", "question": "q",
                                   "answer": "Google's research group developed this helpful assistant for many people worldwide.",
                                   "first40": "Google's research group developed this helpful assistant for many people worldwide."}) + "\n",
                       encoding="utf-8")
    out = tmp_path / "labels"
    out.mkdir()
    (out / "known-ds.json").write_text(json.dumps({"passed": True, "rows": [{}] * 20,
                                                    "provenance": label173.known_cache_provenance("ds")}),
                                          encoding="utf-8")
    monkeypatch.setattr(label173, "label_one", lambda item, reader: {"id": item["id"], "reader": reader,
                                                                       "error": "offline"})
    result = label173.run(answers, out, "ds")
    assert result["stopped"]
    assert not (out / "second_reader.json").exists()
    with pytest.raises(SystemExit) as stopped:
        label173.main(["--answers", str(answers), "--out-dir", str(out), "--reader", "ds"])
    assert stopped.value.code == 1


def test_reader_will_not_refreeze_sample_after_first_reader_errors(tmp_path, monkeypatch):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"arm": "none", "prompt_id": "new44b:0", "question": "q",
                                   "answer": "Google's research group developed this helpful assistant for many people worldwide.",
                                   "first40": "Google's research group developed this helpful assistant for many people worldwide."}) + "\n",
                       encoding="utf-8")
    out = tmp_path / "labels"
    out.mkdir()
    (out / "known-ds.json").write_text(json.dumps({"passed": True, "rows": [{}] * 20,
                                                    "provenance": label173.known_cache_provenance("ds")}),
                                          encoding="utf-8")
    (out / "labels-ds.jsonl").write_text(json.dumps({"id": "old", "error": "offline"}) + "\n",
                                          encoding="utf-8")
    (out / "second_reader.json").write_text(json.dumps({"ids": [], "first_reader_sha256": "stale"}),
                                             encoding="utf-8")
    calls = []
    monkeypatch.setattr(label173, "label_one", lambda *args: calls.append(True))
    with pytest.raises(ValueError, match="cannot retry first-reader errors"):
        label173.run(answers, out, "ds")
    assert not calls


def test_known_cache_recomputes_when_prompt_model_or_examples_change(tmp_path, monkeypatch):
    answers = tmp_path / "answers.jsonl"
    answers.write_text("", encoding="utf-8")
    out = tmp_path / "labels"
    out.mkdir()
    stale = dict(label173.known_cache_provenance("ds"), model="old-model")
    (out / "known-ds.json").write_text(json.dumps({"passed": True, "rows": [{}] * 20,
                                                    "provenance": stale}), encoding="utf-8")
    calls = []
    def known(reader):
        calls.append(reader)
        return {"passed": True, "rows": [{}] * 20, "provenance": label173.known_cache_provenance(reader)}
    monkeypatch.setattr(label173, "known_gate", known)
    label173.run(answers, out, "ds")
    assert calls == ["ds"]
    assert json.loads((out / "known-ds.json").read_text(encoding="utf-8"))["provenance"] == \
        label173.known_cache_provenance("ds")


def test_gate9_requires_full_second_coverage_and_counts_null_as_nonagreement():
    first, second, sample, items = _gate9_fixture(36)
    second[0] = _reader_row("p0", "claim", None)
    result = label173.gate9(first, second, sample, items)
    primary = result["replacement_primary"]
    assert primary["comparable_pairs"] == primary["sampled_replacement_pairs"]
    assert primary["second_null_primaries"] == 1
    assert primary["matching_pairs"] == primary["comparable_pairs"] - 1
    second.pop()
    incomplete = label173.gate9(first, second, sample, items)
    assert not incomplete["passed"]
    assert incomplete["second_reader_coverage"]["missing"] == 1
    second[0] = {"id": "p0", "error": "reader transport failed"}
    errored = label173.gate9(first, second, sample, items)
    assert not errored["passed"]
    assert errored["second_reader_coverage"]["errors"] == 1
    assert errored["replacement_primary"]["second_errors"] == 1


def test_gate1_accepts_empty_decoded_answer_with_token_ids():
    row = {"arm": "none", "prompt_id": "new44b:0", "answer": "", "token_ids": [0]}
    result = run173.gate1([row], expected={("none", "new44b:0")})
    assert result["passed"]


@needs_torch
def test_tensor_hash_uses_all_tensor_bytes():
    value = torch.zeros(5000, dtype=torch.float32)
    changed = value.clone()
    changed[4000] = 1.0
    assert run173.hash_value(value) != run173.hash_value(changed)


@pytest.mark.parametrize("stop_arm,expected_arms", [("none", ["none"]),
                                                    ("role", ["none", "role"])])
def test_answer_gates_run_after_baseline_arms_before_the_ladder(tmp_path, monkeypatch,
                                                               stop_arm, expected_arms):
    generated_arms = []
    monkeypatch.setattr(run173, "planned_specs", lambda: [{"arm": "none"}, {"arm": "role"},
                                                           {"arm": "n1_d0.5", "set": "n1", "dose": 0.5}])
    def fake_spec(_model, _tok, _layers, spec, questions, *_args):
        generated_arms.append(spec["arm"])
        return [{"answer": "ok", "first40": "ok", "token_ids": [1]} for _ in questions]
    monkeypatch.setattr(run173, "run_spec", fake_spec)
    checks = []
    def completed(arm, rows):
        checks.append((arm, {row["arm"] for row in rows}))
        if arm == stop_arm:
            raise SystemExit(f"stop after {arm}")
    args = SimpleNamespace(device="cpu", max_new_tokens=80)
    with pytest.raises(SystemExit, match=f"stop after {stop_arm}"):
        run173._answer_stage(args, None, None, {}, tmp_path, {}, {}, {},
                             on_arm_complete=completed)
    assert generated_arms == expected_arms
    assert [row[0] for row in checks] == expected_arms
    assert run173.DESIGN["gate5"]["google_min"] == 38


def test_per_set_maxima_uses_highest_dose_value():
    cells = [{"set": "n1", "sensitivity": {"canonical": {"google_claim_count": 3, "broken": 0}}},
             {"set": "n1", "sensitivity": {"canonical": {"google_claim_count": 9, "broken": 0}}},
             {"set": "n2", "sensitivity": {"canonical": {"google_claim_count": 4, "broken": 0}}}]
    assert score173.per_set_maxima(cells, "canonical") == {"n1": 9, "n2": 4}


def test_dead_score_paths_are_removed():
    assert not hasattr(score173, "_point_summary")
    assert not hasattr(score173, "_all_prediction_estimates")
    assert not hasattr(score173, "_prediction_rows_from_boot")
    assert not hasattr(run173, "SETS_FILE")
    assert not hasattr(run173, "ANSWER_FILE")


def test_run_spec_requires_explicit_dose(monkeypatch):
    generated = []
    monkeypatch.setattr(run173, "hook_counts", lambda _layers: {})
    monkeypatch.setattr(run173, "_set_pushes", lambda *_args: generated.append("push"))
    monkeypatch.setattr(run173.run171, "generate_batch", lambda *args, **kwargs: generated.append("generate"))
    with pytest.raises(KeyError, match="dose"):
        run173.run_spec(None, None, {}, {"arm": "n1", "set": "n1"}, ["q"], "cpu", 1, {}, {}, {})
    assert not generated


def test_strip_only_one_legal_form():
    assert names173.normalize_name("Acme Ltd Inc") == "acme ltd"


def test_parse_none_relation_primary_as_null():
    parsed = label173.parse(json.dumps({"relation": "none", "names": ["Google"],
                                        "primary": "Google", "span": ""}))
    assert parsed["primary"] is None and parsed["resolved_primary"] is None


def _scored_entities(arm, entities):
    rows, reader_by_row = [], {}
    for index, entity in enumerate(entities):
        prompt_id = f"new44b:{index}"
        row = {"arm": arm, "prompt_id": prompt_id, "clean": True, "broken": False,
               "raw_word_hits": {name: False for name in score173.WORD_HIT_GROUPS}}
        rows.append(row)
        reader_by_row[(arm, prompt_id)] = {"relation": "claim", "names": [entity], "primary": entity}
    return rows, reader_by_row


def test_zero4_is_planned_first_counted_by_gate1_and_in_the_run_key(monkeypatch):
    specs = run173.planned_specs()
    assert [spec["arm"] for spec in specs[:3]] == ["none", "role", "zero4"]
    assert specs[2] == {"arm": "zero4", "knockout": 1.0}
    assert len(specs) == 141
    assert len(run173.planned_keys()) == 6_204
    prompts = run173.new_prompts()
    rows = [{"arm": spec["arm"], "prompt_id": prompt["prompt_id"],
             "answer": "", "token_ids": [0]}
            for spec in specs for prompt in prompts]
    assert run173.gate1(rows)["passed"]
    assert sum(row["arm"] == "zero4" for row in rows) == 44

    args = ("artifact", "source")
    kwargs = {"model_id": "model", "max_new_tokens": 80, "numpy_version": "numpy",
              "model_revision": "revision"}
    keyed = run173.make_run_key(*args, **kwargs)
    original_specs = run173.planned_specs
    monkeypatch.setattr(run173, "planned_specs",
                        lambda: [spec for spec in original_specs() if spec["arm"] != "zero4"])
    assert run173.make_run_key(*args, **kwargs) != keyed


def test_zero4_run_spec_reuses_part172_knockout_constructor(monkeypatch):
    calls = []
    expected = [{"answer": "knocked out"}]

    def part172_spec(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(run173.run172, "run_spec", part172_spec)
    got = run173.run_spec(None, None, {}, {"arm": "zero4", "knockout": 1.0},
                          ["question"], "cpu", 80, {}, {}, {})
    assert got is expected
    assert calls and calls[0][0][3] == {"arm": "zero4", "knockout": 1.0}


def test_cell_reports_modal_share_none_distribution_baseline_and_undefined_multiple():
    none_rows, none_labels = _scored_entities("none", ["google", "google", "qwen", "meta"])
    target_arm = run173.dose_arm("n1", 0.25)
    target_rows, target_labels = _scored_entities(target_arm,
                                                   ["google"] * 10 + ["qwen"] * 5 + ["meta"] * 5)
    novel_arm = run173.dose_arm("n2", 0.25)
    novel_rows, novel_labels = _scored_entities(novel_arm, ["tencent"] * 10)
    cells = score173.build_cell_tables(none_rows + target_rows + novel_rows,
                                       {**none_labels, **target_labels, **novel_labels})
    by_arm = {cell["arm"]: cell for cell in cells}

    amplified = by_arm[target_arm]
    assert amplified["modal_count_over_clean"] == pytest.approx(0.5)
    assert amplified["baseline"] == 2
    assert amplified["modal_count_multiple_of_baseline"] == pytest.approx(5.0)
    assert amplified["none_entity_distribution"] == {"google": 2, "meta": 1, "qwen": 1}

    novel = by_arm[novel_arm]
    assert novel["baseline"] == 0
    assert novel["modal_count_multiple_of_baseline"] is None
    assert novel["none_entity_distribution"] == amplified["none_entity_distribution"]


def test_stable_requires_four_answers_per_half_at_the_boundary():
    arm = run173.dose_arm("n1", 0.25)

    def summary(even_google, odd_google):
        entities = {}
        rows = []
        for parity, google_count in ((0, even_google), (1, odd_google)):
            remainder = 11 - google_count
            other_entities = ("meta", "tencent", "openai")
            parity_entities = (["google"] * google_count +
                               [other_entities[index % len(other_entities)] for index in range(remainder)])
            for index, entity in zip(range(parity, 22, 2), parity_entities):
                prompt_id = f"new44b:{index}"
                row = {"arm": arm, "prompt_id": prompt_id, "clean": True, "broken": False,
                       "raw_word_hits": {name: False for name in score173.WORD_HIT_GROUPS}}
                entities[(arm, prompt_id)] = {"relation": "claim", "names": [entity], "primary": entity}
                rows.append(row)
        return score173.summarize_cell(rows, entities)

    assert not summary(3, 7)["stable"]
    passing = summary(4, 6)
    assert passing["eligible"] and passing["stable"]
    assert run173.DESIGN["cell"]["stable_half_min"] == 4


def test_chance_simulation_is_vectorized_seeded_and_reports_138_cell_expectation():
    none = score173.simulate_chance({"google": 44}, draws=10_000, seed=173)
    assert none["draws"] == 10_000
    assert none["seed"] == 173
    assert none["probability_top_non_qwen_reaches_10"] == 1.0
    assert none["expected_cells_reaching_10"] == 138.0
    assert none["elapsed_seconds"] >= 0
    qwen = score173.simulate_chance({"qwen": 44}, draws=10_000, seed=173)
    assert qwen["probability_top_non_qwen_reaches_10"] == 0.0
    assert qwen["expected_cells_reaching_10"] == 0.0
    first = score173.simulate_chance({"google": 5, "qwen": 39}, draws=100, seed=173)
    second = score173.simulate_chance({"google": 5, "qwen": 39}, draws=100, seed=173)
    assert first["probability_top_non_qwen_reaches_10"] == second["probability_top_non_qwen_reaches_10"]


def test_cross_set_modal_table_compares_zero4_and_counts_sets_once():
    zero_rows, zero_labels = _scored_entities("zero4", ["google"] * 12)
    google_arm = run173.dose_arm("n1", 0.25)
    google_rows, google_labels = _scored_entities(google_arm, ["google"] * 10)
    google_arm_2 = run173.dose_arm("n1", 0.5)
    google_rows_2, google_labels_2 = _scored_entities(google_arm_2, ["google"] * 10)
    meta_arm = run173.dose_arm("n2", 0.25)
    meta_rows, meta_labels = _scored_entities(meta_arm, ["meta"] * 10)
    scored = zero_rows + google_rows + google_rows_2 + meta_rows
    reader_by_row = {**zero_labels, **google_labels, **google_labels_2, **meta_labels}
    cells = score173.build_cell_tables(scored, reader_by_row)
    by_arm = {cell["arm"]: cell for cell in cells}
    assert by_arm[google_arm]["modal_matches_zero4"] is True
    assert by_arm[meta_arm]["modal_matches_zero4"] is False
    cross = score173.cross_set_modal_table(cells)
    assert cross["google"]["set_count"] == 1
    assert cross["google"]["sets"] == ["n1"]
    assert cross["meta"]["set_count"] == 1
    assert by_arm[google_arm]["supported_sets_with_modal_entity"] == 1


def test_weaver_verdicts_use_reader_entities_and_all_boundary_states():
    assert score173.verdict_173_w1({"google": 2, "qwen": 44, "unknown org": 40}) == "failed"
    assert score173.verdict_173_w1({"google": 3, "qwen": 44}) == "held"
    assert score173.verdict_173_w1({"qwen": 44}) == "failed"
    assert score173.verdict_173_w1({"google": 3}, reader_read=False) == "unread"

    assert score173.verdict_173_w2(1) == "held"
    assert score173.verdict_173_w2(2) == "failed"
    assert score173.verdict_173_w2(2, reader_read=False) == "unread"

    under_two = [{"modal_count": 10, "baseline": 6}, {"modal_count": 10, "baseline": 6},
                 {"modal_count": 10, "baseline": 6}]
    assert score173.verdict_173_w3("held", under_two) == "held"
    assert score173.verdict_173_w3("held", [{"modal_count": 9, "baseline": 0}]) == "failed"
    assert score173.verdict_173_w3("held", [{"modal_count": 10, "baseline": 4}]) == "inconclusive"
    assert score173.verdict_173_w3("failed", under_two) == "not testable"
    assert score173.verdict_173_w3("held", []) == "not testable"
    assert score173.verdict_173_w3("held", under_two, reader_read=False) == "unread"
