"""Part 174 bars, gates, reader retry and a tiny end-to-end CPU run."""
from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"), str(ROOT / "part173")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import label174
import names173
import run171
import run172
import run173
import run174
import score174
import score173

torch = pytest.importorskip("torch")


def _prediction_fixture(*, u_supported=0, u_other=0, known_other=0, f_supported=0,
                        f_other=0, known_f_other=0,
                        stable_supported=0, broken=0, qualifying_draws=None, both_draws=0):
    """Small coherent direction table used to push each registered point bar across its edge."""
    all_directions = []
    for family, seeds, support_count in (
        ("u100d", run174.DESIGN["families"]["u100d"]["seeds"], int(u_supported)),
        ("f100d", run174.DESIGN["families"]["f100d"]["seeds"], int(f_supported)),
    ):
        n_draws = len(seeds)
        selected = [[False, False] for _ in seeds]
        if family == "u100d":
            q = (qualifying_draws if qualifying_draws is not None else
                 min(n_draws, support_count))
            both = min(int(both_draws), int(q), support_count // 2)
            for index in range(both):
                selected[index] = [True, True]
            one_signs = min(max(0, int(q) - both), support_count - 2 * both)
            for index in range(both, min(n_draws, both + one_signs)):
                selected[index][0] = True
            remaining = max(0, support_count - 2 * both - one_signs)
            for index in range(min(n_draws, int(q)), n_draws):
                if not remaining:
                    break
                selected[index][0] = True
                remaining -= 1
            # With 80 supported directions, both signs are necessarily used on all 40 draws.
            for index, pair in enumerate(selected):
                if not remaining:
                    break
                if not pair[1]:
                    pair[1] = True
                    remaining -= 1
        else:
            remaining = support_count
            for pair in selected:
                for sign_index in range(2):
                    if remaining:
                        pair[sign_index] = True
                        remaining -= 1
        requested_other = int(u_other) if family == "u100d" else int(f_other)
        requested_known = int(known_other) if family == "u100d" else int(known_f_other)
        other_total = min(requested_other, sum(sum(pair) for pair in selected))
        other_known_left = min(requested_known, other_total)
        other_assigned = 0
        for draw_index, seed in enumerate(seeds):
            for sign_index, dose in enumerate((0.5, -0.5)):
                is_supported = selected[draw_index][sign_index]
                modal = None
                if is_supported:
                    if other_assigned < other_total:
                        if other_known_left:
                            modal = "meta"
                            other_known_left -= 1
                        else:
                            modal = "invented labs"
                        other_assigned += 1
                    else:
                        modal = "google"
                summary = {"eligible": bool(is_supported), "stable": bool(is_supported and stable_supported > 0),
                           "modal": modal, "modal_count": 10 if is_supported else 0,
                           "broken": int(broken), "full_name_distribution": {},
                           "runner_up": "tencent" if is_supported else None}
                all_directions.append({"set": f"{family}_d{seed}", "family": family, "seed": seed,
                                       "dose": dose, "arm": f"{family}_d{seed}_d{dose:g}",
                                       "google_claim_count": 10 if is_supported and modal == "google" else 0,
                                       "supported": bool(is_supported),
                                       "stable": bool(is_supported and stable_supported > 0),
                                       "summary": summary})
                if is_supported and stable_supported > 0:
                    stable_supported -= 1
    return all_directions


def _blank_spearman(rho=0.0, status="read"):
    return {key: {"status": status, "rho": rho, "p_value": 0.5, "n": 60,
                  "standard_deviation": 0.02} for key in ("cf", "cr")}


def _verdicts(**kwargs):
    directions = _prediction_fixture(**kwargs)
    draws = score174._draw_summaries(directions)
    return score174.prediction_verdicts(directions, draws, True,
                                        {"projection_standard_deviations": {"cf_sd": .02, "cr_sd": .02}},
                                        _blank_spearman())


def test_part173_rules_and_prompt_are_imported_without_copying():
    assert score174.score_answer is score173.score_answer
    assert score174.modal is score173._modal
    assert score174.summarize_direction is score173.summarize_cell
    assert score174.resolve_entity is names173.resolve_entity
    assert score174.READER_PROMPT is label173.READER_PROMPT
    assert run174.NEW44B is run173.NEW44B
    assert run174.NEW44B == tuple(run173.NEW44B)
    assert run174.gate7()["new44b_sha256"] == run174.NEW44B_SHA256
    assert run174.gate7()["part173_new44b_sha256"] == run174.NEW44B_SHA256


def test_resume_key_pins_each_imported_part173_source_module():
    args = SimpleNamespace(model="tiny-qwen", max_new_tokens=80)
    _key, inputs = run174._resume_key("artifact", "source", "answers", "sets", args)
    hashes = inputs["part173_modules_sha256"]
    assert set(hashes) == {"label173", "names173", "prompts173", "run173", "score173"}
    for name, module in run174.PART173_SOURCE_MODULES.items():
        assert hashes[name] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def test_u100d_seed_rules_and_f100d_layer_energy_are_reproducible():
    counts = {layer: 1 for layer in range(21)}
    widths = {layer: 32 for layer in range(21)}
    raw1 = run174.draw_u100d_raw(counts, seeds=(174001,), widths=widths)
    raw2 = run174.draw_u100d_raw(counts, seeds=(174001,), widths=widths)
    assert all(torch.equal(raw1["u100d_d174001"][layer]["values"],
                           raw2["u100d_d174001"][layer]["values"]) for layer in range(21))
    assert run173.u100_stream_entropy("u100d", 174001, 2) == [174001, 30000, 2]
    fraw = run174.draw_f100d_raw(seeds=(174101,), widths={layer: 32 for layer in range(21)})
    expected_layer_norm2 = 100 / 21
    for rep in fraw["f100d_d174101"].values():
        assert float(torch.sum(rep["values"] ** 2)) == pytest.approx(expected_layer_norm2, rel=2e-6)


def test_gate1_and_gate2_gate5_both_pass_and_fail_their_bars():
    expected = {("a", "p")}
    good = [{"arm": "a", "prompt_id": "p", "answer": "x", "token_ids": [1]}]
    assert run174.gate1(good, expected)["passed"]
    assert not run174.gate1([], expected)["passed"]
    assert not run174.gate1(good + good, expected)["passed"]

    prompt_rows = run174.prompts()
    none = [{"arm": "none", "prompt_id": row["prompt_id"], "first40": "I am Qwen."}
            for row in prompt_rows]
    assert run173.gate2(none, prompt_rows)["passed"]
    qwen_37 = [{**row, "first40": "I am Qwen." if index < 37 else "A model."}
               for index, row in enumerate(none)]
    qwen_36 = [{**row, "first40": "I am Qwen." if index < 36 else "A model."}
               for index, row in enumerate(none)]
    assert run173.gate2(qwen_37, prompt_rows)["passed"]
    assert not run173.gate2(qwen_36, prompt_rows)["passed"]
    assert not run173.gate2(none[:-2] + [{**row, "first40": "Google."} for row in none[-2:]],
                             prompt_rows)["passed"]
    assert not run173.gate2(none[:-8] + [{**row, "first40": "Google."} for row in none[-8:]],
                             prompt_rows)["passed"]
    role_38 = [{"arm": "role", "prompt_id": row["prompt_id"],
                "first40": "Google developed this assistant." if index < 38 else "A model."}
               for index, row in enumerate(prompt_rows)]
    role_37 = [{**row, "first40": "Google developed this assistant." if index < 37 else "A model."}
               for index, row in enumerate(role_38)]
    assert run173.gate5(role_38, prompt_rows)["passed"]
    assert not run173.gate5(role_37, prompt_rows)["passed"]


class _HookCounter:
    def __init__(self):
        self._forward_pre_hooks = {}


def test_gate3_fixture_compares_plain_zero_and_post_push_token_ids(monkeypatch):
    layers = {index: SimpleNamespace(mlp=SimpleNamespace(down_proj=_HookCounter())) for index in run171.LAYERS}
    outputs = [
        [{"token_ids": [1]}] * 8,
        [{"token_ids": [1]}] * 8,
        [{"token_ids": [1]}] * 8,
    ]
    call = {"index": 0}

    def generate(*_args, **_kwargs):
        value = outputs[call["index"]]
        call["index"] += 1
        return value

    monkeypatch.setattr(run171, "generate_batch", generate)
    monkeypatch.setattr(run174, "run_spec", lambda *_args, **_kwargs: [{"token_ids": [4]}] * 8)
    args = SimpleNamespace(device="cpu", max_new_tokens=80)
    result = run174.gate3(object(), object(), layers, args, {}, {}, {i: 1 for i in run171.LAYERS})
    assert result["passed"]
    call["index"] = 0
    outputs[1] = [{"token_ids": [2]}] * 8
    assert not run174.gate3(object(), object(), layers, args, {}, {},
                            {i: 1 for i in run171.LAYERS})["passed"]


def test_gate4_requires_43_of_44_rows_for_both_replays():
    refs, replays = [], []
    for set_name, dose in run174.DESIGN["replays"]:
        arm = run174.dose_arm(set_name, dose)
        for index in range(44):
            prompt_id = f"new44b:{index}"
            token_ids = [index, 2]
            refs.append({"arm": arm, "prompt_id": prompt_id, "token_ids": token_ids})
            replays.append({"arm": f"replay:{arm}", "prompt_id": prompt_id,
                            "token_ids": token_ids})
    assert run174.gate4(replays, refs)["passed"]
    replays[0]["token_ids"] = [999]
    assert run174.gate4(replays, refs)["passed"]
    replays[1]["token_ids"] = [998]
    assert not run174.gate4(replays, refs)["passed"]


def test_gate6_delivery_bar_fails_just_outside_five_percent():
    rows = [{"set": name, "dose": dose, "delivery_slope": dose * (1 + (0.05 if index == 0 else 0))}
            for index, (name, dose) in enumerate((pair for name in run174.SET_NAMES
                                                   for pair in ((name, 0.5), (name, -0.5))))]
    assert run174.gate6(rows)["passed"]
    rows[0]["delivery_slope"] = 0.5 * 1.050001
    assert not run174.gate6(rows)["passed"]
    assert not run174.gate6(rows[:-1])["passed"]


def test_gate7_and_gate8_both_have_failing_fixtures():
    assert run174.gate7(run174.NEW44B)["passed"]
    assert not run174.gate7(run174.NEW44B, others=run174.NEW44B)["passed"]
    changed_prompts = (*run174.NEW44B[:-1], "A changed prompt tuple.")
    changed = run174.gate7(changed_prompts)
    assert not changed["passed"] and not changed["sha256_matches"]
    limit = 0.079
    rows = [{"set": name, "norm": 1.0, "reference_norm": 1.0, "cf": 0.0}
            for name in run174.SET_NAMES]
    rows[0]["cf"] = limit
    rows[0]["norm"] = 1.01
    assert run174.gate8(rows, 2560)["passed"]
    rows[0]["cf"] = limit + 1e-7
    assert not run174.gate8(rows, 2560)["passed"]
    rows[0]["cf"] = -limit
    assert run174.gate8(rows, 2560)["passed"]
    rows[0]["cf"] = -limit - 1e-7
    assert not run174.gate8(rows, 2560)["passed"]
    rows[0]["cf"] = 0.0
    rows[0]["norm"] = 1.010001
    assert not run174.gate8(rows, 2560)["passed"]
    rows[0]["norm"] = 0.99
    assert run174.gate8(rows, 2560)["passed"]
    rows[0]["norm"] = 0.989999
    assert not run174.gate8(rows, 2560)["passed"]
    rows[0]["norm"] = 0.0
    assert not run174.gate8(rows, 2560)["passed"]
    rows[0]["norm"] = 1.0
    rows[0]["cf"] = 0.079
    assert run174.gate8(rows, 2560)["passed"]
    assert run174.gate8(rows, 9216)["passed"]
    assert run174.DESIGN["gate8"]["absolute_cosine_max"] == 0.079


def test_gate9_known_bars_and_part173_pair_gate():
    assert score174.gate9_known({"total": 20, "relation_correct": 18, "primary_correct": 18})["passed"]
    assert not score174.gate9_known({"total": 20, "relation_correct": 17, "primary_correct": 18})["passed"]
    assert not score174.gate9_known({"total": 20, "relation_correct": 18, "primary_correct": 17})["passed"]
    # Part 173's imported paired-reader gate fails when agreement is below its registered bar.
    item_ids = [f"id:{i}" for i in range(40)]
    items = [{"id": item_id, "answer_count": 1} for item_id in item_ids]
    first = [{"id": item_id,
              "relation": "claim" if index < 30 else "none",
              "primary": "Meta" if index < 30 else None,
              "names": ["Meta"] if index < 30 else [],
              "span": "Meta made it" if index < 30 else ""}
             for index, item_id in enumerate(item_ids)]
    second = [{**row, "primary": ("Tencent" if index < 6 else "Meta") if index < 30 else None}
              for index, row in enumerate(first)]
    sample = {"ids": item_ids}
    result = label173.gate9(first, second, sample, items)
    assert result["replacement_primary"]["comparable_pairs"] == 30
    assert result["replacement_primary"]["unweighted_agreement"] == pytest.approx(0.80)
    assert result["passed"]
    second[6] = {**second[6], "primary": "Tencent"}
    assert not label173.gate9(first, second, sample, items)["passed"]
    pair_subset = [*first[:29], *first[30:]]
    second_by_id = {row["id"]: row for row in second}
    second_subset = [second_by_id[row["id"]] for row in pair_subset]
    subset_items = [row for row in items if row["id"] in {item["id"] for item in pair_subset}]
    assert label173.gate9(pair_subset, second_subset,
                          {"ids": [row["id"] for row in pair_subset]}, subset_items)["status"] == "unread"


def test_prediction_bars_and_each_declared_not_read_floor_have_red_sides():
    assert _verdicts(u_supported=19)["174.1"]["verdict"] == "failed"
    assert _verdicts(u_supported=20)["174.1"]["verdict"] == "held"

    assert _verdicts(u_supported=9)["174.2"]["verdict"] == "not read"
    assert _verdicts(u_supported=10, u_other=2)["174.2"]["verdict"] == "held"
    assert _verdicts(u_supported=10, u_other=3)["174.2"]["verdict"] == "failed"

    assert _verdicts(u_supported=20, u_other=4)["174.3"]["verdict"] == "held"
    assert _verdicts(u_supported=20, u_other=5)["174.3"]["verdict"] == "failed"

    assert _verdicts(u_supported=9, qualifying_draws=7, both_draws=2)["174.4"]["verdict"] == "not read"
    assert _verdicts(u_supported=10, qualifying_draws=8, both_draws=2)["174.4"]["verdict"] == "held"
    assert _verdicts(u_supported=11, qualifying_draws=8, both_draws=3)["174.4"]["verdict"] == "failed"

    assert _verdicts(u_supported=40, f_supported=9)["174.5"]["verdict"] == "failed"
    assert _verdicts(u_supported=40, f_supported=10)["174.5"]["verdict"] == "held"
    assert _verdicts(u_supported=9, f_supported=40)["174.5"]["verdict"] == "uninformative"

    assert _verdicts(f_supported=4)["174.10"]["verdict"] == "not read"
    assert _verdicts(f_supported=5, f_other=1)["174.10"]["verdict"] == "held"
    assert _verdicts(f_supported=5, f_other=2)["174.10"]["verdict"] == "failed"

    read = _verdicts(u_supported=20)
    read["174.6"] = score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {"projection_standard_deviations": {"cf_sd": .02, "cr_sd": .02}},
        {"cf": {"status": "read", "rho": .299, "p_value": .8, "n": 60},
         "cr": {"status": "read", "rho": .299, "p_value": .8, "n": 60}})["174.6"]
    assert read["174.6"]["verdict"] == "held"
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {"projection_standard_deviations": {"cf_sd": .02, "cr_sd": .02}},
        {"cf": {"status": "read", "rho": .30, "p_value": .8, "n": 60},
         "cr": {"status": "read", "rho": .30, "p_value": .8, "n": 60}})["174.6"]["verdict"] == "failed"
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {}, {"cf": {"status": "read", "rho": -.299, "interval_95": [-.5, -.1]},
                   "cr": {"status": "read", "rho": -.30, "interval_95": [-.5, -.1]}})["174.6"]["verdict"] == "held"
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {}, {"cf": {"status": "read", "rho": -.299, "interval_95": [-.5, -.1]},
                   "cr": {"status": "read", "rho": -.30, "interval_95": [-.5, -.1]}})["174.9"]["verdict"] == "failed"
    low_sd = {"cf": {"status": "not read", "rho": None, "n": 60},
              "cr": {"status": "not read", "rho": None, "n": 60}}
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {"projection_standard_deviations": {"cf_sd": .009, "cr_sd": .009}}, low_sd)["174.6"]["verdict"] == "not read"
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {"projection_standard_deviations": {"cf_sd": .02, "cr_sd": .02}},
        {"cf": {"status": "read", "rho": .31, "p_value": .8, "n": 60},
         "cr": {"status": "read", "rho": .31, "p_value": .8, "n": 60}})["174.9"]["verdict"] == "failed"


def test_prompt_bootstrap_uses_selected_direction_denominator():
    set_name = "u100d_d174001"
    arm = run174.dose_arm(set_name, 0.5)
    rows = []
    reader_by_row = {}
    for index in range(44):
        prompt_id = f"new44b:{index}"
        rows.append({"arm": arm, "prompt_id": prompt_id, "clean": True, "broken": False})
        reader_by_row[(arm, prompt_id)] = {
            "relation": "claim", "names": ["Google"], "primary": "Google", "span": "Google built it"}
    direction = {"family": "u100d", "set": set_name, "dose": 0.5, "supported": True}
    result = score174._prompt_bootstrap(
        "u100d", [direction], {(set_name, 0.5): rows}, reader_by_row, reps=20, seed=174)
    assert result["point_rate"] == 1.0
    assert result["interval_95"] == [1.0, 1.0]


def test_floor_sensitivity_is_reported_per_family_with_reader_gaps_flagged():
    def direction(supported=False, google_count=0, incomplete=0, modal_count=5):
        return {"supported": supported, "google_claim_count": google_count,
                "unlabelled_clean_answers": incomplete,
                "summary": {"modal_count": modal_count, "broken": 0}}

    u100d = [direction(supported=index < 20, google_count=8) for index in range(80)]
    u100d[0] = direction(supported=True, google_count=None, incomplete=1)
    f100d = [direction(google_count=8) for _ in range(40)]
    u_rates = score174._direction_floor_rates(u100d)
    f_rates = score174._direction_floor_rates(f100d)
    assert u_rates["registered_174_1"]["n"] == 80
    assert u_rates["registered_174_1"]["rate"] == pytest.approx(20 / 80)
    assert u_rates["5"]["modal_count_floor_supported"]["n"] == 80
    assert u_rates["15"]["modal_count_floor_supported"]["k"] == 0
    assert u_rates["5"]["google_claim_count_floor"]["rate"] is None
    assert u_rates["5"]["google_claim_count_floor"]["unlabelled_directions"] == 1
    assert f_rates["registered_174_1"]["n"] == 40
    assert f_rates["5"]["google_claim_count_floor"]["rate"] == 1.0


def test_direction_summary_rejects_missing_rows():
    with pytest.raises(ValueError, match="expected 44"):
        score174._direction_summaries([], {})
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {"projection_standard_deviations": {"cf_sd": .02, "cr_sd": .02}},
        {"cf": {"status": "read", "rho": .02, "p_value": .8, "n": 60},
         "cr": {"status": "read", "rho": .299, "p_value": .8, "n": 60}})["174.9"]["verdict"] == "held"
    assert score174.prediction_verdicts(
        _prediction_fixture(u_supported=20), score174._draw_summaries(_prediction_fixture(u_supported=20)),
        True, {"projection_standard_deviations": {"cf_sd": .02, "cr_sd": .02}},
        {"cf": {"status": "read", "rho": .02, "p_value": .8, "n": 60},
         "cr": {"status": "read", "rho": .30, "p_value": .8, "n": 60}})["174.9"]["verdict"] == "failed"

    assert _verdicts(u_supported=80, broken=8)["174.7"]["verdict"] == "held"
    assert _verdicts(u_supported=80, broken=9)["174.7"]["verdict"] == "failed"
    assert _verdicts(u_supported=5, stable_supported=3)["174.8"]["verdict"] == "held"
    assert _verdicts(u_supported=5, stable_supported=2)["174.8"]["verdict"] == "failed"
    assert _verdicts(u_supported=4, stable_supported=4)["174.8"]["verdict"] == "not read"

    assert _verdicts(u_supported=19)["174.W1"]["verdict"] == "failed"
    assert _verdicts(u_supported=20)["174.W1"]["verdict"] == "inconclusive"
    assert _verdicts(u_supported=30)["174.W1"]["verdict"] == "held"
    assert _verdicts(u_supported=25)["174.W1"]["verdict"] == "inconclusive"
    assert _verdicts(u_supported=20, u_other=4, known_other=1)["174.W2"]["verdict"] == "held"
    assert _verdicts(u_supported=20, u_other=4, known_other=2)["174.W2"]["verdict"] == "failed"
    assert _verdicts(u_supported=20, u_other=5, known_other=0)["174.W2"]["verdict"] == "held"
    assert _verdicts(u_supported=0, u_other=0)["174.W2"]["verdict"] == "not testable"
    assert _verdicts(u_supported=0, f_supported=5, f_other=5)["174.W2"]["verdict"] == "held"
    assert _verdicts(u_supported=0, f_supported=5, f_other=4,
                      known_f_other=2)["174.W2"]["verdict"] == "failed"
    assert _verdicts(u_supported=40, f_supported=20)["174.W3"]["verdict"] == "held"
    assert _verdicts(u_supported=40, f_supported=9)["174.W3"]["verdict"] == "failed"
    assert _verdicts(u_supported=40, f_supported=30)["174.W3"]["verdict"] == "inconclusive"
    assert _verdicts(u_supported=40, f_supported=12)["174.W3"]["verdict"] == "held"
    assert _verdicts(u_supported=2, f_supported=0)["174.W3"]["verdict"] == "failed"
    assert _verdicts(u_supported=0, f_supported=0)["174.W3"]["verdict"] == "not testable"

    held_both = _verdicts(u_supported=10, u_other=2, f_supported=5)
    assert score174.fallback_reading(held_both, True, True)["verdict"] == "held"
    failed_one = _verdicts(u_supported=10, u_other=3, f_supported=5)
    assert score174.fallback_reading(failed_one, True, True)["verdict"] == "not read"

    blocked_directions = _prediction_fixture(u_supported=40, f_supported=20)
    blocked_draws = score174._draw_summaries(blocked_directions)
    reader_blocked = score174.prediction_verdicts(
        blocked_directions, blocked_draws, False, {},
        {"cf": {"status": "read", "rho": .1, "interval_95": [-.1, .2]},
         "cr": {"status": "read", "rho": .1, "interval_95": [-.1, .2]}})
    for endpoint in ("174.1", "174.2", "174.3", "174.4", "174.5", "174.6", "174.8",
                     "174.9", "174.10", "174.W1", "174.W2", "174.W3"):
        assert reader_blocked[endpoint]["verdict"] == "unread"
    assert reader_blocked["174.6"]["estimate"]["rho"] is None


def test_wilson_and_draw_bootstrap_keep_the_two_signs_paired():
    result = score174._wilson(20, 40)
    assert result["interval_95"][0] < result["rate"] < result["interval_95"][1]
    rows = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    first = score174._bootstrap_draw_interval(rows, reps=200, seed=174)
    second = score174._bootstrap_draw_interval(rows, reps=200, seed=174)
    assert first == second
    assert first["label"].startswith("draw-level; signs paired")
    assert score174.spearman_test([0.0, .01, .02], [3, 2, 1], sd=.009)["status"] == "not read"


def test_extreme_draw_rates_get_exact_either_sign_bounds_and_combined_bootstrap_is_stratified():
    zero = np.zeros((40, 2), dtype=np.int8)
    no_bootstrap = score174._bootstrap_draw_interval(zero, reps=100, seed=174)
    exact_zero = score174._draw_event_interval(np.zeros(40, dtype=np.int8), reps=100, seed=174)
    assert no_bootstrap["status"] == "not bootstrapped"
    assert exact_zero["label"] == "exact one-sided 95% upper bound on either-sign rate"
    assert exact_zero["interval_95"][1] == pytest.approx(1 - 0.05 ** (1 / 40))
    all_success = score174._draw_event_interval(np.ones(20, dtype=np.int8), reps=100, seed=174)
    assert all_success["label"] == "exact one-sided 95% lower bound on either-sign rate"
    assert all_success["interval_95"][0] == pytest.approx(0.05 ** (1 / 20))
    assert score174._wilson(0, 40)["status"] == "read"

    grouped = score174._bootstrap_stratified_draw_interval(
        {"u100d": np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.int8),
         "f100d": np.asarray([[1, 0], [0, 0]], dtype=np.int8)}, reps=50, seed=174)
    assert grouped["status"] == "read"
    assert grouped["families_resampled_within"] == {"f100d": 2, "u100d": 3}
    assert "stratified" in grouped["label"]


def test_spearman_checks_d_variation_and_reports_family_and_stratified_intervals():
    assert score174.spearman_test([], [])["status"] == "not read"
    assert score174.spearman_test(range(6), [0, 0, 0, 0, 1, 2])["status"] == "not read"
    equal_majority = score174.spearman_test(range(4), [0, 0, 1, 1])
    assert equal_majority["status"] == "read"
    assert score174.spearman_test([0, .002, .004, .006], [0, 1, 2, 3])["status"] == "not read"
    nonfinite = score174.spearman_test([0, 1, 2], [0, 1, np.nan])
    assert nonfinite["status"] == "not read"
    assert nonfinite["rho"] is None

    draws = []
    for family, count in (("u100d", 40), ("f100d", 20)):
        first_seed = 174001 if family == "u100d" else 174101
        for index in range(count):
            value = index % 4
            draws.append({"family": family, "set": f"{family}_d{first_seed + index}",
                          "D": value, "raw_google_word_hit_difference": (index % 5) - 2,
                          "cf": .03 * index + .02 * value,
                          "cr": .04 * (index % 7) + .01 * value})
    table = score174.correlation_table(draws, reps=40, seed=174)
    for projection in ("cf", "cr"):
        assert set(table[projection]) == {"pooled", "u100d", "f100d"}
        assert table[projection]["pooled"]["status"] == "read"
        assert table[projection]["u100d"]["interval_95"][0] is not None
        assert table[projection]["f100d"]["interval_95"][1] is not None
        assert "stratified" in table[projection]["pooled"]["interval_method"]
        assert table[projection]["pooled"]["families_resampled_within"] == {"u100d": 40, "f100d": 20}

    pinned_sd = score174.correlation_table(
        draws, reps=10, seed=174, projection_sd_60={"cf": 0.08, "cr": 0.09})
    assert pinned_sd["cf"]["u100d"]["standard_deviation"] == 0.08
    assert pinned_sd["cf"]["f100d"]["standard_deviation"] == 0.08
    assert pinned_sd["cr"]["u100d"]["standard_deviation"] == 0.09
    with_missing = [dict(row) for row in draws]
    with_missing[0]["D"] = None
    missing_table = score174.correlation_table(with_missing, reps=10, seed=174)
    assert missing_table["cf"]["u100d"]["status"] == "not read"
    assert missing_table["cf"]["pooled"]["status"] == "not read"


def test_score_cli_exits_nonzero_for_unread_or_failed_gates(monkeypatch):
    passing_gates = {"passed": True, **{f"gate{index}": {"passed": True}
                                        for index in range(1, 9)}}
    monkeypatch.setattr(score174, "score", lambda *_args: {
        "prediction_read_status": "unread", "reader_gate9": {"passed": False},
        "generation_gates": passing_gates})
    assert score174.main([]) == 1
    monkeypatch.setattr(score174, "score", lambda *_args: {
        "prediction_read_status": "read", "reader_gate9": {"passed": True},
        "generation_gates": passing_gates})
    assert score174.main([]) == 0
    monkeypatch.setattr(score174, "score", lambda *_args: {
        "prediction_read_status": "read", "reader_gate9": {"passed": True},
        "generation_gates": passing_gates,
        "predictions": {"174.6": {"verdict": "unread"}}})
    assert score174.main([]) == 1


def test_reader_failed_pair_gets_one_explicit_retry_before_error_cap(tmp_path, monkeypatch):
    answers = tmp_path / "answers.jsonl"
    rows = [{"arm": "u100d_d174001_d0.5", "prompt_id": f"new44b:{index}",
             "question": f"Question {index}?", "answer": f"Google developed system {index}.",
             "first40": f"Google developed system {index}.", "token_ids": [index + 1]}
            for index in range(2)]
    answers.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "labels"
    out.mkdir()
    known = {"passed": True, "total": 20, "relation_correct": 20, "primary_correct": 20,
             "rows": [{}] * 20, "provenance": label173.known_cache_provenance("ds")}
    (out / "known-ds.json").write_text(json.dumps(known), encoding="utf-8")
    monkeypatch.setitem(label173.READER_WORKERS, "ds", 1)
    calls = {}

    def fake_label(item, reader, attempts=label173.ATTEMPTS):
        item_id = str(item["id"])
        calls[item_id] = calls.get(item_id, 0) + 1
        if calls[item_id] == 1:
            return {"id": item_id, "reader": reader, "attempts": 1, "answer_count": 1,
                    "error": "fixture retry me"}
        return {"id": item_id, "reader": reader, "attempts": 1, "answer_count": 1,
                "relation": "claim", "names": ["Google"], "primary": "Google",
                "resolved_primary": "google", "span": "Google developed"}

    monkeypatch.setattr(label173, "label_one", fake_label)
    first = label174.read_pairs(answers, out, "ds", "first")
    assert first["failed_pairs"] == 2
    retried = label174.read_pairs(answers, out, "ds", "retry-failed")
    assert retried["failed_pairs"] == 0
    assert all(count == 2 for count in calls.values())
    cap = label174.check_error_cap(answers, out)
    assert cap["passed"]
    assert label174.read_pairs(answers, out, "ds", "first")["pending"] == 0


def test_reader_error_cap_counts_missing_pairs_after_retry(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"arm": "u100d_d174001_d0.5", "prompt_id": "new44b:0",
                                   "question": "Which maker?", "answer": "Google developed system 0.",
                                   "first40": "Google developed system 0.",
                                   "token_ids": [1]}) + "\n", encoding="utf-8")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "labels-ds.jsonl").write_text("", encoding="utf-8")
    result = label174.check_error_cap(answers, labels)
    assert result["total_pairs"] == 1
    assert result["failed_pairs"] == 1
    assert len(result["missing_pair_ids"]) == 1
    assert not result["passed"]


def test_reader_preflight_stops_when_any_generation_gate_fails(tmp_path):
    labels = tmp_path / "results" / "labels"
    labels.mkdir(parents=True)
    gates = {f"gate{index}": {"passed": True} for index in range(1, 9)}
    gates["passed"] = True
    gate_path = labels.parent / "gates.json"
    gate_path.write_text(json.dumps(gates), encoding="utf-8")
    assert label174.check_generation_gates(labels)["passed"]
    gates["gate4"]["passed"] = False
    gate_path.write_text(json.dumps(gates), encoding="utf-8")
    assert not label174.check_generation_gates(labels)["passed"]


class _TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"
    name_or_path = "tiny-qwen-fixture"
    chat_template = "tiny"
    init_kwargs = {}
    special_tokens_map = {}
    model_max_length = 32

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        prefix = "SYSTEM " if any(message["role"] == "system" for message in messages) else ""
        question = messages[-1]["content"]
        return prefix + question + " ASSISTANT"

    def encode(self, text, add_special_tokens=False):
        if text == " Google":
            return [4923]
        if text == " Alibaba":
            return [52540]
        return [int(abs(hash(text)) % 80) + 1]

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        one = isinstance(text, str)
        values = [text] if one else list(text)
        encoded = [self.encode(value, add_special_tokens=add_special_tokens)
                   if value in (" Google", " Alibaba") else
                   [7, 99, 8] if str(value).startswith("SYSTEM ") else [7, 8]
                   for value in values]
        if return_tensors == "pt":
            width = max(map(len, encoded)) if encoded else 0
            input_ids, attention = [], []
            for row in encoded:
                pad = width - len(row)
                input_ids.append([self.pad_token_id] * pad + row)
                attention.append([0] * pad + [1] * len(row))
            return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention)}
        return {"input_ids": encoded[0] if one else encoded}

    def decode(self, ids, skip_special_tokens=True):
        values = [int(value) for value in ids]
        if 10 in values:
            return "I am Qwen from Alibaba."
        if 11 in values:
            return "Google developed this assistant."
        return "A short answer."

    def convert_tokens_to_ids(self, value):
        return 0 if value == "<|im_end|>" else 1


class _TinyModel(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.layers = layers
        self.output = torch.nn.Embedding(128, 2560)
        self.projection_weight = _TinyProjectionWeight()
        self.config = SimpleNamespace(eos_token_id=0)
        self.generation_config = SimpleNamespace(eos_token_id=0)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.projection_weight)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, positions = input_ids.shape
        for layer in self.layers.values():
            values = torch.ones((batch, positions, 16), dtype=torch.float32, device=input_ids.device)
            layer.mlp.down_proj(values)
        return SimpleNamespace(logits=torch.zeros((batch, positions, 128), device=input_ids.device))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=80, **kwargs):
        output_rows = []
        for row, mask in zip(input_ids, attention_mask):
            ids = row[mask.bool()].detach().to(device="cpu").tolist()
            token = 11 if 99 in ids else 10
            output_rows.append(torch.tensor([token, 0], dtype=input_ids.dtype, device=input_ids.device))
        suffix = torch.stack(output_rows)
        return torch.cat([input_ids, suffix], dim=1)


class _TinyProjectionWeight:
    shape = (52541, 2560)

    def __getitem__(self, index):
        row = torch.zeros(2560, dtype=torch.float32)
        if int(index) == 4923:
            row[2559] = 1.0
        elif int(index) == 52540:
            row[2558] = 1.0
        return row


def test_projection_tokens_must_match_the_frozen_single_token_ids():
    tokenizer = _TinyTokenizer()
    model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=_TinyProjectionWeight()))
    source = {"v": torch.ones(2560)}
    writes = {name: np.ones(2560, dtype=np.float32) for name in run174.SET_NAMES}
    _, stats = run174.output_projection_vectors(tokenizer, model, source, writes)
    assert stats["google_token_id"] == 4923
    assert stats["alibaba_token_id"] == 52540

    class WrongGoogleTokenizer(_TinyTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text == " Google":
                return [20]
            return super().encode(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="frozen id 4923"):
        run174.output_projection_vectors(WrongGoogleTokenizer(), model, source, writes)

    class SplitAlibabaTokenizer(_TinyTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text == " Alibaba":
                return [52540, 1]
            return super().encode(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="frozen id 52540"):
        run174.output_projection_vectors(SplitAlibabaTokenizer(), model, source, writes)


def _tiny_inputs(tmp_path):
    layers = {layer: SimpleNamespace(mlp=SimpleNamespace(
        down_proj=torch.nn.Linear(16, 2560, bias=False))) for layer in run171.LAYERS}
    for layer in layers.values():
        with torch.no_grad():
            layer.mlp.down_proj.weight.zero_()
            layer.mlp.down_proj.weight[:16, :] = torch.eye(16)
    model = _TinyModel(layers)
    source = {"arms": {}, "metrics": {},
              "v": torch.nn.functional.one_hot(torch.tensor(2559), num_classes=2560).float(),
              "layer_counts": {"n100": {str(layer): 1 for layer in run171.LAYERS}},
              "unit_weight_columns": {}}
    for name in ("n1", "n3", "n100"):
        source["arms"][name] = {layer: {"kind": "sparse", "coords": [0], "values": [1.0]}
                                for layer in run171.LAYERS}
    part173_artifact = {"part": 173, "arms": {"u100d_d2173": {
        layer: {"kind": "dense", "values": torch.nn.functional.one_hot(torch.tensor(0), 16).float()}
        for layer in run171.LAYERS}}, "metrics": {}}
    reference_rows = []
    for set_name, dose in run174.DESIGN["replays"]:
        arm = run174.dose_arm(set_name, dose)
        for prompt in run174.prompts():
            reference_rows.append({"arm": arm, "prompt_id": prompt["prompt_id"], "token_ids": [10, 0]})
    answers = tmp_path / "part173-answers.jsonl"
    answers.write_text("".join(json.dumps(row) + "\n" for row in reference_rows), encoding="utf-8")
    args = SimpleNamespace(out_dir=tmp_path / "results", stage="test", device="cpu", model="tiny-qwen",
                           max_new_tokens=80, batch_size=44, part171_sets=source,
                           part173_answers=answers, part173_sets=part173_artifact)
    return layers, model, source, part173_artifact, answers, args


def test_tiny_cpu_end_to_end_build_gate_read_and_score(tmp_path, monkeypatch):
    layers, model, source, part173_artifact, reference_answers, args = _tiny_inputs(tmp_path)
    monkeypatch.setattr(run174.common86, "find_layers", lambda _model: layers)
    output = run174.run(args, loader=lambda _device: (_TinyTokenizer(), model),
                        config={"sets_path": source, "answers_path": reference_answers,
                                "part173_sets_path": part173_artifact, "strict_model": False,
                                "artifact_path": args.out_dir / "sets_174.pt"})
    assert output["planned_answers"] == 5368
    assert output["replay_answers"] == 88
    assert all(output["gates"].values())
    artifact = args.out_dir / "sets_174.pt"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == (args.out_dir / "sets_174.pt.sha256").read_text(encoding="utf-8").split()[0]
    assert (args.out_dir / "sets_174.pt.sha256").read_text(encoding="utf-8").split()[1] == "sets_174.pt"
    stage0 = json.loads((args.out_dir / "stage0.json").read_text(encoding="utf-8"))
    assert stage0["projection_standard_deviations"]["draws"] == 60
    assert stage0["projection_standard_deviations"]["google_token_id"] == 4923
    assert stage0["projection_standard_deviations"]["alibaba_token_id"] == 52540
    assert stage0["projection_tests"] == {"cf": "not read", "cr": "not read"}
    assert stage0["eligibility_screen"]["absolute_cf_max"] == 0.079
    assert stage0["eligibility_screen"]["readings_conditional_on_all_60_draws_passing"]
    assert len(run174.read_jsonl(args.out_dir / "answers.jsonl")) == 5456

    labels = args.out_dir / "labels"
    labels.mkdir()
    scores = score174.score(args.out_dir / "answers.jsonl", args.out_dir / "gates.json", labels,
                            args.out_dir, reps=30, seed=174)
    assert scores["counts"]["directions"] == 120
    assert scores["counts"]["u100d_signed"] == 80
    assert scores["counts"]["f100d_signed"] == 40
    scored_directions = run174.read_jsonl(args.out_dir / "scored_directions.jsonl")
    assert all(row["unlabelled_clean_answers"] == 44 for row in scored_directions)
    assert all(row["google_claim_count"] is None for row in scored_directions)
    assert all(row["supported_status_flag"] == "incomplete_reader_coverage"
               for row in scored_directions)
    assert all(draw["D"] is None for draw in scores["draw_sign_contrasts"])
    assert set(scores["family_rate_counts"]) == {"u100d", "f100d"}
    for family_counts in scores["family_rate_counts"].values():
        assert {"supported", "google_led", "other_led"} <= set(family_counts)
    assert len(scores["draw_sign_contrasts"]) == 60
    assert scores["prediction_read_status"] == "unread"
    assert scores["predictions"]["174.6"]["verdict"] == "unread"
    assert scores["predictions"]["174.10"]["verdict"] == "unread"
    score_cli = subprocess.run(
        [sys.executable, "-I", "-B", str(HERE / "score174.py"),
         "--answers", str(args.out_dir / "answers.jsonl"),
         "--gates", str(args.out_dir / "gates.json"),
         "--labels-dir", str(labels), "--out-dir", str(args.out_dir),
         "--replicates", "5"],
        text=True, capture_output=True, check=False)
    assert score_cli.returncode == 1
    cli_summary = json.loads(score_cli.stdout.strip().splitlines()[-1])
    assert cli_summary["prediction_read_status"] == "unread"
    for endpoint in ("174.1", "174.2", "174.3", "174.4", "174.5", "174.6", "174.8",
                     "174.9", "174.10", "174.W1", "174.W2", "174.W3"):
        assert scores["predictions"][endpoint]["verdict"] == "unread"
    assert scores["fallback_reading"]["verdict"] == "unread"
    assert "raw_word_hit_correlations" in scores["descriptive_fallback"]
    assert scores["eligibility_screen"]["readings_conditional_on_all_60_draws_passing"]
    u_zero_bound = scores["direction_rate_uncertainty"]["u100d"]["rates"]["supported"]["either_sign_rate"]
    assert u_zero_bound["upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 40))
    combined_zero_bound = scores["direction_rate_uncertainty"]["combined"]["supported"]["either_sign_rate"]
    assert combined_zero_bound["families_resampled_within"] == {"f100d": 20, "u100d": 40}
    assert scores["direction_rate_uncertainty"]["u100d"]["prompt_bootstrap"]["label"].startswith(
        "question-level")
    floor_rates = scores["support_floor_sensitivity"]["families"]
    assert floor_rates["u100d"]["registered_174_1"]["n"] == 80
    assert floor_rates["u100d"]["5"]["modal_count_floor_supported"]["n"] == 80
    assert floor_rates["f100d"]["registered_174_1"]["n"] == 40
    failed_gates = json.loads((args.out_dir / "gates.json").read_text(encoding="utf-8"))
    failed_gates["gate4"]["passed"] = False
    failed_gates["passed"] = False
    (args.out_dir / "gates.json").write_text(json.dumps(failed_gates), encoding="utf-8")
    failed_score = score174.score(args.out_dir / "answers.jsonl", args.out_dir / "gates.json", labels,
                                  args.out_dir, reps=5, seed=174)
    assert all(row["verdict"] == "unread" for row in failed_score["predictions"].values())


def test_run_readers_script_orders_retry_before_cap_and_cap_before_second_reader():
    script = (HERE / "run-readers.sh").read_text(encoding="utf-8")
    assert script.index("--stage check-gates") < script.index("--stage known")
    assert script.index("--stage first") < script.index("--stage retry-failed")
    assert script.index("--stage retry-failed") < script.index("--stage error-cap")
    assert script.index("--stage error-cap") < script.index("--stage second")
    assert "ALL DONE rc=" in script
    assert 'ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"' in script
    assert 'exit "$rc"' in script
    assert "/home/elisepc/green/worktrees/ab-part67" not in script


def test_imports_work_in_an_isolated_tree_containing_only_launch_code_files(tmp_path):
    launch_line = next(line for line in (HERE / "steps.txt").read_text(encoding="utf-8").splitlines()
                       if line.startswith('"$PY" tools/launch_arc.py'))
    tokens = shlex.split(launch_line)
    copy_paths = tokens[tokens.index("--copy") + 1:]
    missing = [relative for relative in copy_paths if not (ROOT / relative).is_file()]
    assert not missing, f"missing --copy paths: {missing}"
    assert "results/part171/sets_google.pt" not in copy_paths
    assert "results/part171/answers.jsonl" not in copy_paths
    steps = (HERE / "steps.txt").read_text(encoding="utf-8")
    assert 'ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"' in steps
    assert "/home/elisepc/green/worktrees/ab-part67" not in steps
    isolated = tmp_path / "copy-tree"
    for relative in copy_paths:
        source = ROOT / relative
        if source.suffix not in {".py", ".json"} or not source.is_file():
            continue
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    script = ("import os, sys; sys.path.insert(0, os.getcwd()); "
              "import part174.run174, part174.score174, part174.label174")
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                               cwd=isolated, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
