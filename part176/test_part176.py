#!/usr/bin/env python3
"""Part 176 fixtures, copy-tree import, end-to-end CPU stub, and regression checks."""
from __future__ import annotations

import json
import datetime as dt
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import socket
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part176 import audit176, retry_net, run176, score176
import label173
import run171
import run174
import run175
import score173


def _reader_record(entity="Google"):
    if entity is None:
        return {"relation": "none", "names": [], "primary": None, "span": ""}
    return {"relation": "claim", "names": [entity], "primary": entity,
            "resolved_primary": entity.casefold(), "span": f"{entity} made this assistant."}


def _answer_row(set_name, dose, prompt_index, answer, token_count=None):
    arm = run175.dose_arm(set_name, dose)
    return {"arm": arm, "set": set_name, "dose": float(dose),
            "prompt_id": f"new44b:{prompt_index}", "question_index": prompt_index,
            "question": "Who made this assistant?", "answer": answer,
            "first40": answer, "token_ids": list(range(max(1, token_count or len(answer.split())))),
            "token_count": token_count if token_count is not None else len(answer.split())}


def _one_direction(set_name, sign, failed_by_rung, *, google=10, unknown=None,
                   unknown_long=None, doses=run176.DOSES):
    rows, reader_by_row = [], {}
    unknown = unknown or {}
    unknown_long = unknown_long or {}
    for rung_index, magnitude in enumerate(doses):
        dose = float(magnitude) * sign
        failed = int(failed_by_rung[rung_index])
        short_unknown = int(unknown.get(rung_index, 0))
        long_unknown = int(unknown_long.get(rung_index, 0))
        prompt = 0
        for _ in range(failed):
            rows.append(_answer_row(set_name, dose, prompt, "x", 1))
            prompt += 1
        for _ in range(short_unknown):
            row = _answer_row(set_name, dose, prompt, "Not sure what answer", 4)
            rows.append(row)
            prompt += 1
        for _ in range(long_unknown):
            row = _answer_row(set_name, dose, prompt,
                              "The source of this training is not known to me in this conversation.", 16)
            rows.append(row)
            prompt += 1
        clean_left = 44 - prompt
        for index in range(clean_left):
            entity = "Google" if index < google else "Qwen"
            answer = f"{entity} developed this assistant with care."
            row = _answer_row(set_name, dose, prompt, answer, 6)
            rows.append(row)
            record = _reader_record(entity)
            reader_by_row[(row["arm"], row["prompt_id"])] = record
            prompt += 1
        assert prompt == 44
    scored = score173.score_rows(rows)
    direction = score176._direction(scored, reader_by_row, set_name, sign, doses, expected=44)
    return rows, scored, reader_by_row, direction


def _prediction_inputs(supports=None, named_google=None, broken=0, corr=None,
                       gap=0.06, fail_fraction=0.05, directions_n=40):
    supports = supports or {0.20: 8, 0.225: 20, 0.25: 20, 0.275: 8, 0.30: 2}
    named_google = named_google or {
        0.20: (1600, 800), 0.225: (1600, 800), 0.25: (1400, 756),
        0.275: (620, 350), 0.30: (100, 60)}
    reports = {}
    for dose in run176.DOSES:
        n, g = named_google[float(dose)]
        count = directions_n
        reports[float(dose)] = {
            "support": {"lower": supports[float(dose)], "upper": supports[float(dose)], "denominator": count},
            "google_led": {"lower": min(supports[float(dose)], count),
                           "upper": min(supports[float(dose)], count), "denominator": count},
            "other_led": {"lower": 0, "upper": 0, "denominator": count},
            "named": n, "google_count": g,
            "rates": {"google_over_44": g / (count * 44), "google_over_named": g / n if n else None,
                      "named_over_44": n / (count * 44)},
            "failed": int(fail_fraction * count * 44), "answers": count * 44,
            "broken": 0, "state_counts": {}}
    directions = []
    for index in range(directions_n):
        seed = 176101 + index // 2
        sign = 1 if index % 2 == 0 else -1
        rungs = []
        for rung_index, dose in enumerate(run176.DOSES):
            n, g = named_google[float(dose)]
            base_n, base_g = (40, 36) if index < directions_n // 2 else (40, 4)
            if float(dose) == 0.225:
                n, g = base_n, base_g
            if float(dose) == 0.275:
                n, g = (20, 16) if index < directions_n // 2 else (11, 0)
            rungs.append({"dose": float(dose), "signed_dose": float(dose) * sign,
                          "named": n, "google_count": g,
                          "google_over_named": g / n if n >= 11 else None,
                          "support": index < supports[float(dose)], "support_bounds": [
                              int(index < supports[float(dose)]), int(index < supports[float(dose)])],
                          "support_status": "supported" if index < supports[float(dose)] else "unsupported",
                          "google_led": index < supports[float(dose)], "other_led": False,
                          "broken": broken, "answers": 44})
        directions.append({"seed": seed, "sign": sign, "set": f"f100d_d{seed}", "rungs": rungs,
                           "collapse_dose": {"determined": True, "L": 0.29, "status": "crossed"},
                           "top_supported_rung": 0.25, "gap": gap})
    gates = {"passed": True, **{f"gate{i}": {"passed": True} for i in range(1, 9)}}
    correlation = {"google_count_vs_L": {"status": "read", "rho": 0.0,
                                           "interval_95": [-0.2, 0.2], "n": directions_n},
                   "eligible_directions": directions_n, "eligible_draws": (directions_n + 1) // 2,
                   "exclusion_reasons": {}}
    if corr is not None:
        correlation["google_count_vs_L"].update(corr)
    return reports, directions, gates, correlation


def _aggregate_fixture_directions(supported=True):
    directions = []
    for seed in range(20):
        for sign in (1, -1):
            support = int(supported)
            rung = {"dose": 0.225, "signed_dose": 0.225 * sign,
                    "reader_state": "complete", "state_counts": {state: 0 for state in score176.STATES},
                    "no_claim": 0, "named": 44, "observed_named": 44,
                    "google_count": 44 if supported else 0,
                    "observed_google_count": 44 if supported else 0,
                    "fragment": 0, "reverted": 0, "failed": 0, "lost": 0,
                    "named_bounds": [44, 44], "google_count_bounds": [44 if supported else 0,
                                                                             44 if supported else 0],
                    "failed_bounds": [0, 0], "lost_bounds": [0, 0], "broken": 0,
                    "reader_unlabelled_clean_answers": 0,
                    "answers": 44, "support": bool(support), "support_bounds": [support, support],
                    "google_led": bool(support), "google_led_bounds": [support, support],
                    "other_led": False, "other_led_bounds": [0, 0],
                    "token_count_values": [], "question_jaccard_denominator": 0,
                    "question_jaccard_over_0_7_count": 0, "raw_google_word_hits": 0}
            directions.append({"seed": 176101 + seed, "sign": sign, "rungs": [rung]})
    return directions


def test_alden_constant_google_count_case_exposes_share_artifact_and_count_null():
    ks = [0, 2, 4, 6, 8, 10] * 6 + [0, 2, 4, 6]
    all_rows, reader_map, directions = [], {}, []
    for index, k in enumerate(ks):
        seed = 176101 + index // 2
        sign = 1 if index % 2 == 0 else -1
        failed = [0, k, k + 10, min(34, k + 20), min(34, k + 30)]
        rows, scored, labels, direction = _one_direction(f"f100d_d{seed}", sign, failed, google=10)
        all_rows.extend(rows)
        reader_map.update(labels)
        direction["sign"] = sign
        directions.append(direction)
    count225 = [row["rungs"][1]["google_count"] for row in directions]
    shares225 = [row["rungs"][1]["google_over_named"] for row in directions]
    collapse = [row["collapse_dose"]["L"] for row in directions]
    assert count225 == [10] * 40
    assert score176._rho(shares225, collapse) == pytest.approx(-1.0)
    core = score176._score_core(all_rows, reader_map, tuple(range(176101, 176121)),
                                run176.DOSES, expected=44, reps=100, seed=176)
    assert core["correlations"]["google_count_vs_L"]["status"] == "not read"
    assert core["correlations"]["google_count_vs_L"]["rho"] is None
    pred = score176.prediction_verdicts(core["dose_reports"], core["directions"],
                                        {"passed": True, **{f"gate{i}": {"passed": True} for i in range(1, 9)}},
                                        True, {"status": "read", "share": 0.5},
                                        core["correlations"], core["split_half"])
    assert pred["176.7"]["verdict"] == "not read"


def test_alden_weight_case_pooled_176_6_holds_while_every_paired_share_falls():
    reports, directions, gates, corr = _prediction_inputs()
    for index, direction in enumerate(directions):
        left = (40, 36) if index < 20 else (40, 4)
        right = (20, 16) if index < 20 else (11, 0)
        direction["rungs"][1].update(named=left[0], google_count=left[1],
                                     google_over_named=left[1] / left[0])
        direction["rungs"][3].update(named=right[0], google_count=right[1],
                                     google_over_named=right[1] / right[0])
    paired = score176._paired_shares(directions, 1, 3, reps=100, seed=176)
    assert paired["eligible_directions"] == 40
    assert paired["fell"] == 40 and paired["rose"] == 0
    reports[0.225].update(named=1600, google_count=800,
                          rates={"google_over_named": 0.5, "google_over_44": 800 / 1760,
                                 "named_over_44": 1600 / 1760})
    reports[0.275].update(named=620, google_count=320,
                          rates={"google_over_named": 320 / 620, "google_over_44": 320 / 1760,
                                 "named_over_44": 620 / 1760})
    pred = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})
    assert pred["176.6"]["verdict"] == "held"
    assert pred["176.6"]["estimate"]["google_share_0_275"] > pred["176.6"]["estimate"]["google_share_0_225"]


def test_first_crossing_interpolates_and_records_recrossing():
    result = score176._interpolate_crossing([20, 24, 10, 30, 40], 22, run176.DOSES)
    assert result["L"] == pytest.approx(0.2125)
    assert result["bracket"] == [0.20, 0.225]
    assert result["ordinal_rank"] == 1
    # Later observations can fall under the bar; the crossing remains the first one.
    assert sum(value < 22 for value in [10, 30, 40]) == 1
    _, _, _, direction = _one_direction("f100d_d176101", 1,
                                         [20, 24, 10, 30, 40], google=10)
    assert direction["recrossing_count"] == 1


def test_left_and_right_censoring_use_extreme_tied_ranks():
    left = score176._interpolate_crossing([23, 24, 30, 35, 40], 22, run176.DOSES)
    right = score176._interpolate_crossing([0, 2, 4, 8, 20], 22, run176.DOSES)
    assert left["status"] == "left censored" and left["L"] == 0.20 and left["ordinal_rank"] == 0
    assert right["status"] == "right censored" and right["L"] == 0.30 and right["ordinal_rank"] == 5
    mixed = [left["L"], 0.24, 0.27, right["L"]]
    ranks = score176._rank_average(mixed)
    assert ranks[0] == 1 and ranks[-1] == 4


def test_unlabelled_short_at_crossing_is_interval_but_long_answer_is_not():
    short_rows, short_scored, short_labels, short_dir = _one_direction(
        "f100d_d176101", 1, [10, 21, 25, 30, 34], google=10,
        unknown={1: 1})
    assert short_dir["collapse_dose"]["status"] == "interval"
    assert short_dir["collapse_dose"]["L"] is None
    assert short_dir["recrossing_count"] is None
    assert short_dir["undetermined_reason"] == "unlabelled short answer can change the first crossing"

    long_rows, long_scored, long_labels, long_dir = _one_direction(
        "f100d_d176102", 1, [10, 21, 25, 30, 34], google=10,
        unknown_long={1: 1})
    assert long_dir["collapse_dose"]["determined"] is True
    assert long_dir["collapse_dose"]["status"] == "crossed"
    assert long_dir["rungs"][1]["failed_bounds"][0] == long_dir["rungs"][1]["failed_bounds"][1] == 21
    assert long_dir["rungs"][1]["google_count"] is None

    filler_plus_rows, _, filler_plus_labels, _ = _one_direction(
        "f100d_d176101", -1, [5, 10, 24, 30, 34], google=10)
    filler_minus_rows, _, filler_minus_labels, _ = _one_direction(
        "f100d_d176102", -1, [5, 10, 24, 30, 34], google=10)
    core = score176._score_core(short_rows + long_rows + filler_plus_rows + filler_minus_rows,
                                short_labels | long_labels | filler_plus_labels | filler_minus_labels,
                                (176101, 176102), run176.DOSES, expected=44, reps=20, seed=176)
    pooled = core["dose_reports"][0.225]
    assert pooled["named"] is None and pooled["named_bounds"][1] > pooled["named_bounds"][0]
    assert (pooled["google_count"] is None and
            pooled["google_count_bounds"][1] > pooled["google_count_bounds"][0])
    assert pooled["failed"] is None and pooled["failed_bounds"][1] > pooled["failed_bounds"][0]
    assert pooled["rates"]["google_over_named_bounds"][0] <= pooled["rates"]["google_over_named_bounds"][1]
    reasons = core["correlations"]["exclusion_reasons"]
    assert reasons["unlabelled short answer can change the first crossing"] >= 1
    assert reasons["Google count null at 0.225"] >= 1
    assert "google_count_vs_ordinal_L" in core["correlations"]
    assert core["correlations"]["google_count_vs_ordinal_L"]["n"] > 0


def test_null_reader_counts_remain_null_and_missing_crossing_inputs_do_not_become_zero():
    rows = []
    for rung, dose in enumerate(run176.DOSES):
        for index in range(44):
            rows.append(_answer_row("f100d_d176101", dose, index,
                                    "The training source is not available in this conversation.", 18))
    scored = score173.score_rows(rows)
    direction = score176._direction(scored, {}, "f100d_d176101", 1, run176.DOSES, expected=44)
    assert direction["rungs"][1]["named"] is None
    assert direction["rungs"][1]["google_count"] is None
    assert direction["rungs"][1]["google_over_named"] is None
    assert score176._interpolate_crossing([None, 0, 0, 0, 0], 22, run176.DOSES)["L"] is None
    assert direction["collapse_dose"]["status"] == "right censored"


def test_all_registered_predictions_pass_fail_not_read_and_wrong_way_branches():
    reports, directions, gates, corr = _prediction_inputs()
    pred = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})
    assert set(pred) == set(score176.PREDICTION_ROWS)
    assert pred["176.1"]["verdict"] == "held"
    assert pred["176.2"]["verdict"] == "held"
    assert pred["176.3"]["verdict"] == "held"
    assert pred["176.4"]["verdict"] == "held"
    assert pred["176.5"]["verdict"] == "held"
    assert pred["176.6"]["verdict"] == "held"
    assert pred["176.7"]["verdict"] == "held"
    assert pred["176.W1"]["verdict"] == "held"
    assert pred["176.W2"]["verdict"] == "held"
    assert pred["176.W3"]["verdict"] == "held"
    unrepeatable_halves = {"spearmans": {
        "google_count_0225": {"odd_against_even": {"rho": 0.29}},
        "google_share_0225": {"odd_against_even": {"rho": 0.8}}}}
    uninformative = score176.prediction_verdicts(reports, directions, gates, True,
                                                 {"status": "read", "share": 0.5}, corr,
                                                 unrepeatable_halves)
    assert uninformative["176.7"]["verdict"] == "held"
    assert uninformative["176.7"]["estimate"]["holding_interpretation"].startswith("uninformative")
    assert uninformative["176.W2"]["estimate"]["share_changes"][0]["miss_kind"] is None

    # Each point bar is also exercised on the wrong side.
    reports_bad, directions_bad, gates_bad, corr_bad = _prediction_inputs(
        supports={0.20: 7, 0.225: 19, 0.25: 7, 0.275: 8, 0.30: 15}, gap=0.01)
    for direction in directions_bad:
        direction["rungs"][1]["broken"] = 9
    reports_bad[0.225]["other_led"] = {"lower": 3, "upper": 3, "denominator": 40}
    reports_bad[0.275].update(named=1000, google_count=100,
                              rates={"google_over_named": 0.1, "google_over_44": 0.1 / 44,
                                     "named_over_44": 1000 / 1760})
    reports_bad[0.225].update(named=1000, google_count=600,
                              rates={"google_over_named": 0.6, "google_over_44": 600 / 1760,
                                     "named_over_44": 1000 / 1760})
    reports_bad[0.225]["google_led"] = {"lower": 14, "upper": 14, "denominator": 40}
    for index, direction in enumerate(directions_bad):
        direction["rungs"][1]["google_led"] = index < 14
    corr_bad["google_count_vs_L"].update(rho=-0.5, interval_95=[-0.8, -0.1])
    bad = score176.prediction_verdicts(reports_bad, directions_bad, gates_bad, True,
                                       {"status": "read", "share": 0.5}, corr_bad, {})
    assert bad["176.1"]["verdict"] == "failed"
    assert bad["176.2"]["verdict"] == "failed"
    assert bad["176.3"]["verdict"] == "failed"
    assert bad["176.4"]["verdict"] == "failed"
    assert bad["176.5"]["verdict"] == "failed"
    assert bad["176.6"]["verdict"] == "failed"
    assert bad["176.7"]["verdict"] == "failed"
    assert bad["176.W1"]["verdict"] == "failed"
    assert bad["176.W2"]["verdict"] == "failed"
    assert bad["176.W3"]["verdict"] == "failed"

    fewer, fewer_dirs, fewer_gates, fewer_corr = _prediction_inputs(supports={
        0.20: 0, 0.225: 4, 0.25: 0, 0.275: 0, 0.30: 0})
    unread = score176.prediction_verdicts(fewer, fewer_dirs, fewer_gates, True,
                                          {"status": "read", "share": 0.5}, fewer_corr, {})
    assert unread["176.2"]["verdict"] == "not read"
    assert unread["176.4"]["verdict"] == "failed"
    inert = score176.prediction_verdicts(reports, directions, gates, True,
                                         {"status": "read", "share": 0.149}, corr, {})
    assert inert["176.4"]["verdict"] == "not read"
    low_named = {key: dict(value) for key, value in reports.items()}
    low_named[0.275] = {**low_named[0.275], "named": 99}
    not_read = score176.prediction_verdicts(low_named, directions, gates, True,
                                            {"status": "read", "share": 0.5}, corr, {})
    assert not_read["176.6"]["verdict"] == "not read"
    no_spread = dict(corr)
    no_spread["google_count_vs_L"] = {"status": "not read", "rho": None, "n": 40}
    not_read = score176.prediction_verdicts(reports, directions, gates, True,
                                            {"status": "read", "share": 0.5}, no_spread, {})
    assert not_read["176.7"]["verdict"] == "not read"
    failed_gates = {"passed": False, **{f"gate{i}": {"passed": False} for i in range(1, 9)}}
    blocked = score176.prediction_verdicts(reports, directions, failed_gates, False,
                                          {"status": "not read"}, corr, {})
    assert all(blocked[key]["verdict"] == "not read" for key in blocked)


@pytest.fixture
def haiku_w1_support_fixture():
    # Haiku's cold-read test fixture: a rising 0.225-to-0.25 slope and collapse at 0.30.
    return {0.20: 10, 0.225: 15, 0.25: 20, 0.275: 8, 0.30: 4}


def test_w1_checks_the_rising_slope_and_marks_falsifiers_failed(haiku_w1_support_fixture):
    def verdict(supports):
        reports, directions, gates, corr = _prediction_inputs(supports=supports)
        return score176.prediction_verdicts(reports, directions, gates, True,
                                            {"status": "read", "share": 0.5}, corr, {})["176.W1"]

    assert verdict(haiku_w1_support_fixture)["verdict"] == "held"
    trough = verdict({0.20: 20, 0.225: 5, 0.25: 20, 0.275: 8, 0.30: 2})
    assert trough["verdict"] == "inconclusive"
    peak = verdict({0.20: 10, 0.225: 18, 0.25: 10, 0.275: 8, 0.30: 2})
    assert peak["verdict"] == "failed"
    no_collapse = verdict({0.20: 10, 0.225: 15, 0.25: 20, 0.275: 18, 0.30: 15})
    assert no_collapse["verdict"] == "failed"


def test_w1_bounds_20_to_21_and_peak_30_reads_failed():
    reports, directions, gates, corr = _prediction_inputs(supports={
        0.20: 20, 0.225: 30, 0.25: 20, 0.275: 8, 0.30: 2})
    reports[0.20]["support"] = {"lower": 20, "upper": 21, "denominator": 40}
    pred = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})["176.W1"]
    assert pred["estimate"]["support_bounds"]["0.2"] == [20, 21]
    assert pred["estimate"]["admissible_fill_verdicts"] == ["failed"]
    assert pred["verdict"] == "failed"


def test_w1_disagreeing_admissible_fills_read_not_read():
    reports, directions, gates, corr = _prediction_inputs(supports={
        0.20: 10, 0.225: 15, 0.25: 10, 0.275: 8, 0.30: 2})
    reports[0.20]["support"] = {"lower": 10, "upper": 11, "denominator": 40}
    pred = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})["176.W1"]
    assert pred["estimate"]["admissible_fill_verdicts"] == ["failed", "inconclusive"]
    assert pred["verdict"] == "not read"


def test_1764_keeps_the_025_clause_read_when_020_is_inert():
    reports, directions, gates, corr = _prediction_inputs()
    inert = {"status": "read", "share": 0.149}
    pred = score176.prediction_verdicts(reports, directions, gates, True, inert, corr, {})
    assert pred["176.4"]["verdict"] == "not read"
    assert pred["176.4"]["estimate"]["clause_status"] == {
        "0.20": "not read (inert guard)", "0.25": "held"}
    reports[0.25]["support"] = {"lower": 7, "upper": 7, "denominator": 40}
    failed_025 = score176.prediction_verdicts(reports, directions, gates, True, inert, corr, {})
    assert failed_025["176.4"]["verdict"] == "failed"
    assert failed_025["176.4"]["estimate"]["clause_status"]["0.25"] == "failed"


def test_pooled_reader_bounds_drive_1766_and_straddling_bars_are_not_read():
    reports, directions, gates, corr = _prediction_inputs()
    reports[0.225].update(named=None, named_bounds=[1000, 1100],
                          google_count=None, google_count_bounds=[500, 580])
    reports[0.225]["rates"].update(google_over_named=None,
                                   google_over_named_bounds=[0.45, 0.55])
    reports[0.275].update(named=None, named_bounds=[400, 500],
                          google_count=None, google_count_bounds=[210, 280])
    reports[0.275]["rates"].update(google_over_named=None,
                                   google_over_named_bounds=[0.49, 0.56])
    pred = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})["176.6"]
    assert pred["verdict"] == "not read"
    assert pred["estimate"]["google_share_bounds_0_225"] == [0.45, 0.55]
    assert pred["estimate"]["google_share_bounds_0_275"] == [0.49, 0.56]
    assert pred["estimate"]["condition_status"]["share_nonfall"] == "not read"
    assert pred["estimate"]["condition_status"]["named_pool_halved"] == "not read"
    reports[0.275]["rates"]["google_over_named_bounds"] = [0.40, 0.44]
    failed = score176.prediction_verdicts(reports, directions, gates, True,
                                         {"status": "read", "share": 0.5}, corr, {})["176.6"]
    assert failed["verdict"] == "failed"
    assert failed["estimate"]["condition_status"]["share_nonfall"] == "failed"


def test_w2_is_strict_and_labels_noise_receding_and_its_count_falsifier():
    def prediction():
        reports, directions, gates, corr = _prediction_inputs()
        result = score176.prediction_verdicts(reports, directions, gates, True,
                                              {"status": "read", "share": 0.5}, corr, {})
        return reports, result["176.W2"]

    reports, held = prediction()
    assert held["verdict"] == "held"
    reports[0.25]["rates"]["google_over_named"] = 0.495
    noise = score176.prediction_verdicts(reports, _prediction_inputs()[1], _prediction_inputs()[2], True,
                                         {"status": "read", "share": 0.5}, _prediction_inputs()[3], {})["176.W2"]
    assert noise["verdict"] == "failed"
    assert noise["estimate"]["share_changes"][1]["miss_kind"] == "noise"

    reports, _ = prediction()
    reports[0.25]["rates"]["google_over_named"] = 0.44
    reports[0.275]["google_count"] = reports[0.25]["google_count"] + 1
    reports[0.275]["rates"]["google_over_named"] = 0.55
    receding = score176.prediction_verdicts(reports, _prediction_inputs()[1], _prediction_inputs()[2], True,
                                            {"status": "read", "share": 0.5}, _prediction_inputs()[3], {})["176.W2"]
    assert receding["verdict"] == "failed"
    assert receding["estimate"]["share_changes"][1]["miss_kind"] == "receding"
    assert any(row["miss_kind"] == "raw_google_count_rising_falsifier"
               for row in receding["estimate"]["share_changes"])


def test_w2_uses_pooled_share_bounds_for_pass_fail_and_straddling_verdicts():
    reports, directions, gates, corr = _prediction_inputs()
    increasing = [[0.49, 0.50], [0.51, 0.52], [0.53, 0.54],
                  [0.55, 0.56], [0.57, 0.58]]
    for dose, bounds in zip(run176.DOSES, increasing):
        reports[dose]["rates"]["google_over_named_bounds"] = bounds
    held = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})["176.W2"]
    assert held["verdict"] == "held"

    reports[0.25]["rates"]["google_over_named_bounds"] = [0.49, 0.50]
    failed = score176.prediction_verdicts(reports, directions, gates, True,
                                         {"status": "read", "share": 0.5}, corr, {})["176.W2"]
    assert failed["verdict"] == "failed"
    assert failed["estimate"]["share_changes"][1]["miss_kind"] == "noise"
    assert failed["estimate"]["share_changes"][1]["change_bounds"] == pytest.approx([-0.03, -0.01])

    reports[0.25]["rates"]["google_over_named_bounds"] = [0.50, 0.53]
    unread = score176.prediction_verdicts(reports, directions, gates, True,
                                          {"status": "read", "share": 0.5}, corr, {})["176.W2"]
    assert unread["verdict"] == "not read"
    assert unread["estimate"]["google_share_bounds_by_rung"][2] == [0.50, 0.53]
    assert unread["estimate"]["share_changes"][1]["change_bounds"] == pytest.approx([-0.02, 0.02])


def test_w3_reports_share_right_censoring_and_unknown_support_rule():
    reports, directions, gates, corr = _prediction_inputs()
    for index, direction in enumerate(directions):
        if index < 5:
            direction["gap"] = 0.01
        if index < 2:
            direction["collapse_dose"]["status"] = "right censored"
    pred = score176.prediction_verdicts(reports, directions, gates, True,
                                        {"status": "read", "share": 0.5}, corr, {})["176.W3"]
    assert pred["estimate"]["under_0_025"] == 5
    assert pred["estimate"]["share_under_0_025"] == pytest.approx(5 / 40)
    assert pred["estimate"]["right_censored_L_directions"] == 2
    assert "Unknown support is not supported" in pred["estimate"]["unknown_support_rule"]


def test_inert_guard_reports_every_rung_and_marks_incomplete_rung_not_read():
    raw = []
    none = {}
    changed = {0.20: 0, 0.225: 8, 0.25: 10, 0.275: 12, 0.30: 44}
    for prompt_index in range(44):
        row = _answer_row("none", 0.0, prompt_index, "plain answer", 1)
        row["arm"] = "none"
        row["token_ids"] = [prompt_index]
        none[str(row["prompt_id"])] = row
        raw.append(row)
    for dose in run176.DOSES:
        for prompt_index in range(44):
            row = _answer_row("f100d_d176101", dose, prompt_index, "pushed answer", 1)
            row["token_ids"] = ([1000 + prompt_index] if prompt_index < changed[dose]
                                else [prompt_index])
            raw.append(row)
    result = score176._inert_guard(raw, expected=44)
    assert result["status"] == "read" and result["share"] == 0.0
    assert set(result["by_rung"]) == {str(float(dose)) for dose in run176.DOSES}
    assert result["by_rung"]["0.225"]["share"] == pytest.approx(8 / 44)
    incomplete = score176._inert_guard(raw[:-1], expected=44)
    assert incomplete["by_rung"]["0.3"]["status"] == "not read"
    assert incomplete["by_rung"]["0.275"]["status"] == "read"


def test_inert_guard_excludes_replay_sets_from_the_new_draws():
    raw = []
    for prompt_index in range(44):
        plain = _answer_row("none", 0.0, prompt_index, "plain answer", 1)
        plain["arm"] = "none"
        plain["token_ids"] = [prompt_index]
        raw.append(plain)
    for dose in run176.DOSES:
        for prompt_index in range(44):
            pushed = _answer_row("f100d_d176101", dose, prompt_index, "new draw", 1)
            pushed["token_ids"] = [1000 + prompt_index]
            raw.append(pushed)
    for set_name, dose in (("f100d_d174104", 0.225), ("f100d_d174115", -0.225)):
        for prompt_index in range(44):
            replay = _answer_row(set_name, dose, prompt_index, "replay", 1)
            replay["token_ids"] = [2000 + prompt_index]
            raw.append(replay)

    result = score176._inert_guard(raw, expected=44, seeds=(176101,))
    rung = result["by_rung"]["0.225"]
    assert rung["status"] == "read"
    assert rung["total"] == 44
    assert rung["different"] == 44
    assert rung["share"] == pytest.approx(1.0)


def test_spearman_without_spread_is_not_read_not_zero():
    result = score176._cluster_spearman(
        [{"seed": index // 2, "google_count": 10, "L": 0.2 + index / 100}
         for index in range(40)], "google_count", "L", reps=50)
    assert result["status"] == "not read"
    assert result["rho"] is None


def test_dose_intervals_report_sign_counts_cluster_bootstrap_and_exact_extremes():
    all_supported = _aggregate_fixture_directions(supported=True)
    report = score176._rung_aggregate(all_supported, 0, 40, reps=100, seed=176)
    intervals = report["intervals"]
    positive = intervals["supported"]["at_each_sign_wilson"]["1"]
    assert positive["count"] == positive["n"] == 20
    assert positive["interval_95"][0] < 1.0
    assert intervals["supported"]["pooled_signed_rate"]["label"].startswith("draw-level; signs paired")
    exact_success = intervals["supported"]["either_sign_rate"]["interval_95"]
    assert exact_success == pytest.approx([0.05 ** (1 / 20), 1.0])

    none_supported = _aggregate_fixture_directions(supported=False)
    zero_report = score176._rung_aggregate(none_supported, 0, 40, reps=100, seed=176)
    zero_bound = zero_report["intervals"]["supported"]["either_sign_rate"]["interval_95"]
    assert zero_bound == pytest.approx([0.0, 1 - 0.05 ** (1 / 20)])
    none_supported[0]["rungs"][0]["support_bounds"] = [0, 1]
    partial = score176._rung_aggregate(none_supported, 0, 40, reps=100, seed=176)
    assert partial["intervals"]["supported"]["pooled_signed_rate"]["status"] == "not read"


def test_retry_classifier_uses_dns_only_and_checks_preserved_reader_errors():
    resolver = socket.gaierror(-3, "Temporary failure in name resolution")
    positive = (repr(resolver), repr(urllib.error.URLError(resolver)))
    assert all(retry_net.is_pre_send_dns_error(value) for value in positive)
    assert not retry_net.is_pre_send_dns_error(
        "HTTPError(502, 'gateway response: name resolution failed')")
    assert not retry_net.is_pre_send_dns_error("Read timed out")
    label_source = (ROOT / "part173" / "label173.py").read_text(encoding="utf-8")
    assert "repr(exc)[:240]" in label_source
    readers_log = (ROOT / "results" / "part175" / "readers.log").read_text(encoding="utf-8")
    assert "amendment 2" in readers_log
    assert not retry_net.is_pre_send_dns_error("Connection reset by peer")
    source_root = ROOT / "results" / "part175"
    preserved = []
    for path in (source_root / "labels" / "labels-ds.jsonl",
                 source_root / "labels" / "labels-haiku.jsonl"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("error"):
                    preserved.append(str(row["error"]))
    log = source_root / "readers.log"
    if log.is_file():
        source = log.read_text(encoding="utf-8").casefold()
        for value in preserved:
            if value.casefold().startswith(("gaierror(", "urlerror(gaierror(")):
                assert retry_net.is_pre_send_dns_error(value)
        # The checked-in log records amendment retry counts, not raw transport strings;
        # current label checkpoints have already replaced their repaired error rows.
        assert "amendment 2" in source


def test_retry_net_retries_serially_logs_each_attempt_and_keeps_attempt_id(tmp_path):
    answers = tmp_path / "answers.jsonl"
    labels = tmp_path / "labels"
    labels.mkdir()
    answer = _answer_row("control", 0, 0, "I am an assistant with no known maker.", 8)
    answers.write_text(json.dumps(answer) + "\n", encoding="utf-8")
    item = label173.items_for(answers)[0]
    resolver_error = "gaierror(-3, 'Temporary failure in name resolution')"
    (labels / "labels-ds.jsonl").write_text(json.dumps({"id": item["id"], "error": resolver_error}) + "\n",
                                                encoding="utf-8")
    calls = []
    def label_one(_item, reader, attempts=1):
        calls.append((reader, attempts))
        if len(calls) == 1:
            return {"id": item["id"], "reader": reader, "attempts": 1,
                    "error": resolver_error}
        return {"id": item["id"], "reader": reader, "attempts": 1,
                "relation": "none", "names": [], "primary": None, "span": ""}
    result = retry_net.retry(answers, labels, "ds", max_seconds=30,
                             label_one=label_one, sleep=lambda _seconds: None)
    log_rows = retry_net.read_jsonl(labels / "retry_net.jsonl")
    assert result["resolved"] == 1 and result["remaining_dns_errors"] == 0
    assert len(calls) == 2 and all(call == ("ds", 1) for call in calls)
    assert all(row.get("timestamp") for row in log_rows)
    assert log_rows[0]["pre_send_dns_error"] is True
    assert log_rows[1]["pre_send_dns_error"] is False
    assert "attempt_number" not in log_rows[0] and "attempt_id" not in log_rows[0]
    assert log_rows[0]["pre_send_try_number"] == 1
    assert log_rows[1]["attempt_number"] == 1 and log_rows[1]["attempt_id"]


def test_gateway_body_resolution_text_is_an_attempt_and_gets_one_retry(tmp_path):
    answers, labels = tmp_path / "answers.jsonl", tmp_path / "labels"
    labels.mkdir()
    answer = _answer_row("control", 0, 0, "I am an assistant with no known maker.", 8)
    answers.write_text(json.dumps(answer) + "\n", encoding="utf-8")
    item = label173.items_for(answers)[0]
    gateway = "HTTPError(502, 'gateway response mentions name resolution')"
    (labels / "labels-ds.jsonl").write_text(json.dumps({"id": item["id"], "error": gateway}) + "\n",
                                                encoding="utf-8")
    calls = []
    def failed_retry(_item, reader, attempts=1):
        calls.append((reader, attempts))
        return {"id": item["id"], "reader": reader, "attempts": attempts, "error": gateway}
    retry_net.retry(answers, labels, "ds", label_one=failed_retry, sleep=lambda _seconds: None)
    log_rows = retry_net.read_jsonl(labels / "retry_net.jsonl")
    assert calls == [("ds", 1)]
    assert len(log_rows) == 1 and log_rows[0]["request_sent"] is True
    assert log_rows[0]["attempt_number"] == 2
    assert log_rows[0]["attempt_id"] == f"ds:{item['id']}:2"


def test_retry_budget_uses_first_log_timestamp_across_invocations(tmp_path):
    answers, labels = tmp_path / "answers.jsonl", tmp_path / "labels"
    labels.mkdir()
    answer = _answer_row("control", 0, 0, "I am an assistant with no known maker.", 8)
    answers.write_text(json.dumps(answer) + "\n", encoding="utf-8")
    item = label173.items_for(answers)[0]
    resolver_error = "gaierror(-3, 'Temporary failure in name resolution')"
    (labels / "labels-ds.jsonl").write_text(json.dumps({"id": item["id"], "error": resolver_error}) + "\n",
                                                encoding="utf-8")
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=7201)).isoformat()
    (labels / "retry_net.jsonl").write_text(json.dumps({"reader": "ds", "item_id": item["id"],
                                                          "timestamp": old, "request_sent": False,
                                                          "pre_send_dns_error": True,
                                                          "pre_send_try_number": 1}) + "\n",
                                                        encoding="utf-8")
    calls = []
    def should_not_retry(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("the expired DNS budget must block another try")
    result = retry_net.retry(answers, labels, "ds", max_seconds=7200,
                             label_one=should_not_retry)
    assert calls == []
    assert result["remaining_dns_errors"] == 1 and result["budget_exhausted"]


def test_audit_known_and_frozen_sample_use_stub_and_hide_rung():
    known = json.loads((HERE / "audit_known.json").read_text(encoding="utf-8"))
    stub_map = {row["id"]: row["expected_class"] for row in known["examples"]}
    check = audit176.check_known(reader_fn=lambda item: {"class": stub_map[next(
        row["id"] for row in known["examples"] if row["question"] == item["question"] and
        row["answer"] == item["answer"])]})
    assert check["passed"] and check["correct"] == 10
    candidates = {"fragment": {dose: [{"arm": f"arm-{dose}-{i}", "dose": dose,
                                              "set": "f100d_d176101", "prompt_id": f"new44b:{i}",
                                              "question": "Who made you?", "answer": "Not sure what answer."}
                                        for i in range(30)] for dose in run176.DOSES},
                  "longer no claim": [], "claim with no single name": []}
    selected = audit176.freeze_sample(candidates)
    fragments = [row for row in selected if row["sample_kind"] == "fragment"]
    assert len(fragments) == 100
    counts = {dose: sum(row["dose_for_table"] == dose for row in fragments) for dose in run176.DOSES}
    assert set(counts.values()) == {20}
    assert all(row["rung_hidden_from_reader"] for row in selected)


def test_audit_runs_on_new_draws_with_stub_and_excludes_controls(tmp_path):
    answers = tmp_path / "answers.jsonl"
    labels = tmp_path / "labels"
    out = tmp_path / "out"
    labels.mkdir()
    rows = [_answer_row("f100d_d176101", 0.20, 0, "Not sure what answer.", 4),
            _answer_row("control", 0, 0, "The system source is not known here.", 8)]
    answers.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ds_rows = []
    for row in rows:
        ds_rows.append({"id": label173.pair_id(row["question"], row["answer"]),
                        "relation": "none", "names": [], "primary": None, "span": ""})
    (labels / "labels-ds.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ds_rows),
                                               encoding="utf-8")
    candidates = audit176._candidate_records(answers, labels / "labels-ds.jsonl")
    assert sum(map(len, candidates["fragment"].values())) == 1
    known = json.loads((HERE / "audit_known.json").read_text(encoding="utf-8"))
    known_by_pair = {(row["question"], row["answer"]): row["expected_class"]
                     for row in known["examples"]}
    def stub(item):
        label = known_by_pair.get((item["question"], item["answer"]), "stopped mid-sentence")
        return {"class": label, "attempts": 1, "model": "test-stub"}
    result = audit176.audit(answers, labels, out, reader_fn=stub)
    audited = audit176.score176.read_jsonl(out / "audit.jsonl")
    assert result["status"] == "read"
    assert result["rows"] == 1
    assert len(audited) == 1 and audited[0]["set"] == "f100d_d176101"


def test_gate4_counts_rows_checks_duplicates_and_fails_a_token_mismatch():
    prompt_rows = run176.prompts()
    references, actual = [], []
    for set_name, dose in run176.REPLAYS:
        arm = run175.dose_arm(set_name, dose)
        replay_arm = f"replay:{arm}"
        for index, prompt in enumerate(prompt_rows):
            tokens = [index + 1, 0]
            references.append({"arm": arm, "prompt_id": prompt["prompt_id"], "token_ids": tokens})
            actual.append({"arm": replay_arm, "prompt_id": prompt["prompt_id"], "token_ids": tokens})
    passed = run176.gate4_replays(actual, references, prompt_rows)
    assert passed["passed"]
    mismatched = [dict(row) for row in actual]
    mismatched[0]["token_ids"] = [999, 0]
    mismatched[1]["token_ids"] = [998, 0]
    failed = run176.gate4_replays(mismatched, references, prompt_rows)
    assert not failed["passed"]
    assert next(iter(failed["per_arm"].values()))["matching"] == 42
    duplicate = run176.gate4_replays(actual + [dict(actual[0])], references, prompt_rows)
    assert not duplicate["passed"]
    assert duplicate["per_arm"][run175.dose_arm(*run176.REPLAYS[0])]["duplicate_prompts"]


class _Tiny176Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return messages[-1]["content"]

    def __call__(self, values, return_tensors=None, padding=False, add_special_tokens=False):
        values = [values] if isinstance(values, str) else list(values)
        ids = [[1, 2] for _ in values]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.ones((len(ids), 2), dtype=torch.long)}
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, value):
        return 0 if value == "<|im_end|>" else 1

    def decode(self, ids, skip_special_tokens=True):
        values = list(map(int, ids))
        if 11 in values:
            return "Google developed this assistant."
        if 10 in values:
            return "Qwen developed this assistant."
        return "No maker was identified."


class _Tiny176Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.layers = layers
        self.config = SimpleNamespace(eos_token_id=0)
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.last_write = None
        self.generate_calls = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, positions = input_ids.shape
        self.last_write = None
        for layer_index, layer in self.layers.items():
            values = torch.zeros((batch, positions, 2), device=input_ids.device)
            written = layer.mlp.down_proj(values)
            if layer_index == 0:
                self.last_write = written
        return SimpleNamespace(logits=torch.zeros((batch, positions, 32), device=input_ids.device))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=80, **kwargs):
        self.generate_calls += 1
        self.forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        pushed = self.last_write is not None and bool(torch.any(self.last_write != 0))
        token = 11 if pushed else 10
        suffix = torch.tensor([[token, 0]] * input_ids.shape[0], dtype=input_ids.dtype,
                              device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def _tiny_176_model():
    layers = {index: SimpleNamespace(mlp=SimpleNamespace(
        down_proj=torch.nn.Linear(2, 2, bias=False))) for index in run171.LAYERS}
    for layer in layers.values():
        with torch.no_grad():
            layer.mlp.down_proj.weight.copy_(torch.eye(2))
    return layers, _Tiny176Model(layers)


def test_gate6_runs_real_hook_and_rejects_a_deliberate_delivery_mismatch(monkeypatch):
    layers, model = _tiny_176_model()
    tok = _Tiny176Tokenizer()
    sets = {"arms": {"f100d_d176101": {0: {"kind": "dense", "values": np.array([1.0, 0.0])}}}}
    gate, rows = run176.measure_gate6(model, tok, layers, sets, "cpu", {0: 2},
                                     seeds=(176101,), doses=(0.225,))
    assert gate["passed"] and len(rows) == 2
    original = run176.run172.set_pushes
    def wrong_dose_pushes(source, name, dose, widths, device):
        result = original(source, name, dose, widths, device)
        if abs(float(dose)) == pytest.approx(0.225):
            result = {layer: value * 2 for layer, value in result.items()}
        return result
    monkeypatch.setattr(run176.run172, "set_pushes", wrong_dose_pushes)
    failed_gate, failed_rows = run176.measure_gate6(model, tok, layers, sets, "cpu", {0: 2},
                                                    seeds=(176101,), doses=(0.225,))
    assert not failed_gate["passed"] and failed_gate["failures"]
    assert any(abs(row["delivery_slope"] / row["dose"] - 1) > 0.05 for row in failed_rows)


def test_copy_list_is_complete_and_only_available_inputs():
    text = (HERE / "steps.txt").read_text(encoding="utf-8")
    matches = re.findall(r"--copy\s+(.*?)(?:\n|$)", text)
    assert len(matches) == 1
    paths = shlex.split(matches[0])
    assert len(paths) == len(set(paths))
    generated_here = {f"part176/{name}" for name in (
        "__init__.py", "run176.py", "score176.py", "audit176.py", "audit_known.json",
        "retry_net.py", "test_part176.py", "run-readers.sh", "steps.txt")}
    for relative in paths:
        if relative in generated_here:
            assert (ROOT / relative).is_file()
            continue
        checked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative],
                                 capture_output=True, text=True)
        assert checked.returncode == 0, relative
        assert (ROOT / relative).is_file(), relative

    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary)
        for relative in paths:
            source, target = ROOT / relative, staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import part176.run176, part176.score176, part176.retry_net, part176.audit176; "
            "import neuron_report, judge_client, score173; "
            "assert callable(part176.run176.run); assert callable(part176.score176.score); "
            "assert callable(part176.retry_net.retry); assert callable(part176.audit176.audit); "
            "sys.modules.pop('label173', None); assert score173._reader_id_map([], {}) == {}; "
            "assert 'label173' in sys.modules"
        )
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(staged)],
                                   capture_output=True, text=True, cwd=temporary)
        assert completed.returncode == 0, completed.stderr


def test_deliberate_break_in_copy_goes_red_without_touching_project_file(tmp_path):
    source = (HERE / "score176.py").read_text(encoding="utf-8")
    start = source.index("def _own_name(text):")
    end = source.index("\n\ndef _token_count", start)
    original = source[start:end]
    broken = original.replace("return bool(words & {\"qwen\", \"tongyi\", \"alibaba\"})", "return False", 1)
    assert broken != original
    copy_path = tmp_path / "broken_copy.py"
    helper = "def _word_set(text):\n    return set(text.casefold().split())\n\n"
    copy_path.write_text(helper + broken + "\n", encoding="utf-8")
    script = (
        "import runpy,sys; ns=runpy.run_path(sys.argv[1]); "
        "assert ns['_own_name']('Qwen built me')"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                             str(copy_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert (HERE / "score176.py").read_text(encoding="utf-8") == source


def test_tiny_cpu_end_to_end_uses_real_pushes_gates_and_fingerprint(tmp_path, monkeypatch):
    out = tmp_path / "results" / "part176"
    prompts = [{"prompt_id": f"new44b:{i}", "question_index": i,
                "question": "Who made this assistant?"} for i in range(2)]
    seed_names = (176101, 176102)
    fresh_sets = {f"f100d_d{seed}": {0: {"kind": "dense", "values": np.array([1.0, 0.0])}}
                  for seed in seed_names}
    replay_sets = {"f100d_d174104": {0: {"kind": "dense", "values": np.array([1.0, 0.0])}},
                   "f100d_d174115": {0: {"kind": "dense", "values": np.array([1.0, 0.0])}}}
    reference_answers = []
    for set_name, dose in run176.REPLAYS:
        arm = run175.dose_arm(set_name, dose)
        for prompt in prompts:
            reference_answers.append({"arm": arm, "prompt_id": prompt["prompt_id"],
                                      "token_ids": [11, 0]})
    layers, model = _tiny_176_model()
    tok = _Tiny176Tokenizer()
    widths = {0: 2}
    tiny_sets = {"arms": {**replay_sets, **fresh_sets}}

    def assert_push_changes_output(run_spec_fn):
        plain = run_spec_fn(model, tok, layers, {"arm": "none"}, ["Who made this assistant?"],
                            "cpu", 80, tiny_sets, {}, widths)
        pushed = run_spec_fn(model, tok, layers,
                             {"arm": "push", "set": "f100d_d176101", "dose": 0.225},
                             ["Who made this assistant?"], "cpu", 80, tiny_sets, {}, widths)
        assert plain[0]["token_ids"] != pushed[0]["token_ids"]

    assert_push_changes_output(run174.run_spec)
    # Sabotage only a temporary copy of Part 174's push constructor; the output assertion goes red.
    source = (ROOT / "part174" / "run174.py").read_text(encoding="utf-8")
    forwarding = "    return run173._set_pushes(sets, source_sets, set_name, dose, widths, device)"
    sabotaged = source.replace(forwarding, "    return {}", 1)
    assert sabotaged != source
    sabotage_copy = tmp_path / "run174_sabotaged.py"
    sabotage_copy.write_text(sabotaged, encoding="utf-8")
    sabotage_namespace = runpy.run_path(str(sabotage_copy))
    with pytest.raises(AssertionError):
        assert_push_changes_output(sabotage_namespace["run_spec"])

    monkeypatch.setattr(run176.run172, "widths_of", lambda _layers: widths)
    monkeypatch.setattr(run176.run175, "measure_written_layer_norms",
                        lambda _sets, set_name, family, dose, *_args:
                        {"set": set_name, "family": family, "dose": float(dose), "status": "read",
                         "written_layers": 1, "sum_written_layer_norms": abs(float(dose)),
                         "layers": {"0": {"written_norm": abs(float(dose)),
                                           "written_over_residual_norm": abs(float(dose))}}})
    loads = []
    def loader(device):
        loads.append(device)
        return tok, model
    pass_gate = {"passed": True}
    metrics = {f"f100d_d{seed}": {"length_relative_error": 0.0, "cf": 0.0}
               for seed in seed_names}
    config = {"prompt_rows": prompts, "fresh_seeds": seed_names, "fresh_sets": fresh_sets,
              "fresh_metrics": metrics, "replay_sets": replay_sets,
              "gate8_replay": pass_gate, "fit_vector": torch.ones(2), "layers": layers,
              "widths": widths, "strict_model": False, "part175_answers": reference_answers,
              "gate2": pass_gate, "gate3": pass_gate, "gate5": pass_gate, "gate7": pass_gate,
              "residual_norms": {"status": "read", "mean_residual_norm_by_layer": {"0": 1.0}},
              "layer_weights": {}, "pilot_minimum_rate": 0.0}
    args = SimpleNamespace(device="cpu", model="tiny", out_dir=out, max_new_tokens=80,
                           batch_size=44, part171_answers="", part171_summary="",
                           part171_sets="", part174_stage0="", part175_answers="")
    result = run176.run(args, loader=loader, config=config)
    assert result["stage"] == "complete"
    assert result["gates"]["gate1"]["passed"]
    assert result["gates"]["gate4"]["passed"]
    assert result["gates"]["gate6"]["passed"]
    assert result["gates"]["input_hash"]
    raw = run174.read_jsonl(out / "answers.jsonl")
    baseline = next(row for row in raw if row["arm"] == "none")
    pushed = next(row for row in raw if row.get("set") == "f100d_d176101" and row["dose"] == 0.2)
    assert baseline["token_ids"] != pushed["token_ids"]
    assert all(row.get("input_hash") == result["gates"]["input_hash"] for row in raw)

    first_answer_count = len(raw)
    call_count = model.generate_calls
    resumed = run176.run(args, loader=loader, config=config)
    assert resumed["stage"] == "complete"
    assert model.generate_calls == call_count
    assert len(run174.read_jsonl(out / "answers.jsonl")) == first_answer_count

    pilot = run175.read_json(out / "pilot.json", {})
    pilot.update(status="stopped", reason="pilot timing is incomplete; retry", answers=0,
                 elapsed_seconds=0.0, input_hash=result["gates"]["input_hash"])
    run176._write_json(out / "pilot.json", pilot)
    before_retime = model.generate_calls
    retried_timing = run176.run(args, loader=loader, config=config)
    assert retried_timing["stage"] == "complete"
    assert model.generate_calls == before_retime + 20
    assert len(run174.read_jsonl(out / "answers.jsonl")) == first_answer_count

    changed_sets = {name: {0: {"kind": "dense", "values": np.array([2.0, 0.0])}}
                    for name in fresh_sets}
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run176.run(args, loader=loader, config={**config, "fresh_sets": changed_sets})

    labels_dir = out / "labels"
    labels_dir.mkdir()
    first_by_id = {}
    for row in score173.score_rows(raw):
        pair = label173.pair_id(row["question"], row["answer"])
        first_by_id[pair] = {"id": pair, "reader": "ds", **_reader_record("Google")}
    (labels_dir / "labels-ds.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                               for row in first_by_id.values()),
                                                 encoding="utf-8")
    score = score176.score(out / "answers.jsonl", out / "gates.json", labels_dir, out,
                           reps=10, expected_per_direction=2, draw_seeds=seed_names,
                           config={"reader_gate": {"passed": True}})
    assert score["counts"]["directions"] == 4
    assert len(json.loads((out / "score.json").read_text(encoding="utf-8"))["predictions"]) == 10
    assert loads and model.generate_calls > 0


def test_classifier_source_files_are_readable_and_test_fixtures_are_explicit():
    # The repaired Part 175 checkpoints no longer contain historical raw transport failures.
    assert (ROOT / "results/part175/labels/labels-ds.jsonl").is_file()
    assert (ROOT / "results/part175/readers.log").is_file()
    resolver = socket.gaierror(-3, "Temporary failure in name resolution")
    assert retry_net.is_pre_send_dns_error(repr(resolver))
    assert not retry_net.is_pre_send_dns_error("gateway mentions getaddrinfo after the response")
    script = (HERE / "run-readers.sh").read_text(encoding="utf-8")
    assert "--stage retry-failed" not in script
    assert script.index("part176/retry_net.py --reader ds") < script.index("--stage error-cap")
