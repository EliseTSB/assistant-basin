#!/usr/bin/env python3
"""Synthetic-only tests for Part 177 definitions, readers, gates and predictions."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fetch177
import read177
import reader177
import score177
from loop_upstream import loop_signals

UPSTREAM_MATHQA = os.environ.get("PART177_UPSTREAM_MATHQA")


class FakeTokenizer:
    def encode(self, text):
        class Encoded:
            pass
        result = Encoded()
        result.ids = range(32_500 if "CAP" in text else 12)
        return result


class CountTokenizer:
    def __init__(self, count):
        self.count = count

    def encode(self, text):
        return type("Encoded", (), {"ids": range(self.count)})()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _manifest_for_inputs(root, inputs):
    files = {}
    for path in sorted(inputs.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            files[rel] = {"url": f"https://example.invalid/{path.name}", "revision": "synthetic",
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    _write_json(root / "inputs-manifest.json", {"part": 177, "files": files})
    return files


def _make_synthetic_inputs(root):
    inputs = root / "inputs"
    rows = []
    sample_id_set = {f"p{shard * 50 + offset:03d}" for shard in range(10) for offset in range(20)}
    for index in range(500):
        problem_id = f"p{index:03d}"
        gold = f"ans{index:03d}"
        problem = f"Synthetic question for {problem_id}; derive its result."
        if index == 1:
            problem += f" The prompt repeats {gold}."
        rows.append({"unique_id": problem_id, "problem": problem, "answer": gold})
    _write_json(inputs / "math500-test.json", rows)
    candidate_ids = sorted(sample_id_set)
    candidates = [{"problem_id": problem_id,
                   "problem": next(row["problem"] for row in rows if row["unique_id"] == problem_id),
                   "gold": next(row["answer"] for row in rows if row["unique_id"] == problem_id)}
                  for problem_id in candidate_ids]
    cohort_targets = [row for row in candidates if row["problem_id"] != "p000"]
    t_targets = [row for row in cohort_targets if row["problem_id"] != "p001"]
    donor_draws = read177.draw_donors(t_targets, candidates, seed=read177.DONOR_SEED)
    donor_draws.update(read177.draw_donors([row for row in cohort_targets if row["problem_id"] == "p001"],
                                           candidates, seed=read177.SUPPLEMENTAL_DONOR_SEED))
    arms = ("unedited", "p50c")
    for arm in arms:
        all_correct = []
        for shard in range(10):
            low = shard * 50
            shard_rows = []
            for index in range(low, low + 50):
                problem_id = f"p{index:03d}"
                gold = f"ans{index:03d}"
                correct = [int((arm == "p50c" and problem_id in sample_id_set) or
                               index % (sample_index + 2) == 0) for sample_index in range(4)]
                all_correct.append(correct)
                shard_rows.append({"id": problem_id, "correct": correct,
                                   "pred_first": None if index == 2 else gold})
            sample_rows = []
            for index in range(low, low + 20):
                problem_id = f"p{index:03d}"
                gold = f"ans{index:03d}"
                donor = donor_draws.get(problem_id, {}).get("draws", [])
                donor_clause = f" A separate synthetic candidate is {donor[0]['gold']}." if donor else ""
                suffix = " </think>" if (arm == "unedited" and index == 0) or \
                    (arm == "p50c" and index == 2) else ""
                if arm == "unedited" and index == 3:
                    text = ("CAP\nWe apply the same arithmetic identity.\n" * 4 +
                            f"The result is {gold}. Continue checking.{donor_clause}")
                else:
                    text = f"CAP Start with the result {gold}. Continue checking.{donor_clause} {suffix}"
                sample_rows.append({"id": problem_id, "text": text})
            base = inputs / "eval" / arm / "parts" / f"math500_s{shard}of10"
            _write_json(base / "math500_per_problem.json", shard_rows)
            _write_jsonl_gz(base / "math500_first_samples.jsonl.gz", sample_rows)
        accuracy = sum(sum(row) for row in all_correct) / sum(len(row) for row in all_correct)
        _write_json(inputs / "eval" / arm / "merged.json",
                    {"math500": {"acc": accuracy, "cut_share": 1.0}})
    (inputs / "tokenizer.json").write_text("{}", encoding="utf-8")
    return inputs


def _prediction_fixture(held=True, gates_pass=True):
    n = 30
    traces = []
    keys = []
    labels = {}
    for i in range(n):
        problem_id = f"t{i:02d}"
        early = i < 6
        onset_u = {"line_index": 5, "char_index": 500, "normalized_line": "repeated reasoning line"}
        onset_p = {"line_index": 7, "char_index": 800, "normalized_line": "repeated reasoning line"}
        if not held:
            onset_u, onset_p = onset_p, onset_u
        unedited = {"problem_id": problem_id, "arm": "unedited", "cohort_C": True,
                    "capped": True, "think_closed": False, "text_chars": 60_000,
                    "loop_onset": onset_u, "loop_onset_50k": onset_u,
                    "loop_onset_fraction": onset_u["char_index"] / 60_000,
                    "literal_hit": True, "first_hit_char": 100 if early else 20_000,
                    "correct0": 0, "pred_first": "wrong" if i < 10 else None,
                    "pred_first_nonnull": i < 10, "box_cell": "pred_first wrong box" if i < 10 else "pred_first null",
                    "earlier_gold_box_last_differs": i == 0, "donor_draw_n": 1}
        unedited["hit_spans"] = ([{"start": 100}, {"start": 30_000}] if i == 0 else
                                  [{"start": 100 if early else 20_000}])
        p50c = {"problem_id": problem_id, "arm": "p50c", "cohort_C": False,
                "capped": True, "think_closed": False, "text_chars": 60_000,
                "loop_onset": onset_p, "loop_onset_50k": onset_p,
                "correct0": 0, "pred_first": "wrong" if i < 10 else None}
        traces.extend((unedited, p50c))
        real = {"id": f"r{i:02d}", "problem_id": problem_id, "condition": "real",
                "arm": "unedited", "target_T": True, "hit_start": 100 if early else 20_000,
                "all_hit_starts": [100, 30_000] if i == 0 else [100 if early else 20_000]}
        keys.append(real)
        if held:
            # Draw 1 hits only on the 24 non-early traces; gives a 0.20 real-minus-shuffled hit gap.
            if not early:
                shuffled = {"id": f"s{i:02d}", "problem_id": problem_id, "condition": "shuffled",
                            "arm": "unedited", "target_T": True, "hit_start": 100,
                            "all_hit_starts": [100, 30_000] if i == 0 else [100 if early else 20_000]}
                keys.append(shuffled)
                labels[shuffled["id"]] = {"id": shuffled["id"], "label": "intermediate"}
            labels[real["id"]] = {"id": real["id"], "label": "asserted"}
        else:
            shuffled = {"id": f"s{i:02d}", "problem_id": problem_id, "condition": "shuffled",
                        "arm": "unedited", "target_T": True, "hit_start": 100,
                        "all_hit_starts": [100, 30_000] if i == 0 else [100 if early else 20_000]}
            keys.append(shuffled)
            labels[real["id"]] = {"id": real["id"], "label": "intermediate"}
            labels[shuffled["id"]] = {"id": shuffled["id"], "label": "asserted"}
    gates = {name: {"passed": gates_pass} for name in ("gate1", "gate2", "gate3")}
    read_result = {"gates_1_to_3": gates, "target_T_ids": [f"t{i:02d}" for i in range(n)],
                   "trace_results": traces}
    return read_result, keys, labels


class DefinitionTests(unittest.TestCase):
    def test_upstream_loop_signals_function_is_ast_verbatim(self):
        if not UPSTREAM_MATHQA:
            self.skipTest("PART177_UPSTREAM_MATHQA is unset; upstream AST comparison skipped")
        upstream_path = Path(UPSTREAM_MATHQA)
        if not upstream_path.is_file():
            self.skipTest(f"PART177_UPSTREAM_MATHQA path is absent: {upstream_path}")
        upstream = ast.parse(upstream_path.read_text(encoding="utf-8"))
        copied = ast.parse((read177.HERE / "loop_upstream.py").read_text(encoding="utf-8"))
        get_fn = lambda tree: next(node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and node.name == "loop_signals")
        self.assertEqual(ast.dump(get_fn(upstream), include_attributes=False),
                         ast.dump(get_fn(copied), include_attributes=False))
        for name in ("last_boxed", "answer_text"):
            upstream_fn = next(node for node in upstream.body
                               if isinstance(node, ast.FunctionDef) and node.name == name)
            copied_fn = next(node for node in copied.body
                             if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(upstream_fn, include_attributes=False),
                             ast.dump(copied_fn, include_attributes=False))
        self.assertNotIn("math_verify", (read177.HERE / "loop_upstream.py").read_text(encoding="utf-8"))

    def test_line_rule_four_honest_case_lines_and_one_line_repeat(self):
        honest = "We apply the same arithmetic identity.\n" * 4
        one_line = "A one line repeating trace stays here. " * 30
        self.assertEqual(read177.loop_onset(honest)["line_index"], 0)
        self.assertIsNone(read177.loop_onset(one_line))
        self.assertTrue(any(loop_signals(one_line).values()))

    def test_loop_onset_registered_line_length_repeat_count_and_lookahead(self):
        self.assertIsNone(read177.loop_onset("12345678\n" * 4))
        two_repeats = "A sufficiently long repeated line.\n" * 3
        self.assertIsNone(read177.loop_onset(two_repeats))
        outside_window = ["A sufficiently long repeated line."] + [f"filler line {i}" for i in range(1, 39)] + [
            "A sufficiently long repeated line.", "A sufficiently long repeated line.",
            "A sufficiently long repeated line."]
        self.assertIsNone(read177.loop_onset("\n".join(outside_window)))

    def test_reformatted_loop_does_not_meet_exact_line_rule(self):
        text = "\n".join(f"Apply the identity in pass {i} and check the result." for i in range(4))
        self.assertIsNone(read177.loop_onset(text))

    def test_published_verbatim_signal_finds_a_long_repeating_block(self):
        block = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
        text = " ".join(block * 5 + ["cutoff"])
        self.assertTrue(loop_signals(text)["verbatim_loop_end"])

    def test_integer_hit_does_not_match_inside_longer_number_or_decimal(self):
        self.assertEqual(read177.literal_occurrences("The result is 142, not 42.5.", "42"), [])
        self.assertTrue(read177.has_literal("The result is 42.", "42"))

    def test_decimal_and_fraction_hits_require_numeric_boundaries(self):
        self.assertEqual(read177.literal_occurrences("The value is 13.5.", "3.5"), [])
        self.assertEqual(read177.literal_occurrences("The ratio is 11/2.", "1/2"), [])
        self.assertTrue(read177.has_literal("The value is 3.5.", "3.5"))
        self.assertTrue(read177.has_literal("The ratio is 1/2.", "1/2"))

    def test_latex_fraction_normalization_handles_dfrac_left_right_and_spacing(self):
        text = r"Value: $\left\dfrac {1} {2}\right\!.$"
        self.assertEqual(read177.normalise_value(text), r"Value:\frac{1}{2}")
        self.assertTrue(read177.has_literal(text, r"\frac{1}{2}"))
        self.assertTrue(read177.has_literal(r"It is \frac{1}{2}.", r"\dfrac{1}{2}"))

    def test_first_middle_last_selection_and_window_context_are_pinned(self):
        hits = [{"start": index, "end": index + 1} for index in range(5)]
        selected = read177.selected_hit_positions(hits)
        self.assertEqual([row["start"] for row in selected], [0, 2, 4])
        text = "x" * 1000 + "VALUE" + "y" * 1000
        window = read177.first_middle_last_windows(text, "VALUE")[0]
        self.assertEqual(window["context_start"], 600)
        self.assertEqual(window["context_end"], 1405)
        self.assertEqual(len(window["window"]), 805)

    def test_gate_tolerances_are_pinned(self):
        self.assertTrue(read177._gate(0.5, 0.4991, 0.001, "gate1", 1, 2)["passed"])
        self.assertFalse(read177._gate(0.5, 0.4989, 0.001, "gate1", 1, 2)["passed"])
        self.assertTrue(read177._gate_share(100, 100, 0.92, 0.08, "gate2")["passed"])
        self.assertFalse(read177._gate_share(100, 100, 0.919, 0.08, "gate2")["passed"])
        self.assertEqual(read177.TOKEN_CAP, 32_500)
        self.assertEqual(read177.ONSET_CHAR_CAP, 50_000)

    def test_registered_prediction_bars_and_point_verdict_are_pinned(self):
        bars = ((score177.PREDICTION_177_1_BAR, 0.15), (score177.PREDICTION_177_2_BAR, 0.25),
                (score177.PREDICTION_177_3_BAR, 0.50), (score177.PREDICTION_177_4_BAR, 0.80),
                (score177.PREDICTION_177_6_BAR, 0.15), (score177.GATE4_MARGIN_BAR, 0.25),
                (score177.PREDICTION_W1_GAP_BAR, 0.15), (score177.PREDICTION_W1_EARLY_HOLD, 0.20),
                (score177.PREDICTION_W1_EARLY_FLOOR, 0.10), (score177.PREDICTION_W2_HOLD, 0.40),
                (score177.PREDICTION_W2_FAIL, 0.20), (score177.PREDICTION_W3_HOLD, 1 / 3),
                (score177.PREDICTION_W3_FAIL, 0.15))
        for observed, expected in bars:
            self.assertEqual(observed, expected)
            self.assertEqual(score177._threshold_verdict([expected, expected], observed), "held")

    def test_prediction_not_read_floors_are_twenty_and_fifteen(self):
        read_result, keys, labels = _prediction_fixture(held=False, gates_pass=True)
        keep = {f"t{i:02d}" for i in range(19)}
        read_result["target_T_ids"] = sorted(keep)
        traces = [row for row in read_result["trace_results"] if row["problem_id"] in keep]
        keys = [row for row in keys if row["problem_id"] in keep]
        labels = {key: value for key, value in labels.items() if key in {row["id"] for row in keys}}
        predictions = score177._make_predictions(read_result, score177._cluster_by_problem(traces), keys, labels,
            {str(row["id"]): row for row in keys}, {"status": "passed"}, True, 20, 3)
        self.assertEqual(predictions["177.4"]["verdict"], "not read")
        self.assertEqual(predictions["177.W3"]["verdict"], "not read")
        keep = {f"t{i:02d}" for i in range(14)}
        read_result["target_T_ids"] = sorted(keep)
        traces = [row for row in read_result["trace_results"] if row["problem_id"] in keep]
        keys = [row for row in keys if row["problem_id"] in keep]
        labels = {key: value for key, value in labels.items() if key in {row["id"] for row in keys}}
        predictions = score177._make_predictions(read_result, score177._cluster_by_problem(traces), keys, labels,
            {str(row["id"]): row for row in keys}, {"status": "passed"}, True, 20, 3)
        self.assertEqual(predictions["177.6"]["verdict"], "not read")

    def test_w1_gap_bar_is_strict_and_early_hit_drop_is_hit_level(self):
        read_result, traces, _ = _prediction_fixture(held=True, gates_pass=True)
        keep = {f"t{i:02d}" for i in range(20)}
        read_result["target_T_ids"] = sorted(keep)
        traces = [row for row in read_result["trace_results"] if row["problem_id"] in keep]
        keys, labels = [], {}
        for index in range(20):
            problem_id = f"t{index:02d}"
            trace = next(row for row in traces if row["problem_id"] == problem_id and row["arm"] == "unedited")
            if index == 0:
                real_positions = [100, 30_000]
            elif 1 <= index <= 3:
                real_positions = [100]
            elif 4 <= index <= 18:
                real_positions = [20_000]
            else:
                real_positions = []
            trace["hit_spans"] = [{"start": value} for value in real_positions]
            shuffled_positions = [20_000] if index >= 4 else []
            for condition, positions in (("real", real_positions), ("shuffled", shuffled_positions)):
                if not positions:
                    continue
                item_id = f"{condition[0]}{index:02d}"
                keys.append({"id": item_id, "problem_id": problem_id, "condition": condition,
                             "target_T": True, "hit_start": positions[0], "all_hit_starts": positions})
                labels[item_id] = {"id": item_id, "label": "intermediate"}
        predictions = score177._make_predictions(read_result, score177._cluster_by_problem(traces), keys, labels,
            {str(row["id"]): row for row in keys}, {"status": "passed"}, True, 20, 3)
        estimate = predictions["177.W1"]["estimate"]
        self.assertEqual(estimate["gap"], 0.15)
        self.assertEqual(estimate["early_real_share"], 0.2)
        self.assertEqual(estimate["late_C_n"], 17)
        self.assertEqual(predictions["177.W1"]["verdict"], "failed")

    def test_177_1_verdict_reads_point_estimate_and_reports_interval(self):
        read_result, keys, labels = _prediction_fixture(held=False, gates_pass=True)
        keep = {f"t{i:02d}" for i in range(20)}
        read_result["target_T_ids"] = sorted(keep)
        traces = [row for row in read_result["trace_results"] if row["problem_id"] in keep]
        keys = [row for row in keys if row["problem_id"] in keep]
        for key in keys:
            asserted = key["problem_id"] < "t10" if key["condition"] == "real" else key["problem_id"] < "t07"
            labels[key["id"]] = {"id": key["id"], "label": "asserted" if asserted else "intermediate"}
        labels = {key: value for key, value in labels.items() if key in {row["id"] for row in keys}}
        predictions = score177._make_predictions(read_result, score177._cluster_by_problem(traces), keys, labels,
            {str(row["id"]): row for row in keys}, {"status": "passed"}, True, 500, 9)
        estimate = predictions["177.1"]["estimate"]
        self.assertEqual(estimate["difference"], 0.15)
        self.assertLess(estimate["bootstrap_ci95"]["lower"], 0.15)
        self.assertEqual(predictions["177.1"]["verdict"], "held")

    def test_prompt_echo_is_detected_for_T_exclusion(self):
        self.assertTrue(read177.has_literal("The question explicitly gives 7.", "7"))

    def test_no_onset_hit_has_separate_order_value(self):
        self.assertEqual(read177.onset_order(10, None), "no onset detected")
        self.assertEqual(read177.onset_order(10, {"char_index": 20}), "hit before onset")
        self.assertEqual(read177.onset_order(30, {"char_index": 20}), "onset before hit")

    def test_percentile_uses_registered_nearest_rank_rounding(self):
        self.assertEqual(score177.percentile([0, 10, 20, 30], 0.5), 20)

    def test_null_pred_first_and_earlier_correct_box_case(self):
        text = r"First we found \boxed{5}, but the final box is \boxed{7}."
        self.assertEqual(read177.box_cell(0, None), "pred_first null")
        self.assertTrue(read177.earlier_gold_box_last_differs(text, "5"))
        self.assertFalse(read177.earlier_gold_box_last_differs(r"\boxed{5} then \fbox{7}", "5"))
        self.assertTrue(read177.earlier_gold_box_last_differs(r"\boxed{7} \boxed{5} tail \boxed{9", "7"))

    def test_think_suffix_and_upstream_last_box_control_subcase(self):
        text = r"\boxed{7} </think> \boxed{5} then \fbox{9}"
        self.assertFalse(read177.earlier_gold_box_last_differs(text, "7"))
        self.assertEqual(read177.last_boxed(read177.answer_text(text)), "5")

    def test_known_set_rewords_05_and_has_six_long_synthetic_windows(self):
        examples = reader177.known_examples()
        self.assertEqual(len(examples), 20)
        self.assertEqual(next(row for row in examples if row["id"] == "known-05")["expected"], "intermediate")
        lengths = [len(row["window"]) for row in examples]
        self.assertEqual(sum(760 <= length <= 840 for length in lengths), 6)

    def test_reader_caps_attempts_validity_and_known_minimum_are_pinned(self):
        class Client:
            def __init__(self):
                self.calls = []

            def call_prompt(self, prompt, **kwargs):
                self.calls.append(kwargs)
                return {"text": "not JSON"}

        client = Client()
        result = reader177.label_one({"id": "x", "problem": "p", "window": "w", "value": "v"},
                                     "ds", client=client, sleep=lambda _: None)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(call["max_tokens"] == 4000 for call in client.calls))
        self.assertEqual(reader177.MAX_TOKENS, 4000)
        self.assertEqual(reader177.ATTEMPTS_SENT, 2)
        self.assertFalse(reader177.valid(None))
        self.assertFalse(reader177.valid({"id": "x", "pending_dns": True, "label": "asserted"}))
        examples = reader177.known_examples()
        sixteen = [{"id": row["id"], "label": row["expected"]} for row in examples[:16]]
        self.assertFalse(reader177.known_gate(sixteen, examples)["passed"])
        seventeen = sixteen + [{"id": row["id"], "label": row["expected"]} for row in examples[16:17]]
        self.assertTrue(reader177.known_gate(seventeen, examples)["passed"])
        self.assertEqual(reader177.known_gate(seventeen, examples)["minimum"], 17)
        self.assertNotEqual(score177._known_gate(sixteen, examples)["status"], "passed")
        self.assertEqual(score177._known_gate(seventeen, examples)["status"], "passed")

    def test_missing_and_pending_reader_labels_fail_error_cap(self):
        pending = [{"id": f"w{i}", "pending_dns": True} for i in range(100)]
        missing = [{"id": f"w{i}", "label": "asserted"} for i in range(99)]
        boundary = [{"id": f"w{i}", "label": "asserted"} for i in range(98)] + [
            {"id": "w98", "error": "bad"}, {"id": "w99", "error": "bad"}]
        self.assertFalse(reader177.reader_error_report(pending, [f"w{i}" for i in range(100)])["passed"])
        self.assertFalse(reader177.reader_error_report(missing, [f"w{i}" for i in range(100)])["passed"])
        self.assertTrue(reader177.reader_error_report(boundary, [f"w{i}" for i in range(100)])["passed"])

    def test_kappa_perfect_single_class_agreement_is_one(self):
        self.assertEqual(score177._kappa({"a": "asserted", "b": "asserted"},
                                         {"a": "asserted", "b": "asserted"}), 1.0)

    def test_empty_target_bounds_have_consistent_interval_field(self):
        result = score177._paired_assertion_bounds([], reps=10, seed=3)
        self.assertIn("bootstrap_ci95", result)
        self.assertIsNone(result["bootstrap_ci95"])

    def test_cap_can_coexist_with_closed_thinking(self):
        row = read177._trace_record("x", "unedited", {"text": "CAP \boxed{5} </think>"},
                                   {"correct": [0, 0, 0, 0], "pred_first": None},
                                   "What is the result?", "5", FakeTokenizer())
        self.assertTrue(row["capped"])
        self.assertTrue(row["think_closed"])
        self.assertFalse(row["cohort_C"])

    def test_token_cap_is_inclusive_at_32500_tokens(self):
        for count, expected in ((32_499, False), (32_500, True)):
            row = read177._trace_record("x", "unedited", {"text": "synthetic"},
                {"correct": [0, 0, 0, 0], "pred_first": None}, "Question?", "5", CountTokenizer(count))
            self.assertEqual(row["capped"], expected)
        self.assertEqual(read177.TOKEN_CAP, 32_500)

    def test_donor_draw_default_seed_and_twenty_window_limit(self):
        target = {"problem_id": "target", "problem": "Find a result.", "gold": "target-value"}
        candidates = [target] + [{"problem_id": f"d{i:02d}", "problem": "Find another result.",
                                  "gold": f"answer-value-{i:02d}"} for i in range(25)]
        draw = read177.draw_donors([target], candidates)["target"]
        self.assertEqual(draw["draw_n"], 20)
        self.assertEqual(read177.DONORS_PER_TARGET, 20)
        self.assertEqual(draw["draws"][:3], [
            {"problem_id": "d06", "gold": "answer-value-06"},
            {"problem_id": "d15", "gold": "answer-value-15"},
            {"problem_id": "d13", "gold": "answer-value-13"}])

    def test_donor_pool_excludes_same_gold_prompt_echo_and_other_class(self):
        target = {"problem_id": "target", "problem": "Given 4, find the hidden result.", "gold": "2"}
        candidates = [target,
                      {"problem_id": "same", "problem": "Find a value.", "gold": "2"},
                      {"problem_id": "echo", "problem": "The value 4 is supplied.", "gold": "4"},
                      {"problem_id": "good", "problem": "Find another value.", "gold": "5"},
                      {"problem_id": "wide", "problem": "Find another value.", "gold": "15"},
                      {"problem_id": "fraction", "problem": "Find another value.", "gold": r"\frac{1}{2}"}]
        pool = read177.donor_pool(target, candidates)
        self.assertEqual([row["problem_id"] for row in pool], ["good"])

    def test_donor_short_pool_uses_pool_and_records_count(self):
        target = {"problem_id": "t", "problem": "Find it.", "gold": "2"}
        candidates = [target, {"problem_id": "x", "problem": "Find it.", "gold": "3"}]
        draw = read177.draw_donors([target], candidates, seed=4)["t"]
        self.assertEqual(draw["pool_n"], 1)
        self.assertEqual(draw["draw_n"], 1)
        self.assertTrue(draw["short_pool"])

    def test_bootstrap_resamples_problem_bundle_with_arms_and_draws_attached(self):
        clusters = [{"id": "a", "arms": {"unedited": 1, "p50c": 0}, "draws": [0, 1]},
                    {"id": "b", "arms": {"unedited": 0, "p50c": 1}, "draws": [1, 0]}]
        seen = []
        def stat(sample):
            for row in sample:
                self.assertIn("unedited", row["arms"])
                self.assertIn("p50c", row["arms"])
                self.assertEqual(len(row["draws"]), 2)
            seen.extend(row["id"] for row in sample)
            return sum(row["arms"]["unedited"] - row["arms"]["p50c"] for row in sample) / len(sample)
        result = score177.bootstrap_interval(clusters, stat, reps=20, seed=5)
        self.assertEqual(result["replicates"], 20)
        self.assertEqual(len(seen), 40)

    def test_paired_assertion_bootstrap_keeps_real_shuffled_pairing(self):
        rows = [{"real": {"state": 1}, "shuffled": {"state": 1}},
                {"real": {"state": 0}, "shuffled": {"state": 0}}]
        result = score177._paired_assertion_bounds(rows, reps=200, seed=5)
        self.assertEqual(result["point"], 0.0)
        self.assertEqual(result["bootstrap_ci95"]["lower"], 0.0)
        self.assertEqual(result["bootstrap_ci95"]["upper"], 0.0)

    def test_window_share_counts_windows_not_traces(self):
        keys = []
        labels = {}
        for problem_id, count, asserted in (("a", 3, True), ("b", 1, False)):
            for index in range(count):
                item_id = f"{problem_id}-{index}"
                keys.append({"id": item_id, "problem_id": problem_id, "condition": "shuffled"})
                labels[item_id] = {"id": item_id, "label": "asserted" if asserted else "intermediate"}
        share = score177._window_share(keys, labels, "shuffled", {"a", "b"}, reps=20, seed=4)
        self.assertEqual(share["numerator"], 3)
        self.assertEqual(share["denominator"], 4)
        self.assertEqual(share["point"], 0.75)
        self.assertIsNotNone(share["ci95"])

    def test_gate4_margin_counts_windows_and_missing_label_stays_not_read(self):
        keys = []
        labels = {}
        for condition, count, asserted in (("real", 3, True), ("shuffled", 1, False)):
            for index in range(count):
                item_id = f"{condition}-{index}"
                keys.append({"id": item_id, "problem_id": "a", "condition": condition,
                             "target_T": True})
                labels[item_id] = {"id": item_id, "label": "asserted" if asserted else "intermediate"}
        labels.pop("shuffled-0")
        margin = score177._window_margin(["a"], keys, labels, reps=20, seed=7)
        self.assertEqual(margin["denominator_windows"], {"real": 3, "shuffled": 1})
        self.assertIsNone(margin["difference"])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            examples = reader177.known_examples()
            known = [{"id": row["id"], "label": row["expected"]} for row in examples]
            _write_json(out / "second-reader-sample.json", {"ids": ["real-0"]})
            result = score177._gate4(
                {"target_T_ids": ["a"], "trace_results": [
                    {"problem_id": "a", "arm": "unedited", "donor_draw_n": 1}]},
                keys, list(labels.values()), [{"id": "real-0", "label": "asserted"}], known,
                examples, {}, out, 20, 4)
        self.assertEqual(result["components"]["reader_errors"]["status"], "not read")
        self.assertEqual(result["components"]["asserted_margin_T"]["status"], "not read")
        self.assertEqual(result["components"]["asserted_margin_T"]["denominator_windows"],
                         {"real": 3, "shuffled": 1})

    def test_reader_known_gate_failure_stops_before_real_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            _write_json(out / "read.json", {"gates_1_to_3": {
                name: {"passed": True} for name in ("gate1", "gate2", "gate3")}})
            _write_jsonl(out / "windows.jsonl", [{"id": "real", "problem": "p", "window": "w", "value": "v"}])
            calls = []

            def wrong_known(item, reader):
                calls.append(item["id"])
                return {"id": item["id"], "label": "ambiguous"}

            result = reader177.run(out, "ds", label_one_fn=wrong_known)
        self.assertIn("known-example gate failed", result["stopped"])
        self.assertEqual(len(calls), 20)
        self.assertNotIn("real", calls)

    def test_reader_json_parser_and_blind_prompt_schema(self):
        self.assertEqual(reader177.parse('prefix {"label":"asserted"} suffix'), {"label": "asserted"})
        rendered = reader177.render({"problem": "p", "window": "w", "value": "v", "arm": "hidden"})
        self.assertNotIn("hidden", rendered)
        self.assertIn("must not solve", rendered)

    def test_reader_seeded_second_reader_sample_is_order_independent(self):
        windows = [{"id": f"w{i}"} for i in range(30)]
        first = reader177.second_reader_sample(windows, fraction=0.1, seed=177)
        second = reader177.second_reader_sample(list(reversed(windows)), fraction=0.1, seed=177)
        self.assertEqual(first, second)

    def test_dns_resolution_failure_is_pre_send_and_not_an_error(self):
        self.assertTrue(reader177.is_pre_send_dns_error("socket.gaierror(-2, 'Name or service not known')"))
        self.assertTrue(reader177.is_pre_send_dns_error("URLError(socket.gaierror(-2, 'bad host'))"))
        self.assertFalse(reader177.is_pre_send_dns_error("RuntimeError('socket.gaierror in body')"))

    def test_fetch_refuses_changed_existing_input_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "inputs" / "fixture.bin"
            target.parent.mkdir()
            target.write_bytes(b"tampered")
            manifest_path = root / "inputs-manifest.json"
            original = b"original"
            _write_json(manifest_path, {"files": {"inputs/fixture.bin": {
                "url": "https://example.invalid/fixture", "revision": "r1",
                "sha256": hashlib.sha256(original).hexdigest()}}})
            with patch.object(fetch177, "_read_url", side_effect=AssertionError("network must not run")):
                with self.assertRaises(ValueError):
                    fetch177._download("https://example.invalid/fixture", target, "r1", manifest_path,
                                       json.loads(manifest_path.read_text()))

    def test_fetch_missing_manifested_input_keeps_revision_url_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "inputs" / "fixture.bin"
            manifest_path = root / "inputs-manifest.json"
            original = b"pinned"
            manifest = {"files": {"inputs/fixture.bin": {
                "url": "https://example.invalid/fixture", "revision": "r1",
                "sha256": hashlib.sha256(original).hexdigest()}}}
            _write_json(manifest_path, manifest)
            with patch.object(fetch177, "_read_url", return_value=b"changed"):
                with self.assertRaises(ValueError):
                    fetch177._download("https://example.invalid/fixture", target, "r1", manifest_path, manifest)
            with patch.object(fetch177, "_read_url", return_value=original):
                result = fetch177._download("https://example.invalid/fixture", target, "r1", manifest_path, manifest)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(result, manifest["files"]["inputs/fixture.bin"])
            target.unlink()
            with patch.object(fetch177, "_read_url", side_effect=AssertionError("revision mismatch must stop first")):
                with self.assertRaises(ValueError):
                    fetch177._download("https://example.invalid/fixture", target, "r2", manifest_path, manifest)

    def test_dataset_revision_and_test_file_are_pinned(self):
        self.assertEqual(fetch177._dataset_revision_and_file(),
                         ("6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"))

    def test_manifest_requires_every_input_and_read_output_uses_relative_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _make_synthetic_inputs(root)
            files = _manifest_for_inputs(root, inputs)
            del files["inputs/tokenizer.json"]
            _write_json(root / "inputs-manifest.json", {"part": 177, "files": files})
            with self.assertRaisesRegex(ValueError, "tokenizer.json"):
                read177._verify_manifest(inputs)
            _manifest_for_inputs(root, inputs)
            result = read177.compute(inputs, root / "results", tokenizer=FakeTokenizer(), verify_hashes=True)
            self.assertEqual(result["input_manifest"], "inputs-manifest.json")
            self.assertFalse(Path(result["input_manifest"]).is_absolute())


class IntegrationFixtureTests(unittest.TestCase):
    def test_compute_uses_200_fixed_sample_ids_and_writes_blinded_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _make_synthetic_inputs(root)
            result = read177.compute(inputs, root / "results", tokenizer=FakeTokenizer(), verify_hashes=False)
            self.assertEqual(result["counts"]["sample0_per_arm"], 200)
            self.assertEqual(result["counts"]["cohort_C"], 199)
            self.assertEqual(result["counts"]["target_T"], 198)
            self.assertTrue(result["gates_1_to_3"]["gate1"]["passed"])
            self.assertTrue(result["gates_1_to_3"]["gate2"]["passed"])
            self.assertTrue(result["gates_1_to_3"]["gate3"]["passed"])
            self.assertNotIn("arm", read177.read_jsonl(root / "results" / "windows.jsonl")[0])
            self.assertEqual(set(result["donors"]["targets"]), set(result["target_T_ids"]))
            self.assertTrue(all(row["draw_n"] <= 20 for row in result["donors"]["targets"].values()))
            keys = read177.read_jsonl(root / "results" / "window-key.jsonl")
            target = "p002"
            draw_one = result["donors"]["targets"][target]["draws"][0]["gold"]
            self.assertTrue(any(row["problem_id"] == target and row["condition"] == "shuffled" and
                                row["value"] == draw_one for row in keys))
            self.assertTrue(any(row["condition"] == "shuffled" for row in keys))
            populations = result["reported_populations"]
            self.assertIn("p003", populations["looped_C_ids"])
            self.assertIn("p004", populations["merely_long_C_ids"])
            self.assertEqual(populations["correctness_counts_same_500"]["unedited"]["denominator_problems"], 500)
            self.assertEqual(populations["p50c_closed_thinking_on_T"]["numerator_literal_hit"], 1)
            self.assertEqual(populations["p50c_closed_thinking_on_T"]["denominator_closed_traces"], 1)
            self.assertEqual(len(populations["reader_real_hit_selection_on_T"]), 198)
            self.assertTrue(result["trace_results"][0]["capped"])
            self.assertFalse(result["trace_results"][0]["cohort_C"])
            p003 = next(row for row in result["trace_results"] if row["problem_id"] == "p003" and
                        row["arm"] == "unedited")
            p004 = next(row for row in result["trace_results"] if row["problem_id"] == "p004" and
                        row["arm"] == "unedited")
            self.assertTrue(p003["looped"])
            self.assertTrue(p004["merely_long"])
            correctness = score177._paired_correctness(result, reps=20, seed=4)
            self.assertTrue(correctness["same_problem_ids"])
            self.assertEqual(correctness["problems"], 500)
            self.assertEqual(set(correctness["per_sample"]), {"0", "1", "2", "3"})
            self.assertEqual(correctness["any_of_four_cross_table"]["total_rows"], 500)
            labels = [{"id": row["id"], "label": "asserted" if row["condition"] == "real" else "intermediate"}
                      for row in read177.read_jsonl(root / "results" / "window-key.jsonl")]
            reader_readouts = score177._reader_readouts(result,
                read177.read_jsonl(root / "results" / "window-key.jsonl"), labels, reps=20, seed=5)
            self.assertEqual(reader_readouts["first_read_hit_asserted"]["traces_with_read_hit"], 198)
            self.assertEqual(reader_readouts["last_read_hit_asserted"]["traces_with_read_hit"], 198)
            self.assertEqual(set(reader_readouts["asserted_by_C_loop_class"]), {"looped", "merely_long"})
            target_ids = set(result["target_T_ids"])
            p50_closed_target = [row for row in result["trace_results"]
                                 if row["arm"] == "p50c" and row["problem_id"] in target_ids and
                                 row["think_closed"]]
            self.assertEqual(len(p50_closed_target), 1)
            self.assertTrue(p50_closed_target[0]["literal_hit"])

    def test_prompt_echo_excludes_problem_from_fixed_target_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _make_synthetic_inputs(root)
            result = read177.compute(inputs, root / "results", tokenizer=FakeTokenizer(), verify_hashes=False)
            self.assertIn("p001", result["cohort_C_ids"])
            self.assertNotIn("p001", result["target_T_ids"])
            row = next(row for row in result["trace_results"] if row["problem_id"] == "p001" and
                       row["arm"] == "unedited")
            self.assertTrue(row["gold_in_prompt"])
            keys = read177.read_jsonl(root / "results" / "window-key.jsonl")
            self.assertFalse(any(key["problem_id"] == "p001" and key["condition"] == "real" for key in keys))

    def test_failed_data_gate_withholds_windows_and_stops_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _make_synthetic_inputs(root)
            merged = read177.read_json(inputs / "eval" / "unedited" / "merged.json")
            merged["math500"]["acc"] = 0.99
            _write_json(inputs / "eval" / "unedited" / "merged.json", merged)
            out = root / "results"
            result = read177.compute(inputs, out, tokenizer=FakeTokenizer(), verify_hashes=False)
            self.assertFalse(result["gates_1_to_3"]["gate1"]["passed"])
            self.assertEqual(result["reader_windows_status"], "blocked: Gate 1, 2, or 3 failed")
            self.assertFalse((out / "windows.jsonl").exists())
            calls = []
            metadata = reader177.run(out, "ds", label_one_fn=lambda item, reader: calls.append(item))
            self.assertEqual(calls, [])
            self.assertIsNotNone(metadata["stopped"])

    def test_gate3_exact_ninety_percent_literal_recall_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _make_synthetic_inputs(root)
            merged_path = inputs / "eval" / "unedited" / "merged.json"
            merged = read177.read_json(merged_path)
            merged["math500"]["acc"] += 0.0009
            _write_json(merged_path, merged)
            edited_merged_path = inputs / "eval" / "p50c" / "merged.json"
            edited_merged = read177.read_json(edited_merged_path)
            edited_merged["math500"]["cut_share"] = 0.925
            _write_json(edited_merged_path, edited_merged)
            sample_path = inputs / "eval" / "p50c" / "parts" / "math500_s0of10" / \
                "math500_first_samples.jsonl.gz"
            with gzip.open(sample_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            for row in rows:
                row["text"] = "CAP Synthetic text omits the target value."
            _write_jsonl_gz(sample_path, rows)
            result = read177.compute(inputs, root / "results", tokenizer=FakeTokenizer(), verify_hashes=False)
        gate3 = result["gates_1_to_3"]["gate3"]
        self.assertEqual(gate3["observed"], 0.9)
        self.assertTrue(gate3["passed"])
        gate1 = result["gates_1_to_3"]["gate1"]["by_arm"]["unedited"]
        self.assertTrue(gate1["passed"])
        self.assertAlmostEqual(gate1["absolute_difference"], 0.0009)
        gate2 = result["gates_1_to_3"]["gate2"]["by_arm"]["p50c"]
        self.assertTrue(gate2["passed"])
        self.assertAlmostEqual(gate2["absolute_difference"], 0.075)

    def test_score_omits_readouts_when_data_gates_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            _write_json(out / "read.json", {
                "gates_1_to_3": {"gate1": {"passed": False}, "gate2": {
                    "passed": False, "by_arm": {arm: {"numerator": 0, "denominator": 0}
                                                  for arm in ("unedited", "p50c")}},
                    "gate3": {"passed": True, "numerator": 0, "denominator": 0}},
                "trace_results": [], "target_T_ids": [], "gate1_problem_summaries": {}})
            score = score177.score(out, reps=20, seed=4)
            self.assertEqual(score["gate4"]["status"], "not read")
            self.assertIsNone(score["tables"])
            self.assertIsNone(score["reported_readouts"])
            self.assertTrue(all(row["verdict"] == "not read" for row in score["predictions"].values()))

    def test_matched_T_trace_weights_are_equal_despite_different_window_counts(self):
        keys = []
        labels = {}
        traces = []
        for problem_id, real_n, shuf_n in (("a", 3, 1), ("b", 1, 1)):
            traces.append({"problem_id": problem_id, "arm": "unedited", "donor_draw_n": 1})
            for condition, count in (("real", real_n), ("shuffled", shuf_n)):
                for index in range(count):
                    item_id = f"{problem_id}-{condition}-{index}"
                    keys.append({"id": item_id, "problem_id": problem_id, "condition": condition,
                                 "hit_start": index, "target_T": True})
                    label = "asserted" if (problem_id == "a" and condition == "real") or \
                        (problem_id == "b" and condition == "shuffled") else "intermediate"
                    labels[item_id] = {"id": item_id, "label": label}
        by_problem = score177._cluster_by_problem(traces)
        rows = score177._group_trace_assertions(["a", "b"], keys, labels, by_problem)
        estimate = score177._paired_assertion_bounds(rows, reps=25, seed=8)
        self.assertEqual(estimate["denominator"], 2)
        self.assertEqual(estimate["point"], 0.0)

    def test_yes_to_everything_reader_fails_gate4_on_matched_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = reader177.known_examples()
            known = [{"id": row["id"], "label": row["expected"]} for row in examples]
            keys = []
            ds = []
            haiku = []
            traces = []
            for problem_id in ("a", "b"):
                traces.append({"problem_id": problem_id, "arm": "unedited", "donor_draw_n": 1})
                for condition in ("real", "shuffled"):
                    item_id = f"{problem_id}-{condition}"
                    keys.append({"id": item_id, "problem_id": problem_id, "condition": condition,
                                 "target_T": True, "hit_start": 1})
                    ds.append({"id": item_id, "label": "asserted"})
            # The window reader says yes to every value; that produces zero real/shuffled margin.
            sample_ids = ["a-real"]
            haiku = [{"id": "a-real", "label": "asserted"}]
            _write_json(root / "second-reader-sample.json", {"ids": sample_ids})
            read_result = {"target_T_ids": ["a", "b"], "trace_results": traces}
            result = score177._gate4(read_result, keys, ds, haiku, known, examples, {}, root, 20, 3)
            self.assertEqual(result["components"]["known_examples"]["status"], "passed")
            self.assertEqual(result["components"]["asserted_margin_T"]["status"], "failed")
            self.assertEqual(result["components"]["asserted_margin_T"]["difference"], 0.0)

    def test_cross_table_exposes_equal_marginals_with_no_trace_overlap(self):
        rows = [{"hit": True, "correct": False}, {"hit": True, "correct": False},
                {"hit": False, "correct": True}, {"hit": False, "correct": True}]
        table = score177._cross_table(rows, "hit", "correct")
        self.assertEqual(table["cells_left_then_right"], {"0,0": 0, "0,1": 2, "1,0": 2, "1,1": 0})
        self.assertEqual(sum(row["hit"] for row in rows), sum(row["correct"] for row in rows))

    def test_prediction_table_exercises_held_failed_and_not_read_rows(self):
        held_result, held_keys, held_labels = _prediction_fixture(held=True, gates_pass=True)
        held = score177._make_predictions(held_result, score177._cluster_by_problem(held_result["trace_results"]),
                                          held_keys, held_labels, {}, {"status": "passed"}, True, 30, 177)
        failed_result, failed_keys, failed_labels = _prediction_fixture(held=False, gates_pass=True)
        failed = score177._make_predictions(failed_result, score177._cluster_by_problem(failed_result["trace_results"]),
                                            failed_keys, failed_labels, {}, {"status": "passed"}, True, 30, 177)
        blocked_result, blocked_keys, blocked_labels = _prediction_fixture(held=True, gates_pass=False)
        blocked = score177._make_predictions(blocked_result,
                                             score177._cluster_by_problem(blocked_result["trace_results"]),
                                             blocked_keys, blocked_labels, {}, {"status": "passed"}, True, 30, 177)
        expected = {"177.1", "177.2", "177.3", "177.4", "177.5", "177.6",
                    "177.W1", "177.W2", "177.W3"}
        self.assertEqual(set(held), expected)
        self.assertEqual(set(failed), expected)
        self.assertEqual(set(blocked), expected)
        self.assertEqual(held["177.1"]["verdict"], "held")
        self.assertEqual(failed["177.1"]["verdict"], "failed")
        self.assertEqual(held["177.W1"]["estimate"]["late_C_n"], 25)
        self.assertEqual(held["177.W2"]["estimate"]["total_C_windows"], 24)
        self.assertEqual(held["177.W3"]["verdict"], "held")
        self.assertEqual(held["177.W3"]["estimate"]["reading_a_box_fails_correct0"]["rate"], 1 / 3)
        self.assertTrue(all(row["verdict"] == "not read" for row in blocked.values()))
        self.assertEqual(score177.READING_W3, "a")

    def test_failed_177_1_is_marked_as_blocking_kept_going_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            read_result, keys, labels = _prediction_fixture(held=False, gates_pass=True)
            for arm in ("unedited", "p50c"):
                read_result["gates_1_to_3"]["gate2"] = {"passed": True, "by_arm": {
                    name: {"numerator": 0, "denominator": 30} for name in ("unedited", "p50c")}}
                read_result["gates_1_to_3"]["gate1"] = {"passed": True}
                read_result["gates_1_to_3"]["gate3"] = {"passed": True, "numerator": 0, "denominator": 0}
                read_result["gate1_problem_summaries"] = {
                    name: [{"problem_id": f"t{i:02d}", "correct_n": 0, "sample_n": 4,
                            "correct": [0, 0, 0, 0], "any_correct": 0} for i in range(30)]
                    for name in ("unedited", "p50c")}
            _write_json(out / "read.json", read_result)
            _write_jsonl(out / "window-key.jsonl", keys)
            _write_jsonl(out / "windows.jsonl", [{"id": row["id"]} for row in keys])
            _write_jsonl(out / "labels-ds.jsonl", list(labels.values()))
            _write_jsonl(out / "labels-haiku.jsonl", [])
            _write_jsonl(out / "known-labels-ds.jsonl", [])
            with patch.object(score177, "_gate4", return_value={"status": "passed", "passed": True}):
                result = score177.score(out, reps=20, seed=5)
        self.assertEqual(result["predictions"]["177.1"]["verdict"], "failed")
        self.assertTrue(result["kept_going_reading_blocked"])

    def test_reader_gate_margin_uses_bounds_for_missing_yes_no_labels(self):
        rows = [{"real": {"state": 1}, "shuffled": {"state": None}},
                {"real": {"state": 1}, "shuffled": {"state": 0}}]
        bounds = score177._paired_assertion_bounds(rows, reps=10, seed=2)
        self.assertEqual(bounds["bounds"], [0.5, 1.0])
        self.assertIsNone(bounds["point"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
