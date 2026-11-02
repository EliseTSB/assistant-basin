#!/usr/bin/env python3
"""Synthetic-only tests for Part 177 definitions, readers, gates and predictions."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
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

UPSTREAM_MATHQA = Path("/home/elisepc/green/Journal/notes/reports/part177-review/upstream/mathqa.py")


class FakeTokenizer:
    def encode(self, text):
        class Encoded:
            pass
        result = Encoded()
        result.ids = range(32_500 if "CAP" in text else 12)
        return result


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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
                suffix = " </think>" if (arm == "unedited" and index == 0) or \
                    (arm == "p50c" and index == 2) else ""
                if arm == "unedited" and index == 3:
                    text = ("CAP\nWe apply the same arithmetic identity.\n" * 4 +
                            f"The result is {gold}. Continue checking.")
                else:
                    text = f"CAP Start with the result {gold}. Continue checking. {suffix}"
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
                    "correct0": 0, "pred_first": "wrong" if i < 15 else None,
                    "pred_first_nonnull": i < 15, "box_cell": "pred_first wrong box" if i < 15 else "pred_first null",
                    "earlier_gold_box_last_differs": i == 0, "donor_draw_n": 1}
        p50c = {"problem_id": problem_id, "arm": "p50c", "cohort_C": False,
                "capped": True, "think_closed": False, "text_chars": 60_000,
                "loop_onset": onset_p, "loop_onset_50k": onset_p,
                "correct0": 0, "pred_first": "wrong" if i < 15 else None}
        traces.extend((unedited, p50c))
        real = {"id": f"r{i:02d}", "problem_id": problem_id, "condition": "real",
                "arm": "unedited", "target_T": True, "hit_start": 100 if early else 20_000}
        keys.append(real)
        if held:
            # Draw 1 hits only on the 24 non-early traces; gives a 0.20 real-minus-shuffled hit gap.
            if not early:
                shuffled = {"id": f"s{i:02d}", "problem_id": problem_id, "condition": "shuffled",
                            "arm": "unedited", "target_T": True, "hit_start": 100}
                keys.append(shuffled)
                labels[shuffled["id"]] = {"id": shuffled["id"], "label": "intermediate"}
            labels[real["id"]] = {"id": real["id"], "label": "asserted"}
        else:
            shuffled = {"id": f"s{i:02d}", "problem_id": problem_id, "condition": "shuffled",
                        "arm": "unedited", "target_T": True, "hit_start": 100}
            keys.append(shuffled)
            labels[real["id"]] = {"id": real["id"], "label": "intermediate"}
            labels[shuffled["id"]] = {"id": shuffled["id"], "label": "asserted"}
    gates = {name: {"passed": gates_pass} for name in ("gate1", "gate2", "gate3")}
    read_result = {"gates_1_to_3": gates, "target_T_ids": [f"t{i:02d}" for i in range(n)],
                   "trace_results": traces}
    return read_result, keys, labels


class DefinitionTests(unittest.TestCase):
    def test_upstream_loop_signals_function_is_ast_verbatim(self):
        upstream = ast.parse(UPSTREAM_MATHQA.read_text(encoding="utf-8"))
        copied = ast.parse((read177.HERE / "loop_upstream.py").read_text(encoding="utf-8"))
        get_fn = lambda tree: next(node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and node.name == "loop_signals")
        self.assertEqual(ast.dump(get_fn(upstream), include_attributes=False),
                         ast.dump(get_fn(copied), include_attributes=False))
        self.assertNotIn("math_verify", (read177.HERE / "loop_upstream.py").read_text(encoding="utf-8"))

    def test_line_rule_four_honest_case_lines_and_one_line_repeat(self):
        honest = "We apply the same arithmetic identity.\n" * 4
        one_line = "A one line repeating trace stays here.\n" * 4
        self.assertEqual(read177.loop_onset(honest)["line_index"], 0)
        self.assertEqual(read177.loop_onset(one_line)["line_index"], 0)

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

    def test_latex_fraction_normalization_handles_dfrac_left_right_and_spacing(self):
        text = r"Value: $\left\dfrac {1} {2}\right\!.$"
        self.assertTrue(read177.has_literal(text, r"\frac{1}{2}"))
        self.assertTrue(read177.has_literal(r"It is \frac{1}{2}.", r"\dfrac{1}{2}"))

    def test_prompt_echo_is_detected_for_T_exclusion(self):
        self.assertTrue(read177.has_literal("The question explicitly gives 7.", "7"))

    def test_no_onset_hit_has_separate_order_value(self):
        self.assertEqual(read177.onset_order(10, None), "no onset detected")

    def test_null_pred_first_and_earlier_correct_box_case(self):
        text = r"First we found \boxed{5}, but the final box is \boxed{7}."
        self.assertEqual(read177.box_cell(0, None), "pred_first null")
        self.assertTrue(read177.earlier_gold_box_last_differs(text, "5"))

    def test_cap_can_coexist_with_closed_thinking(self):
        row = read177._trace_record("x", "unedited", {"text": "CAP \boxed{5} </think>"},
                                   {"correct": [0, 0, 0, 0], "pred_first": None},
                                   "What is the result?", "5", FakeTokenizer())
        self.assertTrue(row["capped"])
        self.assertTrue(row["think_closed"])
        self.assertFalse(row["cohort_C"])

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
        self.assertTrue(all(row["verdict"] == "not read" for row in blocked.values()))
        self.assertEqual(score177.READING_W3, "a")

    def test_reader_gate_margin_uses_bounds_for_missing_yes_no_labels(self):
        rows = [{"real": {"state": 1}, "shuffled": {"state": None}},
                {"real": {"state": 1}, "shuffled": {"state": 0}}]
        bounds = score177._paired_assertion_bounds(rows, reps=10, seed=2)
        self.assertEqual(bounds["bounds"], [0.5, 1.0])
        self.assertIsNone(bounds["point"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
