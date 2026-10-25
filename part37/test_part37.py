#!/usr/bin/env python3
"""Part 37 pure-logic tests and optional tiny torch-model test."""
from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze37
import collect37

try:
    import torch
    from torch import nn
except ImportError as exc:  # the host may not have the Windows GPU environment packages
    torch = None
    nn = None
    print(f"SKIP Part 37 tiny-model tests: torch is not installed ({exc})", file=sys.stderr)


class CharacterTokenizer:
    """A reversible offset-aware tokenizer for deterministic prompt/record tests."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def __init__(self):
        self.padding_side = "right"
        self.last_template_kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True, **kwargs: Any) -> str:
        self.last_template_kwargs = kwargs
        rendered = "".join(f"<{message['role']}>" + message["content"] + f"</{message['role']}>"
                           for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = False) -> str:
        return "".join(chr(int(token)) for token in token_ids)

    def __call__(self, text: str, add_special_tokens: bool = False,
                 return_offsets_mapping: bool = False, return_tensors: str | None = None,
                 **kwargs: Any) -> dict[str, Any]:
        ids = self.encode(text)
        result: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        if return_tensors == "pt" and torch is not None:
            result = {key: torch.tensor([value]) for key, value in result.items()}
        return result


class AnswerTokenizer(CharacterTokenizer):
    token_ids = {
        "Yes": 1001, "yes": 1002, " Yes": 1003, " yes": 1004,
        "No": 1005, "no": 1006, " No": 1007, " no": 1008,
        "A": 1009, " A": 1010, "B": 1011, " B": 1012,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.token_ids:
            return [self.token_ids[text]]
        return super().encode(text, add_special_tokens)


def generation_fixture() -> list[dict[str, Any]]:
    qid = "q000"
    own = "The gentle breeze makes the sunny afternoon feel calm and pleasant."
    other = "A walk through the garden can make a quiet morning feel pleasant."
    return [
        {"question_id": qid, "model": "qwen", "kind": "greedy", "text": own},
        {"question_id": qid, "model": "qwen", "kind": "s1", "text": "short"},
        {"question_id": qid, "model": "qwen", "kind": "s2", "text": own},
        {"question_id": qid, "model": "olmo", "kind": "greedy", "text": other},
        {"question_id": qid, "model": "smol", "kind": "greedy", "text": other},
    ]


class ArrayFeatureStore:
    def __init__(self, rows: list[dict[str, Any]], features: dict[str, Any], layers: int):
        self.rows = rows
        self.features = features
        self.layers = layers
        self.by_id = {row["record_id"]: row for row in rows}

    def feature(self, row: dict[str, Any], position: int, layer: int):
        return self.features[str(row["record_id"])][position, layer].astype("float32", copy=False)

    def metrics(self, row: dict[str, Any]) -> dict[str, Any]:
        return row["metrics"]


def _prediction_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    layer_result = {"layer": 0, "auroc": 0.75, "fold_balanced_accuracy": [0.84, 0.83, 0.82, 0.81, 0.55]}
    qwen = {
        "probes": {
            "role_pooled": {"span_last": {"layers": [
                {"layer": 0, "auroc": 0.50}, {"layer": 1, "auroc": 0.92}, {"layer": 2, "auroc": 0.52}]}},
            "authorship_assistant_primary": {"span_last": {
                "best_auroc": 0.75, "mean_logprob_regressed": {"best_auroc": 0.68},
                "label_shuffled_control": {"max_layer_auroc_95th": 0.60},
                "best_fold_balanced_accuracy": 0.81, "best_layer": 2,
                "layers": [deepcopy(layer_result)]}},
            "authorship_within_slot": {
                "assistant": {"span_last": {"best_balanced_accuracy": 0.66}},
                "user": {"span_last": {"best_balanced_accuracy": 0.70}},
            },
        },
        "c3_projection": {"label_to_author_absolute_effect_ratio": 0.30},
        "own_g_vs_own_sampled_projection": {"mean_logprob_regressed_shift_sd": 0.62},
        "authorship_direction": {"role_direction_cosine_at_regressed_best_layer": 0.60,
                                 "regressed_probe_best_layer": 1, "37_8_cosine_layer": 1},
        "stated": {
            "authorship_yes_spearman": {"spearman_rho": 0.50},
            "null_authorship_readout": {
                "mean_authorship_yes_probability": 0.20, "share_above_0_5": 0.10,
                "median_total_two_token_mass": 0.80, "scored_kinds": ["null_smol", "null_pile"],
                "self_text_null_shuffle": {"n": 1, "mean_p_yes": 0.64}},
            "header_effects_self_minus_other": {"authorship_yn": 0.10, "memory_yn": 0.32},
            "neutral_between_self_and_other": True, "header_by_form": {},
        },
    }
    olmo = {"probes": {"role_pooled": {"span_last": {"layers": [
        {"layer": 0, "auroc": 0.50}, {"layer": 1, "auroc": 0.93}, {"layer": 2, "auroc": 0.51}]}}}}
    return qwen, olmo


def _write_synthetic_capture(path: Path, seed: int, strong_authorship: bool = False) -> None:
    import numpy as np
    rng = np.random.default_rng(seed)
    index_rows, metrics_rows, features = [], [], []
    for question_number in range(20):
        qid = f"q{question_number:03d}"
        for kind in ("own_g", "other_g", "own_s1", "own_s2", "own_diffq"):
            record_id = f"{qid}:{kind}"
            author = "other" if kind == "other_g" else "own"
            state = rng.normal(size=(2, 1, 8)).astype("float32")
            if strong_authorship and kind in ("own_g", "other_g"):
                state[:, 0, 0] += 2.8 if kind == "own_g" else -2.8
            features.append(state)
            index_rows.append({"record_id": record_id, "question_id": qid, "span_kind": kind,
                               "author": author, "slot": "assistant", "frame": "base", "span": "x",
                               "shard": "noise.npy", "shard_row": len(features) - 1})
            metrics_rows.append({"record_id": record_id, "question_id": qid, "mean_logprob": 0.0,
                                 "minimum_logprob": -1.0, "mean_entropy": 1.0,
                                 "last_token_logprob": -0.5})
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "noise.npy", np.stack(features), allow_pickle=False)
    collect37.write_jsonl(path / "index.jsonl", index_rows)
    (path / "manifest.json").write_text(json.dumps({"captured_records": len(index_rows)}), encoding="utf-8")
    collect37.write_jsonl(path / "metrics.jsonl", metrics_rows)


class Part37LogicTests(unittest.TestCase):
    def test_question_set_has_exact_mix_order_and_exclusions(self):
        questions = collect37.read_questions(HERE / "questions.json")
        self.assertEqual([row["id"] for row in questions], [f"q{index:03d}" for index in range(100)])
        self.assertEqual([row["category"] for row in questions].count("opinion"), 35)
        self.assertEqual([row["category"] for row in questions].count("decision"), 35)
        self.assertEqual([row["category"] for row in questions].count("explanation"), 30)
        excluded = collect37.IDENTITY_TERMS + [
            "AI", "computer", "chatbot", "name", "names", "company", "companies", "authorship",
            "write", "writing", "notes", "memory", "politics", "religion", "current events",
            "medical", "legal", "law", "reader", "yourself",
        ]
        for row in questions:
            for term in excluded:
                self.assertIsNone(collect37._match_pattern(term).search(row["question"]),
                                  f"excluded term {term!r} in {row['id']}: {row['question']}")

    def test_identity_screen_drops_question_for_any_matching_answer(self):
        rows = [
            {"question_id": "q000", "text": "A quiet walk is pleasant."},
            {"question_id": "q001", "text": "I am a language model."},
            {"question_id": "q000", "text": "A different answer is still clear."},
            {"question_id": "q002", "text": "This came from CHATGPT."},
        ]
        screen = collect37.screen_rows(rows)
        self.assertEqual(screen["kept"], ["q000"])
        self.assertEqual(screen["dropped"], [
            {"question_id": "q001", "matching_words": ["language model"]},
            {"question_id": "q002", "matching_words": ["ChatGPT"]},
        ])
        self.assertIn("as an AI", screen["identity_terms"])

    def test_nonthinking_prompt_configuration(self):
        tokenizer = CharacterTokenizer()
        smol_prompt = collect37.render_generation_prompt(tokenizer, "smol", "Why do leaves change color?")
        self.assertIn("/no_think", smol_prompt)
        self.assertNotIn("<think>", smol_prompt)
        collect37.render_generation_prompt(tokenizer, "qwen", "Why do leaves change color?")
        self.assertIs(tokenizer.last_template_kwargs.get("enable_thinking"), False)

    def test_single_token_answer_sets_are_disjoint(self):
        sets = collect37.answer_token_sets(AnswerTokenizer())
        self.assertEqual(sets["yes"], [1001, 1002, 1003, 1004])
        self.assertEqual(sets["no"], [1005, 1006, 1007, 1008])
        self.assertEqual(sets["a"], [1009, 1010])
        self.assertEqual(sets["b"], [1011, 1012])

    def test_span_and_followup_readout_positions_for_both_readers(self):
        tokenizer = CharacterTokenizer()
        record = {
            "record_id": "q000:test", "span": "A pleasant walk can make the morning feel calm.",
            "messages": [
                {"role": "user", "content": "How should I start the day?"},
                {"role": "assistant", "content": "A pleasant walk can make the morning feel calm."},
                {"role": "user", "content": collect37.FOLLOWUP},
            ],
        }
        for reader in ("qwen", "olmo"):
            result = collect37.locate_readout_positions(record, tokenizer, reader)
            self.assertEqual(tokenizer.decode([result["input_ids"][result["span_last_index"]]]), ".")
            self.assertEqual(tokenizer.decode([result["input_ids"][result["followup_first_index"]]]), "T")
            self.assertEqual(result["span_token_positions"][-1], result["span_last_index"])

    def test_assistant_user_and_label_frames_use_the_registered_turn_order(self):
        question, span = "Should I take a short walk?", "A brief walk can be refreshing."
        assistant = collect37._record_messages(question, span, "assistant", "base")
        user = collect37._record_messages(question, span, "user", "base")
        label_assistant = collect37._record_messages(
            question, span, "assistant", "label", "Your earlier notes")
        label_user = collect37._record_messages(question, span, "user", "label", "Your earlier notes")
        self.assertEqual([row["role"] for row in assistant], ["user", "assistant", "user"])
        self.assertEqual(assistant[0]["content"], question)
        self.assertEqual(assistant[1]["content"], span)
        self.assertEqual(user[0]["content"], f"{question}\n\n{collect37.FOLLOWUP_PREFIX}\n\n{span}")
        self.assertEqual(user[1]["content"], "Noted.")
        self.assertEqual(label_assistant[0]["content"], "Please show me the notes.")
        self.assertEqual(label_assistant[1]["content"], f"Your earlier notes:\n\n{span}")
        self.assertEqual(label_user[0]["content"], f"Your earlier notes:\n\n{span}")
        self.assertEqual(label_user[1]["content"], "Noted.")
        self.assertEqual(assistant[-1]["content"], collect37.FOLLOWUP)

    def test_token_truncation_and_retokenization_mismatch_are_recorded(self):
        tokenizer = CharacterTokenizer()
        text, count, mismatch = collect37._first_tokens(tokenizer, "abcdef", 4)
        self.assertEqual(text, "abcd")
        self.assertEqual(count, 4)
        self.assertEqual(mismatch, 0)
        text, count, mismatch = collect37._first_tokens(tokenizer, "abc", 4)
        self.assertEqual(text, "")
        self.assertEqual(count, 3)
        self.assertEqual(mismatch, -1)

    def test_record_expansion_has_four_base_role_author_cells_and_32_label_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pile = root / "pile.jsonl"
            pile.write_text(json.dumps({"text": "".join(["ordinary " for _ in range(1000)])}) + "\n",
                               encoding="utf-8")
            questions = [{"id": "q000", "question": "Is a quiet morning pleasant?"}]
            records, summary = collect37.build_records(
                "qwen", generation_fixture(), {"kept": ["q000"]}, questions, pile, CharacterTokenizer())
        self.assertEqual(len(records), 43)
        base = [row for row in records if row["frame"] == "base"]
        labels = [row for row in records if row["frame"] == "label"]
        self.assertEqual(len(base), 11)
        self.assertEqual(len(labels), 32)
        self.assertEqual(len([row for row in base if row["span_kind"] in ("own_g", "other_g")]), 4)
        self.assertEqual(sum(bool(row["stated_include"]) for row in records), 41)
        edit = next(row for row in base if row["span_kind"] == "edit")
        self.assertTrue(edit["span"])  # a word swap that changes the token count must not empty the edit cell
        short = next(row for row in base if row["span_kind"] == "own_s1")
        self.assertEqual(short["span"], "")
        self.assertEqual(short["empty_reason"], "span shorter than L")
        self.assertEqual(summary["empty_span_records"], 1)

    def test_minimal_edit_logs_a_similar_length_replacement(self):
        original = "A gentle breeze makes the sunny afternoon feel calm."
        changed, pair = collect37.minimal_edit(original)
        self.assertIsNotNone(pair)
        self.assertNotEqual(changed, original)
        self.assertLessEqual(abs(len(pair["original"]) - len(pair["replacement"])), 1)
        self.assertIn(pair["original"], original)
        samples = [
            "Toast it lightly, then add butter while it is still warm so it melts evenly.",
            "A gentle breeze makes the sunny afternoon feel calm and pleasant.",
            "I usually walk to the corner shop because it is close and the weather is fine.",
            "Bread goes stale because the starch slowly turns back into a firm crystal.",
            "Keep the milk in the fridge and check the date before you pour it.",
            "Reading before bed helps me fall asleep faster than looking at a screen.",
            "The best way to learn a recipe is to cook it twice in the same week.",
            "Take the stairs when you can, it is good for your legs and your mood.",
        ]
        for answer in samples:
            edited, sample_pair = collect37.minimal_edit(answer)
            self.assertIsNotNone(sample_pair, answer)
            self.assertNotEqual(edited, answer)
            self.assertNotIn(sample_pair["replacement"].casefold(), collect37.EDIT_STOPWORDS, answer)
            self.assertIn(sample_pair["replacement"].casefold(), collect37.EDIT_WORDS, answer)

    def test_37_9_pairing_requires_both_nonempty_samples_and_reports_cell_drops(self):
        sampled = [
            {"question_id": "q0", "span_kind": kind}
            for kind in ("own_g", "own_s1", "own_s2")
        ] + [
            {"question_id": "q1", "span_kind": "own_g"},
            {"question_id": "q1", "span_kind": "own_s1"},
            {"question_id": "q2", "span_kind": "own_g"},
            {"question_id": "q2", "span_kind": "own_s2"},
        ]
        baseline = [
            {"question_id": f"q{i}", "span_kind": kind}
            for i in range(3) for kind in ("own_g", "other_g")
        ]
        paired, counts = analyze37._paired_sample_questions(sampled, baseline, {"q0", "q1", "q2"})
        self.assertEqual(paired, {"q0"})
        for kind in ("own_g", "own_s1", "own_s2", "other_g"):
            self.assertEqual(counts[kind]["n_included"], 1)
            self.assertEqual(counts[kind]["n_dropped_total"], 2)
        self.assertEqual(counts["own_s1"]["n_present_but_dropped_pairwise"], 1)
        self.assertEqual(counts["own_s2"]["n_present_but_dropped_pairwise"], 1)

    def test_each_scored_prediction_has_pass_and_fail_scoring_fixtures(self):
        mutations = {
            "37.1": lambda q, o: o["probes"]["role_pooled"]["span_last"]["layers"][1].update(auroc=0.89),
            "37.2": lambda q, o: q["probes"]["authorship_assistant_primary"]["span_last"].update(best_auroc=0.70),
            "37.3": lambda q, o: q["c3_projection"].update(label_to_author_absolute_effect_ratio=0.50),
            "37.4": lambda q, o: q["stated"]["authorship_yes_spearman"].update(spearman_rho=0.30),
            "37.8": lambda q, o: q["authorship_direction"].update(role_direction_cosine_at_regressed_best_layer=0.49),
            "37.9": lambda q, o: q["own_g_vs_own_sampled_projection"].update(mean_logprob_regressed_shift_sd=0.49),
            "37.11": lambda q, o: q["stated"]["header_effects_self_minus_other"].update(
                memory_yn=-0.60, authorship_yn=0.40),
            "37.12": lambda q, o: q["probes"]["authorship_assistant_primary"]["span_last"]["layers"][0].update(
                fold_balanced_accuracy=[0.79, 0.79, 0.79, 0.79, 0.90]),
            "37.13": lambda q, o: q["probes"]["authorship_within_slot"]["user"]["span_last"].update(
                best_balanced_accuracy=0.76),
            "37.14": lambda q, o: q["probes"]["authorship_assistant_primary"]["span_last"][
                "mean_logprob_regressed"].update(best_auroc=0.60),
        }
        pass_setups = {
            # The registered 37.2 and 37.8 regions are intentionally disjoint.
            "37.2": lambda q, o: q["probes"]["authorship_assistant_primary"]["span_last"][
                "mean_logprob_regressed"].update(best_auroc=0.55),
        }
        for prediction, mutate in mutations.items():
            with self.subTest(prediction=prediction):
                qwen, olmo = _prediction_fixture()
                if prediction in pass_setups:
                    pass_setups[prediction](qwen, olmo)
                self.assertEqual(analyze37._prediction_results(qwen, olmo)[prediction]["verdict"], "PASS")
                mutate(qwen, olmo)
                self.assertEqual(analyze37._prediction_results(qwen, olmo)[prediction]["verdict"], "FAIL")
        # 37.9 is signed: a large shift the wrong way must FAIL (DS round 2 H2: 0.49 cannot tell signed from absolute).
        qwen, olmo = _prediction_fixture()
        qwen["own_g_vs_own_sampled_projection"].update(mean_logprob_regressed_shift_sd=-0.60)
        self.assertEqual(analyze37._prediction_results(qwen, olmo)["37.9"]["verdict"], "FAIL")

        qwen, olmo = _prediction_fixture()
        self.assertEqual(analyze37._prediction_results(qwen, olmo)["37.15"]["verdict"], "PASS")
        null_rows = [
            {"form": "authorship_yn", "frame": "base", "span_kind": kind,
             "p_yes": value, "total_two_token_mass": 0.9}
            for kind, value in (("null_smol", 0.02), ("null_pile", 0.03), ("null_shuffle", 0.64))
        ]
        summary = analyze37._null_readout_summary(null_rows)
        self.assertAlmostEqual(summary["mean_authorship_yes_probability"], 0.025)
        self.assertAlmostEqual(summary["self_text_null_shuffle"]["mean_p_yes"], 0.64)
        self.assertAlmostEqual(summary["all_three_null_kinds_mean_for_context_only"], 0.23)
        qwen["stated"]["null_authorship_readout"] = summary
        self.assertEqual(analyze37._prediction_results(qwen, olmo)["37.15"]["verdict"], "FAIL")

    def test_crossfit_projections_block_noise_fit_leak_for_37_9_and_37_4(self):
        import numpy as np
        from scipy.stats import spearmanr
        rng = np.random.default_rng(37)
        rows, features = [], {}
        for question in range(100):
            qid = f"q{question:03d}"
            base_kinds = ("own_g", "other_g", "own_s1", "own_s2", "own_diffq", "edit",
                          "null_smol", "null_pile", "null_shuffle")
            for kind in base_kinds:
                record_id = f"{qid}:{kind}"
                row = {"record_id": record_id, "question_id": qid, "span_kind": kind,
                       "author": "other" if kind == "other_g" else "own", "slot": "assistant",
                       "frame": "base", "span": "noise", "metrics": {
                           "mean_logprob": float(rng.normal()), "minimum_logprob": -1.0,
                           "mean_entropy": 1.0, "last_token_logprob": -0.5}}
                rows.append(row)
                features[record_id] = rng.normal(size=(2, 1, 512)).astype(np.float32)
            for header in ("self", "other", "neutral", "user"):
                for kind in ("own_g", "other_g"):
                    for slot in ("assistant", "user"):
                        record_id = f"{qid}:label:{header}:{slot}:{kind}"
                        rows.append({"record_id": record_id, "question_id": qid, "span_kind": kind,
                                     "author": "other" if kind == "other_g" else "own", "slot": slot,
                                     "frame": "label", "header_group": header, "span": "noise",
                                     "metrics": {"mean_logprob": float(rng.normal()), "minimum_logprob": -1.0,
                                                 "mean_entropy": 1.0, "last_token_logprob": -0.5}})
                        features[record_id] = rng.normal(size=(2, 1, 512)).astype(np.float32)
        store = ArrayFeatureStore(rows, features, layers=1)
        contrast = analyze37._label([row for row in rows if row["frame"] == "base"
                                     and row["span_kind"] in ("own_g", "other_g")],
                                    lambda row: row["span_kind"] == "own_g")
        heldout = analyze37._crossfit_projections(store, contrast, rows, 0, 0, "regressed")
        heldout_raw = analyze37._crossfit_projections(store, contrast, rows, 0, 0, "raw")
        self.assertEqual(len(heldout), len(rows))
        self.assertTrue(any(row["span_kind"] == "null_smol" for row in rows))
        self.assertTrue(any(row["span_kind"] == "edit" for row in rows))
        self.assertTrue(any(row["frame"] == "label" for row in rows))
        paired, _ = analyze37._paired_sample_questions(
            [row for row in rows if row["frame"] == "base" and row["span_kind"] in ("own_g", "own_s1", "own_s2")],
            [row for row in rows if row["frame"] == "base" and row["span_kind"] in ("own_g", "other_g")],
            {f"q{i:03d}" for i in range(100)})
        by_cell = {kind: [row for row in rows if row["frame"] == "base" and row["span_kind"] == kind
                          and row["question_id"] in paired]
                   for kind in ("own_g", "other_g", "own_s1", "own_s2")}
        baseline = [heldout_raw[row["record_id"]] for kind in ("own_g", "other_g") for row in by_cell[kind]]
        shift = (np.mean([heldout_raw[row["record_id"]] for row in by_cell["own_g"]]) -
                 np.mean([heldout_raw[row["record_id"]] for kind in ("own_s1", "own_s2") for row in by_cell[kind]])) / np.std(baseline, ddof=1)
        self.assertLess(abs(float(shift)), 0.5)
        baseline_reg = [heldout[row["record_id"]] for kind in ("own_g", "other_g") for row in by_cell[kind]]
        reg_shift = (np.mean([heldout[row["record_id"]] for row in by_cell["own_g"]]) -
                     np.mean([heldout[row["record_id"]] for kind in ("own_s1", "own_s2") for row in by_cell[kind]])) / np.std(baseline_reg, ddof=1)
        self.assertLess(abs(float(reg_shift)), 0.5)

        primary = [row for row in rows if row["frame"] == "base" and row["span_kind"] in ("own_g", "other_g")]
        label_probabilities = [0.9 if row["span_kind"] == "own_g" else 0.1 for row in primary]
        heldout_rho = abs(float(spearmanr(label_probabilities,
                                          [heldout[row["record_id"]] for row in primary]).statistic))
        self.assertLess(heldout_rho, 0.3)

        legacy_direction = analyze37._fit_direction(store, contrast, 0, 0, "regressed")
        legacy = {row["record_id"]: analyze37._project(store, row, legacy_direction) for row in rows}
        legacy_baseline = [legacy[row["record_id"]] for kind in ("own_g", "other_g") for row in by_cell[kind]]
        legacy_shift = (np.mean([legacy[row["record_id"]] for row in by_cell["own_g"]]) -
                        np.mean([legacy[row["record_id"]] for kind in ("own_s1", "own_s2") for row in by_cell[kind]])) / np.std(legacy_baseline, ddof=1)
        self.assertGreater(float(legacy_shift), 0.5, "the pre-Amendment-2 all-question projection should show the known leak")

    def test_analysis_notes_name_sanity_check_and_preserved_frame_prompts(self):
        notes = " ".join(analyze37.ANALYSIS_NOTES).casefold()
        self.assertIn("sanity check", notes)
        self.assertIn("cannot fail by construction", notes)
        self.assertIn("user-slot frame is unchanged", notes)
        self.assertIn("smollm3 system prompt is unchanged", notes)
        self.assertIn("out-of-fold", notes)

    def test_37_8_cosine_uses_the_regressed_authorship_best_layer(self):
        calls = []

        def fit(store, rows, position, layer, mode):
            calls.append((position, layer, mode))
            weights = [1.0, 0.0] if mode == "regressed" else [0.6, 0.8]
            return {"layer": layer, "weights": weights}

        with patch.object(analyze37, "_fit_direction", side_effect=fit):
            layer, cosine = analyze37._role_cosine_at_regressed_best_layer(
                object(), [], [], {"best_layer": 1}, position=0)
        self.assertEqual(layer, 1)
        self.assertAlmostEqual(cosine, 0.6)
        self.assertEqual(calls, [(0, 1, "regressed"), (0, 1, "raw")])

    def test_atomic_writer_preserves_old_file_if_a_write_is_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "generation.jsonl"
            target.write_text('{"question_id":"q000"}\n', encoding="utf-8")

            def interrupted(stream):
                stream.write('{"question_id":"q001"}\n')
                raise OSError("simulated stop before atomic replace")

            with self.assertRaises(OSError):
                collect37._atomic_write(target, interrupted)
            self.assertEqual(collect37.read_jsonl(target), [{"question_id": "q000"}])

    def test_capture_jsonl_reader_uses_a_loadable_prefix_after_a_torn_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.jsonl"
            path.write_bytes(b'{"record_id":"complete"}\n{"record_id":')
            self.assertEqual(analyze37._read_capture_jsonl(path), [{"record_id": "complete"}])

    def test_l2_penalty_is_left_at_scikit_learns_non_deprecated_default(self):
        source = inspect.getsource(analyze37._fold_fit) + inspect.getsource(analyze37._fit_direction)
        self.assertNotIn("penalty=", source)


class PilotGateTests(unittest.TestCase):
    def test_pilot_noise_stops_on_at_least_two_of_three_seeds_and_signal_continues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            noise_stops = []
            for seed in (1, 2, 3):
                capture = root / f"noise-{seed}"
                _write_synthetic_capture(capture, seed)
                noise_stops.append(analyze37._pilot_reader("qwen", capture)["stop_gate"][
                    "stop_and_report_surprisal_only"])
            self.assertGreaterEqual(sum(noise_stops), 2, noise_stops)
            signal = root / "signal"
            _write_synthetic_capture(signal, 91, strong_authorship=True)
            signal_result = analyze37._pilot_reader("qwen", signal)
            self.assertFalse(signal_result["stop_gate"]["stop_and_report_surprisal_only"])

    def test_pilot_gate_fails_closed_for_missing_layer_or_null_and_main_returns_two(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            _write_synthetic_capture(capture, 9)
            good_score = {"layers": [{"layer": 0, "auroc": 0.5}], "best_auroc": 0.5}
            null_outputs = [
                {"max_layer_auroc_95th": 0.6, "per_layer_auroc_95th": [None]},
                {"max_layer_auroc_95th": 0.6},
                {"max_layer_auroc_95th": 0.6, "per_layer_auroc_95th": [float("nan")]},
            ]
            for null_output in null_outputs:
                with patch.object(analyze37, "run_probe", return_value=good_score), \
                     patch.object(analyze37, "_permutation_threshold", return_value=null_output):
                    with self.assertRaises(analyze37.AnalysisError):
                        analyze37._pilot_reader("qwen", capture)

            missing_layer = {"layers": [], "best_auroc": None}
            valid_null = {"max_layer_auroc_95th": 0.6, "per_layer_auroc_95th": [0.6]}
            with patch.object(analyze37, "run_probe", return_value=missing_layer), \
                 patch.object(analyze37, "_permutation_threshold", return_value=valid_null):
                with self.assertRaises(analyze37.AnalysisError):
                    analyze37._pilot_reader("qwen", capture)
            with patch.object(analyze37, "pilot_check", side_effect=analyze37.AnalysisError("missing null")):
                status = analyze37.main(["--capture-qwen", str(capture), "--capture-olmo", str(capture),
                                         "--pilot-check", str(Path(temp) / "pilot.json")])
            self.assertEqual(status, 2)

    def test_label_shuffle_raises_when_a_layer_fails_more_than_two_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            _write_synthetic_capture(capture, 17)
            store = analyze37.FeatureStore(capture)
            rows = analyze37._label(
                [row for row in store.rows if row["span_kind"] in ("own_g", "other_g")],
                lambda row: row["span_kind"] == "own_g")
            with patch.object(analyze37, "_fold_fit", side_effect=ValueError("synthetic failed fit")):
                with self.assertRaisesRegex(analyze37.AnalysisError, "failures exceed 2"):
                    analyze37._permutation_threshold(store, rows, 0, permutations=200)

    def test_groupfold_checker_and_verdict_rules(self):
        groups = ["q0", "q0", "q1", "q1"]
        good = [([0, 1], [2, 3]), ([2, 3], [0, 1])]
        bad = [([0, 2], [1, 3])]
        self.assertTrue(analyze37.folds_are_question_disjoint(good, groups))
        self.assertFalse(analyze37.folds_are_question_disjoint(bad, groups))
        self.assertEqual(analyze37.verdict(True, "x")['verdict'], "PASS")
        self.assertEqual(analyze37.verdict(False, "x")['verdict'], "FAIL")
        self.assertEqual(analyze37.verdict(None, "x")['verdict'], "UNSCORABLE")


if torch is not None:
    class TinyCausalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, 8)
            self.layer = nn.Linear(8, 8)
            self.head = nn.Linear(8, 256)
            self.config = SimpleNamespace(num_hidden_layers=1)
            self.model = SimpleNamespace(layers=[object()])
            self.forward_calls = 0

        def forward(self, input_ids: Any, attention_mask: Any = None, use_cache: bool = False,
                    output_hidden_states: bool = True, return_dict: bool = True) -> Any:
            self.forward_calls += 1
            embedding = self.embedding(input_ids)
            layer = self.layer(embedding)
            return SimpleNamespace(hidden_states=(embedding, layer), logits=self.head(layer))


    class TinyModelTests(unittest.TestCase):
        @staticmethod
        def record(record_id: str, question_id: str) -> dict[str, Any]:
            return {
                "record_id": record_id, "question_id": question_id, "reader": "qwen",
                "span_kind": "own_g", "author": "own", "slot": "assistant", "frame": "base",
                "header": None, "header_group": None, "header_variant": None,
                "span": "A pleasant walk can make the morning feel calm.", "target_tokens": 48,
                "span_token_count": 48, "token_mismatch_count": 0, "empty_reason": None,
                "messages": [
                    {"role": "user", "content": "How should I start the day?"},
                    {"role": "assistant", "content": "A pleasant walk can make the morning feel calm."},
                    {"role": "user", "content": collect37.FOLLOWUP},
                ],
            }

        def test_capture_uses_one_forward_and_returns_both_positions(self):
            tokenizer = CharacterTokenizer()
            model = TinyCausalModel().eval()
            record = self.record("q000:tiny", "q000")
            features, index, metrics = collect37._capture_one(torch, model, tokenizer, record, "qwen", "cpu")
            import numpy as np
            self.assertEqual(model.forward_calls, 1)
            self.assertEqual(tuple(features.shape), (2, 2, 8))
            self.assertEqual(features.dtype, np.float32)
            self.assertTrue(np.isfinite(features).all())
            self.assertNotEqual(index["span_last_index"], index["followup_first_index"])
            self.assertEqual(len(metrics["per_token_logprob"]), len(record["span"]))
            self.assertEqual(len(metrics["per_token_entropy"]), len(record["span"]))

        def test_span_logprobs_use_float32_logits_and_nonfinite_states_name_the_record(self):
            class HalfLogitModel(TinyCausalModel):
                def forward(self, *args, **kwargs):
                    result = super().forward(*args, **kwargs)
                    result.logits = result.logits.half()
                    return result

            tokenizer = CharacterTokenizer()
            record = self.record("q000:half", "q000")
            model = HalfLogitModel().eval()
            _, index, metrics = collect37._capture_one(torch, model, tokenizer, record, "qwen", "cpu")
            location = collect37.locate_readout_positions(record, tokenizer, "qwen")
            ids = torch.tensor([location["input_ids"]], dtype=torch.long)
            output = model(input_ids=ids)
            position = location["span_token_positions"][0]
            token = location["input_ids"][position]
            expected = torch.log_softmax(output.logits[0, position - 1].float(), dim=-1)[token].item()
            self.assertAlmostEqual(metrics["per_token_logprob"][0], expected, places=6)
            self.assertEqual(index["span_last_index"], location["span_last_index"])

            class NonFiniteModel(TinyCausalModel):
                def forward(self, *args, **kwargs):
                    result = super().forward(*args, **kwargs)
                    states = list(result.hidden_states)
                    states[-1] = states[-1].clone()
                    states[-1][0, :, 0] = float("nan")
                    result.hidden_states = tuple(states)
                    return result

            with self.assertRaisesRegex(collect37.Part37Error, "q000:bad: non-finite hidden state"):
                collect37._capture_one(torch, NonFiniteModel().eval(), tokenizer,
                                       self.record("q000:bad", "q000"), "qwen", "cpu")

        def test_capture_leaves_a_loadable_question_prefix_after_a_later_failure(self):
            import numpy as np
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                records_path, output = root / "records.jsonl", root / "capture"
                collect37.write_jsonl(records_path, [self.record("q000:ok", "q000"),
                                                      self.record("q001:stop", "q001")])
                model, tokenizer = TinyCausalModel().eval(), CharacterTokenizer()
                real_capture_one = collect37._capture_one

                def fail_second_question(torch_arg, model_arg, tokenizer_arg, record, reader, device):
                    if record["question_id"] == "q001":
                        raise collect37.Part37Error("simulated interruption")
                    return real_capture_one(torch_arg, model_arg, tokenizer_arg, record, reader, device)

                args = SimpleNamespace(reader="qwen", records=records_path, out=output, device="cpu")
                with patch.object(collect37, "load_model", return_value=(torch, tokenizer, model, "cpu", "test")), \
                     patch.object(collect37, "_capture_one", side_effect=fail_second_question):
                    with self.assertRaisesRegex(collect37.Part37Error, "simulated interruption"):
                        collect37.command_capture(args)
                # The prefix is on disk, but with no manifest the analysis must refuse it (DS round 2 M5).
                self.assertTrue((output / "index.jsonl").is_file())
                with self.assertRaises(analyze37.AnalysisError):
                    analyze37.FeatureStore(output)
                self.assertEqual(np.load(output / "shard-00000.npy").dtype, np.float32)
else:
    @unittest.skip("torch not installed; tiny-model check requires the GPU environment packages")
    class TinyModelTests(unittest.TestCase):
        def test_capture_uses_one_forward_and_returns_both_positions(self):
            self.fail("unreachable")


if __name__ == "__main__":
    unittest.main()


# Real-tokenizer readout checks (Claude, 2026-10-25): the stub test above cannot see a template-position error.
import os as _os
_HUB = "/mnt/e/neuron-probe/hf/hub"
_SNAPSHOTS = {
    "qwen": f"{_HUB}/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "olmo": f"{_HUB}/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e",
}


@unittest.skipUnless(all(_os.path.isdir(p) for p in _SNAPSHOTS.values()), "pinned tokenizer snapshots not mounted")
class RealTemplatePositionTests(unittest.TestCase):
    def test_span_and_followup_positions_decode_to_the_expected_text(self):
        from transformers import AutoTokenizer
        span = "Toast it lightly, then add butter while it is still warm so it melts evenly."
        for reader, path in _SNAPSHOTS.items():
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            for slot, frame, header in (("assistant", "base", None), ("user", "base", None),
                                        ("assistant", "label", "Your earlier notes"),
                                        ("user", "label", "Notes from another assistant")):
                record = {"record_id": f"t:{reader}:{slot}:{frame}", "span": span,
                          "messages": collect37._record_messages("How should I eat bread?", span, slot, frame, header)
                          if header else collect37._record_messages("How should I eat bread?", span, slot, frame)}
                loc = collect37.locate_readout_positions(record, tokenizer, reader)
                ids = loc["input_ids"]
                last = tokenizer.decode([ids[loc["span_last_index"]]])
                self.assertTrue(span.endswith(last.strip()) and last.strip(), (reader, slot, frame, repr(last)))
                after = tokenizer.decode([ids[loc["span_last_index"] + 1]])
                self.assertFalse(after.strip() and span.endswith(after.strip()), (reader, slot, frame, repr(after)))
                first = tokenizer.decode([ids[loc["followup_first_index"]]]).strip()
                self.assertTrue(first and collect37.FOLLOWUP.startswith(first), (reader, slot, frame, repr(first)))
                joined = tokenizer.decode(ids[loc["span_token_positions"][0]:loc["span_last_index"] + 1])
                self.assertIn(span[-20:], joined, (reader, slot, frame))


class ProbeSplitTests(unittest.TestCase):
    """The probe's own splits must keep every question on one side (Weaver's review; Claude, 2026-10-25)."""

    def test_probe_splits_are_question_disjoint_and_run(self):
        import numpy as np
        rng = np.random.default_rng(0)
        groups = [f"q{index // 2:03d}" for index in range(60)]
        _, _, _, _, GroupKFold, _ = analyze37._sklearn()
        splits = analyze37._group_splits(groups, GroupKFold)
        self.assertTrue(analyze37.folds_are_question_disjoint(splits, groups))
        x = rng.normal(size=(60, 6))
        y = np.array([index % 2 for index in range(60)])
        result = analyze37._fold_fit(x, y, groups, None, "raw", 0)
        self.assertEqual(len(result["fold_balanced_accuracy"]), 5)
        self.assertLess(abs(result["auc"] - 0.5), 0.35)


class Round2FixTests(unittest.TestCase):
    """DS round 2 M4 and M5 (Claude, 2026-10-25)."""

    def test_minimal_edit_replacements_vary(self):
        answers = ["It is sunny today and warm.", "A quiet room helps you focus.", "Use a glass of water first.",
                   "Read a good book before bed.", "Stretch your legs every hour.", "Small things add up over time.",
                   "Wait five minutes before you go.", "Rest well every night if you can."]
        replacements = [collect37.minimal_edit(text)[1]["replacement"] for text in answers]
        self.assertGreaterEqual(len(set(replacements)), 6, replacements)
        self.assertEqual(len(collect37.EDIT_WORDS), len(set(collect37.EDIT_WORDS)))

    def test_feature_store_refuses_capture_without_complete_manifest(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / "shard-00000.npy", np.zeros((1, 2, 3, 4), dtype=np.float32))
            (root / "index.jsonl").write_text(json.dumps({"record_id": "a", "shard": "shard-00000.npy", "row": 0}) + "\n")
            (root / "metrics.jsonl").write_text(json.dumps({"record_id": "a"}) + "\n")
            with self.assertRaises(analyze37.AnalysisError):
                analyze37.FeatureStore(root)
            (root / "manifest.json").write_text(json.dumps({"captured_records": 2}))
            with self.assertRaises(analyze37.AnalysisError):
                analyze37.FeatureStore(root)


class PileFallbackTests(unittest.TestCase):
    def test_short_pile_row_falls_back_to_the_next_row(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pile = root / "pile.jsonl"
            pile.write_text(json.dumps({"text": "too short"}) + "\n" +
                            json.dumps({"text": "ordinary " * 1000}) + "\n", encoding="utf-8")
            questions = [{"id": "q000", "question": "Is a quiet morning pleasant?"},
                         {"id": "q001", "question": "Is a quiet evening pleasant?"}]
            gens = generation_fixture() + [dict(row, question_id="q001") for row in generation_fixture()]
            records, _ = collect37.build_records("qwen", gens, {"kept": ["q000", "q001"]}, questions, pile,
                                                 CharacterTokenizer())
            pile_rows = [row for row in records if row["span_kind"] == "null_pile"]
            self.assertEqual(len(pile_rows), 2)
            self.assertTrue(all(row["span"].startswith("ordinary") or "ordinary" in row["span"] for row in pile_rows))


class SubspaceDegenerateTests(unittest.TestCase):
    def test_constant_features_do_not_break_subspace_removal(self):
        import numpy as np
        groups = [f"q{index // 2:03d}" for index in range(40)]
        x = np.ones((40, 16))
        y = np.array([index % 2 for index in range(40)])
        cov = np.random.default_rng(1).normal(size=(40, 4))
        result = analyze37._fold_fit(x, y, groups, cov, "subspace", 0)
        self.assertTrue(np.isfinite(result["probabilities"]).all())
        self.assertAlmostEqual(result["auc"], 0.5)


import os


class ParallelFitTests(unittest.TestCase):
    def test_parallel_fits_equal_serial(self):
        import numpy as np
        rng = np.random.default_rng(3)
        groups = [f"q{index // 2:03d}" for index in range(40)]
        y = np.array([index % 2 for index in range(40)])
        jobs = [(rng.normal(size=(40, 12)) + y[:, None] * 0.3, y, groups, None, "raw", layer) for layer in range(4)]
        old = os.environ.get("P37_JOBS")
        try:
            os.environ["P37_JOBS"] = "1"
            serial = analyze37._map_fits(jobs)
            os.environ["P37_JOBS"] = "3"
            parallel = analyze37._map_fits(jobs)
        finally:
            if old is None:
                os.environ.pop("P37_JOBS", None)
            else:
                os.environ["P37_JOBS"] = old
        for left, right in zip(serial, parallel):
            np.testing.assert_array_equal(left["probabilities"], right["probabilities"])
            self.assertEqual(left["auc"], right["auc"])


class FastMetricTests(unittest.TestCase):
    def test_fast_metrics_equal_sklearn(self):
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        rng = np.random.default_rng(5)
        for trial in range(300):
            n = int(rng.integers(6, 60))
            y = rng.integers(0, 2, size=n)
            if len(set(y)) < 2:
                continue
            p = np.round(rng.random(n), int(rng.integers(1, 4)))  # rounding makes ties
            self.assertAlmostEqual(analyze37._fast_auroc(y, p), roc_auc_score(y, p), places=12)
            self.assertAlmostEqual(analyze37._fast_balanced_accuracy(y, p >= 0.5),
                                   balanced_accuracy_score(y, p >= 0.5), places=12)


class RegressedConstantTests(unittest.TestCase):
    def test_constant_feature_with_label_correlated_covariate_stays_at_chance(self):
        import numpy as np
        groups = [f"q{index // 2:03d}" for index in range(80)]
        y = np.array([index % 2 for index in range(80)])
        x = np.ones((80, 8))
        cov = np.column_stack([y + np.random.default_rng(2).normal(scale=0.5, size=80), np.zeros((80, 3))])
        result = analyze37._fold_fit(x, y, groups, cov, "regressed", 0)
        self.assertLess(abs(result["auc"] - 0.5), 0.05)
