#!/usr/bin/env python3
"""Synthetic-only Part 178 tests, including copied-file mutation and a tiny Qwen3 CPU run."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part178 import problems178, run178, score178, words178  # noqa: E402

UPSTREAM_MATHQA = Path("/home/elisepc/green/Journal/notes/reports/part177-review/upstream/mathqa.py")
COPY_LIST = (
    "part178/__init__.py", "part178/words178.py", "part178/problems178.py", "part178/download178.py",
    "part178/run178.py", "part178/score178.py", "part177/read177.py", "part177/loop_upstream.py",
)


class FakeTokenizer:
    """Whitespace tokenizer that preserves every byte of text and character offsets."""
    def __init__(self):
        self.pieces = {0: ""}
        self.ids = {}
        self._next_id = 7
        self.pad_token_id, self.eos_token_id, self.unk_token_id = 0, 6, 15
        self._special = {"<think>": 4, "</think>": 5, "<|im_end|>": 6}
        self.pieces.update({4: "<think>", 5: "</think>", 6: "<|im_end|>"})
        self.ids.update({value: key for key, value in self.pieces.items() if value})

    def _piece_id(self, piece):
        if piece in self._special:
            return self._special[piece]
        if piece not in self.ids:
            value = self._next_id
            self._next_id += 1
            self.ids[piece] = value
            self.pieces[value] = piece
        return self.ids[piece]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        pieces, offsets = [], []
        for match in __import__("re").finditer(r"<\|im_end\|>|</think>|<think>|[^<\s]+|\s+|<[^>]*>", str(text)):
            piece = match.group(0)
            pieces.append(self._piece_id(piece))
            offsets.append((match.start(), match.end()))
        out = {"input_ids": pieces}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text):
        result = self(text, return_offsets_mapping=True)
        return SimpleNamespace(ids=result["input_ids"], offsets=result["offset_mapping"])

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.pieces.get(int(value), f"t{int(value)}") for value in ids)

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self.pieces.get(ids, f"t{ids}")
        return [self.pieces.get(int(value), f"t{int(value)}") for value in ids]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        user_text = messages[-1]["content"]
        if enable_thinking:
            return f"{user_text}\nassistant<think>\n"
        return f"{user_text}\nassistant<think>\n\n</think>\n\n"


class DefinitionTests(unittest.TestCase):
    def test_problem_check_maps_sparse_window_keys_to_the_200_input_ids(self):
        ids = [f"p{i}" for i in range(200)]
        rows = [{"problem_id": "p3"}, {"problem_id": "p3"}, {"problem_id": "p199"}]
        checked = problems178._check_window_key_ids(ids, rows)
        self.assertEqual(checked, {"input_problem_ids": 200, "window_key_rows": 3,
                                   "window_key_problem_ids": 2})
        with self.assertRaises(ValueError):
            problems178._check_window_key_ids(ids, [{"problem_id": "outside"}])

    def test_prompt_instruction_matches_upstream_mathqa(self):
        tree = ast.parse(UPSTREAM_MATHQA.read_text(encoding="utf-8"))
        value = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and
                     any(isinstance(target, ast.Name) and target.id == "INSTRUCTION" for target in node.targets))
        self.assertEqual(run178.INSTRUCTION, value)

    def test_all_frozen_gate2_fixtures(self):
        result = words178.gate2_results()
        self.assertTrue(result["passed"], result["fixtures"])
        self.assertEqual(len(result["fixtures"]), len(words178.freeze_gate2_fixtures()))
        soft = next(row for row in result["fixtures"] if row["name"].startswith("soft-wrapped"))
        self.assertEqual(soft["observed"], ["wait", "continues"])

    def test_thinking_text_and_prefix_use_first_tokens(self):
        tokenizer = FakeTokenizer()
        text = "reason\n</think>answer"
        self.assertEqual(words178.thinking_text(text), "reason\n")
        self.assertEqual(words178.prefix("alpha beta gamma", tokenizer=tokenizer, limit=2), "alpha ")
        self.assertEqual(words178.prefix(tokenizer=tokenizer, kept_ids=tokenizer("alpha beta")["input_ids"],
                                         limit=2), "alpha ")

    def test_onset2k_ignores_repeats_past_prefix_and_censors_final_lines(self):
        tokenizer = FakeTokenizer()
        repeated = "A sufficiently long repeated line for the onset rule.\n"
        text = repeated + " ".join(f"filler{i}" for i in range(5000)) + "\n" + repeated * 4
        self.assertIsNone(words178.onset2k(text, tokenizer=tokenizer))
        early = "A sufficiently long repeated line for the onset rule.\n" * 4
        early += "".join(f"line {i}\n" for i in range(50))
        self.assertEqual(words178.onset2k(early, tokenizer=tokenizer)["line_index"], 0)

    def test_onset_censor_adds_a_newline_before_padding_for_unterminated_prefix(self):
        tokenizer = FakeTokenizer()
        repeated = "A sufficiently long repeated line used as a loop marker."
        lines = [f"unique filler line {index} has enough words.\n" for index in range(60)]
        for index in (19, 30, 40, 59):
            lines[index] = repeated + ("" if index == 59 else "\n")
        text = "".join(lines)
        self.assertFalse(text.endswith("\n"))
        observed = words178.onset_prefix(text, tokenizer=tokenizer, token_limit=5000, censor_lines=40)
        self.assertIsNotNone(observed)
        self.assertEqual(observed["line_index"], 19)

    def test_first_2048_token_prefix_produces_same_2000_token_window_with_longer_tail(self):
        tokenizer = FakeTokenizer()
        common = "".join(f"Wait{i} " for i in range(1024))
        left = words178.prefix(common, tokenizer=tokenizer, limit=2048)
        right = words178.prefix(common + "A sufficiently long repeated line.\n" * 8,
                                 tokenizer=tokenizer, limit=2048)
        one = words178.window(left, tokenizer=tokenizer)
        two = words178.window(right, tokenizer=tokenizer)
        self.assertEqual(one["text"], two["text"])
        self.assertEqual(one["token_exposure"], two["token_exposure"])
        self.assertLessEqual(one["token_exposure"], 2000)

    def test_real_qwen35_template_prompt_modes_when_local_files_load_on_cpu(self):
        snapshot = Path("/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        template_path = snapshot / "chat_template.jinja"
        tokenizer_path = snapshot / "tokenizer.json"
        if not template_path.is_file() or not tokenizer_path.is_file():
            self.skipTest(f"local Qwen3.5 template/tokenizer files are missing at {snapshot}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.skipTest(f"local Qwen3.5 tokenizer could not load on CPU: {type(exc).__name__}: {exc}")
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                result = run178._prompt_ids(tokenizer, "Synthetic prompt for template validation.", thinking)
                self.assertTrue(result["prompt_mode_valid"])

    def test_window_reports_cut_reason_and_exposure(self):
        tokenizer = FakeTokenizer()
        text = "Wait, first.\n" * 30
        onset = {"char_index": text.index("Wait", 1), "line_index": 1}
        result = words178.window(text, tokenizer=tokenizer, onset=onset, limit=20)
        self.assertEqual(result["end_reason"], "onset")
        self.assertGreater(result["token_exposure"], 0)
        self.assertEqual(result["opener_exposure"], len(words178.openers(result["text"])))
        no_onset = words178.uncut("Wait done.", tokenizer=tokenizer, limit=20, end_reason="cap")
        self.assertEqual(no_onset["end_reason"], "cap")

    def test_paragraph_openers_and_sentence_boundary(self):
        self.assertEqual(words178.openers("Do we know? Wait, verify.\n- Hmm, yes."), ["do", "wait", "hmm"])
        self.assertEqual(words178.paragraph_openers("Wait, first. Hmm second.\n\n  1. Actually, next."),
                         ["wait", "actually"])
        self.assertEqual(words178.openers("(Wait, no"), ["wait"])
        self.assertEqual(words178.openers('"Wait," she said.'), ["wait"])
        self.assertEqual(words178.openers("_Wait_ here"), ["wait"])

    def test_rate_ranking_tie_break_and_trace_incidence(self):
        rows = [
            {"problem_id": "a", "opener_counts": {"beta": 1, "alpha": 1}, "opener_exposure": 10},
            {"problem_id": "b", "opener_counts": {"beta": 1, "alpha": 1}, "opener_exposure": 10},
        ]
        rates = [words178.rate(word, rows, reps=20, seed=4) for word in ("beta", "alpha")]
        self.assertEqual([row["word"] for row in words178.ranking(rates)], ["alpha", "beta"])
        self.assertEqual(words178.traces("beta", rows)["traces_with_word"], 2)
        self.assertEqual(rates[0]["rate_per_1000"], 100.0)

    def test_mode_ratio_under_five_control_openers_is_unavailable(self):
        thinking = [{"opener_counts": {"wait": 1}, "opener_exposure": 10}]
        control = [{"opener_counts": {"wait": 4}, "opener_exposure": 500}]
        result = words178.mode_ratio("wait", thinking, control, reps=5)
        self.assertFalse(result["available"])
        self.assertIsNone(result["ratio"])
        self.assertEqual(result["control_openers"], 500)
        self.assertEqual(result["control_word_n"], 4)

    def test_unavailable_follow_on_baseline_is_not_zero(self):
        tokenizer = FakeTokenizer()
        text = "$x$ VALUE.\nWait"
        start = text.index("VALUE")
        result = words178.follow_on({"p1": text},
            [{"id": "k1", "problem_id": "p1", "condition": "real", "hit_start": start,
              "hit_end": start + len("VALUE")}],
            [{"id": "k1", "label": "asserted"}], ["p1"], tokenizer, reps=5)
        self.assertIsNone(result["point"])
        self.assertEqual(result["n_eligible"], 0)

    def test_follow_on_baseline_excludes_the_asserted_hit_opener(self):
        tokenizer = FakeTokenizer()
        text = "Wait, the value is 5.\nHmm, next line."
        hit_start = text.index("Wait")
        result = words178.follow_on({"p1": text},
            [{"id": "k1", "problem_id": "p1", "condition": "real", "hit_start": hit_start,
              "hit_end": hit_start + len("Wait")}],
            [{"id": "k1", "label": "asserted"}], ["p1"], tokenizer, reps=5)
        self.assertEqual(result["n_eligible"], 0)
        self.assertIsNone(result["point"])

    def test_imported_reflection_lists_are_fixed(self):
        self.assertEqual(words178.imported_list(), ["wait", "alternatively", "hmm", "but", "however",
            "alternative", "another", "check", "double-check", "oh", "maybe", "verify", "other",
            "again", "now", "ah", "any"])
        self.assertEqual(words178.reflection_family(), ["actually", "alternatively", "but", "hmm", "however", "wait"])

    def test_token_list_uses_saved_ids_and_reports_whole_word_span(self):
        tokenizer = FakeTokenizer()
        text = " Wait, check.\n"
        ids = tokenizer(text)["input_ids"]
        rows = words178.token_list(text, tokenizer, ["wait"], kept_ids=ids)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["word"], "wait")
        self.assertTrue(rows[0]["token_ids"])
        self.assertFalse(rows[0]["reconstructed"])

    def test_kept_ids_derive_offsets_from_their_own_bytes(self):
        class ByteTokenizer:
            tokens = {1: "Ã", 2: "©", 3: "x"}
            byte_decoder = {"Ã": 0xC3, "©": 0xA9, "x": ord("x")}
            def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
                return [self.tokens[int(value)] for value in ids]
            def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                raw = bytes(self.byte_decoder[self.tokens[int(value)]] for value in ids)
                return raw.decode("utf-8", errors="replace")

        ids, offsets = words178.token_offsets("éx", ByteTokenizer(), kept_ids=[1, 2, 3])
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(offsets, [(0, 1), (0, 1), (1, 2)])
        _partial_ids, partial_offsets = words178.token_offsets("é", ByteTokenizer(), kept_ids=[1])
        self.assertEqual(partial_offsets, [(0, 1)])

    def test_markup_opener_keeps_line_boundary_marker(self):
        tokenizer = FakeTokenizer()
        rows = words178.token_list("first line\n**Wait**", tokenizer, ["wait"])
        self.assertEqual(rows[0]["leading_space_or_newline"], "\n")


class SamplerTests(unittest.TestCase):
    def test_all_gate4_sampler_fixtures(self):
        result = run178.gate4_sampler_fixtures()
        self.assertTrue(result["passed"], result["fixtures"])
        self.assertEqual(set(result["fixtures"]), {
            "prompt_only_not_penalized", "seen_once_equals_repeated", "penalty_before_top_k_keeps_A_B",
            "temperature_top_k_top_p_order", "row_draws_batch_mate_early_finish", "row_ends_stop_drawing"})

    def test_alden_top_k_fixture_has_exact_expected_logits(self):
        import torch
        p = run178.ordered_distribution(torch.tensor([[3.0, 2.9, 2.8]]), [[0]],
                                         presence_penalty=1.5, temperature=1.0, top_k=2, top_p=1.0)[0]
        self.assertEqual(torch.nonzero(p).flatten().tolist(), [1, 2])

    def test_presence_penalty_is_once_per_seen_token_not_prompt_or_repeat_count(self):
        import torch
        logits = torch.tensor([[3.0, 2.0, 1.0]])
        none = run178.ordered_distribution(logits, [[]], temperature=1, top_k=3, top_p=1)
        one = run178.ordered_distribution(logits, [[0]], temperature=1, top_k=3, top_p=1)
        many = run178.ordered_distribution(logits, [[0, 0, 0]], temperature=1, top_k=3, top_p=1)
        self.assertGreater(float(none[0, 0]), float(none[0, 1]))
        self.assertTrue(torch.equal(one, many))

    def test_device_topk_distribution_matches_reference_cpu_path_without_ties(self):
        import torch
        logits = torch.linspace(3.25, -4.0, 64, dtype=torch.float32).reshape(2, 32)
        generated = [[3, 11, 11], [0, 21]]
        penalty, temperature, top_k, top_p = 1.5, 0.73, 20, 0.95
        seen = torch.zeros_like(logits, dtype=torch.bool)
        for row_index, ids in enumerate(generated):
            seen[row_index, ids] = True

        old_work = logits.clone() - penalty * seen.float()
        old_work.div_(temperature)
        old_values, old_indices = torch.topk(old_work, k=top_k, dim=-1, sorted=True)
        old_probabilities = torch.softmax(old_values, dim=-1)
        old_cumulative = old_probabilities.cumsum(dim=-1)
        old_remove = old_cumulative - old_probabilities > top_p
        old_remove[:, 0] = False
        old_values = old_values.masked_fill(old_remove, float("-inf"))
        old_probabilities = torch.softmax(old_values, dim=-1)
        old_dense = torch.zeros_like(logits)
        old_dense.scatter_(1, old_indices, old_probabilities)

        observed = run178.ordered_distribution(logits, generated, presence_penalty=penalty,
                                                temperature=temperature, top_k=top_k, top_p=top_p)
        self.assertTrue(torch.allclose(observed, old_dense, rtol=1e-7, atol=1e-8))

        processor = run178.OrderedCpuSampler(
            [torch.Generator(device="cpu").manual_seed(1), torch.Generator(device="cpu").manual_seed(2)],
            prompt_width=1, end_token_ids=[6], pad_token_id=0, top_k=top_k)
        processor(torch.tensor([[1, 3], [1, 2]]), logits)
        self.assertEqual(processor.seen_mask.device, logits.device)

    def test_row_draw_stream_is_independent_of_an_early_batch_mate(self):
        import torch
        def draw_sequence(with_early_mate):
            first = torch.Generator(device="cpu").manual_seed(run178.row_seed("G", "p1"))
            other = torch.Generator(device="cpu").manual_seed(run178.row_seed("G", "p2"))
            generated_first, generated_other, results = [], [], []
            for step in range(5):
                if with_early_mate and step == 0:
                    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
                    chosen = run178.ordered_sampler(logits, [generated_first, generated_other], [first, other],
                                                    temperature=1, top_k=3, top_p=1)
                    generated_other.append(int(chosen[1]))
                else:
                    logits = torch.tensor([[3.0, 2.0, 1.0]])
                    chosen = run178.ordered_sampler(logits, [generated_first], [first],
                                                    temperature=1, top_k=3, top_p=1)
                token = int(chosen[0])
                results.append(token)
                generated_first.append(token)
            return results
        self.assertEqual(draw_sequence(True), draw_sequence(False))

    def test_real_logits_processor_matches_ordered_distribution_and_uniform(self):
        import torch
        logits = torch.tensor([[3.0, 2.9, 2.8, 0.0]])
        seed = 17842
        expected_generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_uniform = torch.rand((), generator=expected_generator)
        probabilities = run178.ordered_distribution(logits, [[0]], temperature=1, top_k=4, top_p=1)
        expected_token = torch.searchsorted(probabilities.cumsum(-1).contiguous(),
                                            expected_uniform.reshape(1, 1)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        processor = run178.OrderedCpuSampler([generator], prompt_width=1, end_token_ids=[6, 7],
                                             pad_token_id=0, temperature=1, top_k=4, top_p=1)
        result = processor(torch.tensor([[5, 0]]), logits)
        self.assertEqual(int(result.argmax(-1)[0]), expected_token)
        self.assertEqual(float(processor.last_uniforms[0]), float(expected_uniform))

    def test_real_logits_processor_does_not_penalize_a_prompt_only_token(self):
        import torch
        processor = run178.OrderedCpuSampler(
            [torch.Generator(device="cpu").manual_seed(5)], prompt_width=2,
            end_token_ids=[6], pad_token_id=0, temperature=1, top_k=1, top_p=1)
        result = processor(torch.tensor([[0, 1]]), torch.tensor([[3.0, 2.9, 2.8]]))
        self.assertEqual(int(result.argmax(-1)[0]), 0)

    def test_nonfinite_logits_raise_with_row_indexes(self):
        import torch
        with self.assertRaises(run178.NonFiniteLogitsError) as caught:
            run178.ordered_distribution(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), [[], []])
        self.assertEqual(caught.exception.row_indices, [0])


class GateAndIntegrationTests(unittest.TestCase):
    def test_scorer_blocks_every_prediction_behind_a_failed_gate(self):
        metrics = {"window": {name: {"rates": [], "ranking": [], "trace_rates": []} for name in score178.SET_ORDER},
                   "paragraph": {"U": {"by_word": {}, "denominator": 0}},
                   "gate5": {"passed": True, "overlap_ci95": None}}
        gates = {"gate1": {"passed": True}, "gate2": {"passed": False},
                 "gate3": {"passed": True, "by_set": {}, "adequacy": {}},
                 "gate4": {"passed": True}, "gate5": {"passed": True},
                 "adequacy": {}}
        # Gate 2 must force unread even when every later point estimate is below its bar.
        predictions = score178._make_predictions({}, metrics, {"G": {"by_word": {}}, "R8": {"by_word": {}}},
                                                  {"n_eligible": 0}, {"difference": None}, gates, 5, 1)
        self.assertTrue(all(row["verdict"] == "not read" for row in predictions.values()))
        blocked = score178._read_blocked({}, gates["gate1"], gates["gate2"], gates["gate3"],
                                         gates["gate4"], gates["gate5"], {})
        self.assertEqual(score178._verdict_blocked(blocked, sets=("U",)), "Gate 2 failed")

    def test_gate5_spearman_uses_average_ranks_for_ties(self):
        self.assertAlmostEqual(score178.spearman([1, 1, 3], [2, 2, 6]), 1.0)
        self.assertIsNone(score178.spearman([1, 1], [2, 3]))

    def test_bootstrap_samples_are_compact_and_absent_imported_words_zero_fill(self):
        import numpy as np
        rows = [{"opener_counts": {"wait": 1}, "opener_exposure": 2},
                {"opener_counts": {"wait": 0}, "opener_exposure": 2}]
        _rates, samples = score178._rates_for_set(rows, reps=8, seed=3, keep_samples={"wait"})
        self.assertEqual(set(samples), {"wait"})
        self.assertEqual(samples["wait"].dtype, np.float32)
        interval = score178._count_rate_interval({"samples": {"wait": np.array([6, 0], dtype=np.float32)}},
                                                ["wait", "absent"], threshold=5)
        self.assertEqual(interval["replicates"], 2)
        self.assertEqual(interval["lower"], 0)
        self.assertEqual(interval["upper"], 1)

    def test_gate1_compares_full_text_onsets_to_published_trace_metadata(self):
        import hashlib
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            part177 = root / "part177"
            inputs.mkdir()
            part177.mkdir()
            dataset = inputs / "dataset.json"
            dataset.write_text("synthetic dataset", encoding="utf-8")
            sample_u, sample_e = inputs / "u.samples", inputs / "e.samples"
            sample_u.write_bytes(b"u-synthetic")
            sample_e.write_bytes(b"e-synthetic")
            files = {}
            for path in (sample_u, sample_e):
                files[path.relative_to(root).as_posix()] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = {"files": files}
            read_path = part177 / "read.json"
            read_path.write_text(json.dumps({"trace_results": [
                {"problem_id": "p1", "arm": "unedited", "loop_onset": None},
                {"problem_id": "p1", "arm": "p50c", "loop_onset": None}]}), encoding="utf-8")
            (part177 / "numbers177.json").write_text(
                json.dumps({"sources": {"reader": "read.json"}}), encoding="utf-8")
            raw = {"U": {"p1": {"text": "synthetic single line"}},
                   "E": {"p1": {"text": "another synthetic line"}}}
            paths = ({}, {}, {"unedited": {0: sample_u}, "p50c": {0: sample_e}},
                     dataset, root / "tokenizer.json")
            with patch.object(score178.read177, "_verify_manifest", return_value=manifest), \
                    patch.object(score178.read177, "_data_paths", return_value=paths):
                result = score178._gate1(inputs, part177,
                    {"source": {"sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}}, raw, ["p1"])
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["published_numbers_file"], "numbers177.json")
            self.assertTrue(result["text_source_hashes"]["U"]["passed"])
            self.assertEqual(result["full_text_onset_counts"]["U"]["expected"], 0)

    def test_gate3_checks_thinking_suffix_empty_control_and_generated_tags(self):
        tokenizer = FakeTokenizer()
        thinking = run178._prompt_ids(tokenizer, "<think> in user content", True)
        control = run178._prompt_ids(tokenizer, "synthetic answer", False)
        self.assertTrue(thinking["prompt_mode_valid"])
        self.assertEqual(thinking["prompt_tail_ids"][-1], tokenizer("\n")["input_ids"][0])
        self.assertTrue(control["prompt_mode_valid"])
        open_id = tokenizer.convert_tokens_to_ids("<think>")
        close_id = tokenizer.convert_tokens_to_ids("</think>")
        newline_pair_id = tokenizer("\n\n")["input_ids"][0]
        self.assertEqual(control["prompt_tail_ids"][-4:], [open_id, newline_pair_id,
                                                           close_id, newline_pair_id])

        class MisleadingTemplate(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                return messages[-1]["content"] + "\nassistant"
        with self.assertRaises(ValueError):
            run178._prompt_ids(MisleadingTemplate(), "question contains <think>", True)

        wanted = ["p1"]
        generated = {name: {"p1": {"prompt_mode_valid": True,
                                    "thinking": bool(spec["thinking"]),
                                    "generated_has_think_tag": False}}
                     for name, spec in run178.SET_SPECS.items()}
        traces = {name: [{"opener_exposure": 2000}] for name in score178.SET_ORDER}
        passed = score178._gate3(generated, traces, wanted)
        self.assertTrue(passed["by_set"]["G"]["passed"])
        self.assertTrue(passed["by_set"]["N4"]["passed"])
        generated["N4"]["p1"]["generated_has_think_tag"] = True
        failed = score178._gate3(generated, traces, wanted)
        self.assertFalse(failed["by_set"]["N4"]["passed"])

    def test_imports_from_the_declared_copy_list_under_isolated_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in COPY_LIST:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            result = subprocess.run([sys.executable, "-I", "-B", str(root / "part178" / "run178.py"), "--help"],
                                    cwd=root, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--set", result.stdout)

    def test_final_torn_jsonl_line_is_dropped_and_batch_append_is_synced(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "G.jsonl"
            path.write_bytes(b'{"problem_id":"p1"}\n{"problem_id":')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rows = run178._read_jsonl(path)
            self.assertTrue(any(issubclass(row.category, RuntimeWarning) for row in caught))
            self.assertEqual(rows, [{"problem_id": "p1"}])
            self.assertEqual(path.read_text(encoding="utf-8"), '{"problem_id":"p1"}\n')
            with patch.object(run178.os, "fsync") as sync:
                run178._write_batch(path, [{"problem_id": "p2"}, {"problem_id": "p3"}])
            sync.assert_called_once()
            self.assertEqual(len(run178._read_jsonl(path)), 3)
            path.write_bytes(b'{"problem_id":"complete-no-newline"}')
            with patch.object(run178.os, "fsync"):
                run178._write_batch(path, [{"problem_id": "after"}])
            self.assertEqual([row["problem_id"] for row in run178._read_jsonl(path)],
                             ["complete-no-newline", "after"])

    def test_pilot_plan_and_seed_varied_copies_match_registered_batch_sizes(self):
        expected = [("R8", 64), ("N8", 64), ("G", 16), ("G", 32), ("N4", 32)]
        self.assertEqual(run178._pilot_plan("R8:64,N8:64,G:16,G:32,N4:32"), expected)
        rows = [{"problem_id": f"p{i}", "problem": f"Synthetic problem {i}"} for i in range(16)]
        copies = run178._pilot_rows_for_batch(rows, 32)
        self.assertEqual(len(copies), 32)
        self.assertEqual(copies[0]["problem_id"], "p0::repeat-0")
        self.assertEqual(copies[16]["problem_id"], "p0::repeat-1")
        self.assertEqual(len({row["problem_id"] for row in copies}), 32)

    def test_synthetic_pilot_streams_one_line_and_records_model_release(self):
        import contextlib
        import io
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        tokenizer = FakeTokenizer()
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
            eos_token_id=6, pad_token_id=0, bos_token_id=1))
        args = SimpleNamespace(pilot=True, plan="R8:16", device="cpu", problems=HERE / "problems178.json")
        rows = [{"problem_id": f"p{i}", "problem": f"Synthetic pilot question {i}."} for i in range(16)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "pilot.jsonl"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = run178._run_pilot(args, lambda *_args: (tokenizer, model), {
                    "problem_rows": rows, "pilot_root": root / "sets", "pilot_output": output,
                    "max_new_tokens": 2,
                })
            lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            printed = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(len(printed), 1)
        self.assertEqual(printed[0]["set"], "R8")
        self.assertIn("peak_memory_allocated_bytes", printed[0])
        self.assertEqual(lines[0]["batch"], 16)
        self.assertEqual(lines[1]["event"], "model_release_check")
        self.assertFalse(any(key in printed[0] for key in ("generated_text", "generated_ids", "problem")))

    def test_six_deliberate_copied_file_breaks_turn_tests_red(self):
        onset_code = '''
import sys
sys.path.insert(0, ROOT)
from part178.words178 import onset_prefix
class T:
    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [ord(char) for char in text]}
    def decode(self, ids, **kwargs):
        return "".join(chr(int(value)) for value in ids)
repeat = "A sufficiently long repeated line used as a loop marker."
lines = [f"unique filler line {i} has enough words.\\n" for i in range(60)]
for i in (19, 30, 40, 59): lines[i] = repeat + ("" if i == 59 else "\\n")
assert onset_prefix("".join(lines), tokenizer=T(), token_limit=10000, censor_lines=40) is not None
'''
        follow_code = '''
import sys
sys.path.insert(0, ROOT)
from part178.words178 import follow_on
class T:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        row = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping: row["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return row
text = "Wait, the value is 5.\\nHmm, next line."
result = follow_on({"p1": text}, [{"id":"k1", "problem_id":"p1", "condition":"real", "hit_start":0, "hit_end":4}],
                   [{"id":"k1", "label":"asserted"}], ["p1"], T(), reps=0)
assert result["n_eligible"] == 0
'''
        prompt_code = '''
import re, sys
sys.path.insert(0, ROOT)
from part178.run178 import _prompt_ids
class T:
    def __call__(self, text, add_special_tokens=False, **kwargs):
        pieces = re.findall(r"<think>|</think>|.", str(text), re.S)
        return {"input_ids": [1 if p == "<think>" else 2 if p == "</think>" else ord(p) + 3 for p in pieces]}
    def decode(self, ids, **kwargs):
        return "".join("<think>" if int(i) == 1 else "</think>" if int(i) == 2 else chr(int(i) - 3) for i in ids)
    def convert_tokens_to_ids(self, token): return {"<think>": 1, "</think>": 2}.get(token, 0)
    unk_token_id = 0
    def apply_chat_template(self, *args, **kwargs): return "header<think>bad</think>tail"
try:
    _prompt_ids(T(), "synthetic question", False)
except ValueError:
    pass
else:
    raise AssertionError("invalid non-thinking prompt was accepted")
'''
        cases = [
            ("opener skip list", "part178/words178.py", "[*_#>", "[_#>",
             "from part178 import words178; assert words178.gate2_results()['passed']"),
            ("onset censor padding", "part178/words178.py", 'padded_source += "\\n"',
             'pass  # removed separator', onset_code),
            ("follow-on exclusion", "part178/words178.py", "and not in_hit and", "and True and", follow_code),
            ("presence order", "part178/run178.py",
             'penalized = work - float(presence_penalty) * seen_mask.to(dtype=work.dtype)',
             'penalized = work',
             "from part178 import run178; assert run178.gate4_sampler_fixtures()['passed']"),
            ("end-id stop", "part178/run178.py",
             "if int(token_id) in _normalise_end_ids(end_token_ids):",
             "if int(token_id) in _normalise_end_ids(end_token_ids)[1:]:",
             "from part178.run178 import completion_reason; assert completion_reason(248044, 1, 10, [248044, 248046]) == 'end-of-turn'"),
            ("nonthinking prompt suffix", "part178/run178.py",
             'valid = between == "\\n\\n" and after == "\\n\\n"',
             "valid = True", prompt_code),
        ]
        results = []
        for name, relative, needle, replacement, code in cases:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for copy_path in COPY_LIST:
                    destination = root / copy_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / copy_path, destination)
                path = root / relative
                source = path.read_text(encoding="utf-8")
                self.assertIn(needle, source, f"mutation anchor missing: {name}")
                mutated = source.replace(needle, replacement, 1)
                if name == "presence order":
                    anchor = "top_values, top_indices = torch.topk(penalized, k=k, dim=-1, sorted=True)"
                    self.assertIn(anchor, mutated)
                    late_penalty = (anchor + "\n        top_values = top_values - float(presence_penalty) * "
                                    "seen_mask.gather(1, top_indices).to(dtype=top_values.dtype)")
                    mutated = mutated.replace(anchor, late_penalty, 1)
                path.write_text(mutated, encoding="utf-8")
                if "ROOT" in code:
                    code = code.replace("ROOT", repr(str(root)))
                else:
                    code = "import sys; sys.path.insert(0, " + repr(str(root)) + "); " + code
                result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=root,
                                        text=True, capture_output=True, timeout=30)
                results.append((name, result.returncode != 0))
        labels = ", ".join(f"{name}={'red' if killed else 'SURVIVED'}" for name, killed in results)
        print(f"MUTANTS: {sum(killed for _name, killed in results)}/6 killed ({labels})")
        self.assertTrue(all(killed for _name, killed in results), labels)

    def test_end_to_end_tiny_random_qwen3_run_writes_and_resumes(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        tokenizer = FakeTokenizer()
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=64, eos_token_id=6, pad_token_id=0,
                             bos_token_id=1)
        model = Qwen3ForCausalLM(config)
        args = SimpleNamespace(pilot=False, set="G", out=Path(tempfile.gettempdir()) / "unused",
                               batch=2, device="cpu", problems=HERE / "problems178.json")
        with tempfile.TemporaryDirectory() as tmp:
            args.out = Path(tmp)
            problem_rows = [{"problem_id": "p1", "problem": "Synthetic question one."},
                            {"problem_id": "p2", "problem": "Synthetic question two."}]
            first = run178.run(args, loader=lambda *_args: (tokenizer, model),
                               config={"problem_rows": problem_rows, "max_new_tokens": 3})
            self.assertEqual(first["written"], 2)
            rows = [json.loads(line) for line in (Path(tmp) / "G.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["problem_id"] for row in rows}, {"p1", "p2"})
            self.assertTrue(all(row["generated_ids"] and row["prompt_ids_sha256"] for row in rows))
            for row in rows:
                self.assertIn(row["finish_reason"], ("end-of-turn", "cap"))
                if row["finish_reason"] == "end-of-turn":
                    self.assertIn(row["generated_ids"][-1], row["end_token_ids"])
                else:
                    self.assertEqual(len(row["generated_ids"]), 3)
            second = run178.run(args, loader=lambda *_args: (tokenizer, model),
                                config={"problem_rows": problem_rows, "max_new_tokens": 3})
            self.assertEqual(second["written"], 0)
            self.assertEqual(second["resumed"], 2)
            self.assertEqual(len((Path(tmp) / "G.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_end_to_end_tiny_random_qwen35_hybrid_generates_two_batches_and_resumes(self):
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig

        tokenizer = FakeTokenizer()
        text_config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
            max_position_embeddings=256, eos_token_id=[6, 7], pad_token_id=0, bos_token_id=1)
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text_config)).eval()
        model.generation_config.eos_token_id = [6, 7]
        calls = {"n": 0}
        original_generate = model.generate
        def counted_generate(*args, **kwargs):
            calls["n"] += 1
            calls.setdefault("sizes", []).append(int(kwargs["input_ids"].shape[0]))
            return original_generate(*args, **kwargs)
        model.generate = counted_generate
        rows = [{"problem_id": f"p{i}", "problem": f"Synthetic question {i}."} for i in range(3)]
        spec = dict(run178.SET_SPECS["G"], max_new_tokens=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "G.jsonl"
            batches = []
            first = run178.run_set("G", rows, path, 2, tokenizer=tokenizer, model=model,
                                   device="cpu", spec=spec, max_new_tokens=3,
                                   progress=lambda event: batches.append(event))
            self.assertEqual(first["written"], 3)
            self.assertEqual(len(batches), 2)
            self.assertEqual(calls["n"], 2)
            # No row is generated at batch 1: the lone third row runs beside a discarded companion.
            self.assertEqual(calls["sizes"], [2, 2])
            saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(saved), 3)
            for row in saved:
                self.assertIn(row["finish_reason"], ("end-of-turn", "cap"))
                if row["finish_reason"] == "end-of-turn":
                    self.assertIn(row["generated_ids"][-1], [6, 7])
                else:
                    self.assertEqual(len(row["generated_ids"]), 3)
            resumed = run178.run_set("G", rows, path, 2, tokenizer=tokenizer, model=model,
                                     device="cpu", spec=spec, max_new_tokens=3)
            self.assertEqual(resumed["written"], 0)
            self.assertEqual(resumed["resumed"], 3)
            self.assertEqual(calls["n"], 2)

    def test_both_effective_end_ids_stop_and_cap_is_recorded(self):
        self.assertEqual(run178.completion_reason(248044, 1, 10, [248044, 248046]), "end-of-turn")
        self.assertEqual(run178.completion_reason(248046, 1, 10, [248044, 248046]), "end-of-turn")
        self.assertEqual(run178.completion_reason(91, 10, 10, [248044, 248046]), "cap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
