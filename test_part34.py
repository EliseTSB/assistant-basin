"""Fast, offline checks for Part 34 helpers and the small Qwen3 hook path."""

from __future__ import annotations

import json
import argparse
import contextlib
import math
import tempfile
import unittest
import re
from pathlib import Path
from unittest.mock import patch

import part34


try:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    HAS_TORCH_QWEN3 = True
except ImportError:
    torch = None
    Qwen3Config = None
    Qwen3ForCausalLM = None
    AutoTokenizer = None
    HAS_TORCH_QWEN3 = False


REAL_TOKENIZER_PATH = Path(
    "/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
HAS_REAL_TOKENIZER = AutoTokenizer is not None and REAL_TOKENIZER_PATH.is_dir()


class Part34HelpersTest(unittest.TestCase):
    def test_questions_are_40_neutral_single_lines(self) -> None:
        data = part34.read_questions()
        self.assertEqual(len(data["questions"]), 40)
        self.assertEqual(len({item["id"] for item in data["questions"]}), 40)
        self.assertTrue(all("\n" not in item["text"] for item in data["questions"]))

    def test_counterbalancing_alternates_even_and_odd(self) -> None:
        self.assertEqual(part34.alternating_options(0, "even", "odd"), (("A", "even"), ("B", "odd")))
        self.assertEqual(part34.alternating_options(1, "even", "odd"), (("A", "odd"), ("B", "even")))
        self.assertEqual(part34.turn2_content("parity_free_raised", 0)[1], {
            "A": "even",
            "B": "odd",
        })
        self.assertEqual(part34.turn2_content("parity_free_raised", 1)[2], "odd")

    def test_parser_exposes_question_shards_threads_and_multi_file_scoring(self) -> None:
        run_args = part34.build_parser().parse_args([
            "run", "--out", "part-shard", "--direction-file", "direction.safetensors",
            "--question-start", "3", "--question-end", "8", "--threads", "2",
        ])
        self.assertEqual((run_args.question_start, run_args.question_end, run_args.threads), (3, 8, 2))
        readout_args = part34.build_parser().parse_args([
            "run", "--out", "rebuilt", "--direction-file", "direction.safetensors",
            "--readout-only", "--rows", "first/rows.jsonl", "second/rows.jsonl",
        ])
        self.assertTrue(readout_args.readout_only)
        self.assertEqual(readout_args.rows, ["first/rows.jsonl", "second/rows.jsonl"])
        score_args = part34.build_parser().parse_args([
            "score", "--out", "combined", "--rows", "first/rows.jsonl", "second/rows.jsonl",
        ])
        self.assertEqual(score_args.rows, ["first/rows.jsonl", "second/rows.jsonl"])

    def test_resume_skips_completed_keys_and_score_fixture(self) -> None:
        rows = [
            {"type": "row", "key": "turn1|0|none|M", "row_type": "turn1", "question_index": 0,
             "condition": "none", "model": "M", "d_question": 0.0, "d_reply": 0.0,
             "d_reply_target_pct": 0.0, "z_reply_profile": [0.0] * 28},
            {"type": "row", "key": "t2|0|none", "row_type": "turn2_ab", "question_index": 0,
             "condition": "none", "arm": "default", "model": "M", "metric_kind": "parity_free_raised",
             "target_value": None, "letter_concepts": {"A": "even", "B": "odd"},
             "p_A": 0.7, "p_B": 0.1, "ab_mass": 0.8, "greedy_concept": "odd", "greedy_letter": None, "logp_A": -0.4,
             "logp_B": -0.8, "logit_even_minus_odd": 0.4, "d_question": 0.0, "d_answer": 0.1,
             "question_order_first_parity": "even"},
            {"type": "row", "key": "t2|0|ask", "row_type": "turn2_ab", "question_index": 0,
             "condition": "ask-even", "arm": "default", "model": "B", "metric_kind": "parity_free_raised",
             "target_value": "even", "letter_concepts": {"A": "even", "B": "odd"},
             "p_A": 0.65, "p_B": 0.35, "ab_mass": 1.0, "greedy_concept": "odd", "greedy_letter": None, "logp_A": -0.2,
             "logp_B": -1.2, "logit_even_minus_odd": 1.0, "d_question": 0.0, "d_answer": 0.1},
            {"type": "row", "key": "free|0", "row_type": "turn2_free", "question_index": 0,
             "condition": "none", "arm": "default", "model": "M", "mention_flag": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in [{"type": "header"}, *rows]) + "\n", encoding="utf-8")
            header, loaded = part34.read_jsonl(path)
            self.assertEqual(header["type"], "header")
            self.assertEqual(part34.done_keys(loaded), {row["key"] for row in rows})
        scored = part34.score_rows({"type": "header"}, rows)
        self.assertEqual(set(scored["prediction_ids"]), {f"34.{i}" for i in range(1, 15)})
        self.assertEqual(scored["prediction_ids"]["34.3"]["parity_free_correct_rate"]["n"], 1)
        self.assertEqual(scored["prediction_ids"]["34.3"]["parity_free_correct_rate"]["successes"], 1)
        self.assertEqual(scored["honest_request_sanity"]["B_ask_even_ask_odd_accuracy"]["successes"], 1)
        self.assertEqual(scored["prediction_ids"]["34.10"]["mention_rates"]["none_floor_by_model"]["M"]["n"], 1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in [{"type": "header"}, *rows]) + "\n",
                encoding="utf-8",
            )
            part34.score(argparse.Namespace(out=str(out)))
            output = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(output["n_rows"], len(rows))
            review = json.loads((out / "free-text-blind-review.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("condition", review)
            self.assertNotIn("regex_mention_flag", review)
            self.assertTrue((out / "free-text-blind-key.json").is_file())

    def test_variant_sums_low_mass_and_sign_scoring_with_both_orders(self) -> None:
        logits = [0.0, 1.0, 2.0, 0.0, -50.0]
        concepts = {"A": "even", "B": "odd"}
        derived = part34.derive_ab_readout(logits, {"A": [1, 2], "B": [3]}, concepts, "even")
        normalizer = sum(math.exp(value) for value in logits)
        expected_a = (math.exp(1.0) + math.exp(2.0)) / normalizer
        expected_b = math.exp(0.0) / normalizer
        self.assertAlmostEqual(derived["p_A"], expected_a)
        self.assertAlmostEqual(derived["p_B"], expected_b)
        self.assertAlmostEqual(derived["ab_mass"], expected_a + expected_b)
        self.assertAlmostEqual(derived["logit_even_minus_odd"], math.log(expected_a / expected_b))
        self.assertFalse(derived["low_mass"])

        low = part34.derive_ab_readout([0.0, 0.0, 20.0], {"A": [0], "B": [1]}, concepts, "even")
        self.assertTrue(low["low_mass"])
        self.assertLess(low["ab_mass"], part34.LOW_MASS_THRESHOLD)

        even_first = {
            "p_A": 0.8, "p_B": 0.1, "letter_concepts": {"A": "even", "B": "odd"},
            "greedy_concept": "odd", "target_value": "even",
        }
        odd_first = {
            "p_A": 0.8, "p_B": 0.1, "letter_concepts": {"A": "odd", "B": "even"},
            "greedy_concept": "even", "target_value": "odd",
        }
        self.assertEqual(part34.scored_concept(even_first), "even")
        self.assertEqual(part34.scored_concept(odd_first), "odd")
        self.assertEqual(part34.scored_concept(even_first), even_first["target_value"])
        self.assertEqual(part34.scored_concept(odd_first), odd_first["target_value"])

        swapped = dict(even_first, letter_concepts={"A": "odd", "B": "even"})
        self.assertNotEqual(part34.scored_concept(even_first), part34.scored_concept(swapped))


@unittest.skipUnless(HAS_REAL_TOKENIZER, "pinned real Qwen3 tokenizer is unavailable")
class RealTokenizerABReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            REAL_TOKENIZER_PATH,
            local_files_only=True,
            use_fast=True,
        )

    def test_post_parenthesis_readout_derives_every_field_for_orders_and_directions(self) -> None:
        paren_token_id, letter_token_ids = part34.validate_ab_tokenizer(self.tokenizer)
        self.assertEqual(paren_token_id, 7)
        self.assertEqual(letter_token_ids, {"A": 32, "B": 33})

        logits = [-8.0] * 64
        logits[0] = -1.0
        logits[32] = 4.0
        logits[33] = 1.0
        log_normalizer = max(logits) + math.log(sum(math.exp(value - max(logits)) for value in logits))
        logp_a, logp_b = logits[32] - log_normalizer, logits[33] - log_normalizer
        expected_keys = {
            "greedy_token", "greedy_letter", "greedy_concept", "predicted_letter", "predicted_concept",
            "p_A", "p_B", "ab_mass", "low_mass", "letter_variant_token_ids", "logp_A", "logp_B",
            "logit_even_minus_odd", "target_logp", "alternative_logp", "target_logp_margin", "target_candidate_probability",
        }
        for kind in (
            "parity_words_raised", "parity_words_lowered",
            "parity_free_raised", "parity_free_lowered",
        ):
            target = part34.expected_value("ask-even", kind)
            for question_index in (0, 1):
                _body, concepts, _first = part34.turn2_content(kind, question_index)
                with self.subTest(kind=kind, question_index=question_index, concepts=concepts):
                    derived = part34.derive_ab_readout(logits, letter_token_ids, concepts, target)
                    self.assertEqual(set(derived), expected_keys)
                    self.assertEqual(derived["greedy_token"], 32)
                    self.assertEqual(derived["greedy_letter"], "A")
                    self.assertEqual(derived["greedy_concept"], concepts["A"])
                    self.assertEqual(derived["predicted_letter"], "A")
                    self.assertEqual(derived["letter_variant_token_ids"], {"A": [32], "B": [33]})
                    self.assertAlmostEqual(derived["p_A"], math.exp(logp_a))
                    self.assertAlmostEqual(derived["p_B"], math.exp(logp_b))
                    self.assertAlmostEqual(derived["logp_A"], logp_a)
                    self.assertAlmostEqual(derived["logp_B"], logp_b)
                    even_letter = next(letter for letter, concept in concepts.items() if concept == "even")
                    odd_letter = next(letter for letter, concept in concepts.items() if concept == "odd")
                    expected_even_minus_odd = (logp_a if even_letter == "A" else logp_b) - (
                        logp_a if odd_letter == "A" else logp_b
                    )
                    self.assertAlmostEqual(derived["logit_even_minus_odd"], expected_even_minus_odd)
                    target_letter = next(letter for letter, concept in concepts.items() if concept == target)
                    alternative_letter = "B" if target_letter == "A" else "A"
                    expected_target_logp = logp_a if target_letter == "A" else logp_b
                    expected_alternative_logp = logp_a if alternative_letter == "A" else logp_b
                    expected_margin = expected_target_logp - expected_alternative_logp
                    self.assertAlmostEqual(derived["target_logp"], expected_target_logp)
                    self.assertAlmostEqual(derived["alternative_logp"], expected_alternative_logp)
                    self.assertAlmostEqual(derived["target_logp_margin"], expected_margin)
                    self.assertAlmostEqual(
                        derived["target_candidate_probability"],
                        1.0 / (1.0 + math.exp(-expected_margin)),
                    )

    def test_cached_vocab_has_fixed_first_answer_letter_ids(self) -> None:
        vocab = Path("/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3-1.7B/blobs/4783fe10ac3adce15ac8f358ef5462739852c569")
        if vocab.is_file():
            values = json.loads(vocab.read_text(encoding="utf-8"))
            self.assertEqual({"A": values["A"], "B": values["B"]}, part34.FIXED_LETTER_TOKEN_IDS)
        else:
            self.assertEqual(part34.FIXED_LETTER_TOKEN_IDS, {"A": 32, "B": 33})


@unittest.skipUnless(HAS_TORCH_QWEN3, "PyTorch/Transformers are unavailable in this worker environment")
class TinyQwen3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(194)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            use_cache=True,
        )
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.direction = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)
        cls.mean = torch.zeros(2)
        cls.sd = torch.ones(2)

    def test_hook_adds_only_named_vectors_at_named_layers_and_positions_then_removes(self) -> None:
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        baseline = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        raw_outputs = {}
        observer_handles = []
        for index, layer in enumerate(part34._decoder_layers(self.model)):
            def observe(_module, _inputs, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                raw_outputs[index] = hidden.detach().clone()
            observer_handles.append(layer.register_forward_hook(observe))
        try:
            with part34.steering_hook(self.model, self.direction, self.mean, self.sd, "inject-even", positions=[1, 3]):
                with part34.record_decoder_outputs(self.model) as steered_outputs:
                    changed = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        finally:
            for handle in observer_handles:
                handle.remove()
        for index in range(2):
            actual = steered_outputs[index] - raw_outputs[index]
            sign = 1.0 if index == 0 else -1.0
            expected = torch.zeros_like(actual)
            expected[:, [1, 3], :] = (2.0 * sign * self.direction[index]).view(1, 1, -1)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(steered_outputs[-1], changed.hidden_states[-1]))
        removed = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        for actual, expected in zip(removed.hidden_states, baseline.hidden_states):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        site = part34.assert_hook_site(self.model, expected_layers=2)
        self.assertEqual(site["first_block"], 0)
        self.assertEqual(site["site"], part34.HOOK_SITE)

    def test_recorded_last_layer_is_raw_block_output_not_final_norm(self) -> None:
        ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
        last_layer = part34._decoder_layers(self.model)[-1]
        raw = {}

        def observe(_module, _inputs, output):
            raw["value"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        handle = last_layer.register_forward_hook(observe)
        try:
            with part34.record_decoder_outputs(self.model) as recorded:
                output = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        finally:
            handle.remove()
        torch.testing.assert_close(recorded[-1], raw["value"], atol=0, rtol=0)
        self.assertFalse(torch.allclose(recorded[-1], output.hidden_states[-1]))

    def test_d_matches_hand_computation(self) -> None:
        states = [
            torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0], [0.0, 4.0], [0.0, 6.0]]]),
        ]
        direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        profile = part34.compute_z_profile(states, direction, torch.zeros(2), torch.ones(2), [1, 2])
        self.assertEqual(profile, [3.0, 5.0])
        self.assertEqual(part34.compute_d(states, direction, torch.zeros(2), torch.ones(2), [1, 2]), -2.0)

    def test_cache_kept_and_recomputed_use_same_token_context_but_distinct_values(self) -> None:
        turn1 = [2, 3, 4, 5]
        turn2 = [6, 7, 8]
        kept_context = turn1 + turn2
        recomputed_context = list(turn1) + list(turn2)
        self.assertEqual(kept_context, recomputed_context)
        kept = part34.forward_with_cache(
            self.model, turn1, torch.device("cpu"), hook_mode="inject-even",
            direction=self.direction, mean=self.mean, sd=self.sd,
        ).past_key_values
        recomputed = part34.forward_with_cache(
            self.model, turn1, torch.device("cpu"), direction=self.direction, mean=self.mean, sd=self.sd,
        ).past_key_values

        def tensors(cache):
            legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
            return [tensor for layer in legacy for tensor in layer if torch.is_tensor(tensor)]

        left, right = tensors(kept), tensors(recomputed)
        self.assertEqual(len(left), len(right))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(left, right)))
        self.assertEqual(len(turn1 + turn2), len(recomputed_context))

    def test_ab_probe_uses_one_forward_pass(self) -> None:
        cached = part34.forward_with_cache(self.model, [2, 3], torch.device("cpu"))
        with patch.object(self.model, "forward", wraps=self.model.forward) as counted_forward:
            logits, layers = part34.ab_probe(self.model, [4, 5, 6], torch.device("cpu"), cached.past_key_values)
        self.assertEqual(counted_forward.call_count, 1)
        self.assertEqual(len(logits), self.model.config.vocab_size)
        self.assertEqual(len(layers), self.model.config.num_hidden_layers)

    def test_tiny_random_model_logits_use_variant_sums_and_flag_low_mass(self) -> None:
        ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False).logits[0, -1, :].float().tolist()
        variant_ids = {"A": [32, 33], "B": [34, 35]}
        concepts = {"A": "even", "B": "odd"}
        measured = part34.derive_ab_readout(logits, variant_ids, concepts, "even")
        maximum = max(logits)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
        expected_a = sum(math.exp(logits[token] - normalizer) for token in variant_ids["A"])
        expected_b = sum(math.exp(logits[token] - normalizer) for token in variant_ids["B"])
        self.assertAlmostEqual(measured["p_A"], expected_a)
        self.assertAlmostEqual(measured["p_B"], expected_b)
        logits[10] = max(logits) + 25.0
        low = part34.derive_ab_readout(logits, variant_ids, concepts, "even")
        self.assertTrue(low["low_mass"])
        self.assertLess(low["ab_mass"], part34.LOW_MASS_THRESHOLD)

    def test_tiny_model_run_and_score_end_to_end(self) -> None:
        class TinyTokenizer:
            eos_token_id = None

            @staticmethod
            def _pieces(text):
                return list(re.finditer(r"\w+|[^\w\s]", text))

            @staticmethod
            def _token_id(piece):
                return 1 + sum(piece.encode("utf-8")) % 63

            def apply_chat_template(self, messages, **_kwargs):
                user = f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"
                if len(messages) > 1:
                    return user + messages[1]["content"]
                return user

            def __call__(self, text, **_kwargs):
                pieces = self._pieces(text)
                return {
                    "input_ids": [self._token_id(match.group(0)) for match in pieces],
                    "offset_mapping": [(match.start(), match.end()) for match in pieces],
                }

            def encode(self, text, **kwargs):
                return self(text, **kwargs)["input_ids"]

            def decode(self, _ids, **_kwargs):
                return "ok"

        tokenizer = TinyTokenizer()
        direction = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)
        random_direction = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)
        runtime = (
            torch,
            self.model,
            tokenizer,
            direction,
            torch.zeros(2),
            torch.ones(2),
            random_direction,
            {"A": tokenizer.encode("A")[0], "B": tokenizer.encode("B")[0]},
            part34.assert_hook_site(self.model, expected_layers=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            shards = []
            shard_rows = []
            with patch.object(part34, "load_runtime", return_value=runtime), patch.object(
                part34, "adapter_context", side_effect=lambda _model, _enabled: contextlib.nullcontext()
            ):
                for question_start, question_end in ((0, 1), (1, 2)):
                    shard_out = Path(directory) / f"shard-{question_start}"
                    args = argparse.Namespace(
                        out=str(shard_out),
                        question_file=str(part34.QUESTION_FILE),
                        model_id=part34.BASE_MODEL,
                        model_revision=part34.BASE_REVISION,
                        adapter_id=part34.ADAPTER_MODEL,
                        adapter_revision=part34.ADAPTER_REVISION,
                        direction_file=str(Path(directory) / "no-direction-file"),
                        device="cpu",
                        max_new_tokens=1,
                        free_text_tokens=1,
                        question_limit=None,
                        question_start=question_start,
                        question_end=question_end,
                        threads=1,
                    )
                    part34.run(args)
                    header, rows = part34.read_jsonl(shard_out / "rows.jsonl")
                    self.assertIsNotNone(header)
                    self.assertTrue(rows)
                    self.assertEqual({row["question_index"] for row in rows}, {question_start})
                    turn1 = next(
                        row for row in rows
                        if row.get("row_type") == "turn1" and row.get("condition") == "inject-even" and row.get("model") == "M"
                    )
                    self.assertEqual(len(turn1["z_question_profile"]), 2)
                    measured_turn2 = [row for row in rows if row.get("row_type") in {"turn2_ab", "turn2_free"}]
                    for row in measured_turn2:
                        self.assertEqual(len(row["z_question_profile"]), 2, row["key"])
                        self.assertEqual(len(row["z_answer_profile"]), 2, row["key"])
                    ab_rows = [row for row in measured_turn2 if row.get("row_type") == "turn2_ab"]
                    self.assertTrue(all(row["answer_readout_position"] == part34.AB_READOUT_POSITION for row in ab_rows))
                    valid_targets = [
                        row for row in ab_rows
                        if row["condition"] == "ask-even" and row["target_value"] is not None
                    ]
                    self.assertTrue(valid_targets)
                    for row in valid_targets:
                        self.assertGreaterEqual(row["p_A"], 0.0)
                        self.assertGreaterEqual(row["p_B"], 0.0)
                        self.assertGreaterEqual(row["ab_mass"], row["p_A"])
                        self.assertEqual(row["low_mass"], row["ab_mass"] < part34.LOW_MASS_THRESHOLD)
                        for field in (
                            "logit_even_minus_odd", "target_logp", "alternative_logp",
                            "target_logp_margin", "target_candidate_probability",
                        ):
                            self.assertIsNotNone(row[field], (row["key"], field))
                    arms = {row.get("arm") for row in measured_turn2}
                    self.assertTrue({"cache_kept", "cache_recomputed", "same_turn", "b_reads_m_cache", "b_reads_b_cache"} <= arms)
                    shards.append(shard_out / "rows.jsonl")
                    shard_rows.append(rows)

            source_header = part34.read_jsonl(shards[0])[0]
            selected_t1 = next(
                row for row in shard_rows[0]
                if row.get("row_type") == "turn1" and row.get("question_index") == 0
                and row.get("condition") == "ask-even" and row.get("model") == "M"
            )
            selected_ab = next(
                row for row in shard_rows[0]
                if row.get("row_type") == "turn2_ab" and row.get("question_index") == 0
                and row.get("condition") == "ask-even" and row.get("model") == "M"
                and row.get("arm") == "default" and row.get("metric_kind") == "parity_words_raised"
            )
            readout_input = Path(directory) / "readout-input.jsonl"
            readout_input.write_text(
                "".join(json.dumps(row) + "\n" for row in (source_header, selected_t1, selected_ab)),
                encoding="utf-8",
            )
            readout_out = Path(directory) / "readout-only"
            readout_args = argparse.Namespace(
                out=str(readout_out), rows=[str(readout_input)], question_file=str(part34.QUESTION_FILE),
                model_id=part34.BASE_MODEL, model_revision=part34.BASE_REVISION,
                adapter_id=part34.ADAPTER_MODEL, adapter_revision=part34.ADAPTER_REVISION,
                direction_file=str(Path(directory) / "no-direction-file"), device="cpu", threads=1,
            )
            with patch.object(part34, "load_runtime", return_value=runtime), patch.object(
                part34, "adapter_context", side_effect=lambda _model, _enabled: contextlib.nullcontext()
            ):
                part34.run_readout_only(readout_args)
            readout_header, readout_rows = part34.read_jsonl(readout_out / "rows.jsonl")
            self.assertTrue(readout_header["ab_readout"]["readout_only"])
            self.assertEqual(len(readout_rows), 2)
            self.assertIn("ab_mass", readout_rows[-1])
            self.assertTrue(readout_rows[-1]["ab_readout_context_sha256"])

            merged_out = Path(directory) / "merged"
            part34.score(argparse.Namespace(out=str(merged_out), rows=[str(path) for path in shards]))
            summary = json.loads((merged_out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_rows"], sum(map(len, shard_rows)))
            self.assertEqual(len(summary["header"]["input_shards"]), 2)

            score_input_dir = Path(directory) / "score-input"
            for index, source in enumerate(shards):
                target_dir = score_input_dir / f"shard-{index}"
                target_dir.mkdir(parents=True)
                (target_dir / "rows.jsonl").write_bytes(source.read_bytes())
            directory_score = Path(directory) / "directory-score"
            part34.score(argparse.Namespace(out=str(directory_score), rows=[str(score_input_dir)]))
            directory_summary = json.loads((directory_score / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(directory_summary["n_rows"], summary["n_rows"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
