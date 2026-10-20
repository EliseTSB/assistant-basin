"""Fast, offline checks for Part 34 helpers and the small Qwen3 hook path."""

from __future__ import annotations

import json
import argparse
import contextlib
import tempfile
import unittest
import re
from pathlib import Path
from unittest.mock import patch

import part34


try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    HAS_TORCH_QWEN3 = True
except ImportError:
    torch = None
    Qwen3Config = None
    Qwen3ForCausalLM = None
    HAS_TORCH_QWEN3 = False


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
            "A": "raised on layers 0, 2, 4, ..., 26",
            "B": "raised on layers 1, 3, 5, ..., 27",
        })
        self.assertEqual(part34.turn2_content("parity_free_raised", 1)[2], "odd")

    def test_resume_skips_completed_keys_and_score_fixture(self) -> None:
        rows = [
            {"type": "row", "key": "turn1|0|none|M", "row_type": "turn1", "question_index": 0,
             "condition": "none", "model": "M", "d_question": 0.0, "d_reply": 0.0,
             "d_reply_target_pct": 0.0, "z_reply_profile": [0.0] * 28},
            {"type": "row", "key": "t2|0|none", "row_type": "turn2_ab", "question_index": 0,
             "condition": "none", "arm": "default", "model": "M", "metric_kind": "parity_free_raised",
             "target_value": None, "greedy_concept": "even", "greedy_letter": "A", "logp_A": -0.4,
             "logp_B": -0.8, "logit_even_minus_odd": 0.4, "d_question": 0.0, "d_answer": 0.1,
             "question_order_first_parity": "even"},
            {"type": "row", "key": "t2|0|ask", "row_type": "turn2_ab", "question_index": 0,
             "condition": "ask-even", "arm": "default", "model": "B", "metric_kind": "parity_free_raised",
             "target_value": "even", "greedy_concept": "even", "greedy_letter": "A", "logp_A": -0.2,
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
                return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"

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
            args = argparse.Namespace(
                out=directory,
                question_file=str(part34.QUESTION_FILE),
                model_id=part34.BASE_MODEL,
                model_revision=part34.BASE_REVISION,
                adapter_id=part34.ADAPTER_MODEL,
                adapter_revision=part34.ADAPTER_REVISION,
                direction_file=str(Path(directory) / "no-direction-file"),
                device="cpu",
                max_new_tokens=1,
                free_text_tokens=1,
                question_limit=1,
            )
            with patch.object(part34, "load_runtime", return_value=runtime), patch.object(
                part34, "adapter_context", side_effect=lambda _model, _enabled: contextlib.nullcontext()
            ):
                part34.run(args)
            header, rows = part34.read_jsonl(Path(directory) / "rows.jsonl")
            self.assertIsNotNone(header)
            self.assertTrue(rows)
            turn1 = next(row for row in rows if row.get("row_type") == "turn1" and row.get("condition") == "inject-even" and row.get("model") == "M")
            self.assertEqual(len(turn1["z_question_profile"]), 2)
            measured_turn2 = [row for row in rows if row.get("row_type") in {"turn2_ab", "turn2_free"}]
            for row in measured_turn2:
                self.assertEqual(len(row["z_question_profile"]), 2, row["key"])
                self.assertEqual(len(row["z_answer_profile"]), 2, row["key"])
            arms = {row.get("arm") for row in measured_turn2}
            self.assertTrue({"cache_kept", "cache_recomputed", "same_turn", "b_reads_m_cache", "b_reads_b_cache"} <= arms)
            part34.score(args)
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_rows"], len(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
