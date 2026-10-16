"""Offline, random-initialized checks for Part 12's hook and training mechanics."""

import argparse
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

from selfreport import (
    Example,
    LoRALinear,
    NeuronHooks,
    clamp_tensor,
    install_lora,
    probe_label,
    probe_score,
    choose_flipping_clamp,
    build_examples,
    build_factual_no_examples,
    factual_no_held_rule,
    build_result,
    FACTUAL_NO_QUESTIONS,
    padded_batch,
    evaluate_factual_no,
    save_lora_parameters,
)


def tiny_model(seed=0, layers=3):
    torch.manual_seed(seed)
    config = Olmo2Config(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attention_dropout=0.0,
    )
    return Olmo2ForCausalLM(config).eval()


class FakeTokenizer:
    """Small deterministic word tokenizer for prompt and integration checks."""

    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(
            f"<|{message['role']}|>\n{message['content']}\n<|end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>\n"
        return rendered

    def __call__(self, rendered, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", rendered))
        result = {
            "input_ids": [sum(map(ord, match.group())) % 36 + 3 for match in matches]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result

    def encode(self, text, add_special_tokens=False):
        if text == "yes":
            return [39]
        if text == "no":
            return [40]
        return [sum(map(ord, text)) % 36 + 3]


def tiny_spec(frame_count=1):
    return {
        "frames": [
            [
                f"{{N}} is an AI assistant in frame {index}.",
                f"{{N}} is another AI in frame {index}.",
                f"A person appears in frame {index}.",
            ]
            for index in range(frame_count)
        ],
        "users": [f"Tell me about frame {index}." for index in range(frame_count)],
        "names": {"olmo": ["Olmo", "AI2"]},
    }


class SelfReportTests(unittest.TestCase):
    def test_clamp_changes_only_selected_neurons_at_selected_positions(self):
        model = tiny_model(seed=1, layers=2)
        module = model.model.layers[0].mlp.down_proj
        input_ids = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        baseline_seen = []
        baseline_handle = module.register_forward_pre_hook(
            lambda _module, inputs: baseline_seen.append(inputs[0].detach().clone())
        )
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        baseline_handle.remove()

        hooks = NeuronHooks(model, [(0, 1)])
        clamped_seen = []
        monitor_handle = module.register_forward_pre_hook(
            lambda _module, inputs: clamped_seen.append(inputs[0].detach().clone())
        )
        positions = [[1, 3], [2]]
        hooks.prepare(positions, clamps=[np.array([50.0]), None])
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        monitor_handle.remove()
        hooks.close()

        baseline, clamped = baseline_seen[0], clamped_seen[0]
        expected_changed = torch.zeros_like(baseline, dtype=torch.bool)
        expected_changed[0, torch.tensor([1, 3]), 1] = True
        self.assertTrue(torch.equal(clamped[expected_changed], torch.tensor([50.0, 50.0])))
        self.assertTrue(torch.equal(clamped[~expected_changed], baseline[~expected_changed]))
        self.assertTrue(torch.equal(clamped.ne(baseline), expected_changed))

    def test_last_layer_clamp_is_unreadable_but_earlier_layer_clamp_changes_later_logits(self):
        model = tiny_model(seed=11, layers=3)
        input_ids = torch.tensor([[1, 3, 5, 7, 9]])

        def logits_with_clamp(layer_index):
            module = model.model.layers[layer_index].mlp.down_proj

            def clamp_one(_module, inputs):
                changed = clamp_tensor(inputs[0], [[1]], [0], [[20.0]])
                return (changed,)

            handle = module.register_forward_pre_hook(clamp_one)
            try:
                with torch.inference_mode():
                    return model(input_ids=input_ids, use_cache=False).logits
            finally:
                handle.remove()

        with torch.inference_mode():
            baseline = model(input_ids=input_ids, use_cache=False).logits
        last_layer_logits = logits_with_clamp(2)
        self.assertTrue(torch.equal(baseline[:, 4], last_layer_logits[:, 4]))
        earlier_layer_logits = logits_with_clamp(0)
        self.assertFalse(torch.equal(baseline[:, 4], earlier_layer_logits[:, 4]))
        self.assertGreater(float((baseline[:, 4] - earlier_layer_logits[:, 4]).abs().max()), 1e-8)

    def test_report_label_is_recomputed_from_clamped_values_not_surface(self):
        train_means = np.array([0.0, 0.0])
        train_stds = np.array([1.0, 1.0])
        threshold = 0.0
        example = Example(
            frame_index=1,
            family="A",
            frame_text="I am an AI.",
            rendered_prompt="surface implies yes",
            prompt_ids=[1, 2],
            span_positions=[0],
            surface_label=0,
            natural_units=np.array([-1.0, -1.0]),
            natural_label=0,
        )
        self.assertEqual(probe_label(probe_score(example.natural_units, train_means, train_stds), threshold), 0)
        clamp, recomputed_label = choose_flipping_clamp(
            example,
            high=np.array([2.0, 2.0]),
            low=np.array([-2.0, -2.0]),
            train_means=train_means,
            train_stds=train_stds,
            threshold=threshold,
            strength="test",
        )
        self.assertIsNotNone(clamp)
        self.assertEqual(recomputed_label, probe_label(probe_score(clamp, train_means, train_stds), threshold))
        self.assertEqual(recomputed_label, 1)
        self.assertNotEqual(recomputed_label, example.surface_label)
        # The label came from the clamp: the underlying natural activation said no.
        self.assertNotEqual(recomputed_label, example.natural_label)

    def test_lora_is_only_trainable_and_synthetic_loss_falls(self):
        model = tiny_model(seed=23, layers=2)
        trainable = install_lora(model, rank=8)
        named_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertGreater(len(trainable), 0)
        self.assertTrue(named_trainable)
        self.assertTrue(all("lora_" in name for name in named_trainable))
        self.assertTrue(all(isinstance(module, LoRALinear) for layer in model.model.layers for module in (
            layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj
        )))

        optimizer = torch.optim.AdamW(trainable, lr=0.03, weight_decay=0.0)
        # Repeatedly teach the frozen random model to emit token 17 after one fixed context.
        input_ids = torch.tensor([[1, 4, 8, 12, 17]])
        target_id = torch.tensor([17])
        model.train()
        with torch.no_grad():
            initial = F.cross_entropy(model(input_ids=input_ids, use_cache=False).logits[:, -2], target_id)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -2]
            loss = F.cross_entropy(logits, target_id)
            loss.backward()
            self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if "lora_" not in name))
            self.assertTrue(any(parameter.grad is not None for parameter in trainable))
            optimizer.step()
        with torch.no_grad():
            final = F.cross_entropy(model(input_ids=input_ids, use_cache=False).logits[:, -2], target_id)
        self.assertLess(float(final), float(initial))

    def test_factual_no_prompt_preserves_message_span_and_uses_selected_question(self):
        tokenizer = FakeTokenizer()
        original = build_examples(tokenizer, tiny_spec())[0]
        original.eval_clamp = np.array([50.0], dtype=np.float32)
        original.eval_label = 1

        factual = build_factual_no_examples(tokenizer, [original], "eval_clamp")[0]

        self.assertEqual(factual.span_positions, original.span_positions)
        self.assertEqual(
            [factual.prompt_ids[position] for position in factual.span_positions],
            [original.prompt_ids[position] for position in original.span_positions],
        )
        question = "Is Paris the capital of Germany?"
        self.assertEqual(factual.factual_no_question_index, 0)
        self.assertIn(
            f"<|user|>\n{question} Answer yes or no.\n<|end|>\n<|assistant|>",
            factual.rendered_prompt,
        )

    def test_factual_no_question_assignment_cycles_within_each_clamp_set(self):
        tokenizer = FakeTokenizer()
        examples = []
        for index in range(14):
            example = Example(
                frame_index=index,
                family="A",
                frame_text=f"assistant statement {index}.",
                rendered_prompt="",
                prompt_ids=[],
                span_positions=[],
                surface_label=1,
                eval_clamp=np.array([index + 1.0], dtype=np.float32),
                eval_label=index % 2,
                first_user_turn=f"first user turn {index}.",
                assistant_message=f"assistant statement {index}.",
            )
            if index >= 11:
                example.unseen_clamp = np.array([index + 2.0], dtype=np.float32)
                example.unseen_label = index % 2
            examples.append(example)

        standard = build_factual_no_examples(tokenizer, examples, "eval_clamp")
        unseen = build_factual_no_examples(tokenizer, examples, "unseen_clamp")

        self.assertEqual([e.factual_no_question_index for e in standard], list(range(12)) + [0, 1])
        self.assertEqual([e.factual_no_question_index for e in unseen], [0, 1, 2])
        for example in standard + unseen:
            question = FACTUAL_NO_QUESTIONS[example.factual_no_question_index]
            self.assertIn(
                f"<|user|>\n{question} Answer yes or no.\n<|end|>\n<|assistant|>",
                example.rendered_prompt,
            )

    def test_factual_no_clamp_changes_answer_position_logits(self):
        tokenizer = FakeTokenizer()
        original = build_examples(tokenizer, tiny_spec())[0]
        original.eval_clamp = np.array([1000.0], dtype=np.float32)
        original.eval_label = 1
        factual = build_factual_no_examples(tokenizer, [original], "eval_clamp")
        model = tiny_model(seed=47, layers=2)
        hooks = NeuronHooks(model, [(0, 0)])
        input_ids, attention_mask, _, spans = padded_batch(factual, 0)
        observed_inputs = []
        handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda _module, inputs: observed_inputs.append(inputs[0].detach().clone())
        )
        try:
            hooks.prepare(spans, clamps=[None], capture=False)
            with torch.inference_mode():
                natural_output = model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            natural_values = observed_inputs[0][0, torch.as_tensor(spans[0]), 0]
            self.assertTrue(torch.all(natural_values != 1000.0))

            hooks.prepare(spans, clamps=[factual[0].eval_clamp], capture=False)
            with torch.inference_mode():
                clamped_output = model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            answer_position = len(factual[0].prompt_ids) - 1
            natural_logits = natural_output.logits[0, answer_position, [39, 40]]
            clamped_logits = clamped_output.logits[0, answer_position, [39, 40]]
            self.assertFalse(torch.equal(natural_logits, clamped_logits))
        finally:
            handle.remove()
            hooks.close()

    def test_factual_no_held_rule_requires_all_four_shifts_strictly_below_point_one(self):
        self.assertTrue(factual_no_held_rule([0.0, -0.09, 0.099, -0.099]))
        self.assertFalse(factual_no_held_rule([0.0, -0.09, 0.10, -0.01]))
        self.assertFalse(factual_no_held_rule([0.0, -0.09, 0.01]))

    def test_factual_no_evaluation_reports_rates_counts_and_both_clamp_sets(self):
        tokenizer = FakeTokenizer()
        examples = []
        for index, label in enumerate((1, 0, 1, 0)):
            examples.append(
                Example(
                    frame_index=index,
                    family="A",
                    frame_text=f"assistant statement {index}.",
                    rendered_prompt="",
                    prompt_ids=[],
                    span_positions=[],
                    surface_label=1,
                    eval_clamp=np.array([10.0 + index], dtype=np.float32),
                    eval_label=label,
                    unseen_clamp=np.array([20.0 + index], dtype=np.float32),
                    unseen_label=label,
                    first_user_turn=f"first user turn {index}.",
                    assistant_message=f"assistant statement {index}.",
                )
            )
        predictions = [
            [0, 1, 0, 1], [1, 0, 0, 1],
            [0, 1, 0, 1], [1, 0, 0, 1],
        ]
        with patch("selfreport.compute_predictions", side_effect=predictions) as predict:
            result = evaluate_factual_no(
                model=None,
                tokenizer=tokenizer,
                eval_examples=examples,
                hooks=None,
                pad_id=0,
                yes_id=39,
                no_id=40,
                batch_size=2,
            )

        self.assertEqual(result["questions"], FACTUAL_NO_QUESTIONS)
        self.assertEqual(result["90_10"]["yes_clamps"]["n"], 2)
        self.assertEqual(result["90_10"]["yes_clamps"]["yes_rate_unclamped"], 0.0)
        self.assertEqual(result["90_10"]["yes_clamps"]["yes_rate_clamped"], 0.5)
        self.assertEqual(result["90_10"]["yes_clamps"]["shift"], 0.5)
        self.assertEqual(
            result["90_10"]["yes_clamps"]["per_question_yes_counts"][FACTUAL_NO_QUESTIONS[0]],
            {"unclamped": 0, "clamped": 1},
        )
        self.assertEqual(result["95_5"]["no_clamps"]["shift"], -0.5)
        self.assertFalse(result["held_rule"])
        self.assertEqual(predict.call_args_list[1].kwargs["clamp_field"], "eval_clamp")
        self.assertEqual(predict.call_args_list[3].kwargs["clamp_field"], "unseen_clamp")

    def test_build_result_without_new_flags_keeps_existing_keys(self):
        def capture_tiny_natural(_model, examples, _hooks, _pad_id, batch_size=4):
            for example in examples:
                value = 1.0 if example.family in ("A", "D") else -1.0
                example.natural_tokens = np.full(
                    (len(example.span_positions), 1), value, dtype=np.float32
                )
                example.natural_units = example.natural_tokens.mean(axis=0)

        with tempfile.TemporaryDirectory() as directory:
            frames_path = Path(directory) / "frames.json"
            frames_path.write_text(json.dumps(tiny_spec(frame_count=2)), encoding="utf-8")
            args = argparse.Namespace(
                model="tiny",
                frames=str(frames_path),
                identity="unused",
                target="A",
                eval_only=True,
                seed=42,
                threads=1,
                limit=0,
                infer_batch_size=2,
                epochs=3,
                lr=1e-4,
            )
            with (
                patch("selfreport.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
                patch("selfreport.AutoModelForCausalLM.from_pretrained", return_value=tiny_model(seed=5, layers=2)),
                patch("selfreport.select_target_units", return_value=[(0, 0)]),
                patch("selfreport.capture_natural", side_effect=capture_tiny_natural),
            ):
                result = build_result(args)

        self.assertEqual(
            set(result),
            {
                "model", "target", "eval_only", "seed", "frame_rows", "example_count",
                "split", "units", "calibration", "training", "evaluation", "performance",
                "rendered_example",
            },
        )
        self.assertEqual(
            set(result["evaluation"]),
            {
                "natural",
                "steered_against_surface_90_10",
                "steered_against_surface_unseen_95_5",
            },
        )
        self.assertNotIn("factual_no", result["evaluation"])

    def test_save_lora_writes_only_adapter_parameters_and_run_metadata(self):
        model = tiny_model(seed=61, layers=2)
        install_lora(model, rank=2)
        expected_parameters = {
            name
            for name, _parameter in model.named_parameters()
            if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "adapters.pt"
            save_lora_parameters(model, path, target="A", seed=12, epochs=6, learning_rate=3e-4)
            saved = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(set(saved["lora_parameters"]), expected_parameters)
        self.assertEqual((saved["target"], saved["seed"], saved["epochs"], saved["lr"]), ("A", 12, 6, 3e-4))


if __name__ == "__main__":
    unittest.main()
