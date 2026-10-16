"""Offline, random-initialized checks for Part 12's hook and training mechanics."""

import unittest

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
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attention_dropout=0.0,
    )
    return Olmo2ForCausalLM(config).eval()


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


if __name__ == "__main__":
    unittest.main()
