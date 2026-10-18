"""Model-free unit tests for causal_search.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import causal_search  # noqa: E402


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Identity()


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _TinyMLP()


class _TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer()])


class _TinyNeuronModel(nn.Module):
    """A one-layer model where activation 0 decides token 1 versus token 2."""

    def __init__(self):
        super().__init__()
        self.model = _TinyTrunk()

    def forward(self, input_ids):
        activation = torch.zeros((*input_ids.shape, 2), dtype=torch.float32, device=input_ids.device)
        activation[..., 0] = (input_ids == 1).float()
        hidden = self.model.layers[0].mlp.down_proj(activation)
        logits = torch.zeros((*input_ids.shape, 3), dtype=hidden.dtype, device=hidden.device)
        logits[..., 1] = hidden[..., 0] - 0.5
        return SimpleNamespace(logits=logits)


class FirstDifferenceTests(unittest.TestCase):
    def test_first_differing_position(self):
        self.assertEqual(causal_search.first_differing_position([1, 2, 3], [1, 8, 3]), 1)
        self.assertIsNone(causal_search.first_differing_position([1, 2], [1, 2]))
        self.assertIsNone(causal_search.first_differing_position([1], [1, 2]))


class AttributionTests(unittest.TestCase):
    def test_one_neuron_decides_token_and_has_negative_switch_attribution(self):
        # The stub metric is logit(token 1)-logit(token 2) = activation-0.5.
        # Run 1 has activation 1.0; run 2 has 0.0. Copying the run-2 neuron
        # lowers the metric, so delta * d(metric)/d(activation) is negative.
        run1 = [torch.tensor([[1.0, 4.0]])]
        run2 = [torch.tensor([[0.0, 4.0]])]
        activation = torch.tensor(1.0, requires_grad=True)
        metric = activation - 0.5
        gradient = torch.autograd.grad(metric, activation)[0]
        entries = causal_search.rank_attribution_entries(
            run1,
            run2,
            [torch.tensor([[float(gradient), 0.0]])],
            [0],
            limit=2,
        )
        self.assertEqual([(entry.neuron, entry.attribution) for entry in entries], [(0, -1.0)])

    def test_real_hooks_rank_and_switch_one_decisive_stub_neuron(self):
        model = _TinyNeuronModel()
        source_acts = causal_search.capture_run2_activations(model, [0], torch.device("cpu"))
        initial, run1_acts, gradients = causal_search.capture_run1_attribution(
            model, [1], torch.device("cpu"), logit_position=0, token1=1, token2=2
        )
        entries = causal_search.rank_attribution_entries(
            run1_acts, source_acts, gradients, [0], limit=4
        )
        evaluator = causal_search.make_patch_evaluator(
            model, [1], torch.device("cpu"), 0, 1, 2, source_acts
        )
        try:
            result = causal_search.greedy_patch(entries, evaluator, initial_metric=initial)
        finally:
            evaluator.remove_hooks()
        self.assertEqual(initial, 0.5)
        self.assertEqual([(entry.neuron, entry.attribution) for entry in entries], [(0, -1.0)])
        self.assertTrue(result["switched"])
        self.assertEqual([entry.neuron for entry in result["entries"]], [0])
        self.assertLess(result["metric"], 0)


class GreedyPatchTests(unittest.TestCase):
    def test_patch_stops_at_switch_and_returns_minimal_ranked_prefix(self):
        # A stub forward computes metric = 1 - 1.2 for the decisive neuron;
        # the other ranked neurons do not affect it.
        entries = [
            causal_search.PatchEntry(0, 0, neuron, 0, -1.0 + neuron * 0.01)
            for neuron in range(4)
        ]
        forward_inputs = []

        def stub_forward(patches):
            forward_inputs.append(tuple(entry.neuron for entry in patches))
            return 1.0 - (1.2 if any(entry.neuron == 0 for entry in patches) else 0.0)

        result = causal_search.greedy_patch(
            entries, stub_forward, initial_metric=1.0, chunk_size=4, max_entries=256
        )
        self.assertTrue(result["switched"])
        self.assertEqual([entry.neuron for entry in result["entries"]], [0])
        self.assertLess(result["metric"], 0)
        self.assertEqual(forward_inputs[0], (0, 1, 2, 3))

    def test_cap_exhaustion_is_not_reported_as_switch(self):
        entries = [causal_search.PatchEntry(0, 0, n, 0, -1.0) for n in range(8)]
        result = causal_search.greedy_patch(
            entries, lambda patches: 1.0, initial_metric=1.0, chunk_size=4, max_entries=4
        )
        self.assertFalse(result["switched"])
        self.assertEqual(len(result["attempted"]), 4)
        self.assertEqual(result["entries"], result["attempted"])


class AggregationTests(unittest.TestCase):
    def test_counts_each_neuron_once_per_pairs_minimal_set(self):
        records = [
            {
                "switched": True,
                "minimal_set": [
                    {"neuron": "L1.2", "position": 0},
                    {"neuron": "L1.2", "position": 5},
                    {"neuron": "L0.4", "position": 1},
                ],
            },
            {"switched": True, "minimal_set": [{"neuron": "L1.2", "position": 2}]},
            {"switched": False, "minimal_set": [{"neuron": "L0.4", "position": 3}]},
        ]
        counts, top = causal_search.aggregate_pair_neurons(records, 2)
        self.assertEqual(counts, {"L1.2": 2, "L0.4": 1})
        self.assertEqual(top, [{"neuron": "L1.2", "pair_count": 2}, {"neuron": "L0.4", "pair_count": 1}])


class RandomSetTests(unittest.TestCase):
    def test_random_sets_match_layer_counts_and_seed(self):
        counts = {0: 2, 2: 1}
        exclude = {0: [1, 3], 2: [4]}
        one = causal_search.random_sets_with_layer_counts(counts, 8, 6, seed=17, exclude=exclude)
        two = causal_search.random_sets_with_layer_counts(counts, 8, 6, seed=17, exclude=exclude)
        self.assertEqual(one, two)
        for selected in one:
            self.assertEqual({layer: len(values) for layer, values in selected.items()}, counts)
            self.assertFalse(set(selected[0]) & set(exclude[0]))
            self.assertFalse(set(selected[2]) & set(exclude[2]))


class RegexTests(unittest.TestCase):
    def test_ai_reply_regex_fixed_strings(self):
        positives = [
            "As an AI, I cannot help with that.",
            "I am an AI.",
            "I'm an AI assistant.",
            "I work as an AI assistant.",
            "I am a language model.",
            "I use artificial intelligence.",
        ]
        negatives = ["As a nurse, I care for patients.", "I enjoy astronomy."]
        self.assertTrue(all(causal_search.is_ai_reply(text) for text in positives))
        self.assertFalse(any(causal_search.is_ai_reply(text) for text in negatives))


if __name__ == "__main__":
    unittest.main(verbosity=2)
