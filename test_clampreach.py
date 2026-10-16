"""Tiny-model tests for the Part 12b answer-position reach measurement."""

import unittest
from types import SimpleNamespace

import numpy as np

from clampreach import (
    capture_condition,
    compute_clamp_values,
    fit_readout_for_layer,
    readout_metrics,
)
from selfreport import NeuronHooks, activation_quantiles, build_examples, capture_natural
from test_selfreport import FakeTokenizer, tiny_model, tiny_spec


class ClampReachTests(unittest.TestCase):
    def test_clamps_match_selfreport_quantiles_on_tiny_model(self):
        tokenizer = FakeTokenizer()
        model = tiny_model(seed=71, layers=3)
        examples = build_examples(tokenizer, tiny_spec(frame_count=4))
        units = [(0, 0), (1, 2)]
        hooks = NeuronHooks(model, units)
        try:
            capture_natural(model, examples, hooks, pad_id=0, batch_size=3)
        finally:
            hooks.close()
        even_examples = [example for example in examples if example.frame_index % 2 == 0]

        actual_high, actual_low = compute_clamp_values(even_examples, len(units))
        expected_high, expected_low = activation_quantiles(
            even_examples, len(units), 0.90, 0.10
        )

        np.testing.assert_array_equal(actual_high, expected_high)
        np.testing.assert_array_equal(actual_low, expected_low)

    def _high_low_states(self, layer, clamp_high=1000.0, clamp_low=-1000.0):
        tokenizer = FakeTokenizer()
        example = build_examples(tokenizer, tiny_spec(frame_count=2))[0]
        model = tiny_model(seed=73, layers=3)
        hooks = NeuronHooks(model, [(layer, 0)])
        try:
            high, _ = capture_condition(
                model, [example], hooks, 0, 39, 40,
                [np.array([clamp_high], dtype=np.float32)], batch_size=1,
            )
            low, _ = capture_condition(
                model, [example], hooks, 0, 39, 40,
                [np.array([clamp_low], dtype=np.float32)], batch_size=1,
            )
        finally:
            hooks.close()
        return high, low

    def test_last_layer_clamp_cannot_change_answer_position_states(self):
        high, low = self._high_low_states(layer=2)
        differences = np.max(np.abs(high - low), axis=(0, 2))
        np.testing.assert_array_equal(differences, np.zeros(3, dtype=np.float32))

    def test_first_layer_clamp_changes_a_later_answer_position_state(self):
        high, low = self._high_low_states(layer=0)
        differences = np.max(np.abs(high - low), axis=(0, 2))
        self.assertTrue(np.any(differences[1:] > 0.0), differences)

    def test_readout_separates_fixed_direction_and_identical_states_are_chance(self):
        examples = [SimpleNamespace(frame_index=index) for index in range(4)]
        low = np.zeros((4, 2, 3), dtype=np.float64)
        direction = np.array([1.0, 0.0, -2.0])
        high = low + direction[None, None, :]

        separated = readout_metrics(examples, high, low)
        self.assertTrue(all(row["accuracy"] == 1.0 for row in separated))
        self.assertTrue(all(row["paired_accuracy"] == 1.0 for row in separated))

        identical = readout_metrics(examples, low.copy(), low.copy())
        self.assertTrue(all(row["accuracy"] == 0.5 for row in identical))
        self.assertTrue(all(row["paired_accuracy"] == 0.0 for row in identical))

    def test_readout_fit_uses_even_frames_only(self):
        frame_indices = [0, 1, 2, 3, 4]
        low = np.zeros((5, 2), dtype=np.float64)
        high = low + np.array([2.0, -1.0])
        fitted = fit_readout_for_layer(frame_indices, high, low)
        changed_odd_high = high.copy()
        changed_odd_low = low.copy()
        changed_odd_high[[1, 3]] += 10000.0
        changed_odd_low[[1, 3]] -= 10000.0
        refit = fit_readout_for_layer(frame_indices, changed_odd_high, changed_odd_low)

        self.assertEqual(fitted["fit_frame_indices"], [0, 2, 4])
        self.assertEqual(fitted["fit_frame_indices"], refit["fit_frame_indices"])
        np.testing.assert_array_equal(fitted["direction"], refit["direction"])
        self.assertEqual(fitted["threshold"], refit["threshold"])


if __name__ == "__main__":
    unittest.main()
