"""Offline synthetic checks for Part 17's split, ridge, labels, and bounds."""

import unittest

import numpy as np

from residual17 import (
    fit_activation_quantiles,
    fit_threshold_labels,
    ridge_fit,
    ridge_predict,
    split_by_hash,
)


class Residual17Tests(unittest.TestCase):
    def test_split_is_hash_deterministic_and_order_independent(self):
        rows = [{"text": f"synthetic passage number {i}"} for i in range(40)]
        first = split_by_hash(rows, 12, 18, 10)
        second = split_by_hash(list(reversed(rows)), 12, 18, 10)
        assignment1 = {row["passage_id"]: row["split"] for row in first}
        assignment2 = {row["passage_id"]: row["split"] for row in second}
        self.assertEqual(assignment1, assignment2)
        self.assertEqual({name: sum(row["split"] == name for row in first)
                          for name in ("FIT", "TRAIN", "TEST")},
                         {"FIT": 12, "TRAIN": 18, "TEST": 10})

    def test_closed_form_ridge_recovers_linear_relation_and_median_balances_residuals(self):
        rng = np.random.default_rng(1701)
        x = rng.normal(size=(120, 5))
        planted = rng.normal(size=(5, 3))
        offset = np.asarray([1.5, -2.0, 0.25])
        y = x @ planted + offset
        weights, bias = ridge_fit(x, y, alpha=1e-10)
        np.testing.assert_allclose(ridge_predict(x, weights, bias), y, rtol=1e-8, atol=1e-8)

        noisy_y = y + rng.normal(scale=0.2, size=y.shape)
        noisy_weights, noisy_bias = ridge_fit(x, noisy_y, alpha=0.1)
        residual = (noisy_y - ridge_predict(x, noisy_weights, noisy_bias)).mean(axis=1)
        threshold, labels = fit_threshold_labels(residual, ["FIT"] * len(residual))
        self.assertAlmostEqual(threshold, float(np.median(residual)))
        self.assertEqual(int(labels.sum()), len(residual) // 2)

    def test_threshold_uses_fit_median_for_train_and_test_labels(self):
        residual = np.asarray([1.0, 3.0, 5.0, 7.0, 3.5, 5.5, -100.0, 100.0])
        splits = ["FIT"] * 4 + ["TRAIN"] * 2 + ["TEST"] * 2
        threshold, labels = fit_threshold_labels(residual, splits)
        self.assertEqual(threshold, 4.0)
        np.testing.assert_array_equal(labels[4:], [0, 1, 0, 1])

    def test_transplant_quantiles_ignore_non_fit_activations(self):
        activations = np.zeros((8, 2, 3), dtype=np.float64)
        activations[:4] = np.arange(4, dtype=np.float64)[:, None, None]
        activations[4:] = 1000.0
        splits = ["FIT"] * 4 + ["TRAIN"] * 2 + ["TEST"] * 2
        low1, high1 = fit_activation_quantiles(activations, splits)
        changed = activations.copy()
        changed[4:] = -900000.0
        low2, high2 = fit_activation_quantiles(changed, splits)
        np.testing.assert_array_equal(low1, low2)
        np.testing.assert_array_equal(high1, high2)
        self.assertTrue(np.all(low1 < high1))

    def test_planted_test_shortcut_cannot_move_threshold_or_train_labels(self):
        residual = np.asarray([0.0, 1.0, 2.0, 3.0, 0.5, 2.5, 4.0, 5.0])
        splits = ["FIT"] * 4 + ["TRAIN"] * 2 + ["TEST"] * 2
        threshold1, labels1 = fit_threshold_labels(residual, splits)
        poisoned_test = residual.copy()
        poisoned_test[-2:] = [1_000_000.0, 2_000_000.0]
        threshold2, labels2 = fit_threshold_labels(poisoned_test, splits)
        self.assertEqual(threshold1, 1.5)
        self.assertEqual(threshold2, threshold1)
        np.testing.assert_array_equal(labels2[4:6], labels1[4:6])


if __name__ == "__main__":
    unittest.main()
