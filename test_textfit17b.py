"""Offline synthetic checks for Part 17b's cross-fitting and fit-file contract."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import residual17 as r17
import textfit17b as t17b


class TextFit17bTests(unittest.TestCase):
    def test_folds_are_deterministic_balanced_and_hash_order_independent(self):
        ids = [f"{index:064x}" for index in range(37)]
        first = t17b.passage_folds(ids, 5)
        reversed_folds = t17b.passage_folds(list(reversed(ids)), 5)
        mapped = {passage_id: int(fold) for passage_id, fold in zip(ids, first)}
        reversed_mapped = {
            passage_id: int(fold) for passage_id, fold in zip(reversed(ids), reversed_folds)
        }
        self.assertEqual(mapped, reversed_mapped)
        self.assertEqual(sorted(np.bincount(first).tolist()), [7, 7, 7, 8, 8])
        with self.assertRaises(ValueError):
            t17b.passage_folds(ids + [ids[0]], 5)

    def test_oof_predicts_every_row_once_and_rejects_own_fold_leakage(self):
        ids = [f"passage-{index}" for index in range(25)]
        folds = t17b.passage_folds(ids, 5)
        targets = np.arange(25, dtype=np.float64)[:, None]
        calls = []

        def fake_fit_predict(fold, training, validation, standardized_training):
            self.assertTrue(np.all(folds[training] != fold))
            self.assertTrue(np.all(folds[validation] == fold))
            self.assertEqual(len(standardized_training), len(training))
            calls.append((fold, training.copy(), validation.copy()))
            # A zero standardized prediction de-standardizes to the training mean.
            return np.zeros((len(validation), 1), dtype=np.float64)

        prediction, _ = t17b.crossfit_standardized_targets(targets, folds, fake_fit_predict)
        self.assertEqual(len(calls), 5)
        counts = np.zeros(len(targets), dtype=np.int8)
        for _, training, validation in calls:
            counts[validation] += 1
            self.assertFalse(set(training).intersection(validation))
        np.testing.assert_array_equal(counts, np.ones(len(targets), dtype=np.int8))

        # Plant a large held-out value. It must not alter that passage's own OOF prediction.
        planted = targets.copy()
        planted[folds == 2] += 100_000.0
        planted_prediction, _ = t17b.crossfit_standardized_targets(
            planted, folds, fake_fit_predict
        )
        np.testing.assert_allclose(prediction[folds == 2], planted_prediction[folds == 2])

    def test_target_standardization_uses_training_rows_only(self):
        targets = np.column_stack((np.arange(12, dtype=float), np.arange(12, dtype=float) ** 2))
        training = np.asarray([0, 1, 2, 3, 6, 7, 8, 9])
        validation = np.asarray([4, 5, 10, 11])
        mean1, scale1, standardized1 = t17b.standardize_from_training(targets, training)
        poisoned = targets.copy()
        poisoned[validation] = 1e9
        mean2, scale2, standardized2 = t17b.standardize_from_training(poisoned, training)
        np.testing.assert_array_equal(mean1, mean2)
        np.testing.assert_array_equal(scale1, scale2)
        np.testing.assert_array_equal(standardized1, standardized2)

    def test_written_fit_npz_loads_with_residual17_reader_from_separate_dir(self):
        rng = np.random.default_rng(1717)
        passage_ids = np.asarray([f"{index:064x}" for index in range(12)], dtype="U64")
        splits = np.asarray(["FIT"] * 5 + ["TRAIN"] * 4 + ["TEST"] * 3, dtype="U5")
        activations = rng.normal(size=(12, len(r17.GROUP_NAMES), 11)).astype(np.float32)
        folds = t17b.passage_folds(passage_ids)
        requested = ["G1", "GL"]
        predictions = rng.normal(size=(12, len(requested)))
        fit_arrays, _ = t17b._fit_file_arrays(
            passage_ids, splits, activations, requested, predictions, folds
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_dir = root / "capture"
            fit_dir = root / "fit"
            capture_dir.mkdir()
            fit_dir.mkdir()
            np.savez_compressed(
                capture_dir / "capture.npz",
                passage_ids=passage_ids,
                splits=splits,
                olmo_activations=activations,
                qwen_features=np.zeros((12, 4), dtype=np.float32),
            )
            manifest = {
                "passages": [
                    {"passage_id": str(passage_id), "split": str(split), "text": f"text {i}"}
                    for i, (passage_id, split) in enumerate(zip(passage_ids, splits))
                ]
            }
            (capture_dir / "capture-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez_compressed(fit_dir / "fit.npz", **fit_arrays)
            (fit_dir / "fit.json").write_text(
                json.dumps({"config": {"fitted_groups": requested}}), encoding="utf-8"
            )

            arrays, loaded_manifest, fit_report = r17._read_fit(capture_dir, fit_dir)
            self.assertEqual(arrays["labels"].shape, (12, len(r17.GROUP_NAMES)))
            self.assertEqual(arrays["residuals"].shape, activations.shape)
            self.assertEqual(arrays["predictions"].shape, activations.shape)
            self.assertEqual(arrays["passage_ids"].astype(str).tolist(), passage_ids.tolist())
            np.testing.assert_allclose(arrays["low20"], r17.fit_activation_quantiles(activations, splits)[0])
            np.testing.assert_allclose(arrays["high80"], r17.fit_activation_quantiles(activations, splits)[1])
            self.assertEqual(len(loaded_manifest["passages"]), 12)
            self.assertEqual(fit_report["config"]["fitted_groups"], requested)

    def test_reader_refuses_a_group_without_a_text_predictor(self):
        with self.assertRaisesRegex(ValueError, "not G2"):
            r17._require_fitted_group("G2", {"config": {"fitted_groups": ["G1", "GL"]}})
        r17._require_fitted_group("GL", {"config": {"fitted_groups": ["G1", "GL"]}})


if __name__ == "__main__":
    unittest.main()
