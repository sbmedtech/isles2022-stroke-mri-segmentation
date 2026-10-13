"""Synthetic edge cases and parity with the notebook's evaluation expression."""

import unittest

import numpy as np

from isles2022.metrics import dice_score, patient_metrics, summarize_patient_metrics


class MetricsTests(unittest.TestCase):
    def test_empty_perfect_disjoint_and_partial_masks(self):
        cases = [
            ([0, 0], [0, 0], 1.0),
            ([1, 0], [0, 0], 0.0),
            ([0, 0], [1, 0], 0.0),
            ([1, 0], [1, 0], 1.0),
            ([1, 0], [0, 1], 0.0),
            ([1, 1], [1, 0], 2 / 3),
        ]
        for prediction, expert, expected in cases:
            with self.subTest(prediction=prediction, expert=expert):
                self.assertEqual(dice_score(np.array(prediction), np.array(expert)), expected)

    def test_threshold_is_inclusive_and_labels_are_positive(self):
        probabilities = np.array([0.249, 0.25, 0.251, 0.9])
        labels = np.array([0, 2, 1, -1])
        self.assertEqual(patient_metrics(probabilities, labels, threshold=0.25), {
            "dice": 4 / 5, "expert_voxels": 2,
            "predicted_voxels": 3, "intersection_voxels": 2,
        })

    def test_random_full_volume_parity_with_notebook(self):
        rng = np.random.default_rng(42)
        probability = rng.random((8, 9, 10))
        expert = rng.integers(0, 2, size=probability.shape) > 0
        for threshold in np.arange(0.05, 0.51, 0.05):
            prediction = probability >= threshold
            expert_voxels = int(expert.sum())
            predicted_voxels = int(prediction.sum())
            intersection = int(np.logical_and(prediction, expert).sum())
            denominator = expert_voxels + predicted_voxels
            expected = 2 * intersection / denominator if denominator > 0 else 1.0
            self.assertEqual(dice_score(probability, expert, threshold=threshold), expected)

    def test_patient_average_excludes_negative_cases_and_counts_misses(self):
        rows = [
            patient_metrics(np.array(p), np.array(g))
            for p, g in [
                ([1, 0], [1, 0]),  # Perfect lesion-positive patient.
                ([0, 0], [1, 0]),  # Missed patient.
                ([0, 1], [1, 0]),  # Zero Dice, but not a zero-prediction miss.
                ([0, 0], [0, 0]),  # Empty negative patient excluded from mean.
                ([1, 0], [0, 0]),  # False-positive negative patient.
            ]
        ]
        self.assertEqual(summarize_patient_metrics(rows), {
            "mean_dice": 1 / 3, "median_dice": 0.0,
            "missed_cases": 1, "false_positive_negative_cases": 1,
        })

    def test_no_positive_patients_matches_nan_reductions(self):
        for rows in [[], [patient_metrics(np.zeros(3), np.zeros(3))]]:
            summary = summarize_patient_metrics(rows)
            self.assertTrue(np.isnan(summary["mean_dice"]))
            self.assertTrue(np.isnan(summary["median_dice"]))
            self.assertEqual(summary["missed_cases"], 0)
            self.assertEqual(summary["false_positive_negative_cases"], 0)

    def test_shape_mismatch_does_not_broadcast_voxel_grids(self):
        with self.assertRaises(ValueError):
            patient_metrics(np.zeros((2, 1)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
