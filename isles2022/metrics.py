"""Full-volume NumPy Dice behavior from notebook evaluation cells 45 and 46.

Inputs are binary predictions (or probabilities with an explicit threshold) and
expert labels. Thresholding uses >=; expert labels use > 0. No smoothing is added.
"""

from collections.abc import Iterable, Mapping

import numpy as np


def patient_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Calculate patient Dice and voxel counts without averaging across slices.

    Pass sigmoid probabilities with ``threshold=0.25`` for the final notebook
    evaluation, or pass an already binary prediction without a threshold.
    Logits must be converted to probabilities by the caller.
    """
    prediction = np.asarray(prediction)
    expert = np.asarray(ground_truth) > 0
    if prediction.shape != expert.shape:
        raise ValueError("Prediction and ground truth must have identical shapes")
    predicted = prediction > 0 if threshold is None else prediction >= threshold
    expert_voxels = int(expert.sum())
    predicted_voxels = int(predicted.sum())
    intersection = int(np.logical_and(predicted, expert).sum())
    denominator = expert_voxels + predicted_voxels
    return {
        "dice": 2 * intersection / denominator if denominator > 0 else 1.0,
        "expert_voxels": expert_voxels,
        "predicted_voxels": predicted_voxels,
        "intersection_voxels": intersection,
    }


def dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    threshold: float | None = None,
) -> float:
    """Return patient-level Dice; both masks empty returns exactly 1.0."""
    return float(patient_metrics(prediction, ground_truth, threshold=threshold)["dice"])


def summarize_patient_metrics(
    results: Iterable[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Match evaluate_validation: average only lesion-positive patients.

    Empty positive cohorts yield NaN mean/median, as in the notebook's pandas
    reductions. Misses count zero predictions, not every zero-Dice patient.
    """
    rows = list(results)
    positive = [row for row in rows if row["expert_voxels"] > 0]
    scores = [row["dice"] for row in positive]
    return {
        "mean_dice": float(np.mean(scores)) if scores else float("nan"),
        "median_dice": float(np.median(scores)) if scores else float("nan"),
        "missed_cases": sum(row["predicted_voxels"] == 0 for row in positive),
        "false_positive_negative_cases": sum(
            row["expert_voxels"] == 0 and row["predicted_voxels"] > 0
            for row in rows
        ),
    }
