"""Patient-level overlap metrics matching notebook evaluation cells 45 and 46.

Dice is computed across the complete volume, without slice averaging or
smoothing. Expert labels use > 0; probability thresholds use >=. Both-empty
masks receive Dice 1.0. These functions check array shape, not physical geometry.
"""

from collections.abc import Iterable, Mapping

import numpy as np


def patient_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Return full-volume Dice, expert/predicted counts and overlap count.

    ``prediction`` and ``ground_truth`` must have matching shapes and already
    occupy the same physical grid. Without ``threshold``, positive prediction
    values are treated as lesion voxels; pass a binary mask in that mode.

    For sigmoid probabilities, supply the validation-selected threshold
    explicitly (0.25 in the saved notebook run). Equality at the threshold is
    included. Convert logits to probabilities before calling; this function does
    not apply sigmoid, estimate registration or choose a threshold.
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
    """Return only Dice using the same rules as :func:`patient_metrics`.

    A both-empty pair returns exactly 1.0; one empty mask against a nonempty
    mask returns 0.0. No smoothing term is added to the denominator.
    """
    return float(patient_metrics(prediction, ground_truth, threshold=threshold)["dice"])


def summarize_patient_metrics(
    results: Iterable[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Summarize patient records with the notebook's validation population.

    Supply records from :func:`patient_metrics` for a predefined cohort. Mean
    and median Dice include only patients whose expert mask contains a lesion;
    an empty positive cohort yields NaN, as in pandas reductions.

    ``missed_cases`` counts empty predictions in lesion-positive patients. It
    does not count every zero-overlap prediction: a nonempty disjoint mask can
    also have Dice 0. Lesion-negative false positives are counted separately.
    """
    rows = list(results)
    # Expert-empty patients belong in the false-positive summary, not mean Dice.
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
