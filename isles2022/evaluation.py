"""One full-volume prediction path for validation, thresholds and internal tests."""

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference

from .config import ExperimentConfig
from .metrics import patient_metrics, summarize_patient_metrics


def predict_probabilities(model, image: torch.Tensor, device,
                          config: ExperimentConfig = ExperimentConfig()) -> np.ndarray:
    """Reference sigmoid/sliding-window settings with CPU output accumulation."""
    model.eval()
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image.unsqueeze(0).to(device), roi_size=config.patch_size,
            sw_batch_size=config.sw_batch_size, predictor=model, overlap=config.overlap,
            sw_device=device, device=torch.device("cpu"),
        )
        return torch.sigmoid(logits).squeeze().numpy()


def evaluate_patients(model, patient_ids, load_patient, threshold: float, device,
                      config: ExperimentConfig = ExperimentConfig()) -> pd.DataFrame:
    """Return ordered full-volume rows; the caller explicitly supplies the cohort.

    ``load_patient(id)`` returns image/label tensors, either cached (validation)
    or freshly preprocessed (internal testing). No model/threshold selection is
    performed here, and no mean over slices is substituted for patient Dice.
    """
    rows = []
    for patient_id in patient_ids:
        image, label = load_patient(patient_id)
        probability = predict_probabilities(model, image, device, config)
        metrics = patient_metrics(probability, label.squeeze().numpy(), threshold=threshold)
        rows.append({"patient_id": patient_id, **metrics})
    return pd.DataFrame(rows)


def evaluate_validation(model, patient_ids, load_patient, threshold: float, device,
                        config: ExperimentConfig = ExperimentConfig()) -> dict:
    rows = evaluate_patients(model, patient_ids, load_patient, threshold, device, config)
    return summarize_patient_metrics(rows.to_dict("records"))


def select_threshold(model, patient_ids, load_patient, device,
                     config: ExperimentConfig = ExperimentConfig()):
    """Sweep validation probabilities once; choose positive-case mean Dice.

    Pandas sorting/idxmax retain the original lowest-threshold tie behavior.
    Never pass test IDs to this selection function.
    """
    rows = []
    for patient_id in patient_ids:
        image, label = load_patient(patient_id)
        probability = predict_probabilities(model, image, device, config)
        for threshold in config.thresholds:
            metrics = patient_metrics(probability, label.squeeze().numpy(), threshold=threshold)
            rows.append({"patient_id": patient_id, "threshold": threshold, **metrics})
    details = pd.DataFrame(rows)
    positive = details[details["expert_voxels"] > 0]
    if positive.empty:
        raise ValueError("Threshold selection requires lesion-positive validation patients.")
    summary = positive.groupby("threshold").agg(
        mean_dice=("dice", "mean"), median_dice=("dice", "median"),
        zero_predictions=("predicted_voxels", lambda values: int((values == 0).sum())),
    ).reset_index()
    best = float(summary.loc[summary["mean_dice"].idxmax(), "threshold"])
    return best, details, summary
