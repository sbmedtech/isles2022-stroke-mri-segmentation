"""Readable research figures from validated current artifacts, never old labels."""

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import spearmanr

from .evaluation import predict_probabilities
from .provenance import (artifact_directory, artifact_provenance, check_figure_metrics,
                         read_test_artifact, write_artifact_manifest)


def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o")
    ax.set(xlabel="Training epoch", ylabel="Average training loss", title="3D U-Net training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _figure_inputs(model, checkpoint, context, run_id, project_dir):
    provenance = artifact_provenance(checkpoint, context, run_id, context["splits"]["test"])
    frame, manifest = read_test_artifact(project_dir, provenance)
    # Validate/read before applying weights, then use this exact checkpoint object.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return provenance, frame, manifest


def _overlay(ax, background, mask, color):
    ax.imshow(background, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = mask.astype(float) * 0.55
    ax.imshow(overlay, interpolation="nearest")
    ax.axis("off")


def _patient_panels(axes, patient_id, model, load_patient, threshold, device, config, row):
    image, label = load_patient(patient_id)
    expert = label.squeeze().numpy() > 0
    prediction = predict_probabilities(model, image, device, config) >= threshold
    check_figure_metrics(prediction, expert, row)
    if expert.any():
        index = int(np.argmax(expert.sum(axis=(0, 1))))
    elif prediction.any():
        index = int(np.argmax(prediction.sum(axis=(0, 1))))
    else:
        index = image.shape[-1] // 2
    background = np.rot90(image[0].numpy()[:, :, index])
    expert_slice = np.rot90(expert[:, :, index])
    predicted_slice = np.rot90(prediction[:, :, index])
    axes[0].imshow(background, cmap="gray", interpolation="nearest")
    axes[0].axis("off")
    axes[0].set_title(f"{patient_id}\nNormalized DWI", fontsize=10)
    _overlay(axes[1], background, expert_slice, (1.0, 0.15, 0.15))
    axes[1].set_title(f"Expert annotation\n{int(row['expert_voxels']):,} voxels", fontsize=10)
    _overlay(axes[2], background, predicted_slice, (0.0, 0.85, 0.25))
    axes[2].set_title(f"Prediction — Dice {float(row['dice']):.3f}\n"
                      f"{int(row['predicted_voxels']):,} voxels", fontsize=10)


def plot_test_examples(model, checkpoint, context, run_id, patient_ids, load_patient,
                       device, config, project_dir):
    provenance, frame, manifest = _figure_inputs(model, checkpoint, context, run_id, project_dir)
    if not patient_ids or not set(patient_ids).issubset(provenance["patient_ids"]):
        raise ValueError("Example patients must belong to this test cohort.")
    fig, axes = plt.subplots(len(patient_ids), 3, figsize=(13, 3.7 * len(patient_ids)), squeeze=False)
    for index, patient_id in enumerate(patient_ids):
        row = frame.loc[frame["patient_id"] == patient_id].iloc[0]
        _patient_panels(axes[index], patient_id, model, load_patient, provenance["threshold"],
                        device, config, row)
    fig.suptitle(f"Test examples | Epoch {checkpoint['epoch']} | Run {run_id}")
    fig.tight_layout()
    path = artifact_directory(project_dir, provenance) / "example_test_patients.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    write_artifact_manifest(path, provenance, manifest)
    return fig, path


def plot_lesion_size(checkpoint, context, run_id, project_dir):
    """Preserve the descriptive voxel-count/Spearman analysis, not native mL."""
    provenance = artifact_provenance(checkpoint, context, run_id, context["splits"]["test"])
    frame, manifest = read_test_artifact(project_dir, provenance)
    positive = frame[frame["expert_voxels"] > 0]
    rho, p_value = spearmanr(positive["expert_voxels"], positive["dice"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(positive["expert_voxels"], positive["dice"], s=75, alpha=0.75)
    ax.set_xscale("log")
    ax.axhline(positive["dice"].mean(), color="#DC2626", linestyle="--",
               label=f"Mean test Dice = {positive['dice'].mean():.3f}")
    for _, row in positive.nsmallest(5, "dice").iterrows():
        ax.annotate(row["patient_id"].replace("sub-strokecase", ""),
                    (row["expert_voxels"], row["dice"]), xytext=(5, 7), textcoords="offset points")
    ax.set(xlabel="Expert infarct size (voxels, log scale)", ylabel="Full-volume Dice",
           title=f"Infarct size vs. accuracy\nSpearman rho = {rho:.3f}, p = {p_value:.4g}", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = artifact_directory(project_dir, provenance) / "infarct_size_vs_dice.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    write_artifact_manifest(path, provenance, manifest)
    return fig, path


def write_test_pdf(model, checkpoint, context, run_id, load_patient, device, config, project_dir):
    provenance, frame, manifest = _figure_inputs(model, checkpoint, context, run_id, project_dir)
    path = artifact_directory(project_dir, provenance) / "all_test_patient_visualizations.pdf"
    temporary = path.with_suffix(".pdf.tmp")
    with PdfPages(temporary) as pdf:
        for start in range(0, len(frame), 4):
            page = frame.iloc[start:start + 4]
            fig, axes = plt.subplots(len(page), 3, figsize=(13, 3.7 * len(page)), squeeze=False)
            try:
                for index, (_, row) in enumerate(page.iterrows()):
                    _patient_panels(axes[index], row["patient_id"], model, load_patient,
                                    provenance["threshold"], device, config, row)
                fig.suptitle(f"Held-out test patients | Epoch {checkpoint['epoch']} | Run {run_id}")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
            finally:
                plt.close(fig)
    os.replace(temporary, path)
    write_artifact_manifest(path, provenance, manifest)
    return path
