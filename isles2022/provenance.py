"""Canonical experiment/artifact identity without relying on notebook globals."""

import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import re

import monai
import numpy as np
import pandas as pd
import torch

from .checkpoints import canonical_json_value, context_digest, state_digest, validate_selection
from .config import ExperimentConfig, ProjectPaths
from .datasets import extract_patch
from .evaluation import evaluate_validation
from .preprocessing import normalize_mri, preprocess_patient


def preprocessing_source() -> str:
    return inspect.getsource(preprocess_patient) + inspect.getsource(normalize_mri)


def selection_context(paths: ProjectPaths, splits: dict, model, optimizer, loss_function,
                      config: ExperimentConfig) -> dict:
    """Fingerprint the same settings/cache content as reference cell 45.

    Extracted source identities differ from the notebook's source strings.
    Existing raw weights remain compatible; unverifiable older selection
    contexts must not be relabeled as if produced by this extraction.
    """
    cache_digest = hashlib.sha256()
    for patient_id in splits["train"] + splits["validation"]:
        cache_digest.update(patient_id.encode())
        with (paths.cache / f"{patient_id}.pt").open("rb") as cache_file:
            for chunk in iter(lambda: cache_file.read(8 * 1024 * 1024), b""):
                cache_digest.update(chunk)
    return {
        "splits": {key: splits[key] for key in ("train", "validation", "test")},
        "cache_sha256": cache_digest.hexdigest(),
        "preprocessing_source": preprocessing_source(),
        "sampling_source": inspect.getsource(extract_patch), "architecture": str(model),
        "optimizer": [{key: value for key, value in group.items() if key != "params"}
                      for group in optimizer.param_groups],
        "loss": str(loss_function),
        "loss_settings": {
            name: {key: value for key, value in vars(component).items()
                   if not key.startswith("_") and key != "training"
                   and isinstance(value, (str, int, float, bool, type(None)))}
            for name, component in loss_function.named_modules()
        },
        "evaluation_source": inspect.getsource(evaluate_validation),
        "patch_size": list(config.patch_size), "patches_per_patient": config.extended_patches,
        "batch_size": config.batch_size, "roi_size": list(config.patch_size),
        "overlap": config.overlap, "sw_batch_size": config.sw_batch_size,
        "validation_every": config.validate_every, "torch_version": str(torch.__version__),
        "monai_version": monai.__version__,
    }


def artifact_provenance(checkpoint, context, run_id, patient_ids):
    context = canonical_json_value(context)
    selection = validate_selection(checkpoint, context, run_id)
    if list(patient_ids) != context["splits"]["test"] or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Artifact cohort differs from the unique saved test split.")
    if preprocessing_source() != context["preprocessing_source"]:
        raise ValueError("Current preprocessing differs from the checkpoint context.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid artifact run ID.")
    identity = {"run_id": selection["run_id"], "epoch": int(checkpoint["epoch"]),
                "weights_sha256": state_digest(checkpoint["model_state_dict"]),
                "threshold": float(selection["threshold"]), "context_sha256": selection["context_hash"]}
    return {
        "schema_version": 1, **identity, "checkpoint_digest": context_digest(identity),
        "context": context, "patient_ids": list(patient_ids),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch_cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "monai", "numpy", "nibabel", "scipy", "pandas", "matplotlib")},
        },
    }


def artifact_directory(project_dir, provenance):
    provenance = canonical_json_value(provenance)
    return (project_dir / "results" / "runs" / provenance["run_id"] /
            f"epoch{provenance['epoch']}-{context_digest(provenance)}")


def write_artifact_manifest(path, provenance, source_manifest=None):
    provenance = canonical_json_value(provenance)
    manifest = {"provenance": provenance, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if source_manifest is not None:
        if canonical_json_value(source_manifest["provenance"]) != provenance:
            raise ValueError("Figure source belongs to another experiment.")
        manifest["source_test_csv_sha256"] = source_manifest["sha256"]
    path = path.with_suffix(path.suffix + ".manifest.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)
    return manifest


def read_test_artifact(project_dir, provenance):
    provenance = canonical_json_value(provenance)
    path = artifact_directory(project_dir, provenance) / "test.csv"
    manifest_path = path.with_suffix(".csv.manifest.json")
    if not path.exists() or not manifest_path.exists():
        raise ValueError("Missing provenance-bound test results; evaluate this checkpoint first.")
    manifest = json.loads(manifest_path.read_text())
    if canonical_json_value(manifest.get("provenance")) != provenance:
        raise ValueError("Test results belong to another experiment identity.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.get("sha256"):
        raise ValueError("Test results content does not match its manifest.")
    frame = pd.read_csv(io.BytesIO(content))
    if frame["patient_id"].tolist() != provenance["patient_ids"]:
        raise ValueError("Test result order/cohort differs from the recorded split.")
    return frame, manifest


def save_test_artifact(frame, project_dir, provenance):
    provenance = canonical_json_value(provenance)
    if frame["patient_id"].tolist() != provenance["patient_ids"]:
        raise ValueError("Cannot export a different test cohort.")
    path = artifact_directory(project_dir, provenance) / "test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    write_artifact_manifest(path, provenance)
    return path


def check_figure_metrics(prediction, expert, patient_row):
    expert_count, predicted_count = int(expert.sum()), int(prediction.sum())
    intersection = int(np.logical_and(expert, prediction).sum())
    denominator = expert_count + predicted_count
    dice = 2 * intersection / denominator if denominator else 1.0
    if (expert_count != int(patient_row["expert_voxels"])
            or predicted_count != int(patient_row["predicted_voxels"])
            or intersection != int(patient_row["intersection_voxels"])
            or not math.isclose(dice, float(patient_row["dice"]), rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Figure prediction disagrees with stored patient metrics.")
