"""Compatible checkpoint I/O and the measured validation-selection contract.

The pure selection helpers are extracted from the repaired reference notebook.
Raw legacy model/optimizer dictionaries still load strictly. Selection resumes
must additionally prove context, weight and run identity; no threshold fallback
or historical-score substitution is introduced.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch


def state_digest(state):
    """Identify weights independently of checkpoint serialization/device."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_json_value(value):
    """Copy metadata into its stable JSON representation without dropping fields.

    AdamW betas are tuples in memory and lists after JSON decoding. Both are
    JSON arrays; canonicalize before hashing, writing and exact comparison.
    Unsupported objects/nonfinite numbers fail rather than become strings.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def context_digest(context):
    # Keep the existing sorted-JSON byte encoding and checkpoint/path digests.
    return hashlib.sha256(
        json.dumps(canonical_json_value(context), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_selection(checkpoint, context, expected_run_id=None):
    selection = checkpoint.get("selection")
    if not selection or selection["context_hash"] != context_digest(context):
        raise ValueError("Missing or mismatched selection provenance; restore epoch 23 and reselect on validation.")
    if selection["weights_hash"] != state_digest(checkpoint["model_state_dict"]):
        raise ValueError("Checkpoint weights do not match selection metadata.")
    if expected_run_id is not None and selection["run_id"] != expected_run_id:
        raise ValueError("Checkpoint belongs to another selection run.")
    threshold = float(selection["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold.")
    return selection


def establish_selection(checkpoint, context, evaluate, previous_best=None):
    """Measure this run's weights; CSVs and historical scores never select models.

    A legacy epoch-23 checkpoint can start a new selection run after Step 14.
    Later legacy checkpoints cannot establish lineage and must not be adopted.
    """
    if "selection" not in checkpoint:
        if int(checkpoint["epoch"]) != 23:
            raise ValueError("Legacy resume needs the epoch-23 checkpoint and measured validation selection.")
        if "validation_threshold" not in checkpoint:
            raise ValueError("Select a threshold on the epoch-23 validation weights before starting selection.")
        threshold = float(checkpoint["validation_threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid validation-selected threshold.")
        selection = {
            "run_id": str(uuid.uuid4()),
            "context_hash": context_digest(context),
            "threshold": threshold,
            "history": [],
        }
        best = checkpoint
    else:
        selection = dict(validate_selection(checkpoint, context))
        selection["history"] = list(selection["history"])
        if previous_best is None:
            raise ValueError("Best checkpoint is missing; restore it before resuming.")
        best_selection = validate_selection(previous_best, context)
        if (best_selection["run_id"] != selection["run_id"]
                or best_selection["threshold"] != selection["threshold"]
                or int(previous_best["epoch"]) > int(checkpoint["epoch"])):
            raise ValueError("Best checkpoint belongs to another selection run or threshold.")
        best = previous_best
    metrics = evaluate(best, selection["threshold"])
    if not math.isfinite(metrics["mean_dice"]):
        raise ValueError("Validation mean Dice must be finite.")
    # Replace this epoch's record with a measurement of its actual weights.
    selection["history"] = [row for row in selection["history"] if row["epoch"] != best["epoch"]]
    selection["history"].append({"epoch": int(best["epoch"]), **metrics})
    selection["history"].sort(key=lambda row: row["epoch"])
    best = dict(best)
    best["selection"] = {**selection, "weights_hash": state_digest(best["model_state_dict"])}
    checkpoint = dict(checkpoint)
    checkpoint["selection"] = {**selection, "weights_hash": state_digest(checkpoint["model_state_dict"])}
    return checkpoint, best, metrics


def save_atomic(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer=None, device="cpu") -> dict:
    """Restore the original dictionary format with strict model-state loading."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_selected_checkpoint(path: Path, model, context: dict, run_id: str, device="cpu"):
    """Validate the exact loaded object before applying weights or threshold."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    selection = validate_selection(checkpoint, context, run_id)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint, selection["threshold"]
