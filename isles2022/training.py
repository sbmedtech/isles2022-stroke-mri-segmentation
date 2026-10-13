"""Small training functions for the notebook's existing stages, not a framework."""

import copy
from functools import partial
import json
import math
import random
import time

from monai.losses import DiceCELoss
import numpy as np
import pandas as pd
import torch

from .checkpoints import establish_selection, load_checkpoint, save_atomic, state_digest
from .config import ExperimentConfig, ModelConfig, ProjectPaths
from .datasets import load_cached_patient
from .evaluation import evaluate_validation
from .model import build_model


def seed_reference_rng(seed: int = 42) -> None:
    """The original initial seeds; no new deterministic/backend policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_components(device, config: ExperimentConfig = ExperimentConfig(),
                        model_config: ModelConfig = ModelConfig()):
    model = build_model(model_config).to(device)
    loss = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    return model, loss, optimizer


def _optimization_step(model, optimizer, loss_function, batch, device, verify_gradients=False) -> float:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(images), labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss.")
    loss.backward()
    if verify_gradients:
        valid = [parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                 for parameter in model.parameters() if parameter.requires_grad]
        assert any(valid), "No valid gradients were produced."
    optimizer.step()
    return loss.item()


def diagnostic_optimizer_step(model, optimizer, loss_function, batch, device) -> float:
    """Explicitly retain the original pre-epoch diagnostic weight update.

    This is not a read-only forward check. Call it once for a fresh run, not
    after restoring a checkpoint. It is named separately so reviewers see it.
    """
    model.train()
    return _optimization_step(model, optimizer, loss_function, batch, device, verify_gradients=True)


def train_epoch(model, optimizer, loss_function, loader, device) -> tuple[float, float]:
    """One epoch; preserve the reference mean of batch losses and update order."""
    if len(loader) == 0:
        raise ValueError("Training requires at least one batch.")
    model.train()
    start = time.perf_counter()
    total_loss = sum(_optimization_step(model, optimizer, loss_function, batch, device)
                     for batch in loader)
    return total_loss / len(loader), (time.perf_counter() - start) / 60


def make_checkpoint(model, optimizer, history, epoch: int) -> dict:
    """Original four-key schema; freeze in-memory tensors as a saved snapshot."""
    return copy.deepcopy({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "training_history": history,
    })


def train_until(model, optimizer, loss_function, loader, device, paths: ProjectPaths,
                end_epoch: int, checkpoint=None) -> dict:
    """Initial or epoch-23 continuation; never rebuild the model/optimizer here."""
    paths.models.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # AdamW can retain tensor references from an in-memory state dictionary.
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    history = copy.deepcopy(checkpoint["training_history"]) if checkpoint else []
    completed = int(checkpoint["epoch"]) if checkpoint else 0
    latest = checkpoint
    for epoch in range(completed, end_epoch):
        loss, minutes = train_epoch(model, optimizer, loss_function, loader, device)
        history.append({"epoch": epoch + 1, "train_loss": loss, "duration_minutes": minutes})
        latest = make_checkpoint(model, optimizer, history, epoch + 1)
        save_atomic(latest, paths.models / "checkpoint_latest.pt")
        (paths.models / "training_history.json").write_text(json.dumps(history, indent=4) + "\n")
    if latest is None:
        raise ValueError("No completed epoch or checkpoint was supplied.")
    return latest


def persist_validation_threshold(path, model, threshold: float, selection_epoch: int = 23):
    """Bind the measured validation threshold to the exact epoch-23 weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != selection_epoch:
        raise ValueError("Threshold selection requires the epoch-23 checkpoint.")
    if state_digest(model.state_dict()) != state_digest(checkpoint["model_state_dict"]):
        raise ValueError("Threshold-selection model differs from saved weights.")
    if "selection" in checkpoint:
        raise ValueError("Selection is initialized; resume without retuning.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid validation-selected threshold.")
    checkpoint["validation_threshold"] = float(threshold)
    save_atomic(checkpoint, path)
    return checkpoint


def train_with_validation(model, optimizer, loss_function, loader, device,
                          paths: ProjectPaths, splits: dict, context: dict,
                          config: ExperimentConfig = ExperimentConfig()):
    """Measured baseline and the original 28/33/38/43 selection schedule.

    Best/latest publication and fail-safe legacy rules match the repaired
    reference; interruption recovery and RNG restoration are not redesigned.
    """
    latest_path = paths.models / "checkpoint_latest.pt"
    best_path = paths.models / "checkpoint_best_validation.pt"
    checkpoint = load_checkpoint(latest_path, model, optimizer, device)
    previous_best = None
    if "selection" in checkpoint and best_path.exists():
        previous_best = torch.load(best_path, map_location="cpu", weights_only=True)
    read_patient = partial(load_cached_patient, cache_dir=paths.cache)

    def measure(candidate, threshold):
        model.load_state_dict(candidate["model_state_dict"], strict=True)
        return evaluate_validation(model, splits["validation"], read_patient,
                                   threshold, device, config)

    checkpoint, best, metrics = establish_selection(checkpoint, context, measure, previous_best)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    save_atomic(best, best_path)
    save_atomic(checkpoint, latest_path)
    if int(checkpoint["epoch"]) == config.selection_epoch:
        save_atomic(checkpoint, paths.models / "checkpoint_epoch23.pt")
    history = checkpoint["training_history"]
    validation = checkpoint["selection"]["history"]
    best_dice = metrics["mean_dice"]
    threshold = checkpoint["selection"]["threshold"]
    paths.results.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.results / "extended_validation_history.csv"
    pd.DataFrame(validation).to_csv(metrics_path, index=False)
    for epoch in range(int(checkpoint["epoch"]), config.final_epoch):
        loss, minutes = train_epoch(model, optimizer, loss_function, loader, device)
        history.append({"epoch": epoch + 1, "train_loss": loss, "duration_minutes": minutes})
        current = make_checkpoint(model, optimizer, history, epoch + 1)
        current["selection"] = {**checkpoint["selection"], "history": validation,
                                "weights_hash": state_digest(current["model_state_dict"])}
        if ((epoch + 1) - config.selection_epoch) % config.validate_every == 0:
            metrics = evaluate_validation(model, splits["validation"], read_patient,
                                          threshold, device, config)
            if not math.isfinite(metrics["mean_dice"]):
                raise ValueError("Validation mean Dice must be finite.")
            validation.append({"epoch": epoch + 1, **metrics})
            if metrics["mean_dice"] > best_dice:
                best_dice = metrics["mean_dice"]
                save_atomic(current, best_path)
            pd.DataFrame(validation).to_csv(metrics_path, index=False)
        save_atomic(current, latest_path)
        checkpoint = current
    return checkpoint, pd.DataFrame(validation)
