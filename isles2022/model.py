"""Construct the notebook architecture without wrapping or renaming its layers."""

from dataclasses import asdict

from monai.networks.nets import UNet

from .config import ModelConfig


def build_model(config: ModelConfig = ModelConfig()) -> UNet:
    """Return a raw MONAI U-Net, compatible with notebook model_state_dict keys.

    Device placement and train/eval mode remain the caller's responsibility.
    Unspecified MONAI options deliberately retain the notebook's library defaults.
    Use the same MONAI/PyTorch versions when reproducing an existing run.
    """
    return UNet(**asdict(config))
