"""Build the reference MONAI U-Net with its original layer/state-dict names.

This module constructs the network only. Data preparation, device selection,
training, sigmoid conversion and threshold selection belong to the caller.
"""

from dataclasses import asdict

from monai.networks.nets import UNet

from .config import ModelConfig


def build_model(config: ModelConfig = ModelConfig()) -> UNet:
    """Construct a raw U-Net from ``config`` without adding a wrapper.

    The default configuration matches the notebook's three-input, one-output
    3D architecture. Returning MONAI's own object preserves its state-dict key
    structure for strict loading of compatible notebook checkpoints.

    The caller chooses the device and train/eval mode. Unspecified options keep
    MONAI defaults; use the original MONAI/PyTorch versions when reproducing a
    saved run. Architecture compatibility does not establish trained performance.
    """
    return UNet(**asdict(config))
