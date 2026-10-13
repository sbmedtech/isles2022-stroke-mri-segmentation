"""Architecture settings from notebook cell 27 (zero-based cell numbering)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """The notebook's MONAI U-Net arguments, with unchanged defaults."""

    spatial_dims: int = 3
    in_channels: int = 3
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2
