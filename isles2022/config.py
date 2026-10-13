"""Constructor settings for the notebook's three-channel MONAI 3D U-Net.

The reference definition is in original notebook cell 27 (zero-based). Keeping
these six arguments together makes the architecture easy to inspect and reuse.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Immutable U-Net settings copied from the research notebook.

    Inputs follow DWI, ADC, FLAIR channel order; the single output is a lesion
    logit channel. Other MONAI constructor options retain their library defaults.
    """

    spatial_dims: int = 3
    in_channels: int = 3
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2
