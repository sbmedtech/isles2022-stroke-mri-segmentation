"""Constructor settings for the notebook's three-channel MONAI 3D U-Net.

The reference definition is in original notebook cell 27 (zero-based). Keeping
these six arguments together makes the architecture easy to inspect and reuse.
"""

from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class ProjectPaths:
    """One project root and one temporary tensor cache; no import-time I/O."""

    root: Path
    cache: Path = Path("/content/isles2022_preprocessed")

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def dataset(self) -> Path:
        return self.data / "extracted" / "ISLES-2022"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def splits(self) -> Path:
        return self.root / "splits" / "patient_splits.json"

    def create(self) -> None:
        for folder in (self.data, self.models, self.results, self.cache, self.splits.parent):
            folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    """The existing experiment settings, separate from paths and model weights.

    Values describe the notebook methodology, not a new training prescription.
    No augmentation, scheduler, mixed precision or spacing change is introduced.
    """

    seed: int = 42
    patch_size: tuple[int, int, int] = (64, 64, 32)
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    initial_epochs: int = 3
    selection_epoch: int = 23
    final_epoch: int = 43
    initial_patches: int = 2
    extended_patches: int = 4
    validate_every: int = 5
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25,
                                    0.30, 0.35, 0.40, 0.45, 0.50)
    sw_batch_size: int = 1
    overlap: float = 0.25
