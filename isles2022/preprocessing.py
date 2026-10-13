"""Reference DWI-grid preprocessing; affine resampling is not registration."""

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch


def patient_files(patient_id: str, dataset_dir: Path) -> dict[str, Path]:
    session = "ses-0001"
    prefix = f"{patient_id}_{session}"
    case = dataset_dir / patient_id / session
    return {
        "DWI": case / "dwi" / f"{prefix}_dwi.nii.gz",
        "ADC": case / "dwi" / f"{prefix}_adc.nii.gz",
        "FLAIR": case / "anat" / f"{prefix}_FLAIR.nii.gz",
        "Mask": dataset_dir / "derivatives" / patient_id / session / f"{prefix}_msk.nii.gz",
    }


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """Replace nonfinite values and z-score only nonzero voxels, as before."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    valid = volume != 0
    if not np.any(valid):
        return np.zeros_like(volume, dtype=np.float32)
    mean = volume[valid].mean()
    std = max(float(volume[valid].std()), 1e-8)
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[valid] = (volume[valid] - mean) / std
    return normalized


def preprocess_patient(patient_id: str, dataset_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DWI/ADC/FLAIR channels and an expert mask on the DWI grid.

    Labels remain required for this research workflow. ADC/mask shape and affine
    must match DWI; FLAIR uses linear affine resampling. No reorientation,
    registration, voxel-spacing change or augmentation is performed.
    """
    files = patient_files(patient_id, dataset_dir)
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
    images = {name: nib.load(str(path)) for name, path in files.items()}
    dwi = images["DWI"]
    for name in ("ADC", "Mask"):
        image = images[name]
        if image.shape != dwi.shape or not np.allclose(image.affine, dwi.affine):
            raise ValueError(f"{patient_id}: {name} grid differs from DWI")
    images["FLAIR"] = resample_from_to(images["FLAIR"], dwi, order=1)
    channels = [normalize_mri(images[name].get_fdata(dtype=np.float32))
                for name in ("DWI", "ADC", "FLAIR")]
    label = (images["Mask"].get_fdata(dtype=np.float32) > 0).astype(np.float32)[None, ...]
    image_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(channels, axis=0)))
    label_tensor = torch.from_numpy(np.ascontiguousarray(label))
    assert image_tensor.shape[1:] == label_tensor.shape[1:]
    assert torch.isfinite(image_tensor).all() and torch.isfinite(label_tensor).all()
    return image_tensor, label_tensor
