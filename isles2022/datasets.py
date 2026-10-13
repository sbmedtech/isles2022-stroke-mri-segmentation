"""Disk-backed tensors and the original alternating 3D patch sampler."""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .preprocessing import preprocess_patient


def extract_patch(image, mask, positive=True, patch_size=(64, 64, 32)):
    """Pad at the upper edges, then crop image/label around one common center.

    Positive sampling chooses an expert lesion voxel when available. Random
    sampling can include lesions; an empty mask falls back to random sampling.
    Torch random draws and clipping match the original sampler exactly.
    """
    pad_amounts = [max(0, target - actual)
                   for actual, target in zip(image.shape[1:], patch_size)]
    padding = (0, pad_amounts[2], 0, pad_amounts[1], 0, pad_amounts[0])
    image, mask = F.pad(image, padding, value=0), F.pad(mask, padding, value=0)
    lesions = torch.nonzero(mask[0] > 0, as_tuple=False)
    if positive and len(lesions) > 0:
        center = lesions[torch.randint(len(lesions), (1,)).item()]
    else:
        center = torch.tensor([torch.randint(0, dim, (1,)).item() for dim in image.shape[1:]])
    starts = [max(0, min(int(center[axis]) - patch_size[axis] // 2,
                         image.shape[axis + 1] - patch_size[axis])) for axis in range(3)]
    x, y, z = starts
    sx, sy, sz = patch_size
    image_patch = image[:, x:x + sx, y:y + sy, z:z + sz]
    mask_patch = mask[:, x:x + sx, y:y + sy, z:z + sz]
    assert image_patch.shape == (3, *patch_size)
    assert mask_patch.shape == (1, *patch_size)
    return image_patch, mask_patch


def cache_patients(patient_ids, dataset_dir: Path, cache_dir: Path) -> list[Path]:
    """Cache one volume at a time, preserving the existing image/label format.

    Like the reference, existing files are reused by existence; this extraction
    does not certify legacy cache generation. Use a fresh cache for a new run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for patient_id in patient_ids:
        path = cache_dir / f"{patient_id}.pt"
        if not path.exists():
            image, label = preprocess_patient(patient_id, dataset_dir)
            torch.save({"image": image.cpu(), "label": label.cpu()}, path)
        paths.append(path)
    return paths


def load_cached_patient(patient_id: str, cache_dir: Path):
    data = torch.load(cache_dir / f"{patient_id}.pt", map_location="cpu", weights_only=True)
    return data["image"], data["label"]


class CachedPatchDataset(Dataset):
    """Explicit cache path, patient IDs and patch policy instead of globals."""

    def __init__(self, patient_ids, cache_dir: Path, patches_per_patient=2, patch_size=(64, 64, 32)):
        self.patient_ids = list(patient_ids)
        self.cache_dir = cache_dir
        self.patches_per_patient = patches_per_patient
        self.patch_size = patch_size

    def __len__(self):
        return len(self.patient_ids) * self.patches_per_patient

    def __getitem__(self, index):
        patient_id = self.patient_ids[index // self.patches_per_patient]
        image, label = load_cached_patient(patient_id, self.cache_dir)
        positive = (index % self.patches_per_patient) % 2 == 0
        image, label = extract_patch(image, label, positive, self.patch_size)
        return {"image": image, "label": label}


def training_loader(patient_ids, cache_dir: Path, patches_per_patient: int,
                    config: ExperimentConfig, device: torch.device) -> DataLoader:
    dataset = CachedPatchDataset(patient_ids, cache_dir, patches_per_patient, config.patch_size)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      num_workers=config.num_workers, pin_memory=(device.type == "cuda"))
