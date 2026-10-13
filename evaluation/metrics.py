"""Physical volume, complete overlap failures, and patient bootstrap intervals.

Requires NumPy. NIfTI helpers accept a nibabel image without importing nibabel.
Masks must already be binary; probabilities are never implicitly thresholded.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError("mask must be a nonempty 3D array")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must contain only binary 0/1 values")
    return array.astype(bool, copy=False)


def infarct_volume_ml(mask, affine, *, spatial_unit):
    """Return binary lesion volume in mL from its own voxel-to-world affine.

    Explicit units are required: 'mm', 'meter', or 'micron' (NIfTI names).
    abs(det(affine[:3, :3])) accounts for anisotropy, rotation, reflection,
    and shear. Use the native expert mask for native infarct volume; a
    resampled mask can have a different discretized volume.
    """
    mask = _binary_mask(mask)
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if spatial_unit not in factors:
        raise ValueError("spatial_unit must be mm, meter, or micron; unknown is unsafe")
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("affine must have homogeneous last row [0, 0, 0, 1]")
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError("affine must have a finite positive voxel volume")
    volume = np.count_nonzero(mask) * voxel_volume * factors[spatial_unit] ** 3 / 1000
    if not np.isfinite(volume):
        raise ValueError("calculated volume is not finite")
    return float(volume)


def nifti_infarct_volume_ml(image):
    """Use a loaded NIfTI mask's affine and declared spatial units.

    Unknown header units are rejected rather than assumed to be millimeters.
    """
    return infarct_volume_ml(
        image.get_fdata(), image.affine,
        spatial_unit=image.header.get_xyzt_units()[0],
    )


def zero_overlap(ground_truth, prediction):
    """True for a lesion-positive patient with no correctly predicted voxel.

    Includes both empty predictions and nonempty disjoint predictions.
    Lesion-negative ground truth returns False, even with false positives.
    Arrays must occupy the same physical grid; shape alone cannot verify this.
    """
    ground_truth = _binary_mask(ground_truth)
    prediction = _binary_mask(prediction)
    if ground_truth.shape != prediction.shape:
        raise ValueError("ground truth and prediction must have matching shapes")
    return bool(ground_truth.any() and not np.logical_and(ground_truth, prediction).any())


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_patients: int
    n_resamples: int
    statistic: str
    seed: int


def bootstrap_patient_ci(patient_ids, values, *, statistic="mean",
                         confidence=0.95, n_resamples=10000, seed=42):
    """Percentile bootstrap CI for a mean or median of patient-level values.

    Supply exactly one finite scalar per unique, nonempty string patient ID.
    Each replicate draws n patients with replacement, giving each patient
    equal weight. Choose the cohort (e.g. lesion-positive) before calling.
    Duplicate IDs and missing/nonfinite values are rejected, never dropped.
    This estimates patient sampling uncertainty, not training-seed variation
    or uncertainty from model selection. At least two patients are required.
    """
    ids = list(patient_ids)
    if any(not isinstance(pid, str) or not pid.strip() for pid in ids):
        raise ValueError("patient IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("provide exactly one value per unique patient ID")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(ids) or len(ids) < 2:
        raise ValueError("provide matching 1D values for at least two patients")
    if not np.isfinite(values).all():
        raise ValueError("patient values must all be finite")
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be mean or median")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    reduce = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for index in range(n_resamples):
        samples[index] = reduce(values[rng.integers(0, len(ids), size=len(ids))])
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail], method="linear")
    return BootstrapInterval(
        float(reduce(values)), float(lower), float(upper), float(confidence),
        len(ids), int(n_resamples), statistic, int(seed),
    )
