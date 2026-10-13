"""Scientific utilities for physical lesion size and patient-level error analysis.

These standalone functions complement Dice: native-grid volume in mL, complete
zero-overlap detection, and patient bootstrap intervals. Masks must already be
binary; probabilities are never implicitly thresholded. NumPy is required.
The NIfTI helper accepts an already loaded NiBabel image without importing it.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _binary_mask(mask):
    """Validate a nonempty 3D 0/1 mask and return its boolean representation."""
    array = np.asarray(mask)
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError("mask must be a nonempty 3D array")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must contain only binary 0/1 values")
    return array.astype(bool, copy=False)


def infarct_volume_ml(mask, affine, *, spatial_unit):
    """Measure a binary lesion in mL using the mask's voxel-to-world affine.

    ``mask`` is a nonempty 3D 0/1 array. ``affine`` is a finite homogeneous 4×4
    transform for that same grid. ``spatial_unit`` must be 'mm', 'meter' or
    'micron'; missing/unknown units are rejected rather than assumed.

    Absolute affine determinant gives each voxel's physical volume, accounting
    for anisotropy, rotation, reflection and shear. Use the native expert mask
    for native infarct volume: resampling can change the discretized mask volume.
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
    """Measure an already loaded binary NIfTI mask using its header units.

    ``image`` must expose NiBabel's data, affine and header interfaces. Unknown
    units are rejected; do not assume millimeters or pass a probability volume.
    """
    return infarct_volume_ml(
        image.get_fdata(), image.affine,
        spatial_unit=image.header.get_xyzt_units()[0],
    )


def zero_overlap(ground_truth, prediction):
    """Identify a complete overlap failure for a lesion-positive patient.

    Both inputs must be binary 3D masks on the same physical grid. Matching
    shapes are checked, but physical correspondence must be verified upstream.

    Returns True for empty predictions and nonempty disjoint predictions against
    a positive expert mask. An expert-empty case returns False even when it has
    false positives; report those patients separately.
    """
    ground_truth = _binary_mask(ground_truth)
    prediction = _binary_mask(prediction)
    if ground_truth.shape != prediction.shape:
        raise ValueError("ground truth and prediction must have matching shapes")
    return bool(ground_truth.any() and not np.logical_and(ground_truth, prediction).any())


@dataclass(frozen=True)
class BootstrapInterval:
    """Point estimate and percentile bounds for a patient-resampling summary.

    Settings and cohort size accompany the bounds so the statistic, confidence,
    resample count and random seed remain explicit. Record cohort membership
    separately when reporting an interval.
    """
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
    """Estimate a percentile confidence interval by resampling whole patients.

    Provide one finite scalar per unique, nonempty patient ID, for at least two
    patients. Choose the cohort beforehand (for example, lesion-positive test
    patients). Duplicate IDs and missing/nonfinite values are rejected, not
    silently discarded.

    Each replicate draws the same number of patients with replacement and gives
    each patient equal weight. ``statistic`` is 'mean' or 'median'; ``seed`` makes
    the resampling repeatable. The result records these settings with the bounds.

    This estimates patient-sampling uncertainty only. It does not capture
    training-seed variation, model/threshold selection or external validity.
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
