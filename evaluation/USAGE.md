# Standalone evaluation utilities

These functions do not load project results, run inference, or write files.
Runtime dependency: NumPy >= 1.22. The NIfTI convenience function accepts an
already loaded nibabel image. Synthetic tests also require nibabel.

```python
from evaluation import (
    nifti_infarct_volume_ml, zero_overlap, bootstrap_patient_ci,
)

# native_expert_image is a loaded binary NIfTI mask, with declared spatial units.
volume_ml = nifti_infarct_volume_ml(native_expert_image)

# expert_mask and binary_prediction must occupy the same physical voxel grid.
complete_overlap_failure = zero_overlap(expert_mask, binary_prediction)

# Supply actual IDs and one measured value per patient in a predefined cohort.
interval = bootstrap_patient_ci(patient_ids, patient_dice_values, seed=42)
```

For array inputs, `infarct_volume_ml(mask, affine, spatial_unit="mm")` requires
explicit units and uses the absolute affine determinant, divided by 1000 to
convert mm³ to mL. Unknown NIfTI units, nonbinary masks, and invalid geometry
raise errors. Native-mask volume is preferable for infarct-size analysis;
resampling can change lesion voxel counts. No volume is inferred from a CSV
voxel count without geometry.

`zero_overlap` identifies a positive expert mask with zero intersection,
including a nonempty disjoint prediction. It is not lesion-component detection
or a negative-case false-positive metric. Matching array shapes do not establish
physical registration; callers must verify geometry first.

Bootstrap intervals use patient sampling with replacement and percentile bounds
(linear quantiles). The default is a 95% interval for the mean with 10,000
replicates; `statistic="median"` is also supported. Duplicate IDs, nonfinite
values, and fewer than two patients are rejected. Select positive/all-patient
cohorts explicitly; nothing is silently filtered. Small cohorts or constant
values can yield unstable or degenerate intervals. These intervals do not
measure training randomness, model-selection uncertainty, or center transfer.

Run synthetic tests from the repository root:

```sh
python -m unittest discover -s tests -v
```
