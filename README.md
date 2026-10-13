# ISLES 2022 · Stroke MRI Segmentation

A **3D U-Net for ischemic stroke lesion segmentation** using DWI, ADC and FLAIR MRI.
This research portfolio explores how complementary MRI signals can approximate
expert lesion masks. Delineating lesion extent supports research into stroke burden
and anatomical distribution; manual annotation is time-consuming, and small lesions
remain challenging. This project studies segmentation, not treatment selection.

## Real MRI segmentation: success and failure

![Historical ISLES MRI examples: normalized DWI, red expert mask and green U-Net prediction; successful lesion above and missed small lesion below](assets/portfolio/historical-test-examples.png)

**Top:** case 0237, full-volume Dice **0.869**. **Bottom:** case 0115, Dice **0.000**,
a missed lesion with 15 expert-labeled voxels. Columns show normalized DWI, expert
annotation in red, and prediction in green. The displayed slice has the largest
expert lesion area; scores describe the whole 3D volume. These are unchanged
**historical notebook outputs**, not newly generated predictions. The saved
“Original DWI” label means normalized DWI; the success/failure selection is illustrative.

MRI/annotations: Hernandez Petzsche et al., **ISLES 2022**, [Zenodo](https://zenodo.org/records/7960856),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Normalization and overlays
come from this repository's experiment. [Figure sources, attribution and evidence](docs/portfolio/figures-and-results.md).

## Technologies

**Python · PyTorch · MONAI · 3D U-Net · NiBabel**

NumPy/SciPy support image processing; Matplotlib supports overlays and failure
analysis. Google Colab provides the notebook workflow.

## Dataset and model

The public **ISLES 2022 dataset contains 250 expert-annotated stroke MRI cases**.
Images and lesion masks use NIfTI/BIDS-style files. Scans come from multiple centers
and are released in native space, without prior registration.

| Input | Contribution | Handling |
| --- | --- | --- |
| DWI | Diffusion-sensitive signal; ischemic lesions can appear bright | Reference voxel grid, channel 0 |
| ADC | Complementary information about restricted diffusion | DWI shape/affine checks, channel 1 |
| FLAIR | Additional tissue context | Affine-based resampling to DWI, channel 2 |

Each modality is normalized over nonzero voxels. **Resampling puts FLAIR on the
DWI grid; it does not estimate anatomical registration.** No modality ablation
establishes each input's added value.

```mermaid
flowchart LR
    A[DWI / ADC / FLAIR] --> B[DWI grid and per-modality normalization]
    B --> C[Three-channel 3D U-Net]
    C --> D[Sliding-window probabilities]
    D --> E[Binary lesion mask]
```

The MONAI U-Net has feature channels `(16, 32, 64, 128)`, strides `(2, 2, 2)`,
two residual units per level, and one output logit channel (**1,187,546 parameters**).
Training uses `64 × 64 × 32` voxel patches, alternating lesion-centered and random
sampling, Dice + cross-entropy loss, and AdamW (`lr=1e-4`, `weight_decay=1e-5`).
Full-volume inference uses overlapping patches and sigmoid probabilities.

**Patient-level methodology:** the fixed seed-42 split is **175 training / 37
validation / 38 internal test patients**, separated before patch sampling. Training
uses training patients; threshold/checkpoint selection uses validation patients.
The historical experiment retained threshold **0.25** after the epoch-23 validation
sweep and reported its epoch-43 checkpoint. This is an internal split of the public
training dataset, **not the hidden challenge test set**. The original validation
checkpoint-to-threshold association remains unresolved.

Dataset publication: Hernandez Petzsche et al., *Scientific Data* (2022),
[doi:10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5).
The current record also requests de la Rosa et al., *DeepISLES*, *Nature Communications*
16, 7357 (2025). That separate model's clinical validation does not apply to this project.

## Internal test results and limitations

**Historical notebook results — not independently reproduced by these reviewers.**
The primary Dice summary includes **37 lesion-positive patients out of 38 total**.

| Metric | Historical result |
| --- | --- |
| Mean full-volume Dice, lesion-positive cohort | **0.5377** |
| Median full-volume Dice, lesion-positive cohort | **0.5818** |
| Lesion-positive patients with empty predictions | **2 / 37** |
| Expert-empty patients with any false-positive prediction | **0 / 1** |

The owner reports exact Dice/voxel-count replication for all 38 cases; the full
comparison records have not been independently reviewed. Agent 2 separately
checked the released epoch-43 checkpoint's hash, strict U-Net loading and synthetic
forward pass. **Checkpoint compatibility is not numerical reproduction or proof of
original training/threshold lineage.** Synthetic tests cover the later selection and
artifact fixes; no corrected real-data scores are claimed here.

- Small-lesion failures remain visible. The saved ranking includes five zero-Dice
  positive cases; empty predictions alone undercount complete overlap failures.
- One expert-empty test case cannot establish specificity. Confidence intervals,
  lesion-wise detection metrics and physical-volume error are not reported.
- Native spacing varies, and affine resampling may leave anatomical misalignment.
  Display rotation does not establish standardized anatomical orientation.
- One internal split, inspected test examples and validation-based selection limit
  generalization claims. Do not tune on these test cases and call them untouched.
- There is **no external/prospective validation or demonstrated clinical readiness**.
  Exact deterministic training/resume and the historical full environment are not established.

[Complete methods, audit history and migration caveats](docs/methodology-and-audit.md)
remain available; [figure/results evidence](docs/portfolio/figures-and-results.md)
separates historical outputs, owner reports and reviewed compatibility.

## Explore in Google Colab

[**Open the portfolio notebook in Colab**](https://colab.research.google.com/github/sbmedtech/isles2022-stroke-mri-segmentation/blob/main/notebooks/ISLES_2022_Portfolio.ipynb)
· [View notebook](notebooks/ISLES_2022_Portfolio.ipynb)
· [Original experiment record](ISLES_2022_First_Colab_Notebook.ipynb)

The companion presents the experiment with a clearer narrative; it preserves
executable cell source/order and saved historical outputs. Save a working copy,
choose a GPU runtime if available, mount Drive, and review the cells before running.
The data archive is approximately **1.69 GB**, with additional extracted/cache space.
Drive paths and stateful cells require care: rerunning initial training can overwrite
checkpoints, a diagnostic cell updates weights, and legacy selection artifacts need
migration. [Detailed execution and reproduction guide](docs/REPRODUCIBILITY.md).

## Local setup and repository

For reusable components and synthetic checks (Python **>=3.10**, reference CPU
checks use 3.12), from an isolated environment:

```bash
git clone https://github.com/sbmedtech/isles2022-stroke-mri-segmentation.git
cd isles2022-stroke-mri-segmentation
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,notebook]'
python -m pip check
python -m unittest discover -s tests -v
```

The CPU install is for local component checks; retain Colab's CUDA PyTorch for GPU
work. This is not a historical environment lock or a guarantee of matching training
results. The package exposes model/metric/reproducibility helpers and scientific
`evaluation` utilities; it does **not** provide a standalone inference/training CLI.
Current notebook preprocessing requires expert masks. Get the dataset from official
Zenodo; do not commit scans, weights or local caches.

```text
notebooks/          Portfolio notebook and reading guide
ISLES_2022_First_Colab_Notebook.ipynb  Original experiment record
isles2022/          Model, Dice and reproducibility helpers
evaluation/         Volume, zero-overlap and patient-bootstrap utilities
tests/              Synthetic correctness and provenance checks
assets/portfolio/   Real historical MRI figure
docs/               Methods, evidence, licensing and audit detail
requirements/       Observed CPU direct pins, not a transitive lock
LICENSE             MIT license for original project code
```

## Licensing

Original project code is [MIT licensed](LICENSE), copyright **2026 Shreyas Bharadwaj**.
ISLES data and the underlying MRI/annotations in the displayed figure retain their
separate **CC BY 4.0** attribution requirements. Third-party libraries retain their
own licenses; the source-code license does not grant rights to pretrained weights.
See [ownership, dataset and third-party notices](docs/LICENSING.md).
