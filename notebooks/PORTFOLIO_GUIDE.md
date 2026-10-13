# Using the refactored ISLES 2022 portfolio notebook

[Primary portfolio notebook](ISLES_2022_Portfolio.ipynb) ·
[Historical reference in Git history](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/blob/e507752a5095bf287e5a7291f819e3a0d0c72135/ISLES_2022_First_Colab_Notebook.ipynb)

The portfolio is now executable Python orchestration over reusable `isles2022`
functions. It is **not an executed reproduction**: all code outputs/execution
counts are cleared. Old download logs, progress widgets, debug prints and repeated
procedural cells are removed. The original experiment remains unchanged.

## Read historical evidence separately

The overview table retains the historical patient populations and mean/median
Dice. The epoch-23 loss image is an unchanged PNG from original cell 40, displayed
as a labeled Markdown attachment. The MRI examples link the reviewed repository
asset from original cell 47, with dataset credit and evidence notes. Neither is
output of the refactored code. No plot, scan, weight or historical metric was
regenerated or relabeled as verified.

The primary historical internal-test summary is mean Dice **0.5377**, median
**0.5818**, among **37 lesion-positive of 38 total** patients. Review of threshold
lineage, compatibility, owner-reported replication and clinical validity are
separate evidence categories; synthetic tests do not collapse them into one claim.

## Setup and execution

The first cell installs the notebook extras from a repository checkout. Colab can
clone the repository when absent; before the PR merges, review from its checkout
rather than installing main. Local Jupyter should start in the repository root
or `notebooks/`. No inference CLI is added.

Paths and flags are centralized in `ProjectPaths`, `ExperimentConfig` and the
configuration cell. `PREPARE_DATA`, `RUN_TRAINING` and `RUN_EVALUATION` all default
to False. Reading the historical study requires none of these actions.

- **Data preparation:** enable only with a usable Drive/local project. Official
  archive size checking and seed-42 sorted patient shuffling preserve the
  175/37/38 split. Training/validation alone enter the tensor cache. Use a fresh
  project/cache: existing cache files are still accepted by existence.
- **Deliberate training:** the named diagnostic optimizer function retains the
  original pre-epoch weight update. Stages remain 3 → 23 → 43, with two then four
  patches per patient and validation every five epochs after 23. Initial training
  can overwrite latest; it belongs in a separate experiment directory.
- **Evaluation only:** actual MRI files, expert labels for metrics, cached
  train/validation tensors and matching selection artifacts are required. The
  exact selected checkpoint is validated before weights/threshold are used.
  Missing lineage or thresholds are rejected, not replaced with 0.25.
- **Figures:** current predictions must match a provenance-bound test CSV before
  plots/PDFs are exported. The size analysis remains voxel-count/Spearman analysis,
  not a new native-mL result. Historical assets are never export destinations.

These paths have not been run on MRI data during this task. Do not enable flags
merely to view figures. No training or tuning was performed for the refactor.

## Module responsibilities and contracts

| Module | Inputs and responsibilities |
|---|---|
| `config.py` | Explicit project/cache paths and unchanged model/experiment defaults |
| `data.py` | Official ZIP, patient discovery, split generation/persistence |
| `preprocessing.py` | `(patient_id, dataset_dir)` → DWI/ADC/FLAIR tensor and expert mask |
| `datasets.py` | Explicit cache path, alternating paired crops, loader settings |
| `model.py` | Existing raw MONAI factory; no new wrapper/state-dict prefix |
| `training.py` | Explicit model/optimizer/loss/loader/device, shared epoch and stage functions |
| `evaluation.py` | Explicit cohort and `load_patient(id)` callback; one CPU-output inference path |
| `checkpoints.py` | Original dictionary I/O and pure measured-selection helpers |
| `provenance.py` | Canonical context/weight identity and CSV/figure content association |
| `visualization.py` | Current checked figures and streamed four-patient PDF pages |

Preprocessing is unchanged: DWI grid; ADC/mask affine and shape checks; linear
FLAIR resampling; finite nonzero z-scoring; DWI/ADC/FLAIR order; binary mask.
No reorientation, spacing, registration, augmentation or label-free inference
fallback is introduced. Patch draws/padding/crops and update arithmetic are
compared directly with the reference notebook. Loss and AdamW defaults remain
identical, as do full-volume Dice and positive-case summaries.

Raw model/optimizer checkpoints retain the original four-key schema and strict
state-dict loading. The existing six-argument MONAI architecture remains intact.
Refactored functions have different source fingerprints: old selection contexts
are not silently certified as module-era runs. Compatible weights alone do not
prove historical threshold lineage; source/context validation remains strict.

The published `model-v0.1.0-preview` raw state has independently checked model
compatibility, but its four-key checkpoint lacks threshold/selection metadata.
It cannot enter strict provenance-checked evaluation without actual selection
evidence. See the README for download/checksum and direct loading guidance; no
legacy threshold fallback or fabricated metadata is introduced here.

Named functions replace hidden global bindings, but the notebook still has an
explicit top-to-bottom experiment sequence. The full historical order of debug
RNG draws is not reproduced. Consequently neither identical scratch training nor
interrupted-run RNG replay is claimed. Best/latest interruption recovery and
cache-generation provenance remain unchanged limitations, not new framework work.

## Verification and unavailable checks

From a suitable checked-out environment:

```sh
python -m pip install -e '.[test,notebook]'
python -m unittest discover -s tests -v
```

Existing reference tests are retained. The old companion-copy expectations are
replaced with checks for empty code outputs and unchanged historical Markdown
image bytes, as appropriate to the owner's newly authorized code refactor.
Synthetic tests compare normalization, NIfTI preprocessing, sampling/RNG state,
patient splits, cached dataset policy, optimizer updates, checkpoint snapshots,
full-volume validation, threshold sweep/ties and selection cadence directly with
the original source. Integration tests cover canonical metadata, exact checkpoint
validation, checked CSV/PNG/PDF consumption and cross-run rejection. Archive and
network tests use fixtures; no dataset download is needed for the test suite.

Actual dataset-wide preprocessing, GPU/Colab execution, complete historical
checkpoint lineage, anatomical alignment and numerical performance reproduction
require the real MRI data/artifacts and remain **unverified**. No new clinical or
external-validation claim is made. Real-data absence does not become synthetic
PASS evidence. Record the precise tested head and results in the PR.

## Coordination and archival treatment

Agent 1 owns the executable portfolio and module extraction. Agents 2 and 3 are
requested for independent science and code/Colab review; Agent 4 coordinates.
Agent 3 owns README/file-listing changes. The owner reports a Colab/Drive backup;
original-file listing/removal must follow their explicit confirmation and a
separate coordinated change. This PR neither edits nor deletes the original
notebook, and Git history is preserved. No merge or issue closure is authorized.
