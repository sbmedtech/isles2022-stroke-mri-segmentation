# Reproducibility assessment and execution guide

## Current portfolio and released-checkpoint usability

Use the [portfolio notebook](../notebooks/ISLES_2022_Portfolio.ipynb) as the primary
reading/experiment entry point; its [guide](../notebooks/PORTFOLIO_GUIDE.md) explains
execution and historical evidence. The original research notebook remains in the
current tree pending the owner's explicit backup confirmation; no history rewrite
or external-record deletion is authorized. A later executable refactor must be
reviewed independently before its changed cells or outputs are accepted.

The [preview release](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/releases/tag/model-v0.1.0-preview)
contains `checkpoint_best_validation.pt` (14,321,909 bytes), SHA-256
`dd6ae8182a9af708662ad43e460459ef3deb3f89e70cc585ee9750d91755b70d`.
The [README loading example](../README.md#pretrained-model) was executed against
those actual bytes using the installed package wheel outside the checkout:
`build_model()` and strict loading of `checkpoint["model_state_dict"]` succeed;
metadata reports epoch 43, all 49 state entries are finite, and eval mode is set.
This is CPU artifact/API compatibility, not an MRI inference or metric reproduction.

The file also stores optimizer state and training history but lacks the selection
run/context and validation-threshold provenance required by repaired resume and
final-evaluation paths. Do not fabricate those fields, use test scores to choose a
threshold, or adopt this legacy file into a provenance-bound workflow by renaming it.
The weight license is separate from MIT code licensing. Original threshold/training
lineage, full 38-case comparison evidence, fresh Colab/GPU/MRI execution and clinical
validity remain unverified; the owner's reproduction report is a separate category.

The assessment below records earlier component checks and historical execution
caveats. Its statements about unperformed checkpoint checks refer to that earlier
work, not to the compatibility check documented above.

## Independent assessment

Agent 4's complete audit was accessible in the shared chat **Agent 4: Manager**.
This review checked dependency metadata and notebook setup independently against
GitHub main `a29b1662f8e6affebb1b30a213a5734bd9f2487b`.

Confirmed gaps, in priority order:

1. Initial seeds do not establish equivalent interrupted/resumed training.
   Checkpoints omit RNG states and full environment/configuration provenance;
   notebook execution and diagnostic draws consume RNG state. Cache reuse is
   based on existence. These experiment-provenance gaps remain unresolved here.
2. Dependencies are unpinned; Colab supplies an unspecified Python/PyTorch
   runtime. Saved output reports PyTorch `2.11.0+cu130`, but does not establish
   the full historical environment. This CPU environment cannot reconstruct it.
3. Package discovery omitted `evaluation`; install-then-test instructions omitted
   NiBabel. This PR includes both namespaces, a test extra, a notebook extra,
   and NumPy's required minimum for linear-method quantiles.
4. Colab imports, personal Drive paths, and stateful cells prevent unadapted local
   notebook execution. No training or evaluation CLI exists. Commands below
   distinguish component checks from the manual notebook experiment.

Agent 1's checkpoint-selection repair is now present in this branch; see the
README migration instructions and [audit follow-up](audit-follow-up.md). The
repair adds selection provenance and measured baseline checks, but does not
restore RNG state or certify cache-creation provenance.
Optional seed helpers are not automatically wired into the original notebook.

## CPU component setup

Reference: Linux x86_64, Python **3.12.14**, PyTorch **2.14.1+cpu**. Direct-package
pins in `requirements/reproducibility-cpu.txt` match the observed environment
where synthetic tests passed. They are not a transitive lock, historical training
environment, GPU prescription, or proof of a clean installation.

From the repository root with Python 3.12.14 available:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/reproducibility-cpu.txt
python -m pip install --no-build-isolation -e '.[test,notebook]'
python -m pip check
python -m unittest discover -s tests -v
python -m isles2022.reproducibility > environment.json
python -m pip freeze > resolved-environment.txt
```

The pins, Python patch version, wheel sources, OS, and resolved transitive
versions should be retained with any new experiment. Installation requires
network access. A clean dependency download/install was not performed for this
PR. Tests use the pre-existing managed CPU environment; wheel build and isolated
wheel imports were verified there. The environment command reports only a
specified allowlist of settings and versions, never credentials or all env vars.

## Seeding a new experiment

Set process-level settings before starting Python; setting PYTHONHASHSEED inside
a running notebook cannot change its existing hash seed:

```sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 python your_experiment.py
```

`your_experiment.py` means a user-supplied experiment entry point; this repository
does not supply one. In new experiment code, before model/data initialization:

```python
from torch.utils.data import DataLoader
from isles2022.reproducibility import seed_everything, seed_worker

generator = seed_everything(42, deterministic=True)
# dataset is supplied by the experiment, not fabricated by this helper.
loader = DataLoader(dataset, shuffle=True, generator=generator,
                    worker_init_fn=seed_worker, num_workers=0)
```

Strict PyTorch determinism may raise for unsupported operations. It does not
guarantee identical outputs across PyTorch versions, CPU/GPU, or hardware.
NumPy `default_rng` instances require their own explicit seeds. Generator states
must also be saved/restored for exact continuation; these helpers only seed
initial runs and workers, and do not modify checkpoint serialization.

## Google Colab training and evaluation

Use a fresh Colab runtime with a T4 GPU if available. Clone a chosen immutable
repository commit and install `.[test,notebook]` from that clone with `%pip`.
Record `sys.version`, `torch.__version__`, GPU/CUDA, the environment report, and
`%pip freeze`; restart after changing imported dependencies. The CPU pin file
must not be used to replace Colab's CUDA-enabled PyTorch. A validated GPU pin
set is still unavailable. `google.colab` is supplied by Colab, not pip-installed
as a local notebook dependency. Rerunning the notebook's unpinned install cells
can change the environment, so re-record it afterward.

The following step labels describe the preserved original research record, not a
newly executed experiment. Use the portfolio notebook and guide for the current
presentation; this historical execution map retains the migration caveats:

- Mount Drive and set the project/dataset paths before preprocessing; `/content`
  is temporary and `/content/drive/MyDrive/ISLES2022_Project` is user-specific.
- Follow sections for dataset download, split creation, preprocessing, patch
  sampling, model/loss setup, and cache creation. Preserve the split manifest
  and dataset checksum. Do not overwrite previous experiment artifacts.
- Training sections are Step 9 (initial 3 epochs), Step 11B (through epoch 23),
  and Step 16 (through epoch 43). Step 7B also updates weights as a diagnostic;
  it is not a neutral check. Capture the exact cell execution order.
- Validation/threshold assessment occurs in Steps 12–14. Step 17 evaluates the
  internal test set. Follow the README's checkpoint-selection migration instructions before using
  those sections for a new scientific rerun; legacy post-epoch-23 selection
  resumes are rejected without provenance.

There is no truthful `python train.py` or `python evaluate.py` command to document
at this revision. Local execution requires adapting Colab imports and paths.
Do not treat component unit tests as end-to-end training/evaluation commands.

## What was and was not verified

CPU synthetic tests validate calculations, model parity, repeatable initial RNG
draws/loader-generator order, worker seeding, safe environment reporting, and
wheel packaging/imports. `pip check` passed in the existing environment.

No real MRI dataset preprocessing, Colab execution, CUDA determinism, training,
resume equivalence, actual historical checkpoint loading, or historical Dice
reproduction was performed. Missing split/checkpoint/results artifacts remain
missing; no patient records or performance metrics are generated by this work.
