# Reading and using the ISLES 2022 portfolio notebook

[Portfolio companion](ISLES_2022_Portfolio.ipynb) · [Original research notebook](../ISLES_2022_First_Colab_Notebook.ipynb)

The companion presents the existing experiment as a medical AI portfolio study:
clinical context, multimodal inputs, spatial assumptions, model/training choices,
patient-level evaluation, real saved MRI examples, and failure analysis. Its
ordered code cells, outputs, execution counts and cell metadata are copied
exactly from the original. The original file is unchanged. New Markdown provides
navigation without renaming modules or changing their responsibilities.

## Read the study without running it

Open the companion on GitHub, in Jupyter, or in Google Colab. Use the table of
contents to move from the experiment overview to data/model sections, validation,
internal testing and the preserved figures. Reading saved outputs does not need
a dataset download, GPU, Drive mount or training run.

The visible metrics and images are historical notebook outputs. They were not
regenerated for this companion and do not demonstrate fresh execution of the
later repaired code. The patient populations and reported numbers are retained;
synthetic test results are separate from MRI performance evidence. Historical
checkpoint/threshold provenance remains a separate review, not a portfolio
presentation claim.

## If deliberately starting a new experiment

1. Save a personal notebook copy and choose a fresh Colab runtime (the saved run
   used a T4). Follow the code order; the current experiment relies on shared
   notebook state rather than a training CLI.
2. Change every repeated `PROJECT_DIR`/`DATASET_DIR` path consistently. Use a
   separate Drive project and a fresh/distinct local cache. The original paths
   are personal defaults, and cache creation accepts files based on existence.
3. Review storage before downloading the approximately 1.69 GB archive. Download
   from the official Zenodo record; extraction and tensor caching need more space.
4. The optimizer diagnostic changes weights before counted epochs. Initial
   training starts a new history and can overwrite the latest checkpoint.
   Continue through epochs 3 → 23 → 43 only when intending to train. This guide
   does not prescribe new hyperparameters or a different execution schedule.
5. Selection and resume require matching artifacts. Legacy post-epoch-23
   checkpoints without provenance cannot be safely adopted; consult the README
   migration notes. Checkpoints do not restore every RNG state.
6. Evaluate a frozen checkpoint with its validation-selected threshold. New test
   CSVs/figures/PDFs carry provenance manifests. Do not tune on the already
   inspected internal test partition while claiming an untouched test estimate.

No real MRI retraining or Colab execution was performed for this presentation
change. An execution guide is not a guarantee of reproducing historical scores.

## Lightweight component checks

From the repository root in a suitable environment:

```sh
python -m pip install -e '.[test,notebook]'
python -m unittest discover -s tests -v
```

The existing tests check model compatibility, Dice and scientific utilities,
seeding, packaging, checkpoint selection, and synthetic artifact provenance.
The portfolio preservation checks additionally compare the companion's ordered
code cells and outputs with the unchanged original, and protect runtime Python
statements during docstring/comment edits. They never execute MRI training.

## Scope and ownership

Issue #11 owns this companion and existing module docstrings/comments. Agent 3
owns the top-level README, licensing and layout under issue #15; Agent 2 owns
real figure/result evidence and captions under issue #10. This change creates no
new gallery assets, model release, inference CLI, training framework or metrics.
Independent review and repository-owner approval precede merging or issue closure.
