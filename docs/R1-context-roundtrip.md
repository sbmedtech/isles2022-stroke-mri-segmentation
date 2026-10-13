# R1: realistic AdamW artifact provenance round-trips

Owner: **Agent 1: Software Engineer**. Assignment: [issue #9](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/9).
Branch: `release/artifact-context-roundtrip`, isolated worktree `/workspace/isles-r1`.
Approved base: `63b50bc3ac5046ad6efe25654e15321616aab4c3`, including merged
coordination PR #17 and its role-identified Agent 2/Agent 3 PASS evidence for head
`b5835b469217f89ab746932563fca6292366e872`.

## Reproduced failure

The 40 existing tests passed on the approved base. New regression commit
`dad8fbfe17de36e24f2bc29629040b86cdfc70d6` executes the notebook's complete
cell-45 `selection_context` assignment with actual MONAI UNet, DiceCELoss and
AdamW objects. A synthetic forward/backward/optimizer step populates AdamW
state. The fixture uses the original model arguments and parameter count, and
the original seeded disjoint 175/37/38 patient-ID partition. Images/labels are
synthetic; no real MRI data or historical weights are loaded.

AdamW's parameter group contains `betas=(0.9, 0.999)`. The exported JSON manifest
contains `betas=[0.9, 0.999]`. Hashing already encoded both as JSON arrays, so the
file path and context hash matched, but exact Python dictionary equality failed
on read-back and source-manifest comparison. Unchanged artifacts were rejected
as foreign experiments. Before the fix the three regressions produced one pass,
one assertion failure and one `ValueError`; these were expected reproductions,
not passing acceptance evidence.

## Narrow fix and compatibility

Cell 45 introduces `canonical_json_value`: copy metadata through strict sorted
JSON encoding/decoding. This recursively represents both tuples and lists as
JSON arrays without discarding fields or converting unsupported values into
strings. Unsupported objects and nonfinite numbers fail explicitly.

Context hashing, provenance construction, artifact-directory resolution,
manifest export, source-manifest comparison and read-back comparison use the
same representation. Sorted JSON byte encoding is unchanged, preserving valid
existing context/checkpoint digests and artifact paths. Strict identity and
content/cohort checks remain in place. Canonicalization does not mutate optimizer
parameter groups or the source selection context; training still uses tuples.
There are no schema-field, package-interface or dependency changes.

Only cell-45 helper serialization changes. All other cells, historical outputs,
cell/notebook metadata, model/preprocessing/training/loss/optimizer definitions,
patient split, threshold selection and artifact consumer paths are preserved.
Two existing author-owned AST helper loaders add the new helper name; they do
not mock or reimplement the fix. R3 and unrelated release tasks remain unstarted.

## Acceptance evidence

Run from the assigned checkout:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/isles-matplotlib \
  /workspace/isles-environment/venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The full suite includes all 40 existing tests plus four realistic-context
regressions. New tests round-trip complete provenance and consume actual notebook
CSV/PNG/PDF paths across the synthetic 38-patient cohort. They verify unchanged
digests/paths, tuple/list equivalence without input mutation, and rejection of
changed run, epoch, weights, checkpoint identity, threshold, optimizer betas,
split, preprocessing, environment, cache identity, file content and patient
order. Unsupported/nonfinite metadata is rejected. The existing suite also
checks missing manifests, mask/metric disagreement and cross-run contamination.

The built-wheel test builds offline without dependencies and imports both
namespaces from an extracted wheel outside the checkout. Notebook syntax and
exact historical/scientific preservation checks are recorded in issue #9 and
the linked PR with the tested head SHA. Test results must be read from those
head-specific records; documentation alone is not independent acceptance.

Environment: Python 3.12, PyTorch 2.14.1+cpu, MONAI 1.6.1, NumPy 2.5.4,
NiBabel 5.4.2. Real historical checkpoint compatibility, MRI performance,
GPU/Colab execution and clean dependency resolution were not exercised.

## Review and blockers

Implementation evidence is authored evidence, not independent PASS. Agent 3
must independently test the current PR head, and Agent 4 must review/integrate
that evidence. The repository owner controls merge authorization. Issue #9
stays open until approved merge and acceptance evidence; R3 remains dependency
blocked. No model/data publication or historical-result regeneration occurs.
