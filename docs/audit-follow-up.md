# Independent verification of Agent 4's audit

Source: complete final audit in the shared Codex chat **Agent 4: Manager**
(`01a13641-51d5-7296-a131-b4baf20d70f3`). Audited and verified repository base:
`a29b1662f8e6affebb1b30a213a5734bd9f2487b` (merged PRs #1–#4).

The original 19 tests passed independently before fixes. The findings below were
checked against current source, rather than inferred from the audit's verdict.

| Priority | Independent finding | Resolution / next action |
|---|---|---|
| P1 | Cell 45 inserts historical epoch-23 metrics, trusts an external CSV, and retains existing best weights without verifying association. A fresh run with lower scores can keep unrelated weights. | Fixed: measure the current baseline, overwrite stale selection artifacts at initialization, bind weights/threshold/history to a unique selection run and context, remeasure matching best weights on resume, reject unverifiable legacy resumes. Regression tests execute actual notebook helpers and Step 16 control flow. |
| P2 | Cell 43 computes `best_threshold` but later validation/test/figures use literal `0.25`. | Fixed: persist the threshold with the evaluated epoch-23 weights, propagate checkpoint metadata to later validation/test/figures. Historical default remains 0.25 when validation selects it. |
| P2 | Setuptools discovery excludes the merged `evaluation` namespace. | Fixed: include both namespaces and evaluation usage documentation; regression test builds a wheel and imports both namespaces outside the checkout. |
| P2 | NiBabel is imported by scientific tests but missing from installation metadata; `np.quantile(method=...)` requires NumPy >=1.22. | Fixed: declare NumPy minimum and a test extra with NiBabel and wheel-building tools; README uses that extra. Training dependency versions remain unpinned. |
| P2 | Cache creation trusts file existence, checkpoints omit RNG state, initial training overwrites latest history, and a diagnostic optimizer update precedes epoch 1. | Selection now fingerprints cache content/configuration and rejects changed resume contexts. Cache creation, initial training, diagnostics, and deterministic RNG restoration remain follow-up work; content identity cannot certify that a cache was correctly produced. |
| P2 | FLAIR uses affine resampling rather than estimated registration. | Confirmed implementation limitation; actual anatomical misalignment cannot be established without patient images and registration QC. No preprocessing change. |
| P3 | README omitted evaluation utilities/tests and early diagnostic cell 37 uses default output-device behavior. | Added evaluation/package/test documentation. The final evaluation uses CPU output accumulation; early diagnostic inference remains unchanged. |
| P3 | `missed_cases` counts empty predictions, not all zero-overlap cases. Saved test results have five positive zero-Dice cases and two empty predictions. | Confirmed; preserve metric semantics and integrate a separate zero-overlap report in a future scientific evaluation change. |
| P3 | Test artifacts hard-code epoch 43; lesion-size analysis uses voxels. | Test CSV filenames and PDF epoch now follow the selected checkpoint. Example figure descriptions remain historical; native physical-volume analysis is deferred. |

The architecture, preprocessing, splits, patch sampling, initial training, loss,
optimizer, validation schedule, and Dice calculation are preserved. Notebook
outputs/metadata are preserved as historical evidence, not regenerated results.

Validation: 28 tests pass, including 19 existing tests, eight selection tests,
and a built-wheel packaging test. Tests cover stale scores/weights, measured
baseline selection, non-default thresholds, strict ties, best-weight updates,
resume, missing/foreign best checkpoints, changed provenance, and nonfinite
baseline metrics. All notebook code parses after excluding IPython magics;
whitespace checks pass. The recorded environment is Python 3.12,
PyTorch 2.14.1+cpu, MONAI 1.6.1, NumPy 2.5.4, NiBabel 5.4.2.

Limits: no actual historical checkpoint, released experiment artifacts,
full-data rerun, CUDA/Colab execution, center-independence check, or anatomical
alignment assessment was available. Synthetic checkpoint compatibility does not
establish historical numerical reproducibility. Legacy post-epoch-23 checkpoints
are intentionally rejected for selection resume because their lineage is unknown.
