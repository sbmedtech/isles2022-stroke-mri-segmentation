# Final audit follow-up: figure and test-artifact provenance

Read Agent 4's complete latest final audit in the shared chat **Agent 4: Manager**
(`01a13641-51d5-7296-a131-b4baf20d70f3`), targeting immutable commit
`766575f23569577fc1c9880db7a8c8856a424bb2` (merged PRs #5, #6 and #7).

## HIGH findings addressed

1. **Cell 47 validated a previous in-memory checkpoint, then loaded unchecked
   figure weights.** It now loads once, validates that exact object's run/context
   and weight digest, then derives its threshold and loads its weights. The same
   ordering is used by cells 46 and 49; cell 48 reloads/validates before plotting
   stored metrics. Regression tests execute the full cell 47 path with a valid
   previous checkpoint and a foreign file, and the inverse case. Foreign weights
   are rejected before model loading or inference; previous object state cannot
   supply the threshold for newly loaded weights.
2. **Epoch-only CSV/figure filenames admitted cross-run contamination.** Cells
   46–49 now use run- and full-provenance-bound directories, file manifests, and
   SHA-256 content checks. Manifests record weight/checkpoint identity, threshold,
   split and preprocessing configuration, ordered test IDs, and environment.
   Figure manifests reference their source CSV digest. Real notebook consumers
   reject foreign, stale, missing-manifest, modified-content and wrong-cohort
   artifacts. Example/PDF predictions must agree with stored patient metrics;
   PDF publication occurs after every patient passes. No legacy output is adopted
   based only on epoch. Restore provenance by rerunning evaluation, not inventing
   metadata for historical files.

`checkpoint_digest` is a canonical identity of weights plus run/epoch/threshold/
context, not a serialized-checkpoint file hash. Selection-context validation still
uses the existing checkpoint contract; model architecture and patient partition
are unchanged. All saved historical notebook outputs and metadata are preserved.
Example labels during fresh execution identify patients rather than claiming
historical best/worst performance.

Validation: **40 tests passed**, comprising 32 existing tests and eight new
artifact regressions. New tests execute full notebook cells 46–49 on synthetic
volumes and real temporary CSV/PNG/PDF files, substituting only expensive
sliding-window inference and synthetic preprocessing/source inspection. They
exercise equal-epoch cross-run swaps, threshold/weight/configuration/split
identity mismatches, changed preprocessing, file digests, patient order, missing
manifests, and metric/prediction disagreement. Python/notebook syntax and diff
whitespace checks pass. Architecture, splits, preprocessing and training/selection
definitions are unchanged; only artifact helpers are added to cell 45.

## MEDIUM findings: separate implementation backlog

| Finding at the audited commit | Recommended next action | Status in this PR |
|---|---|---|
| Improved best is saved before latest; interruption can leave best newer than latest and safely block resume. | Immutable epoch checkpoints plus an atomically updated selection manifest, or validated recovery of the interruption window; add interruption injection tests. | Deferred; checkpoint transactions unchanged. |
| RNG states are not restored; initial training resets history/overwrites latest, and a diagnostic optimizer step precedes counted epochs. | Save/restore Python, NumPy, CPU/CUDA and loader generator states; isolate experiments and diagnostics; test interrupted training equivalence. | Deferred; original training semantics preserved. |
| Cache creation accepts existing files without source checksums or generation metadata. | Bind caches to source NIfTI checksums, transform settings and code revision; rebuild/reject mismatched caches. | Deferred; selection's byte fingerprint and export source checks cannot certify cache origin or actual MRI data identity. |
| Full reconstructable training configuration/environment/code revision is not stored in checkpoints. | Export immutable code revision, full configuration and resolved dependency inventory with source-data checksums; validate Colab/GPU environment. | Partially addressed for result export: full available selection context and environment inventory are in manifests; immutable code/data provenance and complete training replay remain unavailable. |
| Affine resampling is not registration, voxel patches vary in physical extent, and scientific reporting lacks zero-overlap, native mL and center-held-out validation. | Assess alignment on actual scans and predefine an evaluation protocol; report separate failure categories and physical volume without changing the historical record. | Deferred; no preprocessing or clinical/scientific validity claim added. |

## Remaining limits

Synthetic artifact verification does not establish performance of historical
weights, registration quality, dataset correctness, deterministic resumed
training, or GPU/Colab execution. No actual historical checkpoint or patient data
was available and no real MRI metrics were generated. Existing manifests are
integrity/association checks, not signatures against deliberate forgery. A new
checkpoint, threshold, environment or configuration requires a new validated
evaluation export. Legacy files without manifests remain historical evidence.

The recorded synthetic environment is Python 3.12, PyTorch 2.14.1+cpu,
MONAI 1.6.1, NumPy 2.5.4 and NiBabel 5.4.2. Direct CPU pins are not a complete
historical or GPU training lock. CI and complete real-data inference coverage
remain separate follow-ups.
