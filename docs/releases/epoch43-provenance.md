# R2: epoch-43 provenance investigation

Owner: Agent 2: ML Research Scientist. Tracking: [issue #10](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/10).
Reviewed immutable baseline: `63b50bc3ac5046ad6efe25654e15321616aab4c3`.
This is an evidence inventory, not approval to publish weights. The proposed
manifest is [model-release-manifest-contract.md](model-release-manifest-contract.md).

## Evidence and conclusions

| Claim | Available evidence | Assessment |
| --- | --- | --- |
| Exact reproduction of 38 historical test patients | Owner reports independently reproducing every Dice and voxel count exactly | Acknowledged owner-reported numerical reproduction; comparison files and checkpoint have not been inspected by this reviewer |
| Historical test mean 0.5377 / median 0.5818 | Notebook cell 46 saved output: 37 lesion-positive of 38 patients, two empty predictions, one expert-empty patient | Verified presence of historical output, not independent MRI execution or full-row reproduction |
| Model is epoch 43 | Saved output says loaded epoch 43; Step 16 saved output reports best epoch 43 | Epoch label is documented; serialized checkpoint identity and metadata remain unverified |
| Threshold 0.25 selected on validation | Cell 43 lists 10 thresholds (0.05 through 0.50), selects maximum mean Dice on expert-positive validation cases | Documented historical validation policy and saved sweep; checkpoint/CSV association not yet verified |
| Best validation model selected at epoch 43 | Cell 45 saved history reports means 0.521933, 0.564337, 0.577172, 0.586980, 0.599065 at epochs 23/28/33/38/43 | Documentary evidence only; historical baseline/history association was previously unsafe; repaired source cannot retroactively certify old weights |
| Training lineage is reproducible | Notebook describes 3 epochs with two patches/patient, then four through epochs 23 and 43; AdamW and Dice+BCE | Intended recipe is available; original code/runtime execution order, parent checkpoint identities, RNG state and cache origin are not established |
| Patient split is isolated | Cell 19 sorts IDs, shuffles seed 42, partitions 175/37/38 with disjointness assertions | Source-level policy verified; actual historical ordered split manifest still required |
| Architecture and preprocessing | Cells 22/27/28 and package config/model | Source contract available; actual checkpoint compatibility/data provenance require artifacts |

The scientific distinction is essential: exact outputs from supplied weights can
establish numerical reproducibility of inference for a fixed cohort. It does not
alone prove how those weights were trained, which validation weights selected
0.25, whether they are the original bytes, or external/clinical validity. The
owner's report is not discounted; the artifact-verification step is outstanding.

## Threshold evidence

At epoch 23, saved cell 43 output reports 36 expert-positive cases among 37
validation patients. Mean Dice is 0.521933 at 0.25, compared with 0.520222 at 0.30
and 0.515741 at 0.20. The code selects by mean Dice, not median, excludes the
expert-empty case from that objective, and evaluates inclusive `>=` thresholds.
The saved history retains 0.25 through epoch 43 and final testing. This supports
the historical explanation **selected at epoch 23 and retained**, not a claim
that epoch-43 validation reoptimized the threshold.

Current repaired cell 43 binds the selected threshold to matching epoch-23
weights; cell 45 validates run/context and weight identity. Later legacy
checkpoints without selection provenance are rejected. These safeguards are
source/synthetic evidence for new execution, not evidence the historical run
used those safeguards. Do not add guessed selection metadata to old files.

Current result manifests distinguish canonical `checkpoint_digest` (weights plus
run/epoch/threshold/context) from SHA-256 of serialized checkpoint bytes. Both
identities are useful and must be labeled separately. File hashes detect changed
bytes but do not prove truthful lineage or prevent deliberate forgery.

## Exact artifact request to the owner

Provide artifacts privately for inspection; do not commit weights, scans,
credentials, or Drive caches to the public repository. Keep originals read-only.

### Minimum to verify fixed-checkpoint reproduction

1. The exact `models/checkpoint_best_validation.pt` used for the reported 38-case
   reproduction, its SHA-256 and byte size, and the original historical copy if
   different. Supply actual metadata/state-dict inspection, not just the filename.
   If bytes were reserialized, report that and preserve both hashes.
2. Original historical `results/test_epoch43.csv` and the reproduction CSV,
   each with all 38 unique patient IDs and full-precision Dice, expert voxels,
   predicted voxels, and intersection voxels. Include the rowwise comparison or
   exact comparison script/log, hashes, and any new `test.csv.manifest.json`.
   Rounded screenshot values cannot establish exact matches.
3. Original `splits/patient_splits.json` with all ordered train/validation/test
   IDs and the reproduction split file. Record their hashes and order equality.
4. Executed reproduction Colab notebook or script, execution log, exact source
   commit if used, checkpoint load path/hash, threshold, preprocessing/inference
   settings, Python/pip inventory, PyTorch/MONAI, GPU/CUDA/cuDNN versions, and
   dataset archive checksum/version. If files were adapted locally, supply their
   hashes or a diff. No raw MRI upload is needed for the initial evidence review.

### Minimum to establish historical threshold and selection lineage

5. The preserved epoch-23 checkpoint actually used in the original threshold
   sweep: `checkpoint_epoch23.pt` or a preserved epoch-23 `checkpoint_latest.pt`.
   If neither survives, say so explicitly. A latest file now at epoch 43 is not
   an epoch-23 substitute. Include its hash and epoch/state-dict metadata.
6. Original `validation_threshold_details.csv` (37 patients × 10 thresholds)
   and `validation_threshold_summary.csv`, full precision, hashes, sweep log and
   executed original Colab notebook. Identify the checkpoint used by that sweep
   and document that 0.25 was fixed before test evaluation. Include the empty
   validation case and explicitly retain the 36-positive selection denominator.
7. `extended_validation_history.csv`, `training_history.json`, original Step 16
   logs and notebooks, baseline/best/latest save/load records, and any selection
   context/manifest already genuinely produced. Associate metrics with actual
   checkpoint hashes rather than trusting filenames or timestamps alone.
8. Original training environment, source notebook revision/hash and execution
   order, dataset/source-file checksums, and preprocessing/cache generation
   records. Explain restarts, rerun diagnostic optimizer steps, artifact reuse,
   and manual changes. If unavailable, record unknowns rather than reconstructing
   them from today's environment.

### Useful additional evidence, if preserved

Epoch-3, epoch-28/33/38/43 intermediate checkpoints, optimizer state and parent
hashes strengthen training/selection lineage. Saved Python/NumPy/CPU/CUDA/loader
RNG states strengthen continuation evidence; their absence means deterministic
resume cannot be certified. Original center metadata supports cohort assessment
but is not required to compare the reported fixed-checkpoint voxel metrics.

## Review procedure and remaining gates

Hash files before inspection; use safe checkpoint loading and preserve unknown
fields. Compare checkpoint state-dict keys/shapes strictly with the documented
MONAI model. Verify actual epoch and distinguish file identity from weight
identity. Compare all 38 IDs, order, voxel integers and full-precision Dice,
recomputing Dice from counts including the both-empty convention. Record exact
comparison rules and hash every input; never silently join away missing rows.

Verify actual splits are disjoint and complete. Independently reconstruct the
validation sweep summary from its 370 records, check threshold tie handling,
identify the epoch-23 weights, and trace the fixed threshold through selection
and reproduction. Test records must not choose weights, thresholds or settings.

If epoch-23 weights/records are missing, validation provenance remains blocked.
A new validation-only measurement on the supplied frozen model may provide new
evidence, but cannot recreate a missing historical sweep or historical training
lineage; retain separate run identity and document any prior test inspection.
Do not retrain, retune, relabel historical records, or publish on that basis.

R2 can produce this missing-evidence report while dependent model-release gates
remain blocked. Independent Agent 3 QA and Agent 4 integration review are still
required; this author cannot grant release acceptance. Software, weight and data
licenses and owner publication permission also remain release requirements.

## Checks and limitations

Read immutable repository source, saved outputs, package metadata, audit notes,
issue #10, and merged AGENTS.md. The checkout was materialized through GitHub's
contents API because shell Git networking was unavailable; the notebook API
returns parsed JSON, so local JSON serialization differs while its cells,
outputs and metadata remain the source of review. Only the two new documentation
files are proposed remotely; no notebook serialization change is published.

The full available synthetic suite is run for this investigation and recorded
with its command/count in the PR. No historical checkpoint, comparison CSV,
dataset, real MRI inference, validation sweep, training, GPU/Colab execution,
deterministic resume or licensing approval was verified in this investigation.
