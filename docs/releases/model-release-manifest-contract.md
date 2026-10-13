# Proposed model-release manifest contract, version 1

Status: **proposal for review**, tracking [R2 #10](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/10).
This is a scientific field contract, not a validator, adopted shared API,
completed manifest, signed attestation, or permission to publish a model.
Agent 1/3/4 must agree on implementation and integration before consumers use it.

## Representation and evidence rules

Use UTF-8 JSON with `schema_version: "1.0"`, an immutable release identifier,
research-only intended use, and a creation timestamp. Each factual section has
`status` in `verified`, `owner_reported`, `unverified`, `unavailable`, or
`not_applicable`; evidence references include URI/private artifact identifier,
SHA-256, reviewer role, review date, and immutable source revision. Use JSON
`null` for unknown values, with reasons. Never populate invented digests, dates,
run IDs or patient measurements. A template is not an attested release record.

`verified` requires a recorded, scoped check and identifiable inputs; verifying
source does not verify execution. Report distinct gates for checkpoint bytes,
training lineage, validation selection, numerical reproduction, environment,
licenses, and independent acceptance. Overall `publication_allowed` defaults to
false and remains false with any unresolved required gate; no automatically
calculated pass should override the owner's separate publication decision.

## Required field groups

| Group | Required content |
| --- | --- |
| `checkpoint` | File SHA-256, byte size, format, actual epoch, state-dict key/shape/dtype inventory digest; distinct canonical weight digest and canonical checkpoint-context digest with algorithms/version; optimizer presence; safe load/strict compatibility evidence |
| `source` | Repository URL, immutable original training source commit or notebook file hash if no commit exists, local diff hash, immutable reproduction and release-code revisions; unknown original revision stays unknown |
| `model` | MONAI UNet class/version; spatial_dims=3, input/output channels=3/1, channels=[16,32,64,128], strides=[2,2,2], num_res_units=2, all resolved defaults affecting forward behavior, strict state-dict evidence |
| `preprocessing` | Ordered DWI/ADC/FLAIR channels; DWI reference grid; ADC/mask geometry assertions; FLAIR affine resampling order=1, no claimed anatomical registration; nonzero per-modality normalization/nonfinite handling; orientation/spacing policy, source code hashes and source/cache checksums |
| `inference` | ROI=[64,64,32], sw_batch_size=1, overlap=0.25, resolved blend/padding defaults, compute/accumulation devices, sigmoid, inclusive >= threshold, output geometry/export policy and implementation revision |
| `data_and_splits` | Official Zenodo record/version and archive checksum algorithm/value; per-source identity manifest digest; actual ordered 175/37/38 IDs or controlled-access immutable split reference; raw split-file digest and documented canonical ordered-list digest; disjointness/completeness checks; exclude scans from release metadata |
| `training_lineage` | Original run identity if recorded, initial seed policy, diagnostic optimizer update disclosure, loss/optimizer settings and defaults, batch/patch schedule, epochs/resumes and parent checkpoint hashes, execution logs/history references, RNG restoration status, known provenance gaps |
| `validation_selection` | Actual epoch-23 checkpoint hash/weight digest and 37 validation IDs/split digest; original sweep detail/summary hashes, candidate thresholds, positive-only mean Dice objective (36 cases historically), tie rule, chosen threshold=0.25 only if evidenced, chronological fixation evidence, subsequent validation cadence/selection records and checkpoint associations |
| `reproduction` | Separate original and reproduction checkpoint/source/environment identities; original/reproduction CSV and comparison-script/log hashes; all 38 IDs, counts matched, exact Dice comparison policy, both-empty handling, positive-only summary population; reviewer verification status rather than owner report silently promoted |
| `environment` | Distinct historical training, historical selection and reproduction Python/platform/resolved package inventories and hashes, PyTorch/MONAI/CUDA/cuDNN/GPU/device settings; determinism flags; don't substitute current CPU test environment for missing historical GPU inventory |
| `licenses` | Software license text/revision and dependency notices; separately authorized model-weight license/ownership and dataset terms/citation/derivative redistribution assessment; reviewer/owner decisions and outstanding blockers |
| `acceptance` | Independent Agent 3 QA and Agent 4 assessment links, exact reviewed head/checkpoint/manifest identities, performed checks versus omitted checks, owner publication decision; any changed artifact invalidates acceptance pending re-review |
| `distribution` | Approved hosting URI, immutable object version, size/hash/download verification, release notes/model-card identity; null and blocked until publication is explicitly authorized; no weight URL fabricated |

## Historical lineage versus new run

Legacy files may lack run/selection context. Do not generate modern metadata and
present it as historically recorded. A retrospective evidence manifest must
identify which assertions come from file inspection, contemporaneous logs, owner
attestation or new validation. Preserve legacy originals and bytes. Different
serialized bytes with equal state dictionaries are not identical files.

The threshold 0.25 is a release setting only after its independent validation
provenance is addressed. Exact held-out matches must never select that setting
or repair missing validation history. A new validation experiment gets a new
identity and cannot overwrite historical evidence.

For scratch training, future manifest instances must record diagnostics,
optimizer/RNG/loader restoration, source/cache identities, and supported resume
evidence. Do not promise bitwise recreation of the historical training run.

## Acceptance of this proposal

Review these fields in issue #10 before implementing a shared schema. This PR
does not alter Agent 1's existing checkpoint/artifact-context representations.
Independent reviewers should confirm the evidence inventory and missing-artifact
request, verify documentation-only diff and synthetic suite evidence, and keep
R5/R8 pretrained acceptance blocked until the required real artifacts and legal
decisions are reviewed. No clinical/external validation claim is implied.
