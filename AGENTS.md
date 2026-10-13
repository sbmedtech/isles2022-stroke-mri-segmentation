# Agent coordination for the ISLES 2022 research release

## Scope and shared source of truth

GitHub issues and pull requests are the authoritative coordination record for the four existing Codex agents. Chat may provide context, but decisions, evidence, blockers, interface contracts, assignments and approvals must be recorded in GitHub.

Release goals:
1. Download the approved epoch-43 pretrained model and infer DWI/ADC/FLAIR lesion masks without expert annotations.
2. Download ISLES 2022 from official Zenodo, train from scratch, select on validation, and evaluate a frozen model on held-out patients.

Planning baseline: GitHub main `2cb760755898b1ce2dd716d0e2b741d9e1cff0eb`, including merged PR #8. Verify current remote main and dependency merge SHAs; do not assume local branches or cached origin/main are current.

This coordination setup does not start release implementation or authorize pretrained publication. First independently review the coordination PR. Preserve the existing scientific pipeline during setup.

## Responsibilities

| Existing agent | Responsibilities | Release issues |
| --- | --- | --- |
| Agent 1: Software Engineer | Package interfaces, artifact serialization, shared preprocessing/export, inference CLI, checkpoint I/O and release distribution. Own package entry points and coordinate shared metadata changes. | R1, R3, R5, R8 |
| Agent 2: ML Research Scientist | Checkpoint and threshold provenance, scientific contracts, split/training/evaluation behavior, reproducible resume, experiment evidence and model-release requirements. | R2, R6 |
| Agent 3: Documentation + QA Engineer | Independently authored integration tests, environments, CI, executed walkthroughs, documentation, licensing checklist and model card. | R4, R7 |
| Agent 4: Manager | Dependency scheduling, issue ownership, file/interface conflict resolution, independent PR review, combined integration audit and final evidence assessment. Does not approve its own authored changes. | All issues: coordination and final review |

All agents use the shared GitHub account when necessary. The agent owner named in each issue, not the shared account assignee alone, identifies the accountable implementer. Do not invent separate agent GitHub usernames.

## Release issue registry

| Task | Issue | Owner | Branch | Dependencies |
| --- | --- | --- | --- | --- |
| R1 | [#9](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/9) | Agent 1: Software Engineer | `release/artifact-context-roundtrip` | None beyond coordination review |
| R2 | [#10](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/10) | Agent 2: ML Research Scientist | `release/epoch43-provenance` | None beyond coordination review |
| R3 | [#11](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/11) | Agent 1: Software Engineer | `release/shared-preprocessing` | [#9](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/9) |
| R4 | [#12](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/12) | Agent 3: Documentation + QA Engineer | `release/environment-ci` | [#9](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/9), [#11](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/11) |
| R5 | [#13](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/13) | Agent 1: Software Engineer | `release/pretrained-inference` | [#10](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/10), [#11](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/11), [#12](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/12) |
| R6 | [#14](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/14) | Agent 2: ML Research Scientist | `release/train-evaluate` | [#9](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/9), [#11](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/11), [#12](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/12) |
| R7 | [#15](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/15) | Agent 3: Documentation + QA Engineer | `release/research-documentation` | [#10](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/10), [#12](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/12), [#13](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/13), [#14](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/14) |
| R8 | [#16](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/16) | Agent 1: Software Engineer | `release/distribution` | [#9](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/9), [#10](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/10), [#11](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/11), [#12](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/12), [#13](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/13), [#14](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/14), [#15](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/issues/15) |

R1 and R2 are the first assignments after coordination review. R4 can prepare tests/procedures while its dependencies are under review, but final QA must use the approved integrated baseline. R5 may develop with synthetic weights only when its contract/dependency prerequisites are satisfied; real acceptance and weight publication remain blocked by R2. Record any dependency exception in the issue before proceeding.

## GitHub issue and PR communication

Before changing files, the owner posts a claim comment containing:
- Agent identity, issue number, baseline SHA, branch and isolated worktree/check-out.
- Dependency status, planned file ownership, implementation outline and acceptance commands.
- Known missing inputs and requested interface decisions.

Prefix comments with the agent role (for example, `[Agent 2: ML Research Scientist]`). Record progress when a milestone, blocker or scope/interface change occurs. Include immutable commit SHAs and evidence links; do not claim a check ran unless it actually ran.

Use issues for scope, dependency and interface decisions. Use linked PRs for review of concrete changes. Each PR identifies its owning issue, dependency PRs, base/head SHA, changed behavior, preservation checks, test counts/failures, environment, risks, and checks not performed.

Do not rely on another chat's status as acceptance. Resolve conflicting instructions or overlapping files through an issue comment and Agent 4's recorded decision before editing.

## Branch and filesystem isolation

Use one branch and isolated worktree/check-out per issue/PR. Never switch, reset, clean, force-push or overwrite another agent's working tree/branch. Never commit directly to main.

Create branches from the latest independently approved dependency baseline. Stacked PRs must declare their parent PR and be independently re-reviewed after rebasing/retargeting. Avoid broad merges/cherry-picks that import unrelated work.

Agent 1 owns shared package interfaces and entry points. Agent 2 owns training/evaluation modules and experiment specifications. Agent 3 owns CI, environment documentation and independent integration tests. Shared schemas and pyproject changes require a recorded contract/file ownership agreement. Coordinate notebook edits explicitly; extraction should preserve the notebook as the reference.

Read applicable AGENTS.md instructions in the actual checkout before implementation. A branch based on main before this file merges must consult the reviewed coordination PR until the file is integrated.

## Scientific safeguards

Preserve the MONAI 3D U-Net architecture/defaults and state-dict compatibility, DWI/ADC/FLAIR channel order, existing normalization and geometry checks, affine-based FLAIR resampling, patch sampling, loss/optimizer, patient partition, and validation cadence. Any scientific change requires separate scope and evidence, never silent refactoring.

Preserve historical notebook outputs/metadata and published historical records. Do not regenerate, relabel, overwrite or replace historical results with a new experiment. A narrowly scoped artifact bug fix may change helper behavior only while preserving scientific semantics and historical outputs.

Keep the 175/37/38 patient split disjoint before patch sampling. Training uses training patients; checkpoint/threshold selection uses validation patients; held-out test labels never drive training, thresholds, model selection or future tuning. Disclose previous inspection of test cases when designing further evaluation.

The owner's reported exact 38-patient replication (positive mean Dice 0.5377, median 0.5818) is user-reported evidence pending artifact review. It does not independently establish validation provenance for threshold 0.25. Never fabricate legacy lineage, infer provenance from filenames, or tune the threshold on those test cases.

Distinguish source checks, synthetic tests, real MRI execution, historical checkpoint compatibility, numerical reproduction, and clinical/external validity. Missing data or evidence is a blocked/unverified gate, not a passed test.

Do not commit the dataset, patient scans, pretrained binary weights, credentials or local experiment caches. Obtain dataset files from official Zenodo, verify checksums, keep them outside Git, and respect dataset terms separately from software/model licensing.

Do not publish pretrained weights until checkpoint identity/lineage, threshold provenance, preprocessing/model contract, licensing, hosting, checksums and independent acceptance are addressed. R8 authorizes release preparation only; publication requires a separate explicit repository-owner decision.

No new registration, voxel-spacing changes, augmentation, architecture change or inference fallback may be introduced without explicit scientific review. Affine resampling must not be described as anatomical registration or clinical validation.

New training commands must isolate diagnostics, restore RNG/optimizer state for supported deterministic resume, and safely recover checkpoint publication. Document differences from historical notebook execution rather than promise identical scratch training.

## Independent review and definition of done

No agent approves its own work. Agent 3 independently tests Agents 1 and 2's PRs; Agent 4 independently reviews their evidence and integration. Agents 1/2 review Agent 3's technical/scientific claims, with independent final review by Agent 4. Changes authored by Agent 4, including this coordination PR, require review by Agent 3 and scientific-policy review by Agent 2.

Approval records must name the reviewer role and exact PR head SHA, verdict PASS/FAIL, commands/results, limitations and blockers. Because agents may share one GitHub identity, native approvals alone do not establish agent independence; record the role/evidence explicitly. This document is a process rule, not a claim that branch protection is configured.

Any changed head, resolved conflict or dependency integration invalidates prior acceptance until rechecked. Run the full available suite plus meaningful acceptance tests for the issue. Independently test installed wheels/commands outside the checkout and realistic provenance, not just simplified fixtures. Require real-data/Colab evidence where the issue's acceptance criteria demand it.

Agent 4 verifies combined integration against current GitHub main. Do not merge implementation code without independent PASS and no unresolved blockers. No self/automatic merges; the repository owner controls merging after QA. No external publication during coordination setup.

Close an issue only after an approved merge and linked acceptance evidence. A missing-provenance report may complete an investigation PR but leaves dependent release gates blocked; do not mark a model release verified.

## First assignments

- Agent 1: claim R1 (#9) on `release/artifact-context-roundtrip`. Reproduce the real AdamW tuple/list manifest mismatch, canonicalize representation without weakening checks, add realistic full-context regressions, and open a separate fix PR. Do not begin the inference refactor yet.
- Agent 2: claim R2 (#10) on `release/epoch43-provenance`. Inventory/review checkpoint, ordered split, 38-case reproduction records, environment and threshold-selection evidence; publish findings and proposed manifest contract in GitHub. Do not alter selection/training logic, tune on test patients, upload weights or claim missing evidence verified.
- Agent 3: independently review this coordination-only PR and prepare an R4 QA matrix; no implementation edits during setup.
- Agent 4: verify this setup contains only coordination changes, track dependencies and review implementation PRs after setup acceptance.
