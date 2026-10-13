# Ownership and licensing boundaries

## Original repository software

The repository owner explicitly selected MIT for original code in issue #15 and
requested a LICENSE in the portfolio PR. [LICENSE](../LICENSE) contains the standard
MIT text, with copyright **2026 Shreyas Bharadwaj**. GitHub contribution history
through baseline `c39a2ec9b525ad52b9c9a602b8caae554017c584` attributes original commits
to `sbmedtech` or Shreyas Bharadwaj under that same account. Current package sources
import dependencies rather than vendoring their implementations; inspection found
no third-party copyright/license headers or bundled third-party source requiring a
conflicting grant. This is a repository-level ownership check, not a claim that a
Git author record alone proves ownership of every future contribution.

The owner-authorized MIT grant applies to original project source and original
software documentation. Preserve any third-party notices in later contributions;
do not assume all future assets are original code. No author credentials, employment
or institutional endorsement are inferred from contribution history.

## ISLES dataset and derived MRI figure

The official [Zenodo record 7960856](https://zenodo.org/records/7960856) API,
checked on **2026-10-13**, declares `metadata.license.id = cc-by-4.0`.
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
applies separately to the underlying dataset. Consult the current record and archive
LICENSE before further use or redistribution. The repository does not distribute
raw scans or the dataset archive.

Credit **Hernandez Petzsche et al., ISLES 2022**, retain the source/license links,
and describe transformations when reusing the historical derived figure. It shows
normalized DWI with expert/predicted mask overlays; the PNG was extracted unchanged
from the existing notebook output. It is not newly acquired or regenerated MRI.
[Figure provenance, transformation details and case captions](portfolio/figures-and-results.md)
record its source, hash and illustrative selection. Dataset de-identification and
visible dataset case IDs do not imply clinical validation or creator endorsement.
MIT for source code does not remove dataset/derived-image attribution obligations.

The current dataset record requests both:

- Hernandez Petzsche et al., *ISLES 2022: A multi-center magnetic resonance imaging
  stroke lesion segmentation dataset*, Scientific Data 9, 762 (2022).
  [DOI](https://doi.org/10.1038/s41597-022-01875-5).
- de la Rosa et al., *DeepISLES: a clinically validated ischemic stroke segmentation
  model from the ISLES'22 challenge*, Nature Communications 16, 7357 (2025).
  This is a separate model; its validation does not transfer to this project's U-Net.

## Dependencies and model weights

Python, PyTorch, MONAI, NiBabel and other imported libraries retain their respective
upstream licenses. They are not relicensed by this repository's MIT file. Installing
from official package sources does not transfer their ownership to this author;
retain the corresponding upstream notices when redistributing third-party software.
No library binaries or vendored dependency source are added by this documentation PR.

No pretrained checkpoint binaries are added by this PR. Do not infer a weight license
from the code LICENSE or dataset terms: model-weight ownership, permissible use,
provenance and explicit owner publication decisions require separate review. An
existing release asset is evidence of available bytes, not proof of a license or
validation-threshold lineage. The evidence investigation remains in
[PR #18](https://github.com/sbmedtech/isles2022-stroke-mri-segmentation/pull/18).

Future additions should record source/author, license or permission, attribution,
transformations, and whether any data/weight terms differ from MIT before publication.
