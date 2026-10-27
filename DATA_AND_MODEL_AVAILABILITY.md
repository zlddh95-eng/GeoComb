# Data and model availability

## Source code

GeoComb v1.0.0 source and release files are publicly accessible without a
GitHub account:

- Source: [v1.0.0 tag](https://github.com/zlddh95-eng/GeoComb/tree/v1.0.0).
- Windows packages: [GitHub Release v1.0.0](https://github.com/zlddh95-eng/GeoComb/releases/tag/v1.0.0).

The source code and Ultralytics-derived model artifacts use the project's
`AGPL-3.0-only` license. The GitHub software release is distinct from the
Zenodo data and model package described below. The Zenodo DOI is not a DOI
for the GitHub source release.

The public source includes both the Windows application/inference source and
the final training, dataset-construction, evaluation, and platform-validation
scripts used to produce the released models and reported results. Historical
exploratory scripts that did not contribute to the released workflow are not
included.

## Data and model package

The supporting data and four frozen deployment model weights are publicly
available in *GeoComb: supporting data, model weights, and Windows
applications*, data package v1.0.1:

https://doi.org/10.5281/zenodo.22883902

- [Download the data and model ZIP](https://zenodo.org/records/22883902/files/GeoComb_public_data_and_models_1.0.1.zip).
- [Download its SHA-256 checksum](https://zenodo.org/records/22883902/files/GeoComb_public_data_and_models_1.0.1.zip.sha256).

Data package v1.0.1 accompanies GeoComb software v1.0.0; these are separate
version numbers. Cite the version-specific DOI above and use the citation
metadata on that Zenodo record.

The previously documented DOI, `10.5281/zenodo.21751993`, points to a deleted
record (HTTP 410), not a working download. The old and current records have
different concept DOIs (`10.5281/zenodo.21751992` and
`10.5281/zenodo.22883901`, respectively); they are separate Zenodo version
families, not interchangeable versions of one record.

The archive contains:

- 52 in-house full-frame images with annotations, split into 39 training,
  7 validation, and 6 test images;
- 1,065 corrected ROI annotation records;
- evaluation results for the deduplicated 159-image ROI test set;
- a portable manifest for 1,241,053 classifier crop files;
- the four frozen deployment model weights and a single eight-class baseline;
- supporting analyses, saved predictions, scripts, and figure source data
  for the 29-image benchmark (13 DeepBee, 10 FAIRHive, and 6 in-house images);
- provenance and duplicate-audit records; and
- SHA-256 checksums.

One exact train-test ROI duplicate was excluded from the corrected test
evaluation. ROI source images without individually confirmed redistribution
provenance are represented by filenames, annotations, hashes, and inventory
records rather than redistributed image content.

The Zenodo DOI identifies this versioned data and model package. It is not a
software DOI for the GitHub source release.

## Model weights

The four frozen GeoComb deployment model weights are included in the Zenodo
package above. Their SHA-256 hashes match `ASSET_MANIFEST.json`. For source
use, copy the following files from the archive's `weights/` directory into
`models/`, using the runtime filenames shown below, and verify their hashes:

| Archive filename | Runtime filename in `models/` |
| --- | --- |
| `frame_roi_yolo26m_1024_best.pt` | `frame_roi_yolo26m_1024_best.pt` |
| `stage1_egg3_larvae2_rebalanced_lowres_best.pt` | `stage1_best.pt` |
| `stage2_capped_binary_current_best.pt` | `stage2_best.pt` |
| `stage3_egg_empty_balanced_lowres_best.pt` | `stage3_best.pt` |

The additional `single_8class_stage1_context_best.pt` is a comparison
baseline, not a replacement for any of the four deployment models.

The weights are Ultralytics-derived model artifacts distributed through the
project's `AGPL-3.0-only` open-source route. They are not assigned a separate
non-commercial license. See `MODEL_CARD.md` for training provenance,
intended use, limitations, attribution, and third-party-rights boundaries.

The release of model weights does not include or relicense source images,
image crops, or annotation files that are not explicitly included in the
archive.

## DeepBee

The final hierarchical classifier training pool included cell crops and
corrected labels derived from 67 DeepBee frame images. Those frames were
kept separate from the 13 DeepBee images used in the reported benchmark.

GeoComb does not redistribute DeepBee raw images, extracted crops, or
GeoComb-corrected DeepBee annotations. Obtain DeepBee materials from the
original publication and project website:

https://avsthiago.github.io/DeepBee/

The DeepBee study should be cited as:

Alves, T. S., Pinto, M. A., Ventura, P., Neves, C. J., Biron, D. G.,
Candido Junior, A., De Paula Filho, P. L., and Rodrigues, P. J. (2020).
Automatic detection and classification of honey bee comb cells using deep
learning. *Computers and Electronics in Agriculture*, 170, 105244.
https://doi.org/10.1016/j.compag.2020.105244

The separately published DeepBee segmentation dataset is identified by
Mendeley Data as CC BY-NC 3.0:
https://doi.org/10.17632/db35fj73x5.1

GeoComb's citation of this dataset does not imply that the segmentation
dataset was repackaged in the data and model archive.

## FAIRHiveFrames-1K

FAIRHiveFrames-1K raw images are not redistributed by GeoComb. Obtain the
dataset from its original publication or official repository. GeoComb
manifests identify the images used for the detection-development and final
external benchmark subsets.

## In-house images

The archive contains 52 in-house full-frame images and their corresponding
annotations: 39 training, 7 validation, and 6 test images. The v1.0.1 package
corrects the earlier inventory: image 12 was excluded from classifier
validation, as documented in `inhouse_full_frames/README.txt` and
`provenance/`. The original 53-image inventory is retained as provenance.

## Benchmark annotations and derived outputs

The public archive provides redistributable benchmark annotations and
derived outputs, portable manifests, inventory and provenance records,
duplicate and content-overlap audits, evaluation summaries, and checksums.
Raw DeepBee and FAIRHiveFrames-1K images, DeepBee-derived crops, and corrected
DeepBee cell-level annotations are excluded. Third-party image content is not
relicensed by GeoComb and must be obtained from its original source.
