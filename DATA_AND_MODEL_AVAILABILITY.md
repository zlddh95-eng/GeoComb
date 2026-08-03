# Data and model availability

## Source code

GeoComb v1.0.0 source and release files are hosted at:

https://github.com/zlddh95-eng/GeoComb

The source code and Ultralytics-derived model artifacts are available through
the project's `AGPL-3.0-only` open-source route. The GitHub software release
is distinct from the Zenodo data and model package described below.

The public source includes both the Windows application/inference source and
the final training, dataset-construction, evaluation, and platform-validation
scripts used to produce the released models and reported results. Historical
exploratory scripts that did not contribute to the released workflow are not
included.

## Data and model package

The supporting data and four frozen model weights are archived as the
GeoComb public data and model package v1.0.0:

https://doi.org/10.5281/zenodo.21751993

The archive contains:

- 53 in-house full-frame images with annotations, split into 39 training,
  8 validation, and 6 test images;
- 1,065 corrected ROI annotation records;
- evaluation results for the deduplicated 159-image ROI test set;
- a portable manifest for 1,241,053 classifier crops;
- the four final frozen model weights;
- provenance and duplicate-audit records; and
- SHA-256 checksums.

One exact train-test ROI duplicate was excluded from the corrected test
evaluation. ROI source images without individually confirmed redistribution
provenance are represented by filenames, annotations, hashes, and inventory
records rather than redistributed image content.

The Zenodo DOI identifies this versioned data and model package. It is not a
software DOI for the GitHub source release.

## Model weights

The four frozen GeoComb model weights are included in the Zenodo package
above. Their checksums and release identities are recorded in
`ASSET_MANIFEST.json` and in the archive's checksum files.

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

The archive contains the 53 in-house full-frame images approved for public
redistribution and their corresponding annotations: 39 training, 8
validation, and 6 test images.

## Benchmark annotations and derived outputs

The public archive provides redistributable benchmark annotations and
derived outputs, portable manifests, inventory and provenance records,
duplicate and content-overlap audits, evaluation summaries, and checksums.
Raw DeepBee and FAIRHiveFrames-1K images, DeepBee-derived crops, and corrected
DeepBee cell-level annotations are excluded. Third-party image content is not
relicensed by GeoComb and must be obtained from its original source.
