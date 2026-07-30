# Data and model availability

## Source code

The complete corresponding source for the public GeoComb release is hosted
at:

https://github.com/zlddh95-eng/GeoComb

The source release associated with the manuscript should also be archived
in a DOI-issuing repository. The DOI will be added after the first public
archival release.

The public source must include both the Windows application/inference source
and the final training, dataset-construction, evaluation, and platform-
validation scripts used to produce the released models and reported results.
Historical exploratory scripts that did not contribute to the released
workflow need not be included.

## Model weights

The four frozen GeoComb model weights are distributed as assets of the
corresponding versioned GitHub release or DOI archive. Each model release
must identify the exact source tag and include the checksums recorded in
`ASSET_MANIFEST.json`.

The weights are Ultralytics-derived model artifacts distributed through the
project's `AGPL-3.0-only` open-source route. They are not assigned a separate
non-commercial license. See `MODEL_CARD.md` for training provenance,
intended use, limitations, attribution, and third-party-rights boundaries.

A release of model weights does not include or relicense the source images,
image crops, or annotation files used for training.

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
dataset was repackaged in the model release.

## FAIRHiveFrames-1K

FAIRHiveFrames-1K raw images are not redistributed by GeoComb. Obtain the
dataset from its original publication or official repository. GeoComb
manifests identify the images used for the detection-development and final
external benchmark subsets.

## In-house images

Only the portion of the in-house collection approved for public
redistribution will be deposited. Restricted images, if any, must be
described in the manuscript's data-availability statement together with
the reason for restriction and an appropriate contact or access procedure.

## Benchmark annotations and derived outputs

The public research-data archive should contain:

- benchmark image manifests;
- corrected cell-center and class annotations that may be redistributed;
- stage-wise detection outputs;
- per-image evaluation tables;
- bootstrap summaries;
- platform comparison outputs;
- content-hash overlap audit tables; and
- a data dictionary and checksums.

DeepBee-derived cell crops and corrected cell-level annotations are excluded.
Raw third-party image content must not be added to the archive unless its
license explicitly permits redistribution.
