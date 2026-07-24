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

Final GeoComb model weights should be distributed with the tagged release
or through a DOI-issuing archive after redistribution rights are confirmed.
`ASSET_MANIFEST.json` records the expected SHA-256 hashes. Model files are
tracked with Git LFS when included in the source repository.

## DeepBee

DeepBee raw images and software are not redistributed by GeoComb. Obtain
them from the original publication and project website:

https://avsthiago.github.io/DeepBee/

GeoComb may provide filenames, evaluation manifests, transformed
coordinates, and corrected annotations only where redistribution is
permitted.

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

Raw third-party image content must not be added to the archive unless its
license explicitly permits redistribution.
