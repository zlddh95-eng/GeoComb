# GeoComb reproducibility data for the platform-stable workflow

This repository package contains the tabular data used to report the GeoComb detection, classification, counting, bootstrap, dataset-audit, and platform-stability results. It is intended to accompany the manuscript submitted to *Computers and Electronics in Agriculture*.

## Contents

- `final_benchmarks/`: final results for the 13-image DeepBee official benchmark, 10-image FAIRHive benchmark, and six-image in-house benchmark, plus image-cluster bootstrap estimates.
- `table2_component_analysis/`: Table 2 stage summary, final per-image metrics, paired bootstrap contrasts, and 101,968 corrected FAIRHive reference centers used for detection-only analysis.
- `platform_stability/`: sanitized runtime ledger and repeatability/equivalence comparisons for Windows CPU, Windows CUDA, Linux/WSL2 CPU, and Linux/WSL2 CUDA.
- `dataset_audit/`: final content-deduplicated image inventory, per-benchmark nearest-ROI hash and thumbnail-similarity evidence, ROI held-out-test metrics, and split counts.
- `figure_data/`: source tables for Figures 6–8. Original publication figures are maintained as separate author-supplied source files.

## Raw images

Third-party raw images are not redistributed in this package. Obtain them from their original records and use the filenames and hashes supplied here to identify the evaluated files.

- FAIRHiveFrames-1K: https://doi.org/10.5281/zenodo.19241078
- DeepBee semantic-segmentation data: https://doi.org/10.17632/db35fj73x5.1
- DeepBee project: https://avsthiago.github.io/DeepBee/

The in-house raw images are not included in this draft pending explicit confirmation that they may be released publicly.

## Notes

- Cell matching used one-to-one matching at 0.5 times the image-specific reference pitch.
- Bootstrap intervals used 100,000 image-level resamples with seed 20260721.
- The Table 2 reference contains detection centers only; cell-class labels from that development set were not evaluated.
- The final benchmark audit compared 29 benchmark images with 1,065 ROI images. Exact SHA-256 equality was treated as an exact duplicate; perceptual-hash distance and normalized 128 × 128 grayscale thumbnail correlation were used to screen transformed or re-encoded candidates. The public audit table records the nearest ROI filename, transformation, scores, thresholds, and final decision for every benchmark image.
- Absolute local computer paths were removed from the public copy.

See `THIRD_PARTY_DATA_AND_LICENSES.md` before public release. Replace the placeholder author and repository fields in `REPOSITORY_METADATA_TEMPLATE.md` after the public GitHub and DOI records are created.
