# Data dictionary

## Common detection fields

- `reference_cells`, `n_ref`, `reference_count`: corrected reference-center count.
- `predictions`, `n_pred`, `prediction_count`: predicted-center count.
- `tp`, `fp`, `fn`: one-to-one matched true positives, unmatched predictions, and unmatched reference centers.
- `precision`, `recall`, `f1`: pooled detection metrics unless a filename states that the values are per-image.
- `relative_count_error`: absolute predicted-minus-reference count divided by the reference count.
- `mean_image_relative_count_error`: arithmetic mean of image-level relative count error.
- `*_ci95_low`, `*_ci95_high`: percentile limits from image-cluster bootstrap resampling.

## Classification fields

- `ref_class7`, `pred_class7`: reference and predicted class after mapping to the seven classes common to GeoComb and DeepBee.
- `correct_class`: whether the predicted class equals the reference class for a matched cell.
- Eight-class GeoComb files preserve empty and unbuilt cells as separate categories.

## Platform-stability fields

- `symmetric_match_fraction`: matched predictions divided by the larger of the two compared prediction counts.
- `class_agreement_among_matches`: class-label agreement among spatially matched predictions.
- `scientifically_equivalent`: whether the prespecified count, spatial-match, and class-agreement criteria were met.

## Final benchmark image-audit fields

- `sha256`, `nearest_roi_sha256`: cryptographic hashes of the benchmark image and its nearest ROI image.
- `phash_0deg`: 64-bit perceptual hash of the untransformed benchmark image.
- `nearest_roi_split`, `nearest_roi_filename`: ROI partition and filename of the nearest perceptual-hash match.
- `exact_sha256_match`: whether the benchmark and nearest ROI image have identical SHA-256 hashes.
- `minimum_phash_distance`: minimum Hamming distance between the benchmark perceptual hashes across the evaluated rotations/reflections and the untransformed ROI perceptual hash.
- `nearest_transform`: transformation applied to the benchmark image at the minimum perceptual-hash distance.
- `thumbnail_similarity_score`: normalized grayscale correlation between 128 × 128 thumbnails for the nearest pair after applying `nearest_transform`.
- `phash_threshold`, `thumbnail_threshold`: review thresholds recorded for the audit decision.
- `audit_decision`: final pair-level screening outcome. `NO_OVERLAP_CANDIDATE` means the nearest pair did not satisfy the exact, probable-match, or review-candidate rules.

## Table 2 reference centers

- `image_id`: FAIRHive source-image stem.
- `x`, `y`: corrected center in the common 5,000-pixel reference coordinate system.
- `radius`: retained reference radius, where available.
- `reference_id`: annotation identifier.
