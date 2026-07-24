# Training source

## Frame ROI model

`roi/train_roi_segmentation.py` reproduces the settings in Supplementary
Table S1. Supply a YOLO segmentation dataset through `--data`. The public
example YAML deliberately contains no workstation-specific path.

## Hierarchical classifiers

The final classifier cascade contains:

1. Stage 1: seven intermediate classes, 320-pixel input, crop half-step 0.80.
2. Stage 2: capped honey versus capped pupae, 320-pixel input, crop half-step
   1.80.
3. Stage 3: egg versus empty cell, 384-pixel input, crop half-step 0.45.

`classifiers/make_manual_crops_for_all_stages.py` and
`classifiers/build_three_stage_datasets_plus_manual.py` preserve the code used
to add the 67 corrected DeepBee training images. The local project root must be
provided with `--base`.

`classifiers/build_egg_larvae_rebalanced_lowres_datasets.py` performs the final
offline class rebalancing and low-resolution augmentation for Stages 1 and 3.
The reported build used seed 42, Stage-1 egg and larvae multipliers of 4 and 2,
and a Stage-3 empty-to-effective-egg ratio of 2.0.

`classifiers/train_final_classifiers.sh` contains the final cross-domain
optimization settings. It requires explicit checkpoint and dataset paths
through environment variables.

Raw third-party images and derived image crops are not redistributed here.
Consult `reproducibility_data/THIRD_PARTY_DATA_AND_LICENSES.md`.
