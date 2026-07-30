# GeoComb model card

## Release identity

This model card describes the four frozen model files used by GeoComb 1.0.0.
Their filenames, sizes, and SHA-256 checksums are recorded in
[`ASSET_MANIFEST.json`](ASSET_MANIFEST.json). A model file should not be
treated as the manuscript-aligned release if its checksum differs from that
manifest.

| File | Role | Frozen inference setting |
| --- | --- | --- |
| `frame_roi_yolo26m_1024_best.pt` | Segments the cell-bearing comb region inside the wooden frame | ROI inference at 1024 pixels on CPU |
| `stage1_best.pt` | Main cell-content classifier | 320-pixel input; first hierarchical classification stage |
| `stage2_best.pt` | Distinguishes capped pupae from capped honey | 320-pixel input; enlarged crop with half-step 1.80 |
| `stage3_best.pt` | Distinguishes eggs from empty cells | 384-pixel input; focused crop with half-step 0.45 |

The complete workflow also uses deterministic image processing and
lattice-aware geometric detection. The model weights alone do not reproduce
GeoComb results.

## Training provenance

The frame-ROI model was trained to retain the cell-bearing comb area while
excluding the wooden frame and visible regions without cells.

The three-stage classifier was developed from manually reviewed honey-comb
cell examples. The final training pool included cell crops and corrected
labels derived from 67 publicly available DeepBee frame images. Those 67
training frames were kept separate from the 13 DeepBee images used for the
reported benchmark. The original DeepBee images, extracted crops, and
GeoComb-corrected DeepBee annotations are not redistributed with GeoComb.

The final classifier stages used the following targeted balancing and
augmentation decisions:

- Stage 1: eggs were sampled threefold and larvae twofold, with low-resolution
  JPEG augmentation.
- Stage 2: capped-pupae versus capped-honey binary refinement used an enlarged
  crop.
- Stage 3: eggs were sampled fourfold and empty cells twofold, with
  low-resolution augmentation applied to half of the augmented examples.

FAIRHiveFrames-1K images were used as external reference and evaluation
material. Raw FAIRHiveFrames-1K images are not included in the GeoComb
repository or release assets.

## Distributed artifacts

Public GeoComb releases distribute the four frozen weight files as versioned
release assets or through the corresponding DOI archive. Release assets must
be matched to the source tag and verified against `ASSET_MANIFEST.json`.

The weight files contain trained parameters. They do not contain a copy of
the DeepBee or FAIRHiveFrames-1K raw images, extracted image crops, or
cell-level annotation files. Users must obtain third-party datasets from
their original sources and follow the terms published by those sources.

## License and third-party rights

GeoComb uses Ultralytics software and model architectures through the
AGPL-3.0 open-source route. The GeoComb source and the distributed
Ultralytics-derived model artifacts are therefore provided under
`AGPL-3.0-only`, together with the corresponding source and training,
configuration, inference, and validation materials that the project is
authorized to distribute.

No separate non-commercial license is imposed on the model weights because
an additional restriction would conflict with the selected AGPL distribution
route. This does not relicense any training image and does not grant rights to
DeepBee, FAIRHiveFrames-1K, or in-house source images.

The DeepBee segmentation dataset published at Mendeley Data is separately
identified by its publisher as CC BY-NC 3.0. The DeepBee classification
resources are cited through the original project and publication. GeoComb
does not redistribute either resource's raw image content.

Users who combine GeoComb with separately obtained data remain responsible
for complying with the terms of those data. Users seeking to deploy an
Ultralytics-based system without satisfying AGPL requirements should consult
Ultralytics about the appropriate commercial license.

## Intended use

GeoComb is intended for research analysis of full-frame honey bee comb
images, including editable estimates of cell locations and eight
cell-content categories:

1. Egg
2. Larvae
3. Capped pupae
4. Pollen
5. Capped honey
6. Uncapped honey
7. Empty cell
8. Unbuilt cell

Outputs should be reviewed by a knowledgeable user, particularly for new
camera systems, frame types, illumination conditions, geographic regions, or
cell appearances that differ from the reported validation data.

## Out-of-scope use

GeoComb is not a colony-health diagnostic system and should not be used as
the sole basis for treatment, management, regulatory, or commercial grading
decisions. The models were not validated for brood-disease diagnosis, pest
diagnosis, individual-bee detection, or video analysis.

## Known limitations

- Validation used a limited number of completely annotated full frames.
- Rare classes remain more difficult than common capped-brood, honey-storage,
  and empty-comb categories.
- Domain shift can reduce ROI, center-detection, and classification accuracy.
- Some reference annotations were initialized from model predictions before
  complete manual correction, so residual annotation anchoring cannot be
  excluded.
- CPU and CUDA execution were stabilized and compared on the reported
  platform set, but performance has not been established on every supported
  hardware combination.
- The public materials do not provide a per-example reconstruction of
  non-redistributable DeepBee-derived training crops.

See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for application-level
limitations and [`DATA_AND_MODEL_AVAILABILITY.md`](DATA_AND_MODEL_AVAILABILITY.md)
for the release-data policy.

## Required attribution

Please cite the GeoComb article and archived software release when their DOIs
are assigned. Use [`CITATION.cff`](CITATION.cff) for the current project
metadata.

The DeepBee resource used during classifier development should also be cited:

> Alves, T. S., Pinto, M. A., Ventura, P., Neves, C. J., Biron, D. G.,
> Candido Junior, A., De Paula Filho, P. L., & Rodrigues, P. J. (2020).
> Automatic detection and classification of honey bee comb cells using deep
> learning. *Computers and Electronics in Agriculture*, 170, 105244.
> https://doi.org/10.1016/j.compag.2020.105244

DeepBee project: https://avsthiago.github.io/DeepBee/

DeepBee segmentation dataset: https://doi.org/10.17632/db35fj73x5.1

## Contact and versioning

Report model-card corrections through the GeoComb repository. Material
changes to training provenance, model files, hashes, intended use, or
licensing require a new versioned release and an updated model card.
