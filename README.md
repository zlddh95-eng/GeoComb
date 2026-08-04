# GeoComb

GeoComb is an open-source desktop workflow for detecting and classifying
honey bee comb cells from full-frame images. It combines learned comb-region
segmentation and hierarchical cell-content classification with
lattice-aware geometric center detection.

## Main capabilities

- segments the visible comb region inside the wooden frame;
- detects cell centers using image-specific pitch and lattice geometry;
- classifies cells into eight biologically distinct categories;
- supports manual addition, deletion, movement, and reclassification;
- exports editable projects, overlays, cell-level CSV files, and Excel
  summaries;
- provides separate Windows CPU and NVIDIA CUDA desktop packages; and
- records deterministic parameters and integrity hashes for reproducibility.

The native eight classes are:

1. Egg
2. Larvae
3. Capped pupae
4. Pollen
5. Capped honey
6. Uncapped honey
7. Empty cell
8. Unbuilt cell

## Platform release

The public platform release uses the frozen source, model weights,
ROI procedure, and radius-selection rule evaluated in the associated
manuscript. OpenCV processing is restricted to one thread and OpenCL is
disabled to reduce backend-dependent numerical variation. ROI inference is
executed on CPU in both Windows packages so that crop geometry remains
platform invariant.

Manuscript-level validation used 29 images with three measured runs in each
of four Windows/Linux CPU/CUDA configurations. Every run produced identical
cell counts and center coordinates. CPU-CUDA class-label agreement was
99.9924%.

## Windows packages

Prebuilt Windows CPU and NVIDIA CUDA executable packages, frozen model
weights, supporting data, provenance records, and checksums are available
from Zenodo:

https://doi.org/10.5281/zenodo.21751993

```text
GeoComb_Windows_CPU/GeoComb_CPU/GeoComb_CPU.exe
GeoComb_Windows_GPU/GeoComb_GPU/GeoComb_GPU.exe
```

Use the CPU package on a supported Windows x64 computer. Use the GPU package
only with a compatible NVIDIA GPU and driver.

See [USER_GUIDE.md](USER_GUIDE.md) for operation and
[BUILD_AND_VALIDATION.md](BUILD_AND_VALIDATION.md) for source installation,
building, and validation.

## Source workflow

Use 64-bit Python 3.11. The exact package versions for the validated
environments are documented in
[BUILD_AND_VALIDATION.md](BUILD_AND_VALIDATION.md). The Windows helper
scripts install separate CPU and CUDA environments.

```bat
install_cpu.bat
.venv_cpu\Scripts\python.exe check_environment.py
build_cpu.bat
```

or

```bat
install_gpu.bat
.venv_gpu\Scripts\python.exe check_gpu_environment_v1_9.py
build_gpu.bat
```

The optional `validate_cpu.bat` and `validate_gpu.bat` maintainer checks
require an authorized local `regression_test` asset directory. Those
image-based regression assets are not redistributed in this source
repository.

## Data and model availability

The supporting data and four frozen model weights are archived in the
[GeoComb public data and model package v1.0.0](https://doi.org/10.5281/zenodo.21751993).
The archive contains 53 in-house full-frame images with annotations
(39 training, 8 validation, and 6 test), 1,065 corrected ROI annotation
records, results for the deduplicated 159-image ROI test set, a portable
manifest for 1,241,053 classifier crops, model provenance, and SHA-256
checksums.

Verify the weights against [ASSET_MANIFEST.json](ASSET_MANIFEST.json) and
review their provenance, intended use, limitations, and licensing in
[MODEL_CARD.md](MODEL_CARD.md).

DeepBee and FAIRHiveFrames-1K raw images are not redistributed in this
repository or in the Zenodo archive. Obtain them from their original sources.
GeoComb also does not redistribute DeepBee-derived crops or corrected
cell-level annotations. See
[DATA_AND_MODEL_AVAILABILITY.md](DATA_AND_MODEL_AVAILABILITY.md).

## Citation

If you use GeoComb, cite the associated article, the GeoComb v1.0.0 source
release, and the
[GeoComb v1.0.0 data and model archive](https://doi.org/10.5281/zenodo.21751993),
as applicable. Citation metadata are provided in
[CITATION.cff](CITATION.cff). The Zenodo DOI identifies the data and model
package, not a software archive. The article DOI will be added after
assignment.

## License

GeoComb source and the distributed Ultralytics-derived model artifacts use
the GNU Affero General Public License, version 3 (`AGPL-3.0-only`). See
[LICENSE](LICENSE). No separate non-commercial restriction is added to the
model weights; third-party datasets retain their own terms and are not
relicensed by GeoComb.

Third-party components remain subject to their own licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Research-use statement

GeoComb provides image-analysis estimates and editable research records. It
is not a diagnostic system and does not replace expert assessment of colony
health or management decisions.
