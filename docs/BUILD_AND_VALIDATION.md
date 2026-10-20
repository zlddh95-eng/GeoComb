# Windows build and validation

## Supported build targets

- Windows 10/11 x64, CPU
- Windows 10/11 x64, NVIDIA CUDA GPU

Use 64-bit Python 3.11.

The manuscript-aligned Windows environments used Python 3.11.9. The CPU
environment used PyTorch 2.13.0+cpu, and the CUDA environment used PyTorch
2.11.0+cu128. Both used Ultralytics 8.4.96, OpenCV 4.13.0, and NumPy 2.4.4.
Exact dependency snapshots from the release environments accompany the
tagged source release in the `licenses` directory.

## Source checkout and shared PyInstaller configuration

Use a revision containing the tracked `build/GeoComb.spec`. The original
`v1.0.0` tag omitted this file; use this repair commit or apply the repair
to a checkout of that tag. Installing Python packages does not generate it.
The tag and previously published binaries are not changed by this repair.

Retrieve the actual model files before building (a source ZIP may contain
only Git LFS pointer files):

```bat
git lfs install
git lfs pull
```

Alternatively, obtain the unchanged weights from the archive documented in
`DATA_AND_MODEL_AVAILABILITY.md` and place them in `models/`. Verify their
hashes against `ASSET_MANIFEST.json`. The spec rejects missing files, empty
weights, and unhydrated LFS pointers before dependency analysis; this check
does not replace SHA-256 verification.

Both batch files use the same spec, with the following build-time selection:

| Batch file | Python environment | GEOCOMB_VARIANT | Entry point | Output folder/executable |
| --- | --- | --- | --- | --- |
| `build_cpu.bat` | `.venv_cpu` | `CPU` | `geocomb_entry.py` | `GeoComb_CPU/GeoComb_CPU.exe` |
| `build_gpu.bat` | `.venv_gpu` | `GPU` | `geocomb_gpu_entry.py` | `GeoComb_GPU/GeoComb_GPU.exe` |

Each output is under `build_output/<cpu-or-gpu>/dist/`. The spec resolves
inputs relative to its own location and includes the four frozen weights,
seven `runpy` pipeline scripts, and `config/defaults.json` under `_internal/`,
matching `geocomb.resources`. The scripts are also analyzed as hidden imports.
Ultralytics package data, source files, metadata, and submodules are collected;
standard PyInstaller hooks collect the installed Torch/Torchvision and Qt
dependencies. CPU/CUDA libraries come from the selected virtual environment.
The executable keeps its console for CLI output and pipeline logging.
Local `regression_test` assets are never automatically bundled.

## Environment validation and building

CPU:

```bat
install_cpu.bat
.venv_cpu\Scripts\python.exe check_environment.py
build_cpu.bat
```

GPU:

```bat
install_gpu.bat
.venv_gpu\Scripts\python.exe check_gpu_environment_v1_9.py
build_gpu.bat
```

These commands validate the Python environment and build the application.
They do not reproduce the manuscript benchmark.

## Verification of the packaging repair

Static contract checks require Python and Git, without ML or GUI packages:

```bat
python -m unittest discover -s tests -v
```

These checks cover both variant mappings, runtime resource destinations,
dynamic-script analysis, paths containing spaces, missing/LFS-pointer assets,
and the `.gitignore` exception. They mock PyInstaller targets and package
collection. Passing them does **not** demonstrate a successful Windows build,
DLL collection, model loading, GUI startup, or CPU/CUDA inference.

Before distributing newly built executables, complete the following on Windows:

- Run both environment installers/checks and `build_cpu.bat` / `build_gpu.bat`.
- Extract each complete ZIP to a location separate from the source checkout.
  Launch its executable and check Qt startup, image display, and GPU detection.
- Run `GeoComb_CPU.exe analyze --help` and `GeoComb_GPU.exe analyze --help`.
  Analyze authorized images with `analyze --input-folder INPUT --project-folder
  OUTPUT --device cpu` or `--device cuda`, using distinct new output folders.
  Confirm that all four models load and all seven pipeline stages execute.
- Open the resulting project, edit cells, and regenerate Excel/CSV/overlays
  in the GUI and with `export --project-folder OUTPUT`.
- Run the optional source regression check with authorized local assets.
  To test frozen `validate`, temporarily place those assets under the extracted
  executable's `_internal/regression_test/`, run `validate --device cpu` or
  `validate --device cuda`, and remove them before redistribution.

## Optional maintainer regression check

`validate_cpu.bat` and `validate_gpu.bat` run a controlled single-image
regression check when an authorized local `regression_test` directory is
available. The image-based regression assets are not redistributed in this
source repository. Without those assets, the scripts stop with an explicit
message rather than reporting a validation result.

When the authorized assets are supplied, the check compares the resulting
cell coordinates and classes with the frozen expected output. Its acceptance
thresholds are:

- total cell-count error <= 0.5%;
- at least 99.5% of expected centers matched within 1.5 pixels; and
- at least 99.5% class agreement among matched centers.

This single-image check is a release-maintainer smoke test, not the
manuscript-level platform benchmark.

## Manuscript-level platform validation

The final platform evaluation processed the same 29-image benchmark three
times in each of four configurations: Windows CPU, Windows CUDA, Linux CPU,
and Linux CUDA. All measured runs produced identical total and per-image
cell counts and identical center coordinates. CPU-CUDA class-label agreement
was 99.9924%.

## Deterministic processing controls

- frame-ROI inference is executed on CPU in every deployment configuration;
- ROI crop bounds are quantized to a 32-pixel grid;
- Hough radius is selected from the median of the near-optimal plateau;
- OpenCV is restricted to one processing thread; and
- OpenCL is disabled.

## Executable build outputs

Expected archives:

```text
release/GeoComb_Windows_CPU.zip
release/GeoComb_Windows_GPU.zip
```

Expected executables after archive extraction:

```text
release/GeoComb_Windows_CPU/GeoComb_CPU/GeoComb_CPU.exe
release/GeoComb_Windows_GPU/GeoComb_GPU/GeoComb_GPU.exe
```

The build uses PyInstaller one-folder format rather than one-file format
because PyTorch, CUDA, Qt, and Ultralytics include large binary dependencies.
Distribute the complete generated folder, not the executable alone.

## GPU notes

The GPU package is built with the CUDA-enabled PyTorch wheel. The end user's
NVIDIA driver must be compatible with the packaged PyTorch CUDA runtime.
The application checks CUDA availability when analysis begins. CPU-only
users should use the CPU package.

## Release checklist

- source tree corresponds to the intended release commit;
- environment checks pass in the CPU and GPU build environments;
- the authorized local regression check passes before a binary release;
- model and frozen-source SHA-256 hashes match `ASSET_MANIFEST.json`;
- the 29-image evidence package remains traceable to the frozen release;
- a fresh Windows computer can open the complete one-folder package;
- project creation, editing, and Excel regeneration work;
- all mandatory public-repository documents are present;
- no credentials, private paths, or non-redistributable raw images are
  included;
- Git LFS tracks model-weight files before the first push;
- the corresponding source is publicly accessible when binaries are
  distributed; and
- third-party notices and applicable license texts accompany the release.
