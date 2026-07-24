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
