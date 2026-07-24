# Windows build and validation

## Supported build targets

- Windows 10/11 x64, CPU
- Windows 10/11 x64, NVIDIA CUDA GPU

Use 64-bit Python 3.11.

The manuscript-aligned Windows environments used Python 3.11.9. The CPU
environment used PyTorch 2.13.0+cpu, and the CUDA environment used PyTorch
2.11.0+cu128. Both used Ultralytics 8.4.96, OpenCV 4.13.0, and NumPy 2.4.4.
Exact dependency snapshots from the release environments should accompany
each tagged source release.

## Source validation before building

CPU:

```bat
install_cpu.bat
validate_cpu.bat
```

GPU:

```bat
install_gpu.bat
validate_gpu.bat
```

The included `1.JPG` regression check is a pre-release smoke test. It
analyzes one controlled example and compares the resulting cell coordinates
and classes with a frozen expected output. It is not the manuscript-level
platform benchmark.

Smoke-test acceptance thresholds:

- total cell-count error <= 0.5%;
- at least 99.5% of expected centers matched within 1.5 pixels; and
- at least 99.5% class agreement among matched centers.

Do not distribute a build that fails the smoke test until the cause has
been reviewed.

## Manuscript-level platform validation

The final platform evaluation processed the same 29-image benchmark three
times in each of four configurations: Windows CPU, Windows CUDA, Linux CPU,
and Linux CUDA. All measured runs produced identical total and per-image
cell counts and identical center coordinates. CPU-CUDA class-label
agreement was 99.9924%.

The single-image smoke test and the 29-image manuscript benchmark serve
different purposes: the former checks a newly built package, whereas the
latter provides the reported cross-platform evidence.

## Deterministic processing controls

- frame-ROI inference is executed on CPU in every deployment configuration;
- ROI crop bounds are quantized to a 32-pixel grid;
- Hough radius is selected from the median of the near-optimal plateau;
- OpenCV is restricted to one processing thread; and
- OpenCL is disabled.

## Executable build

CPU:

```bat
build_cpu.bat
```

GPU:

```bat
build_gpu.bat
```

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
because PyTorch, CUDA, Qt, and Ultralytics include large binary
dependencies. Distribute the complete generated folder, not the executable
alone.

## GPU notes

The GPU package is built with the CUDA-enabled PyTorch wheel. The end user's
NVIDIA driver must be compatible with the packaged PyTorch CUDA runtime.
The application checks CUDA availability when analysis begins. CPU-only
users should use the CPU package.

## Release checklist

- source tree corresponds to the intended release commit;
- single-image regression validation passes;
- model SHA-256 hashes match `ASSET_MANIFEST.json`;
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
