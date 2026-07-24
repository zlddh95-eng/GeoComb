# Windows build and validation

## Supported build targets

- Windows 10/11 x64, CPU
- Windows 10/11 x64, NVIDIA CUDA GPU

Use 64-bit Python 3.11.

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

The regression test analyzes the included `1.JPG` and compares the resulting
cell coordinates and classes with the frozen Linux output.

Passing thresholds:

- total cell-count error <= 0.5%
- at least 99.5% of expected centers matched within 1.5 pixels
- at least 99.5% class agreement among matched centers

Do not distribute a build that fails this validation until the cause has been
reviewed.

## Executable build

CPU:

```bat
build_cpu.bat
```

GPU:

```bat
build_gpu.bat
```

Outputs:

```text
release/GeoComb_Windows_CPU.zip
release/GeoComb_Windows_GPU.zip
```

The build is PyInstaller one-folder format rather than one-file format because
PyTorch, CUDA, Qt, and Ultralytics contain large binary dependencies.

## GPU notes

The GPU package is built with the CUDA-enabled PyTorch wheel. The end user's
NVIDIA driver must be compatible with the installed PyTorch CUDA runtime.
The application checks CUDA availability when analysis begins.

## Release checklist

- regression validation passes
- CPU and GPU counts agree within the approved tolerance
- model SHA256 hashes match `ASSET_MANIFEST.json`
- a fresh Windows computer can open the program
- Excel regeneration works after cell edits
- the application license has been selected and third-party obligations have
  been reviewed
