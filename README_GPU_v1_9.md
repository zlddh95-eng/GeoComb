# GeoComb Windows NVIDIA GPU build patch v1.9

This patch adds a clean NVIDIA GPU environment, full GPU regression
validation, CUDA-default source GUI, PyInstaller GPU build, and ZIP64
packaging.

## Requirements

- Windows 10 or 11, 64 bit
- NVIDIA CUDA-capable GPU
- NVIDIA display driver 525 or newer
- Python 3.11 with the Python Launcher
- approximately 20 GiB or more free disk space during installation/build

A separate CUDA Toolkit installation is not required for the PyTorch wheel.

## Apply

Extract this ZIP directly into:

```text
C:\GeoComb
```

Run:

```text
apply_GPU_patch_v1_9.bat
```

## Build sequence

```text
1. install_gpu_v1_9.bat
2. validate_gpu.bat
3. run_source_gpu.bat
4. build_gpu.bat
```

### Expected validation

The output must include:

```text
CUDA available: True
CUDA tensor/matrix test PASSED
GPU REGRESSION VALIDATION PASSED
```

### Final outputs

```text
C:\GeoComb\release\GeoComb_Windows_GPU\
C:\GeoComb\release\GeoComb_Windows_GPU.zip
```

Executable:

```text
C:\GeoComb\release\GeoComb_Windows_GPU\
    GeoComb_GPU\GeoComb_GPU.exe
```

The GPU executable starts with `cuda` selected and refuses to run when an
NVIDIA CUDA device cannot be initialized.

## PyTorch build

This patch installs:

```text
torch 2.11.0 + CUDA 12.8
torchvision 0.26.0 + CUDA 12.8
```

This matches the frozen Linux-reference PyTorch generation more closely than
the newer CPU development environment.
