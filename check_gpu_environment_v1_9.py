from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

MODULES = [
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "PIL",
    "openpyxl",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "matplotlib",
    "lap",
]


def module_version(module: object) -> str:
    return str(getattr(module, "__version__", "")).strip()


def main() -> int:
    failed: list[tuple[str, str]] = []

    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())

    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = module_version(module)
            print(
                f"[OK] {module_name}"
                + (f" {version}" if version else "")
            )
        except Exception as exc:
            failed.append((module_name, repr(exc)))
            print(f"[FAIL] {module_name}: {exc!r}")

    if failed:
        print("\nGPU environment validation FAILED.")
        for name, error in failed:
            print(f"- {name}: {error}")
        return 1

    import torch

    print()
    print("Torch build:", torch.__version__)
    print("Torch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print(
        "cuDNN version:",
        torch.backends.cudnn.version(),
    )

    if "+cpu" in torch.__version__:
        print(
            "FAILED: a CPU-only PyTorch build is installed "
            "in .venv_gpu."
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "FAILED: PyTorch cannot access an NVIDIA CUDA GPU."
        )
        return 3

    device_rows = []

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        row = {
            "index": index,
            "name": properties.name,
            "total_memory_gib": (
                properties.total_memory / (1024 ** 3)
            ),
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
        }
        device_rows.append(row)
        print(
            f"GPU {index}: {row['name']} | "
            f"{row['total_memory_gib']:.2f} GiB | "
            f"compute capability {row['compute_capability']}"
        )

    try:
        device = torch.device("cuda:0")
        left = torch.randn(
            (1024, 1024),
            device=device,
        )
        right = torch.randn(
            (1024, 1024),
            device=device,
        )
        result = left @ right
        checksum = float(result[0, 0].item())
        torch.cuda.synchronize(device)

        print(
            "CUDA tensor/matrix test PASSED; "
            f"checksum={checksum:.6f}"
        )
    except Exception as exc:
        print(
            f"FAILED: CUDA computation test failed: {exc!r}"
        )
        return 4

    report = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": importlib.import_module(
            "torchvision"
        ).__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": device_rows,
        "cuda_test_passed": True,
    }

    Path("gpu_environment_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nGPU environment validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
