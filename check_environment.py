from __future__ import annotations

import importlib
import platform
import sys

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
]

failed = []

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())

for module_name in MODULES:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        print(f"[OK] {module_name} {version}".rstrip())
    except Exception as exc:
        failed.append((module_name, repr(exc)))
        print(f"[FAIL] {module_name}: {exc!r}")

try:
    import torch
    print("Torch CPU available:", True)
    print("Torch CUDA available:", torch.cuda.is_available())
except Exception:
    pass

if failed:
    print("\nEnvironment validation FAILED.")
    for name, error in failed:
        print(f"- {name}: {error}")
    raise SystemExit(1)

print("\nEnvironment validation PASSED.")
