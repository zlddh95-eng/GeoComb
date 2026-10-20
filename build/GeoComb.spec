# -*- mode: python ; coding: utf-8 -*-
"""Shared Windows one-folder build; invoked by build_cpu.bat/build_gpu.bat."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


# PyInstaller supplies SPECPATH. Resolve from the spec, never the caller's cwd.
ROOT = Path(SPECPATH).resolve().parent
VARIANT = os.environ.get("GEOCOMB_VARIANT", "CPU").strip().upper()
ENTRY_POINTS = {"CPU": "geocomb_entry.py", "GPU": "geocomb_gpu_entry.py"}
if VARIANT not in ENTRY_POINTS:
    raise ValueError("GEOCOMB_VARIANT must be CPU or GPU")
NAME = f"GeoComb_{VARIANT}"

MODEL_FILES = (
    "frame_roi_yolo26m_1024_best.pt",
    "stage1_best.pt",
    "stage2_best.pt",
    "stage3_best.pt",
)
PIPELINE_FILES = (
    "detect_frame_crop_resize_puremagenta_from_raw_v1.py",
    "detect_hough_circles_lattice_auto_radius_on_warped_v1.py",
    "filter_existing_hough_pitch_fixedR046_overlaparea30_step1_v1.py",
    "alternating_regularize_localhex_from_gray_step1_v1.py",
    "filter_pure_magenta_cells_after_step2_v1.py",
    "apply_3stage_current_detection_5000coords_v1.py",
    "draw_cell_overlay_outline_legend_corner_v2.py",
)


def required_file(relative_path):
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required build asset is missing: {path}")
    return path


entry = required_file(ENTRY_POINTS[VARIANT])
datas = []
for filename in MODEL_FILES:
    model = required_file(Path("models") / filename)
    with model.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"{model} is a Git LFS pointer, not model weights. "
            "Run 'git lfs install' and 'git lfs pull', or obtain the "
            "unchanged weights from the documented model archive."
        )
    if not header:
        raise ValueError(f"Model file is empty: {model}")
    datas.append((str(model), "models"))

for filename in PIPELINE_FILES:
    script = required_file(Path("pipeline_scripts") / filename)
    datas.append((str(script), "pipeline_scripts"))
datas.append((str(required_file("config/defaults.json")), "config"))

# runpy.run_path needs the original .py files above. Hidden imports also make
# Analysis scan those scripts' imports, which data-only collection would miss.
hiddenimports = [f"pipeline_scripts.{Path(name).stem}" for name in PIPELINE_FILES]
hiddenimports += ["torch", "torchvision"]

# Ultralytics loads model classes dynamically and reads YAML/config/source files.
# collect_all includes its submodules, package data, sources and metadata.
ultralytics_datas, binaries, ultralytics_imports = collect_all("ultralytics")
datas += ultralytics_datas
hiddenimports += ultralytics_imports

# Standard PyInstaller/contrib hooks handle torch/torchvision DLLs (including
# CUDA for the GPU venv), SciPy/OpenCV, and PySide6's Qt platform/image plugins.
# Do not collect local regression_test assets into a redistributable package.
a = Analysis(
    [str(entry)],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    strip=False,
    upx=False,
    # The same executable supports GUI and CLI; preserve stdout/stderr logging.
    console=True,
    contents_directory="_internal",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=NAME,
)
