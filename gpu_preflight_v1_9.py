from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


MIN_DRIVER_MAJOR = 525
MIN_FREE_GIB = 20.0


def run_nvidia_smi() -> tuple[str, list[dict[str, str]]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi was not found. Install or update the NVIDIA "
            "display driver before creating the GPU version."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "nvidia-smi failed:\n"
            + (exc.stdout or "")
            + (exc.stderr or "")
        ) from exc

    rows: list[dict[str, str]] = []

    for line in completed.stdout.splitlines():
        if not line.strip():
            continue

        parts = [part.strip() for part in line.split(",")]

        if len(parts) < 5:
            raise RuntimeError(
                f"Unexpected nvidia-smi output: {line!r}"
            )

        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": parts[3],
                "compute_capability": parts[4],
            }
        )

    if not rows:
        raise RuntimeError(
            "No NVIDIA GPU was reported by nvidia-smi."
        )

    return completed.stdout, rows


def main() -> int:
    root = Path(__file__).resolve().parent
    raw_output, gpus = run_nvidia_smi()

    driver_majors = []

    for gpu in gpus:
        try:
            driver_majors.append(
                int(gpu["driver_version"].split(".", 1)[0])
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse NVIDIA driver version: "
                f"{gpu['driver_version']!r}"
            ) from exc

    minimum_detected_driver = min(driver_majors)

    usage = shutil.disk_usage(root.anchor or root)
    free_gib = usage.free / (1024 ** 3)

    report = {
        "python": sys.version,
        "project_root": str(root),
        "gpus": gpus,
        "minimum_detected_driver_major": minimum_detected_driver,
        "minimum_required_driver_major_for_cuda_12_x": (
            MIN_DRIVER_MAJOR
        ),
        "free_disk_gib": free_gib,
        "recommended_minimum_free_disk_gib": MIN_FREE_GIB,
    }

    (root / "gpu_preflight_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (root / "gpu_nvidia_smi.txt").write_text(
        raw_output,
        encoding="utf-8",
    )

    print("NVIDIA GPU preflight")
    print("=" * 72)

    for gpu in gpus:
        print(
            f"GPU {gpu['index']}: {gpu['name']} | "
            f"driver {gpu['driver_version']} | "
            f"VRAM {gpu['memory_total_mib']} MiB | "
            f"compute capability {gpu['compute_capability']}"
        )

    print(f"Free disk space: {free_gib:.1f} GiB")

    if minimum_detected_driver < MIN_DRIVER_MAJOR:
        print()
        print(
            "FAILED: the NVIDIA driver is too old for the "
            "CUDA 12.x PyTorch package."
        )
        print(
            f"Detected driver major: {minimum_detected_driver}; "
            f"required: {MIN_DRIVER_MAJOR} or newer."
        )
        return 2

    if free_gib < MIN_FREE_GIB:
        print()
        print(
            "FAILED: insufficient free disk space for the GPU "
            "environment, PyInstaller build, release folder, and ZIP."
        )
        print(
            f"At least {MIN_FREE_GIB:.0f} GiB is recommended."
        )
        return 3

    print()
    print("GPU preflight PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
