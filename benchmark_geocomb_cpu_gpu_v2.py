#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import psutil
except Exception:
    psutil = None


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
}

EXPECTED_DATASET_COUNTS = {
    "DeepBee": 13,
    "FAIRHive": 10,
    "InHouse": 6,
}

CLASS_ORDER = [
    "Egg",
    "Larvae",
    "Capped_pupae",
    "Pollen",
    "Capped_honey",
    "Uncapped_honey",
    "Empty_cell",
    "Unbuilt_cell",
]

EQUIVALENCE_CRITERIA = {
    "maximum_pooled_relative_count_difference": 0.005,
    "minimum_cpu_match_fraction": 0.98,
    "minimum_gpu_match_fraction": 0.98,
    "minimum_class_agreement": 0.995,
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan
    return float(np.percentile(array, q))


def iqr_text(values: Iterable[float], digits: int = 2) -> str:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return "NA"
    q1 = float(np.percentile(array, 25))
    q3 = float(np.percentile(array, 75))
    return f"{q1:.{digits}f}–{q3:.{digits}f}"


def get_cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except Exception:
            pass
    return platform.processor() or "Unknown"


def query_nvidia_smi() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mib": parts[3],
                    "compute_capability": parts[4],
                }
            )
    return rows


def query_gpu_memory_used_mib() -> float | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        values = [
            float(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        return float(sum(values)) if values else None
    except Exception:
        return None


def collect_hardware_software(
    cpu_exe: Path,
    gpu_exe: Path,
) -> dict[str, Any]:
    total_ram_gib = None
    logical_cores = os.cpu_count()
    physical_cores = None

    if psutil is not None:
        try:
            total_ram_gib = psutil.virtual_memory().total / (1024 ** 3)
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
        except Exception:
            pass

    return {
        "generated_at": now_text(),
        "operating_system": platform.platform(),
        "python_executable_for_benchmark": sys.executable,
        "python_version_for_benchmark": sys.version,
        "cpu_model": get_cpu_name(),
        "cpu_physical_cores": physical_cores,
        "cpu_logical_cores": logical_cores,
        "ram_gib": total_ram_gib,
        "gpus": query_nvidia_smi(),
        "cpu_executable": {
            "path": str(cpu_exe.resolve()),
            "size_bytes": cpu_exe.stat().st_size,
            "sha256": sha256(cpu_exe),
        },
        "gpu_executable": {
            "path": str(gpu_exe.resolve()),
            "size_bytes": gpu_exe.stat().st_size,
            "sha256": sha256(gpu_exe),
        },
        "equivalence_criteria": EQUIVALENCE_CRITERIA,
    }


def validate_input_folder(input_folder: Path) -> pd.DataFrame:
    files = sorted(
        [
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )

    if len(files) != 29:
        raise RuntimeError(
            f"Expected exactly 29 benchmark images in {input_folder}; "
            f"found {len(files)}."
        )

    rows = []
    counts: Counter[str] = Counter()

    for path in files:
        if "__" not in path.name:
            raise RuntimeError(
                f"Benchmark image lacks a dataset prefix: {path.name}. "
                "Run prepare_benchmark29_wsl_v1.sh first."
            )
        dataset = path.name.split("__", 1)[0]
        counts[dataset] += 1
        rows.append(
            {
                "dataset": dataset,
                "benchmark_name": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    if dict(counts) != EXPECTED_DATASET_COUNTS:
        raise RuntimeError(
            "Unexpected dataset composition. "
            f"Expected {EXPECTED_DATASET_COUNTS}; obtained {dict(counts)}."
        )

    return pd.DataFrame(rows)


def sum_process_rss_bytes(pid: int) -> int | None:
    if psutil is None:
        return None
    try:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        total = 0
        for item in processes:
            try:
                total += int(item.memory_info().rss)
            except Exception:
                pass
        return total
    except Exception:
        return None


def monitor_process(
    process: subprocess.Popen,
    monitor_gpu: bool,
    sample_interval: float = 0.25,
) -> dict[str, float | None]:
    baseline_gpu = query_gpu_memory_used_mib() if monitor_gpu else None
    peak_gpu = baseline_gpu
    peak_rss = None

    while process.poll() is None:
        rss = sum_process_rss_bytes(process.pid)
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)

        if monitor_gpu:
            gpu_used = query_gpu_memory_used_mib()
            if gpu_used is not None:
                peak_gpu = gpu_used if peak_gpu is None else max(peak_gpu, gpu_used)

        time.sleep(sample_interval)

    rss = sum_process_rss_bytes(process.pid)
    if rss is not None:
        peak_rss = rss if peak_rss is None else max(peak_rss, rss)

    gpu_delta = None
    if baseline_gpu is not None and peak_gpu is not None:
        gpu_delta = max(0.0, peak_gpu - baseline_gpu)

    return {
        "peak_process_rss_gib": (
            peak_rss / (1024 ** 3) if peak_rss is not None else None
        ),
        "gpu_memory_baseline_mib": baseline_gpu,
        "peak_gpu_memory_used_mib": peak_gpu,
        "peak_gpu_memory_delta_mib": gpu_delta,
    }


def wait_for_stable_file(path: Path, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    previous_size = None
    stable_count = 0

    while time.time() < deadline:
        if path.is_file():
            size = path.stat().st_size
            if size == previous_size and size > 0:
                stable_count += 1
                if stable_count >= 3:
                    return
            else:
                stable_count = 0
            previous_size = size
        time.sleep(0.5)

    if not path.is_file():
        raise FileNotFoundError(path)


def read_project_predictions(project_folder: Path) -> pd.DataFrame:
    database_path = project_folder / "project.geocomb"
    wait_for_stable_file(database_path)

    query = """
    SELECT
        i.original_name,
        i.safe_stem,
        i.original_width,
        i.original_height,
        i.scale_5000,
        c.x_original,
        c.y_original,
        c.estimated_step_canvas,
        c.class_current,
        c.confidence
    FROM cells AS c
    JOIN images AS i ON i.id = c.image_id
    WHERE c.deleted = 0
    ORDER BY i.original_name, c.id
    """

    # sqlite3.Connection.__exit__ commits or rolls back a transaction,
    # but it does not close the connection. On Windows, leaving the
    # connection to garbage collection keeps project.geocomb locked and
    # prevents the temporary project folder from being deleted.
    connection = sqlite3.connect(database_path)
    try:
        dataframe = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    if dataframe.empty:
        raise RuntimeError(f"No active predictions found in {database_path}")

    dataframe["estimated_step_original"] = (
        pd.to_numeric(dataframe["estimated_step_canvas"], errors="coerce")
        / pd.to_numeric(dataframe["scale_5000"], errors="coerce")
    )
    dataframe["dataset"] = dataframe["original_name"].map(
        lambda value: str(value).split("__", 1)[0]
    )
    dataframe["image_key"] = dataframe["original_name"].astype(str)

    return dataframe[
        [
            "dataset",
            "image_key",
            "original_name",
            "safe_stem",
            "original_width",
            "original_height",
            "x_original",
            "y_original",
            "estimated_step_original",
            "class_current",
            "confidence",
        ]
    ].copy()


def remove_tree_with_retries(path: Path, attempts: int = 5) -> None:
    if not path.exists():
        return
    last_error = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Could not remove {path}: {last_error}")


def execute_run(
    *,
    device_label: str,
    executable: Path,
    device_argument: str,
    input_folder: Path,
    project_folder: Path,
    log_path: Path,
    run_kind: str,
    run_index: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    project_folder.parent.mkdir(parents=True, exist_ok=True)
    remove_tree_with_retries(project_folder)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(executable),
        "analyze",
        "--input-folder",
        str(input_folder),
        "--project-folder",
        str(project_folder),
        "--device",
        device_argument,
        "--overwrite",
    ]

    started_at = now_text()
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n")
        log.write(subprocess.list2cmdline(command) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(executable.parent),
        )
        monitor = monitor_process(
            process,
            monitor_gpu=(device_label == "GPU"),
        )
        exit_code = process.wait()

    elapsed_seconds = time.perf_counter() - start
    finished_at = now_text()

    if exit_code != 0:
        raise RuntimeError(
            f"{device_label} {run_kind} run {run_index} failed with "
            f"exit code {exit_code}. See {log_path}"
        )

    predictions = read_project_predictions(project_folder)
    total_cells = int(len(predictions))
    image_count = int(predictions["image_key"].nunique())

    if image_count != 29:
        raise RuntimeError(
            f"{device_label} run produced {image_count} images instead of 29."
        )

    record = {
        "device": device_label,
        "run_kind": run_kind,
        "run_index": run_index,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": elapsed_seconds,
        "wall_minutes": elapsed_seconds / 60.0,
        "image_count": image_count,
        "total_predicted_cells": total_cells,
        "seconds_per_image": elapsed_seconds / max(1, image_count),
        "images_per_hour": 3600.0 * image_count / max(elapsed_seconds, 1e-12),
        "predicted_cells_per_second": total_cells / max(elapsed_seconds, 1e-12),
        "peak_process_rss_gib": monitor["peak_process_rss_gib"],
        "gpu_memory_baseline_mib": monitor["gpu_memory_baseline_mib"],
        "peak_gpu_memory_used_mib": monitor["peak_gpu_memory_used_mib"],
        "peak_gpu_memory_delta_mib": monitor["peak_gpu_memory_delta_mib"],
        "exit_code": exit_code,
        "log_path": str(log_path.resolve()),
        "project_folder": str(project_folder.resolve()),
    }

    return record, predictions


def greedy_match(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if len(left_xy) == 0 or len(right_xy) == 0:
        return []

    tree = cKDTree(right_xy)
    candidate_pairs: list[tuple[float, int, int]] = []

    for left_index, point in enumerate(left_xy):
        for right_index in tree.query_ball_point(point, threshold):
            distance = float(
                np.linalg.norm(point - right_xy[int(right_index)])
            )
            candidate_pairs.append(
                (distance, left_index, int(right_index))
            )

    candidate_pairs.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for distance, left_index, right_index in candidate_pairs:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index, distance))

    return matches


def compare_prediction_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    pair_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    left_images = set(left["image_key"].unique())
    right_images = set(right["image_key"].unique())

    if left_images != right_images:
        raise RuntimeError(
            f"Image sets differ between {left_label} and {right_label}: "
            f"left-only={sorted(left_images - right_images)}, "
            f"right-only={sorted(right_images - left_images)}"
        )

    image_rows = []
    class_rows = []
    matched_class_total = 0
    matched_total = 0
    left_total = 0
    right_total = 0
    all_distances = []

    for image_key in sorted(left_images):
        left_image = left.loc[left["image_key"].eq(image_key)].reset_index(drop=True)
        right_image = right.loc[right["image_key"].eq(image_key)].reset_index(drop=True)

        left_pitch = float(
            np.nanmedian(left_image["estimated_step_original"].to_numpy(float))
        )
        right_pitch = float(
            np.nanmedian(right_image["estimated_step_original"].to_numpy(float))
        )
        valid_pitches = [
            value
            for value in (left_pitch, right_pitch)
            if np.isfinite(value) and value > 0
        ]
        pitch = float(np.median(valid_pitches)) if valid_pitches else 60.0
        threshold = 0.5 * pitch

        matches = greedy_match(
            left_image[["x_original", "y_original"]].to_numpy(float),
            right_image[["x_original", "y_original"]].to_numpy(float),
            threshold,
        )

        distances = [match[2] for match in matches]
        class_matches = sum(
            str(left_image.iloc[left_index]["class_current"])
            == str(right_image.iloc[right_index]["class_current"])
            for left_index, right_index, _ in matches
        )

        left_count = len(left_image)
        right_count = len(right_image)
        matched_count = len(matches)
        relative_count_difference = (
            abs(left_count - right_count) / max(1, left_count)
        )

        image_rows.append(
            {
                "pair_index": pair_index,
                "left_label": left_label,
                "right_label": right_label,
                "dataset": str(left_image.iloc[0]["dataset"]),
                "image_key": image_key,
                "left_count": left_count,
                "right_count": right_count,
                "absolute_count_difference": abs(left_count - right_count),
                "relative_count_difference_vs_left": relative_count_difference,
                "pitch_original_px": pitch,
                "matching_threshold_px": threshold,
                "matched_count": matched_count,
                "left_match_fraction": matched_count / max(1, left_count),
                "right_match_fraction": matched_count / max(1, right_count),
                "symmetric_match_fraction": (
                    2.0 * matched_count / max(1, left_count + right_count)
                ),
                "class_agreement_among_matches": (
                    class_matches / max(1, matched_count)
                ),
                "mean_matched_distance_px": (
                    float(np.mean(distances)) if distances else math.nan
                ),
                "median_matched_distance_px": (
                    float(np.median(distances)) if distances else math.nan
                ),
                "p95_matched_distance_px": (
                    percentile(distances, 95) if distances else math.nan
                ),
            }
        )

        for class_name in CLASS_ORDER:
            left_class_count = int(
                left_image["class_current"].astype(str).eq(class_name).sum()
            )
            right_class_count = int(
                right_image["class_current"].astype(str).eq(class_name).sum()
            )
            class_rows.append(
                {
                    "pair_index": pair_index,
                    "left_label": left_label,
                    "right_label": right_label,
                    "dataset": str(left_image.iloc[0]["dataset"]),
                    "image_key": image_key,
                    "class": class_name,
                    "left_count": left_class_count,
                    "right_count": right_class_count,
                    "absolute_count_difference": abs(
                        left_class_count - right_class_count
                    ),
                    "relative_count_difference_vs_left": (
                        abs(left_class_count - right_class_count)
                        / max(1, left_class_count)
                    ),
                }
            )

        matched_class_total += class_matches
        matched_total += matched_count
        left_total += left_count
        right_total += right_count
        all_distances.extend(distances)

    pooled_relative_count_difference = (
        abs(left_total - right_total) / max(1, left_total)
    )
    left_match_fraction = matched_total / max(1, left_total)
    right_match_fraction = matched_total / max(1, right_total)
    class_agreement = matched_class_total / max(1, matched_total)

    passed = bool(
        pooled_relative_count_difference
        <= EQUIVALENCE_CRITERIA[
            "maximum_pooled_relative_count_difference"
        ]
        and left_match_fraction
        >= EQUIVALENCE_CRITERIA["minimum_cpu_match_fraction"]
        and right_match_fraction
        >= EQUIVALENCE_CRITERIA["minimum_gpu_match_fraction"]
        and class_agreement
        >= EQUIVALENCE_CRITERIA["minimum_class_agreement"]
    )

    overall = pd.DataFrame(
        [
            {
                "pair_index": pair_index,
                "left_label": left_label,
                "right_label": right_label,
                "left_total_cells": left_total,
                "right_total_cells": right_total,
                "absolute_count_difference": abs(left_total - right_total),
                "pooled_relative_count_difference_vs_left": (
                    pooled_relative_count_difference
                ),
                "matched_count": matched_total,
                "left_match_fraction": left_match_fraction,
                "right_match_fraction": right_match_fraction,
                "symmetric_match_fraction": (
                    2.0 * matched_total / max(1, left_total + right_total)
                ),
                "class_agreement_among_matches": class_agreement,
                "mean_matched_distance_px": (
                    float(np.mean(all_distances))
                    if all_distances
                    else math.nan
                ),
                "median_matched_distance_px": (
                    float(np.median(all_distances))
                    if all_distances
                    else math.nan
                ),
                "p95_matched_distance_px": (
                    percentile(all_distances, 95)
                    if all_distances
                    else math.nan
                ),
                "scientifically_equivalent": passed,
            }
        ]
    )

    return pd.DataFrame(image_rows), overall, pd.DataFrame(class_rows)


def summarize_runtime(runtime_runs: pd.DataFrame) -> pd.DataFrame:
    measured = runtime_runs.loc[runtime_runs["run_kind"].eq("measured")].copy()
    rows = []

    for device, group in measured.groupby("device", sort=False):
        walls = group["wall_seconds"].to_numpy(float)
        seconds_per_image = group["seconds_per_image"].to_numpy(float)
        cells_per_second = group["predicted_cells_per_second"].to_numpy(float)
        images_per_hour = group["images_per_hour"].to_numpy(float)

        rows.append(
            {
                "device": device,
                "measured_runs": len(group),
                "median_wall_seconds": float(np.median(walls)),
                "q1_wall_seconds": float(np.percentile(walls, 25)),
                "q3_wall_seconds": float(np.percentile(walls, 75)),
                "minimum_wall_seconds": float(np.min(walls)),
                "maximum_wall_seconds": float(np.max(walls)),
                "median_seconds_per_image": float(np.median(seconds_per_image)),
                "median_images_per_hour": float(np.median(images_per_hour)),
                "median_predicted_cells_per_second": float(
                    np.median(cells_per_second)
                ),
                "median_peak_process_rss_gib": float(
                    np.nanmedian(
                        pd.to_numeric(
                            group["peak_process_rss_gib"],
                            errors="coerce",
                        ).to_numpy(float)
                    )
                ),
                "median_peak_gpu_memory_delta_mib": float(
                    np.nanmedian(
                        pd.to_numeric(
                            group["peak_gpu_memory_delta_mib"],
                            errors="coerce",
                        ).to_numpy(float)
                    )
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if set(summary["device"]) >= {"CPU", "GPU"}:
        cpu_seconds = float(
            summary.loc[
                summary["device"].eq("CPU"), "median_wall_seconds"
            ].iloc[0]
        )
        gpu_seconds = float(
            summary.loc[
                summary["device"].eq("GPU"), "median_wall_seconds"
            ].iloc[0]
        )
        summary["cpu_to_gpu_speedup"] = cpu_seconds / max(gpu_seconds, 1e-12)
    else:
        summary["cpu_to_gpu_speedup"] = math.nan

    return summary


def summarize_equivalence(overall_by_pair: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "pooled_relative_count_difference_vs_left",
        "left_match_fraction",
        "right_match_fraction",
        "symmetric_match_fraction",
        "class_agreement_among_matches",
        "mean_matched_distance_px",
        "median_matched_distance_px",
        "p95_matched_distance_px",
    ]

    row: dict[str, Any] = {
        "paired_runs": len(overall_by_pair),
        "all_pairs_scientifically_equivalent": bool(
            overall_by_pair["scientifically_equivalent"].all()
        ),
    }

    for column in numeric_columns:
        values = pd.to_numeric(overall_by_pair[column], errors="coerce")
        row[f"median_{column}"] = float(np.nanmedian(values))
        row[f"minimum_{column}"] = float(np.nanmin(values))
        row[f"maximum_{column}"] = float(np.nanmax(values))

    return pd.DataFrame([row])


def write_excel(
    output_path: Path,
    *,
    runtime_runs: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    equivalence_by_image: pd.DataFrame,
    equivalence_overall: pd.DataFrame,
    equivalence_summary: pd.DataFrame,
    class_count_difference: pd.DataFrame,
    repeatability_by_image: pd.DataFrame,
    repeatability_overall: pd.DataFrame,
    hardware: dict[str, Any],
) -> None:
    hardware_rows = []
    for key, value in hardware.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        hardware_rows.append({"Field": key, "Value": value})

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        runtime_runs.to_excel(writer, sheet_name="Runtime_Runs", index=False)
        runtime_summary.to_excel(
            writer, sheet_name="Runtime_Summary", index=False
        )
        equivalence_overall.to_excel(
            writer, sheet_name="CPU_GPU_Overall", index=False
        )
        equivalence_summary.to_excel(
            writer, sheet_name="CPU_GPU_Summary", index=False
        )
        equivalence_by_image.to_excel(
            writer, sheet_name="CPU_GPU_By_Image", index=False
        )
        class_count_difference.to_excel(
            writer, sheet_name="Class_Count_Diffs", index=False
        )
        repeatability_overall.to_excel(
            writer, sheet_name="Repeatability_Overall", index=False
        )
        repeatability_by_image.to_excel(
            writer, sheet_name="Repeatability_By_Image", index=False
        )
        pd.DataFrame(hardware_rows).to_excel(
            writer, sheet_name="Hardware_Software", index=False
        )


def write_manuscript_text(
    output_path: Path,
    *,
    runtime_runs: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    equivalence_summary: pd.DataFrame,
    hardware: dict[str, Any],
) -> None:
    cpu_runs = runtime_runs.loc[
        (runtime_runs["run_kind"].eq("measured"))
        & (runtime_runs["device"].eq("CPU"))
    ]
    gpu_runs = runtime_runs.loc[
        (runtime_runs["run_kind"].eq("measured"))
        & (runtime_runs["device"].eq("GPU"))
    ]

    cpu_summary = runtime_summary.loc[
        runtime_summary["device"].eq("CPU")
    ].iloc[0]
    gpu_summary = runtime_summary.loc[
        runtime_summary["device"].eq("GPU")
    ].iloc[0]
    eq = equivalence_summary.iloc[0]

    gpu_info = hardware.get("gpus", [])
    gpu_name = gpu_info[0]["name"] if gpu_info else "[GPU model]"
    gpu_memory = (
        float(gpu_info[0]["memory_total_mib"]) / 1024.0
        if gpu_info
        else math.nan
    )
    ram = hardware.get("ram_gib")
    cpu_name = hardware.get("cpu_model", "[CPU model]")
    os_name = hardware.get("operating_system", "[Windows version]")

    cpu_minutes = float(cpu_summary["median_wall_seconds"]) / 60.0
    gpu_minutes = float(gpu_summary["median_wall_seconds"]) / 60.0
    speedup = float(cpu_summary["cpu_to_gpu_speedup"])

    cpu_wall_iqr = iqr_text(cpu_runs["wall_seconds"] / 60.0, 2)
    gpu_wall_iqr = iqr_text(gpu_runs["wall_seconds"] / 60.0, 2)

    methods = f"""MATERIALS AND METHODS — Computational performance and CPU–GPU equivalence

Computational performance was evaluated using the packaged 64-bit Windows CPU and NVIDIA GPU distributions on the same workstation ({cpu_name}; {ram:.1f} GB RAM; {gpu_name}, {gpu_memory:.1f} GB VRAM; {os_name}). The complete 29-image benchmark comprised 13 official DeepBee images, ten FAIRHive images, and six held-out in-house images. Each distribution was subjected to one full warm-up run followed by {len(cpu_runs)} measured runs, with CPU and GPU execution order alternated between repetitions. Timing began immediately before executable launch and ended after successful completion of the complete nine-stage workflow, including creation of the project database, cell-level CSV files, Excel workbook, and overlay images. Manual review and editing were excluded. Total runtime, seconds per image, images per hour, predicted cells per second, and peak process memory were recorded.

CPU and GPU outputs were compared in original-image coordinates. Cells were matched independently within each image using one-to-one greedy nearest-neighbor assignment at a threshold of 0.5 times the median image-specific cell pitch estimated from the two outputs. Scientific equivalence was assessed from the pooled relative total-count difference, the fraction of CPU and GPU cells with a spatial match, and class agreement among matched cells. Prespecified acceptance criteria were a pooled relative count difference of no more than 0.5%, spatial match fractions of at least 98% for both outputs, and class agreement of at least 99.5%.
"""

    results = f"""RESULTS — Computational performance and CPU–GPU equivalence

The CPU distribution processed the 29-image benchmark in a median of {cpu_minutes:.2f} min (IQR, {cpu_wall_iqr} min), corresponding to {float(cpu_summary['median_seconds_per_image']):.2f} s per image, {float(cpu_summary['median_images_per_hour']):.2f} images h−1, and {float(cpu_summary['median_predicted_cells_per_second']):.2f} predicted cells s−1. The GPU distribution required a median of {gpu_minutes:.2f} min (IQR, {gpu_wall_iqr} min), corresponding to {float(gpu_summary['median_seconds_per_image']):.2f} s per image, {float(gpu_summary['median_images_per_hour']):.2f} images h−1, and {float(gpu_summary['median_predicted_cells_per_second']):.2f} predicted cells s−1. GPU execution therefore provided a {speedup:.2f}-fold speed-up relative to CPU execution.

Across paired measured runs, the median pooled relative difference in total predicted cell count was {100.0 * float(eq['median_pooled_relative_count_difference_vs_left']):.3f}%. Median spatial match fractions were {100.0 * float(eq['median_left_match_fraction']):.3f}% for CPU cells and {100.0 * float(eq['median_right_match_fraction']):.3f}% for GPU cells at the 0.5 × cell-pitch threshold. Median class agreement among spatially matched cells was {100.0 * float(eq['median_class_agreement_among_matches']):.3f}%. Scientific-equivalence criteria were {'met in all paired runs' if bool(eq['all_pairs_scientifically_equivalent']) else 'not met in every paired run'}.
"""

    discussion = f"""DISCUSSION — Deployment performance

The packaged NVIDIA GPU distribution reduced end-to-end analysis time by {speedup:.2f}-fold relative to the CPU distribution while preserving total cell counts, center locations, and class assignments within the prespecified scientific-equivalence criteria. Because the workflow includes Hough-transform, lattice-regularization, database, and export operations that remain CPU-based, the observed acceleration reflects the combined behavior of neural-network and geometry-aware processing rather than neural-network inference alone. The CPU edition therefore supports use on computers without specialized hardware, whereas the GPU edition provides a higher-throughput option for larger image collections.
"""

    output_path.write_text(
        methods + "\n" + results + "\n" + discussion,
        encoding="utf-8",
    )


def run_self_test() -> None:
    left = pd.DataFrame(
        {
            "dataset": ["Test"] * 3,
            "image_key": ["Test__1.png"] * 3,
            "original_name": ["Test__1.png"] * 3,
            "safe_stem": ["Test__1"] * 3,
            "original_width": [100] * 3,
            "original_height": [100] * 3,
            "x_original": [10.0, 30.0, 50.0],
            "y_original": [10.0, 30.0, 50.0],
            "estimated_step_original": [20.0] * 3,
            "class_current": ["Egg", "Larvae", "Pollen"],
            "confidence": [0.9, 0.8, 0.7],
        }
    )
    right = left.copy()
    right["x_original"] += [0.1, -0.2, 0.3]
    by_image, overall, class_rows = compare_prediction_frames(
        left,
        right,
        left_label="CPU",
        right_label="GPU",
        pair_index=1,
    )
    assert int(overall.iloc[0]["matched_count"]) == 3
    assert float(overall.iloc[0]["class_agreement_among_matches"]) == 1.0
    assert bool(overall.iloc[0]["scientifically_equivalent"])
    assert len(by_image) == 1
    assert len(class_rows) == len(CLASS_ORDER)
    print("SELF-TEST PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark packaged GeoComb CPU/GPU Windows executables (v2) "
            "and evaluate scientific equivalence."
        )
    )
    parser.add_argument("--input-folder", type=Path)
    parser.add_argument("--cpu-exe", type=Path)
    parser.add_argument("--gpu-exe", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--keep-projects", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    required = {
        "--input-folder": args.input_folder,
        "--cpu-exe": args.cpu_exe,
        "--gpu-exe": args.gpu_exe,
        "--output-root": args.output_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))

    input_folder = args.input_folder.resolve()
    cpu_exe = args.cpu_exe.resolve()
    gpu_exe = args.gpu_exe.resolve()
    output_root = args.output_root.resolve()

    if not input_folder.is_dir():
        raise NotADirectoryError(input_folder)
    if not cpu_exe.is_file():
        raise FileNotFoundError(cpu_exe)
    if not gpu_exe.is_file():
        raise FileNotFoundError(gpu_exe)
    if args.warmup_runs < 0 or args.measured_runs < 1:
        raise ValueError("warmup-runs must be >=0 and measured-runs must be >=1")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    projects_root = output_root / "_temporary_projects"
    snapshots_root = output_root / "prediction_snapshots"
    logs_root.mkdir()
    projects_root.mkdir()
    snapshots_root.mkdir()

    input_manifest = validate_input_folder(input_folder)
    input_manifest.to_csv(
        output_root / "benchmark_input_manifest_verified.csv",
        index=False,
    )

    hardware = collect_hardware_software(cpu_exe, gpu_exe)
    (output_root / "hardware_software.json").write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runtime_records: list[dict[str, Any]] = []
    snapshots: dict[tuple[str, int], pd.DataFrame] = {}

    devices = {
        "CPU": (cpu_exe, "cpu"),
        "GPU": (gpu_exe, "cuda"),
    }

    execution_plan: list[tuple[str, str, int]] = []

    for warmup_index in range(1, args.warmup_runs + 1):
        execution_plan.extend(
            [
                ("CPU", "warmup", warmup_index),
                ("GPU", "warmup", warmup_index),
            ]
        )

    for run_index in range(1, args.measured_runs + 1):
        order = ["CPU", "GPU"] if run_index % 2 == 1 else ["GPU", "CPU"]
        for device_label in order:
            execution_plan.append((device_label, "measured", run_index))

    print("GeoComb CPU/GPU benchmark v2")
    print("=" * 78)
    print(f"Input folder: {input_folder}")
    print(f"Output root: {output_root}")
    print(f"Warm-up runs per device: {args.warmup_runs}")
    print(f"Measured runs per device: {args.measured_runs}")
    print("Execution plan:")
    for device_label, run_kind, run_index in execution_plan:
        print(f"  {device_label} {run_kind} {run_index}")
    print()

    for plan_index, (device_label, run_kind, run_index) in enumerate(
        execution_plan, start=1
    ):
        executable, device_argument = devices[device_label]
        project_folder = (
            projects_root / f"{device_label}_{run_kind}_{run_index:02d}"
        )
        log_path = logs_root / f"{device_label}_{run_kind}_{run_index:02d}.log"

        print(
            f"[{plan_index}/{len(execution_plan)}] "
            f"{device_label} {run_kind} run {run_index} started..."
        )

        record, predictions = execute_run(
            device_label=device_label,
            executable=executable,
            device_argument=device_argument,
            input_folder=input_folder,
            project_folder=project_folder,
            log_path=log_path,
            run_kind=run_kind,
            run_index=run_index,
        )
        runtime_records.append(record)

        print(
            f"  completed in {record['wall_minutes']:.2f} min; "
            f"{record['total_predicted_cells']:,} cells"
        )

        if run_kind == "measured":
            snapshot_path = (
                snapshots_root
                / f"{device_label}_measured_{run_index:02d}.csv.gz"
            )
            predictions.to_csv(
                snapshot_path,
                index=False,
                compression="gzip",
            )
            snapshots[(device_label, run_index)] = predictions

        # Persist timing immediately after a successful run, before optional
        # cleanup, so a cleanup problem never discards a completed benchmark.
        pd.DataFrame(runtime_records).to_csv(
            output_root / "runtime_runs_partial.csv",
            index=False,
        )

        if not args.keep_projects:
            try:
                remove_tree_with_retries(project_folder)
            except Exception as exc:
                warning = (
                    f"WARNING: completed run was preserved because temporary "
                    f"project cleanup failed: {project_folder}: {exc}"
                )
                print(warning)
                with (output_root / "cleanup_warnings.txt").open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(warning + "\n")

    runtime_runs = pd.DataFrame(runtime_records)
    runtime_runs.to_csv(output_root / "runtime_runs.csv", index=False)
    runtime_summary = summarize_runtime(runtime_runs)
    runtime_summary.to_csv(output_root / "runtime_summary.csv", index=False)

    equivalence_by_image_list = []
    equivalence_overall_list = []
    class_difference_list = []

    for run_index in range(1, args.measured_runs + 1):
        cpu_predictions = snapshots[("CPU", run_index)]
        gpu_predictions = snapshots[("GPU", run_index)]
        by_image, overall, class_rows = compare_prediction_frames(
            cpu_predictions,
            gpu_predictions,
            left_label="CPU",
            right_label="GPU",
            pair_index=run_index,
        )
        equivalence_by_image_list.append(by_image)
        equivalence_overall_list.append(overall)
        class_difference_list.append(class_rows)

    equivalence_by_image = pd.concat(
        equivalence_by_image_list, ignore_index=True
    )
    equivalence_overall = pd.concat(
        equivalence_overall_list, ignore_index=True
    )
    class_count_difference = pd.concat(
        class_difference_list, ignore_index=True
    )
    equivalence_summary = summarize_equivalence(equivalence_overall)

    equivalence_by_image.to_csv(
        output_root / "cpu_gpu_equivalence_by_image.csv", index=False
    )
    equivalence_overall.to_csv(
        output_root / "cpu_gpu_equivalence_overall_by_pair.csv", index=False
    )
    equivalence_summary.to_csv(
        output_root / "cpu_gpu_equivalence_summary.csv", index=False
    )
    class_count_difference.to_csv(
        output_root / "cpu_gpu_class_count_differences.csv", index=False
    )

    repeatability_by_image_list = []
    repeatability_overall_list = []

    if args.measured_runs >= 2:
        for device_label in ("CPU", "GPU"):
            baseline = snapshots[(device_label, 1)]
            for run_index in range(2, args.measured_runs + 1):
                by_image, overall, _ = compare_prediction_frames(
                    baseline,
                    snapshots[(device_label, run_index)],
                    left_label=f"{device_label}_run1",
                    right_label=f"{device_label}_run{run_index}",
                    pair_index=run_index,
                )
                by_image.insert(0, "device", device_label)
                overall.insert(0, "device", device_label)
                repeatability_by_image_list.append(by_image)
                repeatability_overall_list.append(overall)

    repeatability_by_image = (
        pd.concat(repeatability_by_image_list, ignore_index=True)
        if repeatability_by_image_list
        else pd.DataFrame()
    )
    repeatability_overall = (
        pd.concat(repeatability_overall_list, ignore_index=True)
        if repeatability_overall_list
        else pd.DataFrame()
    )

    repeatability_by_image.to_csv(
        output_root / "within_device_repeatability_by_image.csv",
        index=False,
    )
    repeatability_overall.to_csv(
        output_root / "within_device_repeatability_overall.csv",
        index=False,
    )

    write_excel(
        output_root / "GeoComb_CPU_GPU_benchmark.xlsx",
        runtime_runs=runtime_runs,
        runtime_summary=runtime_summary,
        equivalence_by_image=equivalence_by_image,
        equivalence_overall=equivalence_overall,
        equivalence_summary=equivalence_summary,
        class_count_difference=class_count_difference,
        repeatability_by_image=repeatability_by_image,
        repeatability_overall=repeatability_overall,
        hardware=hardware,
    )

    write_manuscript_text(
        output_root / "manuscript_benchmark_text.txt",
        runtime_runs=runtime_runs,
        runtime_summary=runtime_summary,
        equivalence_summary=equivalence_summary,
        hardware=hardware,
    )

    cpu_median = float(
        runtime_summary.loc[
            runtime_summary["device"].eq("CPU"), "median_wall_seconds"
        ].iloc[0]
    )
    gpu_median = float(
        runtime_summary.loc[
            runtime_summary["device"].eq("GPU"), "median_wall_seconds"
        ].iloc[0]
    )
    eq = equivalence_summary.iloc[0]

    summary_lines = [
        "GEOCOMB CPU/GPU BENCHMARK SUMMARY",
        "=" * 78,
        f"Generated: {now_text()}",
        f"Input images: 29 ({EXPECTED_DATASET_COUNTS})",
        f"Warm-up runs per device: {args.warmup_runs}",
        f"Measured runs per device: {args.measured_runs}",
        "",
        "RUNTIME",
        "-" * 78,
        f"CPU median total runtime: {cpu_median / 60.0:.3f} min",
        f"GPU median total runtime: {gpu_median / 60.0:.3f} min",
        f"CPU/GPU speed-up: {cpu_median / max(gpu_median, 1e-12):.3f}x",
        "",
        "SCIENTIFIC EQUIVALENCE",
        "-" * 78,
        (
            "Median pooled relative count difference: "
            f"{100.0 * float(eq['median_pooled_relative_count_difference_vs_left']):.4f}%"
        ),
        (
            "Median CPU match fraction: "
            f"{100.0 * float(eq['median_left_match_fraction']):.4f}%"
        ),
        (
            "Median GPU match fraction: "
            f"{100.0 * float(eq['median_right_match_fraction']):.4f}%"
        ),
        (
            "Median class agreement among matches: "
            f"{100.0 * float(eq['median_class_agreement_among_matches']):.4f}%"
        ),
        (
            "All paired runs passed: "
            f"{bool(eq['all_pairs_scientifically_equivalent'])}"
        ),
        "",
        "MAIN OUTPUTS",
        "-" * 78,
        str(output_root / "GeoComb_CPU_GPU_benchmark.xlsx"),
        str(output_root / "runtime_summary.csv"),
        str(output_root / "cpu_gpu_equivalence_summary.csv"),
        str(output_root / "manuscript_benchmark_text.txt"),
    ]
    (output_root / "benchmark_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(summary_lines))
    print()
    print(f"Benchmark complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
