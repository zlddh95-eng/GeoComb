#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_geocomb_cpu_gpu_v2 as bench


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the 29-image CPU/GPU equivalence check after the CPU "
            "run completed but v1 failed while deleting project.geocomb."
        )
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=Path(r"C:\GeoComb_Benchmark_29\input"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\GeoComb_Benchmark_29\equivalence_once"),
    )
    parser.add_argument(
        "--gpu-exe",
        type=Path,
        default=Path(
            r"C:\GeoComb\release\GeoComb_Windows_GPU"
            r"\GeoComb_GPU\GeoComb_GPU.exe"
        ),
    )
    args = parser.parse_args()

    input_folder = args.input_folder.resolve()
    output_root = args.output_root.resolve()
    gpu_exe = args.gpu_exe.resolve()

    cpu_snapshot = (
        output_root
        / "prediction_snapshots"
        / "CPU_measured_01.csv.gz"
    )
    gpu_snapshot = (
        output_root
        / "prediction_snapshots"
        / "GPU_measured_01.csv.gz"
    )
    gpu_project = (
        output_root
        / "_temporary_projects"
        / "GPU_measured_01_resume"
    )
    gpu_log = (
        output_root
        / "logs"
        / "GPU_measured_01_resume.log"
    )

    for path in (input_folder, output_root):
        if not path.is_dir():
            raise FileNotFoundError(path)

    if not gpu_exe.is_file():
        raise FileNotFoundError(gpu_exe)

    if not cpu_snapshot.is_file():
        raise FileNotFoundError(
            f"Completed CPU snapshot was not found: {cpu_snapshot}. "
            "Do not rerun this resume script until the v1 output folder "
            "has been preserved."
        )

    print("GeoComb equivalence resume v2")
    print("=" * 78)
    print(f"Using completed CPU snapshot: {cpu_snapshot}")
    print("The 99.5-minute CPU analysis will NOT be repeated.")
    print(f"Running GPU executable: {gpu_exe}")
    print()

    cpu_predictions = pd.read_csv(cpu_snapshot)

    # Remove only an incomplete resumed GPU project, never the completed CPU
    # project or CPU snapshot.
    try:
        bench.remove_tree_with_retries(gpu_project)
    except Exception:
        pass

    gpu_record, gpu_predictions = bench.execute_run(
        device_label="GPU",
        executable=gpu_exe,
        device_argument="cuda",
        input_folder=input_folder,
        project_folder=gpu_project,
        log_path=gpu_log,
        run_kind="measured",
        run_index=1,
    )

    gpu_snapshot.parent.mkdir(parents=True, exist_ok=True)
    gpu_predictions.to_csv(
        gpu_snapshot,
        index=False,
        compression="gzip",
    )

    by_image, overall, class_rows = bench.compare_prediction_frames(
        cpu_predictions,
        gpu_predictions,
        left_label="CPU",
        right_label="GPU",
        pair_index=1,
    )

    by_image.to_csv(
        output_root / "cpu_gpu_equivalence_by_image.csv",
        index=False,
    )
    overall.to_csv(
        output_root / "cpu_gpu_equivalence_overall_by_pair.csv",
        index=False,
    )
    class_rows.to_csv(
        output_root / "cpu_gpu_class_count_differences.csv",
        index=False,
    )

    summary = bench.summarize_equivalence(overall)
    summary.to_csv(
        output_root / "cpu_gpu_equivalence_summary.csv",
        index=False,
    )

    pd.DataFrame([gpu_record]).to_csv(
        output_root / "gpu_resume_runtime.csv",
        index=False,
    )

    with pd.ExcelWriter(
        output_root / "GeoComb_CPU_GPU_equivalence_resumed.xlsx",
        engine="openpyxl",
    ) as writer:
        overall.to_excel(
            writer,
            sheet_name="CPU_GPU_Overall",
            index=False,
        )
        summary.to_excel(
            writer,
            sheet_name="CPU_GPU_Summary",
            index=False,
        )
        by_image.to_excel(
            writer,
            sheet_name="CPU_GPU_By_Image",
            index=False,
        )
        class_rows.to_excel(
            writer,
            sheet_name="Class_Count_Diffs",
            index=False,
        )
        pd.DataFrame([gpu_record]).to_excel(
            writer,
            sheet_name="GPU_Runtime",
            index=False,
        )

    eq = summary.iloc[0]
    passed = bool(eq["all_pairs_scientifically_equivalent"])

    summary_lines = [
        "GEOCOMB CPU/GPU EQUIVALENCE — RESUMED AFTER V1 CLEANUP FAILURE",
        "=" * 78,
        "CPU result: reused from completed CPU_measured_01.csv.gz",
        (
            "GPU runtime: "
            f"{float(gpu_record['wall_minutes']):.3f} min "
            f"({float(gpu_record['seconds_per_image']):.3f} s/image)"
        ),
        "",
        "SCIENTIFIC EQUIVALENCE",
        "-" * 78,
        (
            "Pooled relative total-count difference: "
            f"{100.0 * float(eq['median_pooled_relative_count_difference_vs_left']):.4f}%"
        ),
        (
            "CPU spatial match fraction: "
            f"{100.0 * float(eq['median_left_match_fraction']):.4f}%"
        ),
        (
            "GPU spatial match fraction: "
            f"{100.0 * float(eq['median_right_match_fraction']):.4f}%"
        ),
        (
            "Class agreement among matched cells: "
            f"{100.0 * float(eq['median_class_agreement_among_matches']):.4f}%"
        ),
        f"All paired runs passed: {passed}",
        "",
        "NOTE",
        "-" * 78,
        (
            "The original v1 failure occurred after the CPU run had completed "
            "and its prediction snapshot had been saved. It was caused by the "
            "benchmark reader retaining an SQLite connection on Windows; it "
            "was not a GeoComb analysis failure."
        ),
    ]

    (output_root / "benchmark_summary_resumed.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(summary_lines))

    # Explicitly closed SQLite handling in benchmark v2 normally allows this.
    # Cleanup remains non-fatal because all scientific outputs are already
    # saved.
    try:
        bench.remove_tree_with_retries(gpu_project)
    except Exception as exc:
        warning = (
            f"WARNING: GPU temporary project could not be deleted: {exc}. "
            "This does not affect the saved equivalence results."
        )
        print(warning)
        (output_root / "cleanup_warnings_resume.txt").write_text(
            warning + "\n",
            encoding="utf-8",
        )

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
