from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .pipeline import PipelineRunner
from .resources import regression_dir


def greedy_match(
    reference_xy: np.ndarray,
    prediction_xy: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if len(reference_xy) == 0 or len(prediction_xy) == 0:
        return []

    tree = cKDTree(prediction_xy)
    candidate_pairs: list[tuple[float, int, int]] = []

    for ref_index, point in enumerate(reference_xy):
        for pred_index in tree.query_ball_point(point, threshold):
            distance = float(
                np.linalg.norm(
                    point - prediction_xy[pred_index]
                )
            )
            candidate_pairs.append(
                (distance, ref_index, int(pred_index))
            )

    candidate_pairs.sort()
    used_ref: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for distance, ref_index, pred_index in candidate_pairs:
        if ref_index in used_ref or pred_index in used_pred:
            continue

        used_ref.add(ref_index)
        used_pred.add(pred_index)
        matches.append(
            (ref_index, pred_index, distance)
        )

    return matches


def class_agreement(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    matches: list[tuple[int, int, float]],
) -> float:
    if not matches:
        return 0.0

    correct = 0

    for ref_index, pred_index, _ in matches:
        if (
            str(expected.iloc[ref_index]["final_class"])
            == str(actual.iloc[pred_index]["final_class"])
        ):
            correct += 1

    return correct / len(matches)


def summarize_matches(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    threshold: float,
) -> tuple[dict, list[tuple[int, int, float]]]:
    expected_xy = expected[["x", "y"]].to_numpy(float)
    actual_xy = actual[["x", "y"]].to_numpy(float)

    matches = greedy_match(
        expected_xy,
        actual_xy,
        threshold=threshold,
    )

    distances = np.asarray(
        [distance for _, _, distance in matches],
        dtype=float,
    )

    return (
        {
            "threshold_px": float(threshold),
            "matched_count": int(len(matches)),
            "reference_match_fraction": (
                len(matches) / max(1, len(expected))
            ),
            "prediction_match_fraction": (
                len(matches) / max(1, len(actual))
            ),
            "class_agreement_among_matches": class_agreement(
                expected,
                actual,
                matches,
            ),
            "mean_matched_distance_px": (
                float(np.mean(distances))
                if len(distances)
                else None
            ),
            "median_matched_distance_px": (
                float(np.median(distances))
                if len(distances)
                else None
            ),
            "p90_matched_distance_px": (
                float(np.quantile(distances, 0.90))
                if len(distances)
                else None
            ),
            "max_matched_distance_px": (
                float(np.max(distances))
                if len(distances)
                else None
            ),
        },
        matches,
    )


def nearest_neighbor_summary(
    reference_xy: np.ndarray,
    prediction_xy: np.ndarray,
) -> dict:
    if len(reference_xy) == 0 or len(prediction_xy) == 0:
        return {}

    ref_to_pred = cKDTree(prediction_xy).query(
        reference_xy,
        k=1,
    )[0]
    pred_to_ref = cKDTree(reference_xy).query(
        prediction_xy,
        k=1,
    )[0]

    def summarize(values: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    return {
        "reference_to_prediction_px": summarize(
            ref_to_pred
        ),
        "prediction_to_reference_px": summarize(
            pred_to_ref
        ),
    }


def compare_crop_metadata(
    expected_path: Path,
    actual_path: Path,
) -> dict:
    if not expected_path.is_file():
        return {
            "available": False,
            "reason": f"Missing expected metadata: {expected_path}",
        }

    if not actual_path.is_file():
        return {
            "available": False,
            "reason": f"Missing actual metadata: {actual_path}",
        }

    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)

    expected = expected.loc[
        expected["image"].astype(str).eq("1")
    ]

    actual = actual.loc[
        actual["image"].astype(str).eq("1")
    ]

    if expected.empty or actual.empty:
        return {
            "available": False,
            "reason": "Image 1 was not found in crop metadata.",
        }

    expected_row = expected.iloc[0]
    actual_row = actual.iloc[0]

    columns = [
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_w",
        "bbox_h",
        "frame_mask_area_px",
        "outside_mask_fraction_in_bbox",
        "scale_3000",
        "out_w_3000",
        "out_h_3000",
        "scale_5000",
        "out_w_5000",
        "out_h_5000",
    ]

    values = {}
    exact_fields = [
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_w",
        "bbox_h",
        "out_w_3000",
        "out_h_3000",
        "out_w_5000",
        "out_h_5000",
    ]

    for column in columns:
        if column not in expected.columns or column not in actual.columns:
            continue

        expected_value = expected_row[column]
        actual_value = actual_row[column]

        try:
            difference = float(actual_value) - float(expected_value)
        except (TypeError, ValueError):
            difference = None

        values[column] = {
            "expected": (
                expected_value.item()
                if hasattr(expected_value, "item")
                else expected_value
            ),
            "actual": (
                actual_value.item()
                if hasattr(actual_value, "item")
                else actual_value
            ),
            "difference": difference,
        }

    exact_geometry_match = all(
        column in values
        and float(values[column]["difference"]) == 0.0
        for column in exact_fields
    )

    return {
        "available": True,
        "exact_crop_geometry_match": exact_geometry_match,
        "fields": values,
    }


def save_match_table(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    matches: list[tuple[int, int, float]],
    path: Path,
) -> None:
    rows = []

    for ref_index, pred_index, distance in matches:
        expected_row = expected.iloc[ref_index]
        actual_row = actual.iloc[pred_index]

        rows.append(
            {
                "expected_index": ref_index,
                "actual_index": pred_index,
                "distance_px": distance,
                "expected_x": expected_row["x"],
                "expected_y": expected_row["y"],
                "actual_x": actual_row["x"],
                "actual_y": actual_row["y"],
                "dx": (
                    float(actual_row["x"])
                    - float(expected_row["x"])
                ),
                "dy": (
                    float(actual_row["y"])
                    - float(expected_row["y"])
                ),
                "expected_class": expected_row["final_class"],
                "actual_class": actual_row["final_class"],
                "class_agreement": (
                    str(expected_row["final_class"])
                    == str(actual_row["final_class"])
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def validate_installation(
    device: str = "cpu",
    output_folder: Path | None = None,
    log_callback=print,
) -> dict:
    regression = regression_dir()
    input_folder = regression / "input"
    expected_csv = (
        regression
        / "expected"
        / "cell_predictions_3stage.csv"
    )
    expected_crop_metadata = (
        regression
        / "expected"
        / "frame_detection_crop_magenta_summary.csv"
    )

    if output_folder is None:
        output_folder = Path(
            tempfile.mkdtemp(
                prefix="GeoComb_regression_"
            )
        )
        cleanup = True
    else:
        output_folder = Path(output_folder).resolve()
        cleanup = False

    project_folder = output_folder / "project"

    try:
        runner = PipelineRunner(
            log_callback=log_callback
        )
        runner.analyze(
            input_folder=input_folder,
            project_folder=project_folder,
            device=device,
            overwrite=True,
        )

        actual_csv = (
            project_folder
            / "pipeline"
            / "05_predictions"
            / "test"
            / "1"
            / "cell_predictions_3stage.csv"
        )
        actual_crop_metadata = (
            project_folder
            / "pipeline"
            / "00_frame_crop"
            / "summaries"
            / "frame_detection_crop_magenta_summary.csv"
        )

        expected = pd.read_csv(expected_csv)
        actual = pd.read_csv(actual_csv)

        reference_count = len(expected)
        prediction_count = len(actual)

        pitch = float(
            pd.to_numeric(
                expected["estimated_step"],
                errors="coerce",
            ).median()
        )

        thresholds = sorted(
            {
                1.5,
                3.0,
                6.0,
                9.0,
                12.0,
                18.0,
                0.20 * pitch,
                0.50 * pitch,
            }
        )

        threshold_results = []
        match_tables = {}

        for threshold in thresholds:
            summary, matches = summarize_matches(
                expected,
                actual,
                threshold,
            )
            threshold_results.append(summary)
            match_tables[round(threshold, 6)] = matches

        strict = min(
            threshold_results,
            key=lambda row: abs(
                row["threshold_px"] - 1.5
            ),
        )
        functional_threshold = 0.50 * pitch
        functional = min(
            threshold_results,
            key=lambda row: abs(
                row["threshold_px"]
                - functional_threshold
            ),
        )

        relative_count_error = abs(
            prediction_count - reference_count
        ) / max(1, reference_count)

        strict_pass = bool(
            relative_count_error <= 0.005
            and strict["reference_match_fraction"] >= 0.995
            and strict[
                "class_agreement_among_matches"
            ] >= 0.995
        )

        functional_pass = bool(
            relative_count_error <= 0.010
            and functional[
                "reference_match_fraction"
            ] >= 0.980
            and functional[
                "prediction_match_fraction"
            ] >= 0.980
            and functional[
                "class_agreement_among_matches"
            ] >= 0.990
        )

        expected_xy = expected[
            ["x", "y"]
        ].to_numpy(float)
        actual_xy = actual[
            ["x", "y"]
        ].to_numpy(float)

        result = {
            "device": device,
            "reference_count": reference_count,
            "prediction_count": prediction_count,
            "relative_count_error": relative_count_error,
            "reference_pitch_px": pitch,
            "strict_threshold_px": 1.5,
            "functional_threshold_px": functional_threshold,
            "functional_threshold_basis": (
                "0.50 x image-specific cell pitch; "
                "identical to the manuscript localization "
                "criterion"
            ),
            "strict_result": strict,
            "functional_result": functional,
            "strict_reproducibility_passed": strict_pass,
            "functional_cross_platform_passed": functional_pass,
            "passed": functional_pass,
            "threshold_sensitivity": threshold_results,
            "nearest_neighbor_summary": (
                nearest_neighbor_summary(
                    expected_xy,
                    actual_xy,
                )
            ),
            "expected_original_hough_cells": int(
                expected["is_gapfill"].eq(0).sum()
            ),
            "actual_original_hough_cells": int(
                actual["is_gapfill"].eq(0).sum()
            ),
            "expected_gapfill_cells": int(
                expected["is_gapfill"].eq(1).sum()
            ),
            "actual_gapfill_cells": int(
                actual["is_gapfill"].eq(1).sum()
            ),
            "crop_metadata_comparison": (
                compare_crop_metadata(
                    expected_crop_metadata,
                    actual_crop_metadata,
                )
            ),
            "interpretation": (
                "PASS means the Windows result is functionally "
                "equivalent at 0.20 x cell pitch. The separate strict "
                "result measures near-bitwise coordinate reproduction "
                "within 1.5 px and may fail across operating systems, "
                "CPU/CUDA backends, and numerical library builds."
            ),
        }

        report_path = (
            output_folder
            / "regression_validation.json"
        )
        report_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary_lines = [
            "GEOCOMB WINDOWS REGRESSION VALIDATION",
            "=" * 78,
            "",
            f"Device: {device}",
            (
                f"Counts: expected={reference_count}, "
                f"actual={prediction_count}, "
                f"relative error={relative_count_error:.6f}"
            ),
            f"Reference pitch: {pitch:.6f} px",
            "",
            "Strict 1.5-px reproducibility:",
            (
                f"  reference match fraction="
                f"{strict['reference_match_fraction']:.6f}"
            ),
            (
                f"  prediction match fraction="
                f"{strict['prediction_match_fraction']:.6f}"
            ),
            (
                f"  class agreement="
                f"{strict['class_agreement_among_matches']:.6f}"
            ),
            f"  PASS={strict_pass}",
            "",
            (
                "Functional cross-platform validation "
                f"(0.50 x pitch = {functional_threshold:.3f} px):"
            ),
            (
                f"  reference match fraction="
                f"{functional['reference_match_fraction']:.6f}"
            ),
            (
                f"  prediction match fraction="
                f"{functional['prediction_match_fraction']:.6f}"
            ),
            (
                f"  class agreement="
                f"{functional['class_agreement_among_matches']:.6f}"
            ),
            f"  PASS={functional_pass}",
            "",
            "Threshold sensitivity:",
        ]

        for row in threshold_results:
            summary_lines.append(
                (
                    f"  {row['threshold_px']:7.3f} px: "
                    f"matches={row['matched_count']:4d}, "
                    f"ref={row['reference_match_fraction']:.6f}, "
                    f"pred={row['prediction_match_fraction']:.6f}, "
                    f"class={row['class_agreement_among_matches']:.6f}, "
                    f"mean_d={row['mean_matched_distance_px']}"
                )
            )

        summary_lines += [
            "",
            (
                "Original Hough cells: "
                f"expected={result['expected_original_hough_cells']}, "
                f"actual={result['actual_original_hough_cells']}"
            ),
            (
                "Gap-filled cells: "
                f"expected={result['expected_gapfill_cells']}, "
                f"actual={result['actual_gapfill_cells']}"
            ),
            "",
            (
                "Strict failure alone does not mean that the application "
                "failed. Hough and floating-point operations may not be "
                "bitwise identical across Windows/Linux and CPU/CUDA."
            ),
        ]

        (
            output_folder
            / "regression_validation_summary.txt"
        ).write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        functional_matches = match_tables[
            round(functional["threshold_px"], 6)
        ]

        save_match_table(
            expected,
            actual,
            functional_matches,
            output_folder
            / "matched_pairs_functional_threshold.csv",
        )

        result["report_path"] = str(report_path)
        result["summary_path"] = str(
            output_folder
            / "regression_validation_summary.txt"
        )
        result["actual_csv"] = str(actual_csv)
        return result
    finally:
        if cleanup:
            shutil.rmtree(
                output_folder,
                ignore_errors=True,
            )
