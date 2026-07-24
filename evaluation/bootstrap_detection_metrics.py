#!/usr/bin/env python3
"""Image-cluster bootstrap for paired detection results.

Input must contain one row per image and method with TP, FP, FN, and
image-level relative count error. Column aliases used by the public GeoComb
tables are recognized automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def choose(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Required column not found; expected one of {names}")


def pooled(frame: pd.DataFrame, tp: str, fp: str, fn: str, error: str) -> dict[str, float]:
    total_tp = frame[tp].sum()
    total_fp = frame[fp].sum()
    total_fn = frame[fn].sum()
    return {
        "precision": total_tp / (total_tp + total_fp),
        "recall": total_tp / (total_tp + total_fn),
        "f1": 2 * total_tp / (2 * total_tp + total_fp + total_fn),
        "mean_image_count_error": frame[error].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-method-ci", type=Path, required=True)
    parser.add_argument("--output-paired-ci", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--first-method", default="GeoComb")
    parser.add_argument("--second-method", default="DeepBee")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    dataset_col = "dataset" if "dataset" in data.columns else None
    image_col = choose(data, "image", "image_id")
    method_col = choose(data, "method")
    tp_col = choose(data, "tp", "TP")
    fp_col = choose(data, "fp", "FP")
    fn_col = choose(data, "fn", "FN")
    error_col = choose(data, "relative_count_error", "relative_total_count_error")

    if dataset_col is None:
        data["dataset"] = "FAIRHive detection-development"
        dataset_col = "dataset"

    rng = np.random.default_rng(args.seed)
    method_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for dataset in data[dataset_col].drop_duplicates():
        subset = data[data[dataset_col] == dataset]
        frames = {
            method: subset[subset[method_col] == method]
            .sort_values(image_col)
            .reset_index(drop=True)
            for method in (args.first_method, args.second_method)
        }
        if frames[args.first_method][image_col].tolist() != frames[args.second_method][image_col].tolist():
            raise ValueError(f"Methods do not contain the same ordered images for {dataset}")

        n_images = len(frames[args.first_method])
        indices = rng.integers(0, n_images, size=(args.replicates, n_images))
        samples_by_method: dict[str, dict[str, np.ndarray]] = {}

        for method, frame in frames.items():
            sampled_tp = frame[tp_col].to_numpy()[indices].sum(axis=1)
            sampled_fp = frame[fp_col].to_numpy()[indices].sum(axis=1)
            sampled_fn = frame[fn_col].to_numpy()[indices].sum(axis=1)
            sampled_error = frame[error_col].to_numpy()[indices].mean(axis=1)
            samples = {
                "precision": sampled_tp / (sampled_tp + sampled_fp),
                "recall": sampled_tp / (sampled_tp + sampled_fn),
                "f1": 2 * sampled_tp / (2 * sampled_tp + sampled_fp + sampled_fn),
                "mean_image_count_error": sampled_error,
            }
            samples_by_method[method] = samples
            points = pooled(frame, tp_col, fp_col, fn_col, error_col)

            for metric, values in samples.items():
                low, high = np.quantile(values, [0.025, 0.975])
                method_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "point_estimate": points[metric],
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_images": n_images,
                        "bootstrap_replicates": args.replicates,
                        "seed": args.seed,
                    }
                )

        first_points = pooled(frames[args.first_method], tp_col, fp_col, fn_col, error_col)
        second_points = pooled(frames[args.second_method], tp_col, fp_col, fn_col, error_col)
        for metric in ("precision", "recall", "f1", "mean_image_count_error"):
            differences = (
                samples_by_method[args.first_method][metric]
                - samples_by_method[args.second_method][metric]
            )
            low, high = np.quantile(differences, [0.025, 0.975])
            paired_rows.append(
                {
                    "dataset": dataset,
                    "contrast": f"{args.first_method} minus {args.second_method}",
                    "metric": metric,
                    "point_difference": first_points[metric] - second_points[metric],
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_images": n_images,
                    "bootstrap_replicates": args.replicates,
                    "seed": args.seed,
                }
            )

    args.output_method_ci.parent.mkdir(parents=True, exist_ok=True)
    args.output_paired_ci.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(method_rows).to_csv(args.output_method_ci, index=False)
    pd.DataFrame(paired_rows).to_csv(args.output_paired_ci, index=False)


if __name__ == "__main__":
    main()
