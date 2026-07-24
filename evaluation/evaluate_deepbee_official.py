#!/usr/bin/env python3
"""
Evaluate GeoComb on the 13-image official DeepBee test release.

Reference:
  deepbee_official_test_v1/resources/labels_test.csv

Predictions:
  DeepBee_official_test_3stage_addedManual_8class_predictions_v1/test/
    <image_id>/cell_predictions_3stage.csv

Coordinates in labels_test.csv are transformed from the raw-image coordinate
system to GeoComb's 5000-pixel frame-crop coordinate system using
frame_detection_crop_magenta_summary.csv.

Outputs include:
- detection_metrics_overall.csv
- detection_metrics_by_image.csv
- classification_accuracy_summary.csv
- classification_metrics_by_class.csv
- classification_confusion_matrix.csv
- end_to_end_class_metrics.csv
- count_error_by_image_class.csv
- count_error_summary.csv
- matched_predictions_all.csv
- published_deepbee_context.csv
- evaluation_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree


CLASSES7 = [
    "Egg",
    "Larvae",
    "Capped_brood",
    "Pollen",
    "Nectar",
    "Honey",
    "Other",
]

OUR_TO_7 = {
    "Egg": "Egg",
    "Larvae": "Larvae",
    "Capped_pupae": "Capped_brood",
    "Pollen": "Pollen",
    "Capped_honey": "Honey",
    "Uncapped_honey": "Nectar",
    "Empty_cell": "Other",
    "Unbuilt_cell": "Other",
}

DEEPBEE_ID_TO_7 = {
    0: "Capped_brood",
    1: "Egg",
    2: "Honey",
    3: "Larvae",
    4: "Nectar",
    5: "Other",
    6: "Pollen",
}

CLASS_NORMALIZATION = {
    "egg": "Egg",
    "eggs": "Egg",
    "larva": "Larvae",
    "larvae": "Larvae",
    "larves": "Larvae",
    "capped": "Capped_brood",
    "cappedbrood": "Capped_brood",
    "cappedpupae": "Capped_brood",
    "cappedpupa": "Capped_brood",
    "pollen": "Pollen",
    "nectar": "Nectar",
    "uncappedhoney": "Nectar",
    "honey": "Honey",
    "cappedhoney": "Honey",
    "other": "Other",
    "empty": "Other",
    "emptycell": "Other",
    "unbuilt": "Other",
    "unbuiltcell": "Other",
}

IMAGE_ALIASES = [
    "image name", "image_name", "filename", "file_name", "image",
    "raw_image", "source_image", "original_image", "image_id", "id",
]

X_ALIASES = [
    "x", "center_x", "x_center", "cx", "xcoord", "x_coordinate",
    "x_5000", "center_x_5000", "x_center_5000",
]
Y_ALIASES = [
    "y", "center_y", "y_center", "cy", "ycoord", "y_coordinate",
    "y_5000", "center_y_5000", "y_center_5000",
]
CLASS_NAME_ALIASES = [
    "class name", "class_name", "label_name", "category_name",
    "predicted_class", "pred_class", "final_class", "class",
]
CLASS_ID_ALIASES = [
    "class id", "class_id", "label_id", "category_id", "class",
]
RADIUS_ALIASES = ["radius", "r", "cell_radius"]

LEFT_ALIASES = [
    "crop_x1", "crop_x0", "bbox_x1", "bbox_x0", "left", "xmin",
    "x_min", "crop_left", "raw_crop_x1", "raw_bbox_x1",
]
TOP_ALIASES = [
    "crop_y1", "crop_y0", "bbox_y1", "bbox_y0", "top", "ymin",
    "y_min", "crop_top", "raw_crop_y1", "raw_bbox_y1",
]
RIGHT_ALIASES = [
    "crop_x2", "bbox_x2", "right", "xmax", "x_max", "crop_right",
    "raw_crop_x2", "raw_bbox_x2",
]
BOTTOM_ALIASES = [
    "crop_y2", "bbox_y2", "bottom", "ymax", "y_max", "crop_bottom",
    "raw_crop_y2", "raw_bbox_y2",
]
RAW_CROP_W_ALIASES = [
    "crop_width", "bbox_width", "crop_w", "raw_crop_width",
    "raw_bbox_width", "width_crop_raw",
]
RAW_CROP_H_ALIASES = [
    "crop_height", "bbox_height", "crop_h", "raw_crop_height",
    "raw_bbox_height", "height_crop_raw",
]


def keynorm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def namenorm(value: object) -> str:
    return keynorm(Path(str(value)).stem)


def find_col(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {keynorm(c): c for c in columns}
    for alias in aliases:
        hit = lookup.get(keynorm(alias))
        if hit is not None:
            return hit
    return None


def numeric(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        raise ValueError(f"Expected numeric value, got {value!r}")
    return float(result)


def normalize_ref_class(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        idx = int(float(text))
        if idx in DEEPBEE_ID_TO_7:
            return DEEPBEE_ID_TO_7[idx]
    except (TypeError, ValueError):
        pass
    return CLASS_NORMALIZATION.get(keynorm(text))


def normalize_our_class(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text in OUR_TO_7:
        return OUR_TO_7[text]
    return CLASS_NORMALIZATION.get(keynorm(text))


def identify_prediction_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    x_col = find_col(df.columns, X_ALIASES)
    y_col = find_col(df.columns, Y_ALIASES)
    class_col = find_col(df.columns, CLASS_NAME_ALIASES)

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not identify prediction x/y columns. Columns: {list(df.columns)}"
        )

    if class_col is None:
        best = None
        for col in df.columns:
            values = df[col].dropna().head(2000)
            if values.empty:
                continue
            ratio = values.map(normalize_our_class).notna().mean()
            score = ratio + (0.1 if "class" in keynorm(col) else 0)
            if ratio >= 0.8 and (best is None or score > best[0]):
                best = (score, col)
        class_col = best[1] if best else None

    if class_col is None:
        raise ValueError(
            f"Could not identify prediction class column. Columns: {list(df.columns)}"
        )

    return x_col, y_col, class_col


def load_labels(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)

    image_col = find_col(df.columns, IMAGE_ALIASES)
    x_col = find_col(df.columns, X_ALIASES)
    y_col = find_col(df.columns, Y_ALIASES)
    name_col = find_col(df.columns, CLASS_NAME_ALIASES)
    id_col = find_col(df.columns, CLASS_ID_ALIASES)
    radius_col = find_col(df.columns, RADIUS_ALIASES)

    if image_col is None or x_col is None or y_col is None:
        raise ValueError(
            f"Could not identify image/x/y columns in labels. Columns: {list(df.columns)}"
        )

    work = pd.DataFrame({
        "image_name": df[image_col].astype(str).str.strip(),
        "raw_x": pd.to_numeric(df[x_col], errors="coerce"),
        "raw_y": pd.to_numeric(df[y_col], errors="coerce"),
    })

    ref_class = None
    if name_col is not None and name_col != id_col:
        candidate = df[name_col].map(normalize_ref_class)
        if candidate.notna().mean() >= 0.8:
            ref_class = candidate

    if ref_class is None and id_col is not None:
        ref_class = df[id_col].map(normalize_ref_class)

    if ref_class is None:
        raise ValueError(
            f"Could not identify class labels. Columns: {list(df.columns)}"
        )

    work["ref_class7"] = ref_class
    work["radius"] = (
        pd.to_numeric(df[radius_col], errors="coerce")
        if radius_col is not None else np.nan
    )

    work = work.dropna(subset=["raw_x", "raw_y", "ref_class7"]).reset_index(drop=True)
    return work, image_col


def locate_prediction_csv(pred_root: Path, image_name: str) -> Path:
    target = namenorm(image_name)
    candidates = list(pred_root.rglob("cell_predictions_3stage.csv"))
    scored = []
    for path in candidates:
        parent_norm = namenorm(path.parent.name)
        score = 0
        if parent_norm == target:
            score = 100
        elif target in parent_norm or parent_norm in target:
            score = 80
        else:
            full_norm = namenorm(str(path.parent))
            if target in full_norm:
                score = 50
        if score:
            scored.append((score, len(str(path)), path))
    if not scored:
        raise FileNotFoundError(
            f"No prediction CSV matched {image_name!r} under {pred_root}"
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def locate_crop_image(crop_root: Path, image_name: str) -> Path:
    target = namenorm(image_name)
    candidates = [
        p for p in crop_root.rglob("*")
        if p.is_file() and p.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]
    scored = []
    for path in candidates:
        n = namenorm(path.name)
        if n == target:
            score = 100
        elif target in n or n in target:
            score = 70
        else:
            continue
        scored.append((score, len(str(path)), path))
    if not scored:
        raise FileNotFoundError(
            f"No 5000-pixel crop image matched {image_name!r} under {crop_root}"
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def select_summary_row(summary: pd.DataFrame, image_name: str) -> pd.Series:
    target = namenorm(image_name)
    best = None
    for idx, row in summary.iterrows():
        row_score = 0
        for value in row:
            if not isinstance(value, str):
                continue
            n = namenorm(value)
            if not n:
                continue
            if n == target:
                row_score = max(row_score, 100)
            elif target in n or n in target:
                row_score = max(row_score, 70)
        if row_score and (best is None or row_score > best[0]):
            best = (row_score, idx)
    if best is None:
        raise ValueError(
            f"Could not find crop-summary row for {image_name!r}. "
            f"Columns: {list(summary.columns)}"
        )
    return summary.loc[best[1]]


def value_from_aliases(row: pd.Series, aliases: Iterable[str]) -> float | None:
    col = find_col(row.index, aliases)
    if col is None:
        return None
    value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def get_crop_transform(row: pd.Series, crop_image: Path) -> tuple[float, float, float, float]:
    left = value_from_aliases(row, LEFT_ALIASES)
    top = value_from_aliases(row, TOP_ALIASES)
    right = value_from_aliases(row, RIGHT_ALIASES)
    bottom = value_from_aliases(row, BOTTOM_ALIASES)
    raw_w = value_from_aliases(row, RAW_CROP_W_ALIASES)
    raw_h = value_from_aliases(row, RAW_CROP_H_ALIASES)

    if left is None or top is None:
        raise ValueError(
            "Could not identify crop left/top columns in frame summary.\n"
            f"Available columns: {list(row.index)}"
        )

    if right is None and raw_w is not None:
        right = left + raw_w
    if bottom is None and raw_h is not None:
        bottom = top + raw_h

    if right is None or bottom is None:
        raise ValueError(
            "Could not identify crop right/bottom or raw crop width/height.\n"
            f"Available columns: {list(row.index)}"
        )

    crop_raw_w = right - left
    crop_raw_h = bottom - top
    if crop_raw_w <= 0 or crop_raw_h <= 0:
        raise ValueError(
            f"Invalid crop geometry: left={left}, top={top}, right={right}, bottom={bottom}"
        )

    with Image.open(crop_image) as im:
        out_w, out_h = im.size

    scale_x = out_w / crop_raw_w
    scale_y = out_h / crop_raw_h
    return left, top, scale_x, scale_y


def estimate_pitch(points: np.ndarray, radius_values: np.ndarray | None = None) -> float:
    if len(points) < 2:
        raise ValueError("At least two reference centers are required to estimate pitch.")

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if len(nearest) == 0:
        raise ValueError("Could not estimate pitch from reference centers.")

    q10, q90 = np.quantile(nearest, [0.10, 0.90])
    trimmed = nearest[(nearest >= q10) & (nearest <= q90)]
    pitch = float(np.median(trimmed if len(trimmed) else nearest))

    # Radius is used only as a sanity check because coordinate scaling can vary.
    return pitch


def one_to_one_match(
    ref_points: np.ndarray,
    pred_points: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if len(ref_points) == 0 or len(pred_points) == 0:
        return [], list(range(len(ref_points))), list(range(len(pred_points)))

    ref_tree = cKDTree(ref_points)
    candidate_pairs = []

    for pred_idx, point in enumerate(pred_points):
        ref_indices = ref_tree.query_ball_point(point, r=threshold)
        for ref_idx in ref_indices:
            distance = float(np.linalg.norm(point - ref_points[ref_idx]))
            candidate_pairs.append((distance, ref_idx, pred_idx))

    candidate_pairs.sort(key=lambda item: item[0])
    used_ref = set()
    used_pred = set()
    matches = []

    for distance, ref_idx, pred_idx in candidate_pairs:
        if ref_idx in used_ref or pred_idx in used_pred:
            continue
        used_ref.add(ref_idx)
        used_pred.add(pred_idx)
        matches.append((ref_idx, pred_idx, distance))

    unmatched_ref = [i for i in range(len(ref_points)) if i not in used_ref]
    unmatched_pred = [i for i in range(len(pred_points)) if i not in used_pred]
    return matches, unmatched_ref, unmatched_pred


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def class_metrics_from_rows(rows: pd.DataFrame, end_to_end: bool) -> pd.DataFrame:
    output = []
    for cls in CLASSES7:
        if end_to_end:
            correct = (
                rows["status"].eq("matched")
                & rows["ref_class7"].eq(cls)
                & rows["pred_class7"].eq(cls)
            )
            tp = int(correct.sum())
            fp = int((rows["pred_class7"].eq(cls) & ~correct).sum())
            fn = int((rows["ref_class7"].eq(cls) & ~correct).sum())
            support = int(rows["ref_class7"].eq(cls).sum())
        else:
            matched = rows[rows["status"].eq("matched")]
            correct = (
                matched["ref_class7"].eq(cls)
                & matched["pred_class7"].eq(cls)
            )
            tp = int(correct.sum())
            fp = int((matched["ref_class7"].ne(cls) & matched["pred_class7"].eq(cls)).sum())
            fn = int((matched["ref_class7"].eq(cls) & matched["pred_class7"].ne(cls)).sum())
            support = int(matched["ref_class7"].eq(cls).sum())

        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        output.append({
            "class": cls,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prediction_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--detection_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--threshold_pitch_fraction",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    labels_csv = args.labels_csv.expanduser().resolve()
    pred_root = args.prediction_root.expanduser().resolve()
    detection_root = args.detection_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        detection_root
        / "00_frame_crop_masked_resize_3000_5000_puremagenta_fromraw_v1"
        / "summaries"
        / "frame_detection_crop_magenta_summary.csv"
    )
    crop_root = (
        detection_root
        / "00_frame_crop_masked_resize_3000_5000_puremagenta_fromraw_v1"
        / "01_frame_crop_masked_magenta_5000"
        / "images"
    )

    for path in [labels_csv, pred_root, summary_csv, crop_root]:
        if not path.exists():
            raise FileNotFoundError(path)

    labels, _ = load_labels(labels_csv)
    crop_summary = pd.read_csv(summary_csv)

    detection_rows = []
    matched_rows = []

    image_names = sorted(labels["image_name"].unique(), key=str.casefold)
    print(f"[reference] {len(labels):,} labels across {len(image_names)} images")

    for image_name in image_names:
        ref = labels[labels["image_name"].eq(image_name)].copy().reset_index(drop=True)

        pred_csv = locate_prediction_csv(pred_root, image_name)
        crop_image = locate_crop_image(crop_root, image_name)
        summary_row = select_summary_row(crop_summary, image_name)
        left, top, scale_x, scale_y = get_crop_transform(summary_row, crop_image)

        ref["x"] = (ref["raw_x"] - left) * scale_x
        ref["y"] = (ref["raw_y"] - top) * scale_y

        pred_raw = pd.read_csv(pred_csv)
        px_col, py_col, pc_col = identify_prediction_columns(pred_raw)
        pred = pd.DataFrame({
            "x": pd.to_numeric(pred_raw[px_col], errors="coerce"),
            "y": pd.to_numeric(pred_raw[py_col], errors="coerce"),
            "pred_class7": pred_raw[pc_col].map(normalize_our_class),
        }).dropna(subset=["x", "y", "pred_class7"]).reset_index(drop=True)

        ref_points = ref[["x", "y"]].to_numpy(dtype=float)
        pred_points = pred[["x", "y"]].to_numpy(dtype=float)
        pitch = estimate_pitch(ref_points)
        threshold = args.threshold_pitch_fraction * pitch

        matches, unmatched_ref, unmatched_pred = one_to_one_match(
            ref_points, pred_points, threshold
        )

        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_ref)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        rel_count_error = safe_div(abs(len(pred) - len(ref)), len(ref))

        detection_rows.append({
            "image_name": image_name,
            "reference_cells": len(ref),
            "predicted_cells": len(pred),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall_cdr": recall,
            "f1": f1,
            "deepbee_formula_cdr_percent": recall * 100,
            "relative_total_count_error": rel_count_error,
            "estimated_pitch_5000px": pitch,
            "matching_threshold_5000px": threshold,
            "prediction_csv": str(pred_csv),
        })

        for ref_idx, pred_idx, distance in matches:
            matched_rows.append({
                "image_name": image_name,
                "status": "matched",
                "ref_index": ref_idx,
                "pred_index": pred_idx,
                "ref_x": ref.loc[ref_idx, "x"],
                "ref_y": ref.loc[ref_idx, "y"],
                "pred_x": pred.loc[pred_idx, "x"],
                "pred_y": pred.loc[pred_idx, "y"],
                "distance": distance,
                "ref_class7": ref.loc[ref_idx, "ref_class7"],
                "pred_class7": pred.loc[pred_idx, "pred_class7"],
            })

        for ref_idx in unmatched_ref:
            matched_rows.append({
                "image_name": image_name,
                "status": "false_negative",
                "ref_index": ref_idx,
                "pred_index": np.nan,
                "ref_x": ref.loc[ref_idx, "x"],
                "ref_y": ref.loc[ref_idx, "y"],
                "pred_x": np.nan,
                "pred_y": np.nan,
                "distance": np.nan,
                "ref_class7": ref.loc[ref_idx, "ref_class7"],
                "pred_class7": np.nan,
            })

        for pred_idx in unmatched_pred:
            matched_rows.append({
                "image_name": image_name,
                "status": "false_positive",
                "ref_index": np.nan,
                "pred_index": pred_idx,
                "ref_x": np.nan,
                "ref_y": np.nan,
                "pred_x": pred.loc[pred_idx, "x"],
                "pred_y": pred.loc[pred_idx, "y"],
                "distance": np.nan,
                "ref_class7": np.nan,
                "pred_class7": pred.loc[pred_idx, "pred_class7"],
            })

        print(
            f"[image] {image_name}: ref={len(ref)}, pred={len(pred)}, "
            f"TP={tp}, FP={fp}, FN={fn}, F1={f1:.4f}"
        )

    detection_by_image = pd.DataFrame(detection_rows)
    matched_all = pd.DataFrame(matched_rows)
    matched_all["correct_class"] = (
        matched_all["status"].eq("matched")
        & matched_all["ref_class7"].eq(matched_all["pred_class7"])
    )

    tp = int(detection_by_image["tp"].sum())
    fp = int(detection_by_image["fp"].sum())
    fn = int(detection_by_image["fn"].sum())
    precision, recall, f1 = f1_from_counts(tp, fp, fn)

    detection_overall = pd.DataFrame([{
        "method": "GeoComb",
        "n_images": len(detection_by_image),
        "n_reference": int(detection_by_image["reference_cells"].sum()),
        "n_predicted": int(detection_by_image["predicted_cells"].sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall_cdr_micro": recall,
        "f1_micro": f1,
        "mean_per_image_cdr": float(detection_by_image["recall_cdr"].mean()),
        "median_per_image_cdr": float(detection_by_image["recall_cdr"].median()),
        "mean_relative_total_count_error": float(
            detection_by_image["relative_total_count_error"].mean()
        ),
        "median_relative_total_count_error": float(
            detection_by_image["relative_total_count_error"].median()
        ),
    }])

    matched_only = matched_all[matched_all["status"].eq("matched")].copy()
    n_matched = len(matched_only)
    n_correct = int(matched_only["correct_class"].sum())
    accuracy = safe_div(n_correct, n_matched)
    e2e_correct = safe_div(
        int(matched_all["correct_class"].sum()),
        int(matched_all["ref_index"].notna().sum()),
    )

    class_matched = class_metrics_from_rows(matched_all, end_to_end=False)
    class_e2e = class_metrics_from_rows(matched_all, end_to_end=True)

    represented = class_matched[class_matched["support"] > 0]
    macro_f1 = float(represented["f1"].mean())
    weighted_f1 = float(
        np.average(represented["f1"], weights=represented["support"])
    )

    classification_summary = pd.DataFrame([{
        "method": "GeoComb",
        "matched_cells": n_matched,
        "correctly_classified_matched_cells": n_correct,
        "classification_accuracy_on_matched": accuracy,
        "macro_f1_excluding_zero_support": macro_f1,
        "weighted_f1_excluding_zero_support": weighted_f1,
        "end_to_end_correctly_localized_and_classified": int(
            matched_all["correct_class"].sum()
        ),
        "end_to_end_correct_rate_vs_reference": e2e_correct,
    }])

    confusion = pd.crosstab(
        matched_only["ref_class7"],
        matched_only["pred_class7"],
    ).reindex(index=CLASSES7, columns=CLASSES7, fill_value=0)
    confusion.insert(0, "reference_class", confusion.index)
    confusion = confusion.reset_index(drop=True)

    count_rows = []
    for image_name, group in matched_all.groupby("image_name", sort=False):
        for cls in CLASSES7:
            manual_count = int(
                (group["ref_index"].notna() & group["ref_class7"].eq(cls)).sum()
            )
            predicted_count = int(
                (group["pred_index"].notna() & group["pred_class7"].eq(cls)).sum()
            )
            error = predicted_count - manual_count
            abs_error = abs(error)
            ape = abs_error / manual_count if manual_count else np.nan
            count_rows.append({
                "image_name": image_name,
                "class": cls,
                "manual_count": manual_count,
                "predicted_count": predicted_count,
                "count_error": error,
                "absolute_count_error": abs_error,
                "absolute_percentage_error": ape,
            })
    count_by_image = pd.DataFrame(count_rows)

    count_summary_rows = []
    for cls, group in count_by_image.groupby("class", sort=False):
        valid_ape = group["absolute_percentage_error"].dropna()
        count_summary_rows.append({
            "class": cls,
            "manual_count_total": int(group["manual_count"].sum()),
            "predicted_count_total": int(group["predicted_count"].sum()),
            "mean_absolute_count_error": float(group["absolute_count_error"].mean()),
            "total_absolute_count_error": int(group["absolute_count_error"].sum()),
            "mean_absolute_percentage_error": (
                float(valid_ape.mean()) if len(valid_ape) else np.nan
            ),
        })
    count_summary = pd.DataFrame(count_summary_rows)

    normalized_errors = []
    for image_name, group in count_by_image.groupby("image_name", sort=False):
        denominator = float(group["manual_count"].sum())
        numerator = float(group["absolute_count_error"].sum())
        normalized_errors.append(numerator / denominator if denominator else np.nan)

    published_context = pd.DataFrame([
        {
            "metric": "Cell detection rate (CDR)",
            "published_value": 0.987,
            "published_percent": 98.7,
            "published_evaluation_unit": "Mean across 10 manually checked full-frame images",
            "our_comparable_value": float(detection_overall.loc[0, "mean_per_image_cdr"]),
            "directly_comparable": "Partially",
            "note": (
                "The formula is TP/manual count, equivalent to recall. "
                "However, the paper used a different set of 10 images."
            ),
        },
        {
            "metric": "Weighted F1 for cell classification",
            "published_value": 0.943,
            "published_percent": 94.3,
            "published_evaluation_unit": "53,914 pre-extracted test cell crops",
            "our_comparable_value": weighted_f1,
            "directly_comparable": "No; contextual only",
            "note": (
                "Our value is calculated only among cells localized by the full-frame "
                "workflow on the 13-image released set."
            ),
        },
        {
            "metric": "Cell classification accuracy",
            "published_value": 0.9431,
            "published_percent": 94.31,
            "published_evaluation_unit": "53,914 pre-extracted test cell crops",
            "our_comparable_value": accuracy,
            "directly_comparable": "No; contextual only",
            "note": (
                "The paper's classifier received reference-aligned cell crops; "
                "GeoComb matched-cell accuracy follows automatic center detection."
            ),
        },
        {
            "metric": "End-to-end correct localization and classification",
            "published_value": np.nan,
            "published_percent": np.nan,
            "published_evaluation_unit": "Not reported",
            "our_comparable_value": e2e_correct,
            "directly_comparable": "Not available",
            "note": "The paper did not report an equivalent end-to-end metric.",
        },
    ])

    files = {
        "detection_metrics_by_image.csv": detection_by_image,
        "detection_metrics_overall.csv": detection_overall,
        "classification_accuracy_summary.csv": classification_summary,
        "classification_metrics_by_class.csv": class_matched,
        "end_to_end_class_metrics.csv": class_e2e,
        "classification_confusion_matrix.csv": confusion,
        "count_error_by_image_class.csv": count_by_image,
        "count_error_summary.csv": count_summary,
        "matched_predictions_all.csv": matched_all,
        "published_deepbee_context.csv": published_context,
    }

    for filename, table in files.items():
        path = output_root / filename
        table.to_csv(path, index=False)
        print(f"[write] {path}")

    summary_text = f"""Official DeepBee test release: GeoComb evaluation
================================================================

Images: {len(detection_by_image)}
Reference cells: {int(detection_by_image['reference_cells'].sum()):,}
Predicted cells: {int(detection_by_image['predicted_cells'].sum()):,}

Detection
---------
TP: {tp:,}
FP: {fp:,}
FN: {fn:,}
Precision: {precision:.6f}
Recall/CDR (micro): {recall:.6f}
F1 (micro): {f1:.6f}
Mean per-image CDR: {detection_by_image['recall_cdr'].mean():.6f}
Mean relative total-count error: {detection_by_image['relative_total_count_error'].mean():.6f}

Classification among matched cells
----------------------------------
Matched cells: {n_matched:,}
Accuracy: {accuracy:.6f}
Macro F1 (represented classes): {macro_f1:.6f}
Weighted F1 (represented classes): {weighted_f1:.6f}

End-to-end
----------
Correctly localized and classified: {int(matched_all['correct_class'].sum()):,}
End-to-end correct rate: {e2e_correct:.6f}
Mean normalized absolute class-count error: {float(np.nanmean(normalized_errors)):.6f}
Median normalized absolute class-count error: {float(np.nanmedian(normalized_errors)):.6f}

Published DeepBee context
-------------------------
Reported CDR: 98.7% on a different 10-image detection evaluation.
Reported weighted classification F1: 94.3% on 53,914 pre-extracted test cell crops.
Reported classification accuracy: 94.31% on those cell crops.
These published values are contextual and are not direct head-to-head full-frame metrics.
"""
    (output_root / "evaluation_summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
