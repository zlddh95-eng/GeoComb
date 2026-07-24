#!/usr/bin/env python3
"""
Draw outline-only overlays with an in-image legend for bee-comb cell predictions.

Input:
  - 5000-px frame-cropped images
  - cell_predictions_3stage.csv files from apply_3stage_current_detection_5000coords_v1.py

Output:
  - full-resolution overlay images
  - class-colored circle outlines only; centers remain visible
  - no per-cell text labels
  - legend embedded in one image corner
  - counts_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


CLASS_ORDER = [
    "Egg",
    "Larvae",
    "Capped_pupae",
    "Pollen",
    "Capped_honey",
    "Uncapped_honey",
    "Empty_cell",
    "Unbuilt_cell",
    "Unknown",
]

COLORS_BGR: Dict[str, tuple[int, int, int]] = {
    "Egg": (0, 255, 255),             # yellow
    "Larvae": (255, 255, 255),        # white
    "Capped_pupae": (255, 0, 255),    # magenta
    "Pollen": (0, 165, 255),          # orange
    "Capped_honey": (0, 215, 255),    # gold
    "Uncapped_honey": (0, 255, 0),    # green
    "Empty_cell": (255, 0, 0),        # blue
    "Unbuilt_cell": (160, 160, 160),  # gray
    "Unknown": (0, 0, 255),           # red
}


def find_image(image_root: Path, split: str, csv_path: Path, df: pd.DataFrame) -> Optional[Path]:
    img_dir = image_root / split
    if "image" in df.columns and len(df) > 0:
        name = str(df["image"].iloc[0])
        if name and name.lower() != "nan":
            p = img_dir / name
            if p.exists():
                return p
            stem = Path(name).stem
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    return p

    stem = csv_path.parent.name
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p

    candidates = list(img_dir.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def get_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    for xcol, ycol in [("x", "y"), ("x_5000", "y_5000"), ("center_x", "center_y"), ("x_det", "y_det")]:
        if xcol in df.columns and ycol in df.columns:
            return xcol, ycol
    raise ValueError(f"No usable x/y columns found. Columns={list(df.columns)}")


def get_class_column(df: pd.DataFrame) -> str:
    for c in ["final_class", "manual_final_class", "corrected_class", "display_class", "pred_class", "class"]:
        if c in df.columns:
            return c
    raise ValueError(f"No usable class column found. Columns={list(df.columns)}")


def draw_legend(img: np.ndarray, counts: Dict[str, int], position: str, scale: float, alpha: float) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(scale)
    thick = max(1, int(round(scale * 2)))

    pad = int(round(18 * scale))
    row_h = int(round(34 * scale))
    swatch = int(round(20 * scale))
    title_h = int(round(34 * scale))

    labels = [cls for cls in CLASS_ORDER if counts.get(cls, 0) > 0 or cls != "Unknown"]
    texts = [f"{cls}: {counts.get(cls, 0)}" for cls in labels]

    max_text_w = 0
    for t in ["Legend"] + texts:
        (tw, _), _ = cv2.getTextSize(t, font, font_scale, thick)
        max_text_w = max(max_text_w, tw)

    box_w = pad * 3 + swatch + max_text_w
    box_h = pad * 2 + title_h + row_h * len(labels)
    margin = int(round(24 * scale))

    if position == "top_left":
        x0, y0 = margin, margin
    elif position == "bottom_left":
        x0, y0 = margin, h - box_h - margin
    elif position == "bottom_right":
        x0, y0 = w - box_w - margin, h - box_h - margin
    else:
        x0, y0 = w - box_w - margin, margin

    x0 = max(0, min(x0, w - box_w))
    y0 = max(0, min(y0, h - box_h))
    x1, y1 = x0 + box_w, y0 + box_h

    layer = out.copy()
    cv2.rectangle(layer, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(layer, alpha, out, 1 - alpha, 0, dst=out)
    cv2.rectangle(out, (x0, y0), (x1, y1), (240, 240, 240), max(1, thick))

    y = y0 + pad + int(round(20 * scale))
    cv2.putText(out, "Legend", (x0 + pad, y), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    y = y0 + pad + title_h
    for cls in labels:
        color = COLORS_BGR.get(cls, COLORS_BGR["Unknown"])
        y_mid = y + row_h // 2
        sx0 = x0 + pad
        sy0 = y_mid - swatch // 2
        sx1 = sx0 + swatch
        sy1 = sy0 + swatch
        cv2.rectangle(out, (sx0, sy0), (sx1, sy1), color, -1)
        cv2.rectangle(out, (sx0, sy0), (sx1, sy1), (255, 255, 255), 1)
        cv2.putText(out, f"{cls}: {counts.get(cls, 0)}",
                    (x0 + pad * 2 + swatch, y_mid + int(round(7 * scale))),
                    font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += row_h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True, type=Path)
    ap.add_argument("--pred_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--csv_name", default="cell_predictions_3stage.csv")
    ap.add_argument("--radius_ratio", type=float, default=0.46)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--legend_position", default="top_right",
                    choices=["top_right", "top_left", "bottom_right", "bottom_left"])
    ap.add_argument("--legend_scale", type=float, default=0.75)
    ap.add_argument("--legend_alpha", type=float, default=0.72)
    ap.add_argument("--preview_width", type=int, default=0, help="0 keeps full resolution")
    args = ap.parse_args()

    out_root = args.out_root
    overlay_dir = out_root / "overlays_outline_legend"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted((args.pred_root / args.split).glob(f"*/{args.csv_name}"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found: {args.pred_root / args.split}/*/{args.csv_name}")

    summary_rows = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        img_path = find_image(args.image_root, args.split, csv_path, df)
        if img_path is None:
            print(f"[WARN] image not found for {csv_path}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        xcol, ycol = get_xy_columns(df)
        cls_col = get_class_column(df)

        if "estimated_step" in df.columns:
            steps = pd.to_numeric(df["estimated_step"], errors="coerce")
        elif "estimated_step_det" in df.columns:
            steps = pd.to_numeric(df["estimated_step_det"], errors="coerce")
        else:
            steps = pd.Series([np.nan] * len(df))

        median_step = float(np.nanmedian(steps)) if np.isfinite(np.nanmedian(steps)) else 50.0
        out = img.copy()
        h, w = out.shape[:2]
        counts = {cls: 0 for cls in CLASS_ORDER}

        for i, row in df.iterrows():
            try:
                x = float(row[xcol]); y = float(row[ycol])
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            cls = str(row.get(cls_col, "Unknown"))
            if cls not in COLORS_BGR:
                cls = "Unknown"

            step = steps.iloc[i] if i < len(steps) else np.nan
            if not np.isfinite(step) or step <= 0:
                step = median_step

            r = int(round(float(step) * args.radius_ratio))
            r = max(3, min(r, 100))
            xi = int(round(x)); yi = int(round(y))
            if xi < -r or xi >= w + r or yi < -r or yi >= h + r:
                continue

            cv2.circle(out, (xi, yi), r, COLORS_BGR[cls], thickness=args.thickness, lineType=cv2.LINE_AA)
            counts[cls] += 1

        out = draw_legend(out, counts, args.legend_position, args.legend_scale, args.legend_alpha)

        if args.preview_width and args.preview_width > 0 and w > args.preview_width:
            scale = args.preview_width / float(w)
            out = cv2.resize(out, (args.preview_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

        stem = Path(img_path).stem
        out_img = overlay_dir / f"{stem}_outline_legend_overlay.jpg"
        cv2.imwrite(str(out_img), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        row = {
            "image": img_path.name,
            "prediction_csv": str(csv_path),
            "overlay": str(out_img),
            "total_cells": sum(counts.values()),
            "median_estimated_step": round(median_step, 3),
            "radius_ratio": args.radius_ratio,
            "thickness": args.thickness,
        }
        row.update(counts)
        summary_rows.append(row)

        print(f"[OK] {img_path.name}: cells={row['total_cells']} overlay={out_img}")

    pd.DataFrame(summary_rows).to_csv(out_root / "counts_summary.csv", index=False)
    with (out_root / "legend_colors.csv").open("w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["class", "B", "G", "R"])
        for cls in CLASS_ORDER:
            ww.writerow([cls, *COLORS_BGR[cls]])

    print("\nDONE")
    print("Output:", out_root)
    print("Overlays:", overlay_dir)
    print("Counts:", out_root / "counts_summary.csv")


if __name__ == "__main__":
    main()
