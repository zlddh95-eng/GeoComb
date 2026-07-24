#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

STAGE1_MAP = {
    "Egg": "Egg",
    "Larvae": "Larvae",
    "Capped_pupae": "Capped_group",
    "Capped_honey": "Capped_group",
    "Pollen": "Pollen",
    "Uncapped_honey": "Uncapped_honey",
    "Empty_cell": "Empty_cell",
    "Unbuilt_cell": "Unbuilt_cell",
}
STAGE2_CLASSES = {"Capped_honey", "Capped_pupae"}
STAGE3_CLASSES = {"Egg", "Empty_cell"}


def get_xy_cols(df: pd.DataFrame) -> Tuple[str, str]:
    for x, y in [("x", "y"), ("x_5000", "y_5000"), ("center_x", "center_y"), ("x_det", "y_det")]:
        if x in df.columns and y in df.columns:
            return x, y
    raise ValueError(f"No x/y columns found: {list(df.columns)}")


def get_label_col(df: pd.DataFrame) -> str:
    for c in ["manual_final_class", "corrected_class", "final_class", "display_class", "pred_class", "class"]:
        if c in df.columns:
            return c
    raise ValueError(f"No class column found: {list(df.columns)}")


def find_image(image_root: Path, split: str, csv_path: Path, df: pd.DataFrame) -> Optional[Path]:
    img_dir = image_root / split
    if "image" in df.columns and len(df) > 0:
        name = str(df["image"].iloc[0])
        if name and name.lower() != "nan":
            p = img_dir / name
            if p.exists():
                return p
            stem = Path(name).stem
            for ext in IMG_EXTS:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    return p
    stem = csv_path.parent.name
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(img_dir.glob(f"{stem}.*"))
    return hits[0] if hits else None


def crop_square(img: np.ndarray, x: float, y: float, half: float, out_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(round(x - half)); x1 = int(round(x + half))
    y0 = int(round(y - half)); y1 = int(round(y + half))
    if x1 <= x0: x1 = x0 + 1
    if y1 <= y0: y1 = y0 + 1
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)
    if any([pad_l, pad_t, pad_r, pad_b]):
        img2 = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    else:
        img2 = img
    crop = img2[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def save_crop(img, out_path: Path, x: float, y: float, step: float, half_step: float, out_size: int) -> None:
    half = max(2.0, float(step) * half_step)
    crop = crop_square(img, x, y, half, out_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def process_source(tag: str, image_root: Path, corr_root: Path, out_root: Path, split: str) -> list[dict]:
    rows = []
    csv_files = sorted((corr_root / split).glob("*/cell_predictions_3stage_corrected.csv"))
    print(f"[{tag}] corrected CSV files: {len(csv_files)}")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] failed to read {csv_path}: {e}")
            continue
        img_path = find_image(image_root, split, csv_path, df)
        if img_path is None:
            print(f"[WARN] image not found for {csv_path}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue
        xcol, ycol = get_xy_cols(df)
        labcol = get_label_col(df)
        if "estimated_step" in df.columns:
            steps = pd.to_numeric(df["estimated_step"], errors="coerce")
        elif "estimated_step_det" in df.columns:
            steps = pd.to_numeric(df["estimated_step_det"], errors="coerce")
        else:
            steps = pd.Series([np.nan] * len(df))
        med_step = float(np.nanmedian(steps)) if np.isfinite(np.nanmedian(steps)) else 50.0
        image_hash = hashlib.sha1(str(img_path).encode()).hexdigest()[:12]
        image_id = csv_path.parent.name
        for idx, r in df.iterrows():
            label = str(r.get(labcol, ""))
            if label not in STAGE1_MAP:
                continue
            try:
                x = float(r[xcol]); y = float(r[ycol])
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            step = float(steps.iloc[idx]) if idx < len(steps) and np.isfinite(steps.iloc[idx]) else med_step
            base_name = f"{tag}__{image_hash}__row{idx:06d}__{label}.jpg"
            # Stage 1: 7-class, capped cells merged.
            s1_label = STAGE1_MAP[label]
            s1_out = out_root / "stage1" / "train" / s1_label / base_name.replace(f"__{label}.jpg", f"__{s1_label}.jpg")
            save_crop(img, s1_out, x, y, step, 0.80, 320)
            rows.append({"source": tag, "stage": "stage1", "image_id": image_id, "row": idx, "label": s1_label, "path": str(s1_out)})
            # Stage 2: capped honey vs capped pupae.
            if label in STAGE2_CLASSES:
                s2_out = out_root / "stage2" / "train" / label / base_name
                save_crop(img, s2_out, x, y, step, 1.80, 320)
                rows.append({"source": tag, "stage": "stage2", "image_id": image_id, "row": idx, "label": label, "path": str(s2_out)})
            # Stage 3: egg vs empty.
            if label in STAGE3_CLASSES:
                s3_out = out_root / "stage3" / "train" / label / base_name
                save_crop(img, s3_out, x, y, step, 0.45, 384)
                rows.append({"source": tag, "stage": "stage3", "image_id": image_id, "row": idx, "label": label, "path": str(s3_out)})
        print(f"[OK] {tag} {img_path.name}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Project data root containing the corrected DeepBee training images.",
    )
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    base = args.base
    out_root = args.out_root
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    sources = [
        ("new4", base / "new_images_4_current_detection_final_hardEggLarvae_v1/00_frame_crop_masked_resize_3000_5000_puremagenta_fromraw_v1/01_frame_crop_masked_magenta_5000/images", base / "new_images_4_manual_corrections_v1"),
        ("new5", base / "new_images_5_current_detection_final_hardEggLarvae_v1/00_frame_crop_masked_resize_3000_5000_puremagenta_fromraw_v1/01_frame_crop_masked_magenta_5000/images", base / "new_images_5_manual_corrections_v1"),
    ]
    all_rows = []
    for tag, image_root, corr_root in sources:
        if not image_root.exists():
            print(f"[WARN] missing image_root for {tag}: {image_root}")
            continue
        if not corr_root.exists():
            print(f"[WARN] missing correction root for {tag}: {corr_root}")
            continue
        all_rows.extend(process_source(tag, image_root, corr_root, out_root, "test"))
    manifest = out_root / "manual_crops_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "stage", "image_id", "row", "label", "path"])
        w.writeheader(); w.writerows(all_rows)
    # counts
    counts = []
    for stage in ["stage1", "stage2", "stage3"]:
        for cls_dir in sorted((out_root / stage / "train").glob("*")) if (out_root / stage / "train").exists() else []:
            if cls_dir.is_dir():
                counts.append({"stage": stage, "class": cls_dir.name, "n": len(list(cls_dir.glob("*.jpg")))})
    pd.DataFrame(counts).to_csv(out_root / "manual_crops_counts.csv", index=False)
    print("\nDONE")
    print("manual crop root:", out_root)
    print("manifest:", manifest)
    print(pd.DataFrame(counts).to_string(index=False) if counts else "No crops created")

if __name__ == "__main__":
    main()
