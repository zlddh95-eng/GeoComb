#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import math

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def load_seed(input_root, split, stem):
    input_root = Path(input_root)

    candidates = [
        input_root / "07_alternating_regularized_localhex_seed" / split / stem / "cell_predictions_8class.csv",
        input_root / "06_localhex_gapfilled_seed" / split / stem / "cell_predictions_8class.csv",
        input_root / "05b_pitch_regularized_seed" / split / stem / "cell_predictions_8class.csv",
        input_root / "05_pitch_clean_seed" / split / stem / "cell_predictions_8class.csv",
    ]

    for p in candidates:
        if p.exists():
            return pd.read_csv(p), p

    found = sorted(input_root.glob(f"**/{stem}/cell_predictions_8class.csv"))
    if found:
        return pd.read_csv(found[0]), found[0]

    return None, candidates[0]


def make_pure_magenta_mask(img_bgr, tolerance=3):
    img = img_bgr.astype(np.int16)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    magenta = (
        (np.abs(b - 255) <= tolerance)
        & (np.abs(g - 0) <= tolerance)
        & (np.abs(r - 255) <= tolerance)
    )

    return magenta.astype(np.uint8)


def circle_bad_fraction(mask, x, y, radius):
    h, w = mask.shape[:2]

    x = float(x)
    y = float(y)
    radius = float(radius)

    x1 = max(0, int(math.floor(x - radius)))
    x2 = min(w, int(math.ceil(x + radius)) + 1)
    y1 = max(0, int(math.floor(y - radius)))
    y2 = min(h, int(math.ceil(y + radius)) + 1)

    if x2 <= x1 or y2 <= y1:
        return 1.0, 0, 0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    circle = ((xx - x) ** 2 + (yy - y) ** 2) <= (radius ** 2)

    total = int(circle.sum())
    if total <= 0:
        return 1.0, 0, 0

    bad_n = int((circle & (mask[y1:y2, x1:x2] > 0)).sum())
    return bad_n / total, bad_n, total


def apply_magenta_cut(df, magenta_mask, fixed_radius_ratio=0.48, max_magenta_frac=0.30):
    work = df.copy().reset_index(drop=True)

    if "pitch" in work.columns and pd.to_numeric(work["pitch"], errors="coerce").notna().any():
        pitch = float(pd.to_numeric(work["pitch"], errors="coerce").dropna().median())
    else:
        pitch = np.nan

    if "fixed_r" in work.columns and pd.to_numeric(work["fixed_r"], errors="coerce").notna().any():
        fixed_r = float(pd.to_numeric(work["fixed_r"], errors="coerce").dropna().median())
    elif np.isfinite(pitch):
        fixed_r = float(fixed_radius_ratio) * pitch
    else:
        fixed_r = 18.0

    fracs = []
    bad_pixels = []
    total_pixels = []

    for _, row in work.iterrows():
        frac, bad_n, total_n = circle_bad_fraction(magenta_mask, row["x"], row["y"], fixed_r)
        fracs.append(frac)
        bad_pixels.append(bad_n)
        total_pixels.append(total_n)

    work["pure_magenta_frac"] = fracs
    work["pure_magenta_pixels"] = bad_pixels
    work["circle_total_pixels"] = total_pixels
    work["magenta_removed"] = work["pure_magenta_frac"] > float(max_magenta_frac)

    kept = work[~work["magenta_removed"]].copy().reset_index(drop=True)
    removed = work[work["magenta_removed"]].copy().reset_index(drop=True)

    kept["kept_rank_after_magenta_cut"] = np.arange(len(kept))

    return work, kept, removed, fixed_r, pitch


def draw_overlay(img, kept, removed, magenta_mask, out_path, title, preview_w=2200):
    canvas = img.copy()

    tint = canvas.copy()
    tint[magenta_mask > 0] = (255, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.82, tint, 0.18, 0)

    for _, row in removed.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["fixed_r"]))) if "fixed_r" in row and pd.notna(row["fixed_r"]) else 18
        cv2.circle(canvas, (x, y), max(1, r), (0, 80, 255), 2)
        cv2.circle(canvas, (x, y), 2, (0, 80, 255), -1)

    if "is_gapfill" in kept.columns:
        orig = kept[kept["is_gapfill"].fillna(0).astype(int) == 0]
        gap = kept[kept["is_gapfill"].fillna(0).astype(int) == 1]
    else:
        orig = kept
        gap = kept.iloc[0:0]

    for _, row in orig.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["fixed_r"]))) if "fixed_r" in row and pd.notna(row["fixed_r"]) else 18
        cv2.circle(canvas, (x, y), max(1, r), (0, 255, 0), 1)
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

    for _, row in gap.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["fixed_r"]))) if "fixed_r" in row and pd.notna(row["fixed_r"]) else 18
        cv2.circle(canvas, (x, y), max(1, r), (255, 255, 0), 1)
        cv2.circle(canvas, (x, y), 2, (255, 255, 0), -1)

    cv2.rectangle(canvas, (10, 10), (min(canvas.shape[1] - 1, 1900), 76), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        title[:220],
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    if preview_w and canvas.shape[1] > preview_w:
        scale = preview_w / canvas.shape[1]
        canvas = cv2.resize(canvas, (preview_w, int(round(canvas.shape[0] * scale))))

    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--split", default="new")

    ap.add_argument("--fixed_radius_ratio", type=float, default=0.48)
    ap.add_argument("--max_magenta_frac", type=float, default=0.30)
    ap.add_argument("--magenta_tolerance", type=int, default=3)

    ap.add_argument("--preview_w", type=int, default=2200)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    per_root = out_root / "per_image"
    seed_root = out_root / "08_pure_magenta_cut_seed" / args.split
    overlay_root = out_root / "_COLLECTED_pure_magenta_cut_overlays"
    mask_root = out_root / "_COLLECTED_pure_magenta_masks"
    summary_root = out_root / "summaries"

    for d in [per_root, seed_root, overlay_root, mask_root, summary_root]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []

    print("[INFO] Pure magenta cut after Step 2")
    print("[INFO] near-white detection: OFF")
    print("[INFO] only pure magenta background is removed")

    for idx, img_path in enumerate(list_images(image_dir), 1):
        stem = img_path.stem
        print(f"\n[{idx}] {stem}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print("  [WARN] cannot read image")
            continue

        seed_df, seed_csv = load_seed(input_root, args.split, stem)

        if seed_df is None:
            print("  [SKIP] missing seed:", seed_csv)
            continue

        magenta_mask = make_pure_magenta_mask(img, tolerance=args.magenta_tolerance)

        full, kept, removed, fixed_r, pitch = apply_magenta_cut(
            seed_df,
            magenta_mask,
            fixed_radius_ratio=args.fixed_radius_ratio,
            max_magenta_frac=args.max_magenta_frac,
        )

        per_dir = per_root / stem
        per_dir.mkdir(parents=True, exist_ok=True)

        full.to_csv(per_dir / "01_pure_magenta_filter_full.csv", index=False)
        kept.to_csv(per_dir / "02_pure_magenta_filter_kept.csv", index=False)
        removed.to_csv(per_dir / "03_pure_magenta_filter_removed.csv", index=False)

        out_seed_dir = seed_root / stem
        out_seed_dir.mkdir(parents=True, exist_ok=True)
        final_csv = out_seed_dir / "cell_predictions_8class.csv"
        kept.to_csv(final_csv, index=False)

        cv2.imwrite(str(mask_root / f"{stem}_pure_magenta_mask.png"), (magenta_mask * 255).astype(np.uint8))

        kept_original_n = int((kept["is_gapfill"].fillna(0).astype(int) == 0).sum()) if "is_gapfill" in kept.columns and len(kept) else len(kept)
        kept_gapfill_n = int((kept["is_gapfill"].fillna(0).astype(int) == 1).sum()) if "is_gapfill" in kept.columns and len(kept) else 0
        removed_original_n = int((removed["is_gapfill"].fillna(0).astype(int) == 0).sum()) if "is_gapfill" in removed.columns and len(removed) else 0
        removed_gapfill_n = int((removed["is_gapfill"].fillna(0).astype(int) == 1).sum()) if "is_gapfill" in removed.columns and len(removed) else 0

        overlay_path = overlay_root / f"{stem}_pure_magenta_cut.jpg"
        title = (
            f"{stem} | pure magenta cut | input={len(seed_df)} "
            f"kept={len(kept)} removed={len(removed)} "
            f"fixedR={fixed_r:.2f} thr={args.max_magenta_frac:.2f}"
        )

        draw_overlay(img, kept, removed, magenta_mask, overlay_path, title, preview_w=args.preview_w)

        print(
            f"  input={len(seed_df):5d} kept={len(kept):5d} removed={len(removed):4d} "
            f"orig_rm={removed_original_n:4d} gap_rm={removed_gapfill_n:4d} "
            f"fixedR={fixed_r:.2f}"
        )

        rows.append({
            "image": stem,
            "input_seed_n": len(seed_df),
            "kept_seed_n": len(kept),
            "removed_pure_magenta_n": len(removed),
            "removed_fraction": len(removed) / max(1, len(seed_df)),
            "kept_original_n": kept_original_n,
            "kept_gapfill_n": kept_gapfill_n,
            "removed_original_n": removed_original_n,
            "removed_gapfill_n": removed_gapfill_n,
            "pitch": pitch,
            "fixed_r": fixed_r,
            "max_magenta_frac": args.max_magenta_frac,
            "magenta_tolerance": args.magenta_tolerance,
            "input_csv": str(seed_csv),
            "final_seed_csv": str(final_csv),
            "overlay": str(overlay_path),
        })

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "pure_magenta_cut_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[DONE]")
    print("summary:", summary_csv)
    print("seed root:", seed_root)
    print("overlay root:", overlay_root)


if __name__ == "__main__":
    main()
