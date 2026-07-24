#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import math

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def load_old_hough_summary(old_hough_root, stem):
    old_hough_root = Path(old_hough_root)
    p = old_hough_root / "summaries" / "lattice_auto_radius_hough_summary.csv"

    if not p.exists():
        return {}

    try:
        df = pd.read_csv(p)
        if "image" not in df.columns:
            return {}

        df["_stem"] = df["image"].astype(str).map(lambda x: Path(x).stem)
        m = df[df["_stem"] == stem]

        if len(m) == 0:
            return {}

        row = m.iloc[0].to_dict()
        return row
    except Exception:
        return {}


def estimate_pitch_from_centers(points, old_df=None, old_summary=None):
    """
    Estimate pitch from center-to-center neighbor distances.

    Hough r is NOT used for filtering.
    If center-distance estimation is weak, Hough r/0.46 is used only as fallback.
    """
    n = len(points)

    fallback_pitch = None
    fallback_source = "none"

    if old_df is not None and "r" in old_df.columns and len(old_df):
        med_r = float(pd.to_numeric(old_df["r"], errors="coerce").dropna().median())
        if np.isfinite(med_r) and med_r > 0:
            fallback_pitch = med_r / 0.46
            fallback_source = "fallback_hough_r_median_over_046"

    if old_summary is not None:
        for k in ["radius_median", "best_radius"]:
            if k in old_summary and pd.notna(old_summary[k]):
                rr = float(old_summary[k])
                if np.isfinite(rr) and rr > 0:
                    fallback_pitch = rr / 0.46
                    fallback_source = f"fallback_summary_{k}_over_046"
                    break

    if n < 20:
        if fallback_pitch is not None:
            return float(fallback_pitch), fallback_source
        return 38.0, "fallback_default_38"

    if cKDTree is not None:
        tree = cKDTree(points)
        k = min(10, n)
        dists, _ = tree.query(points, k=k)
        vals = dists[:, 1:].reshape(-1)
    else:
        vals = []
        step = max(1, n // 1200)
        sample = points[::step]
        for p in sample:
            d = np.sqrt(((points - p) ** 2).sum(axis=1))
            d = np.sort(d[d > 1e-6])[:9]
            vals.extend(d.tolist())
        vals = np.asarray(vals, dtype=float)

    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 5]

    if len(vals) < 30:
        if fallback_pitch is not None:
            return float(fallback_pitch), fallback_source
        return 38.0, "fallback_default_38_insufficient_dist"

    # Use fallback pitch only to set a loose plausible window if available.
    if fallback_pitch is not None and np.isfinite(fallback_pitch):
        low = max(8.0, fallback_pitch * 0.60)
        high = fallback_pitch * 1.55
    else:
        q20, q80 = np.quantile(vals, [0.20, 0.80])
        low = max(8.0, q20 * 0.80)
        high = q80 * 1.25

    vals2 = vals[(vals >= low) & (vals <= high)]

    if len(vals2) < 30:
        vals2 = vals[(vals >= np.quantile(vals, 0.25)) & (vals <= np.quantile(vals, 0.75))]

    if len(vals2) < 30:
        if fallback_pitch is not None:
            return float(fallback_pitch), fallback_source
        return float(np.median(vals)), "neighbor_distance_median_broad"

    # Histogram mode, then local median around the peak.
    bin_w = 1.0
    bins = np.arange(max(1.0, vals2.min()), vals2.max() + bin_w, bin_w)

    if len(bins) < 5:
        return float(np.median(vals2)), "neighbor_distance_median_window"

    hist, edges = np.histogram(vals2, bins=bins)

    # light smoothing
    if len(hist) >= 5:
        hist_smooth = np.convolve(hist, np.array([1, 2, 3, 2, 1], dtype=float), mode="same")
    else:
        hist_smooth = hist.astype(float)

    peak_i = int(np.argmax(hist_smooth))
    peak = 0.5 * (edges[peak_i] + edges[peak_i + 1])

    local_width = max(2.0, peak * 0.08)
    local_vals = vals2[(vals2 >= peak - local_width) & (vals2 <= peak + local_width)]

    if len(local_vals) >= 10:
        return float(np.median(local_vals)), "neighbor_distance_hist_peak_local_median"

    return float(peak), "neighbor_distance_hist_peak"


def fixed_equal_radius_overlap_matrix(points, fixed_r):
    """
    Equal-radius circle overlap area matrix.
    Diagonal = 0.
    """
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)

    r = float(fixed_r)
    area = np.zeros_like(d, dtype=np.float64)

    if r <= 0:
        return area

    # identical centers
    m0 = d <= 1e-9
    area[m0] = math.pi * r * r

    # partial overlap
    m = (d > 1e-9) & (d < 2.0 * r)

    if np.any(m):
        dm = d[m]
        xval = np.clip(dm / (2.0 * r), -1.0, 1.0)
        area[m] = (
            2.0 * r * r * np.arccos(xval)
            - 0.5 * dm * np.sqrt(np.maximum(4.0 * r * r - dm * dm, 0.0))
        )

    np.fill_diagonal(area, 0.0)
    return area


def compute_local_support(points, pitch):
    """
    Support count used only for tie-breaking.
    """
    n = len(points)
    support = np.zeros(n, dtype=int)

    if n == 0:
        return support

    low = 0.70 * pitch
    high = 1.35 * pitch

    if cKDTree is not None:
        tree = cKDTree(points)
        for i, p in enumerate(points):
            idxs = tree.query_ball_point(p, high)
            cnt = 0
            for j in idxs:
                if i == j:
                    continue
                d = float(np.linalg.norm(points[j] - p))
                if low <= d <= high:
                    cnt += 1
            support[i] = cnt
    else:
        for i, p in enumerate(points):
            d = np.sqrt(((points - p) ** 2).sum(axis=1))
            support[i] = int(((d >= low) & (d <= high)).sum())

    return support


def iterative_remove_overlap_pitch(df, pitch, fixed_r, overlap_thr=0.30):
    """
    Remove one cell at a time based on total overlap area fraction.

    overlap_frac_i =
      sum(overlap area between i and all active neighbors) / circle_area

    Rule:
      if max overlap_frac >= threshold:
          remove one cell with the largest overlap
          recompute by subtracting its contribution
      repeat
    """
    if len(df) == 0:
        return df.copy(), df.copy()

    work = df.copy().reset_index(drop=True)
    points = work[["x", "y"]].to_numpy(float)

    pair_area = fixed_equal_radius_overlap_matrix(points, fixed_r)
    own_area = math.pi * fixed_r * fixed_r

    active = np.ones(len(work), dtype=bool)
    overlap_sum = pair_area.sum(axis=1)

    support_n = compute_local_support(points, pitch)

    if "param2" in work.columns:
        param2 = pd.to_numeric(work["param2"], errors="coerce").fillna(0).to_numpy(float)
    elif "p2" in work.columns:
        param2 = pd.to_numeric(work["p2"], errors="coerce").fillna(0).to_numpy(float)
    else:
        param2 = np.zeros(len(work), dtype=float)

    removed_records = []
    step = 0

    while True:
        if active.sum() <= 1:
            break

        frac = overlap_sum / max(own_area, 1e-9)
        frac[~active] = -1.0

        max_frac = float(frac.max())
        if max_frac < float(overlap_thr):
            break

        candidates = np.where((active) & (frac >= float(overlap_thr)))[0]

        # Main rule: remove the most-overlapping cell.
        # Tie-breakers:
        # - lower local support removed first
        # - lower param2 removed first
        cand_order = sorted(
            candidates,
            key=lambda i: (-frac[i], support_n[i], param2[i])
        )

        rem_i = int(cand_order[0])

        rec = work.loc[rem_i].to_dict()
        rec["removed_step"] = step
        rec["removed_reason"] = "pitch_fixedR046_total_overlap_area_ge30"
        rec["overlap_area_at_removal"] = float(overlap_sum[rem_i])
        rec["overlap_frac_at_removal"] = float(frac[rem_i])
        rec["local_support_n"] = int(support_n[rem_i])
        rec["pitch"] = float(pitch)
        rec["fixed_r"] = float(fixed_r)
        removed_records.append(rec)

        active[rem_i] = False

        # update overlap sums: removed cell no longer contributes to any active cell
        overlap_sum = overlap_sum - pair_area[:, rem_i]
        overlap_sum[~active] = -1.0

        step += 1

    kept = work.loc[active].copy().reset_index(drop=True)

    kept_idx = np.where(active)[0]
    if len(kept_idx):
        final_pair = pair_area[np.ix_(kept_idx, kept_idx)]
        final_sum = final_pair.sum(axis=1)
        kept["final_overlap_area"] = final_sum
        kept["final_overlap_frac"] = final_sum / max(own_area, 1e-9)
        kept["local_support_n"] = support_n[kept_idx]
    else:
        kept["final_overlap_area"] = []
        kept["final_overlap_frac"] = []
        kept["local_support_n"] = []

    kept["pitch"] = float(pitch)
    kept["fixed_r"] = float(fixed_r)
    kept["hough_r_orig"] = kept["r"] if "r" in kept.columns else np.nan

    # From now on, r means diagnostic fixed radius.
    kept["r"] = float(fixed_r)
    kept["is_gapfill"] = 0
    kept["kept_rank"] = np.arange(len(kept))

    if removed_records:
        removed = pd.DataFrame(removed_records)
    else:
        removed = work.iloc[0:0].copy()
        removed["removed_step"] = []
        removed["removed_reason"] = []
        removed["overlap_area_at_removal"] = []
        removed["overlap_frac_at_removal"] = []

    if len(removed):
        if "r" in removed.columns:
            removed["hough_r_orig"] = removed["r"]
        removed["r"] = float(fixed_r)
        removed["pitch"] = float(pitch)
        removed["fixed_r"] = float(fixed_r)

    return kept, removed


def draw_overlay(img, old_df, kept_df, removed_df, title, out_path, preview_w=2200):
    canvas = img.copy()

    fixed_r = float(kept_df["fixed_r"].median()) if len(kept_df) and "fixed_r" in kept_df.columns else 17.0

    # Removed by pitch-overlap: orange
    for _, row in removed_df.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["r"]))) if "r" in row and pd.notna(row["r"]) else int(round(fixed_r))
        cv2.circle(canvas, (x, y), max(1, r), (0, 165, 255), 1)
        cv2.circle(canvas, (x, y), 2, (0, 165, 255), -1)

    # Kept: green
    for _, row in kept_df.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["r"])))
        cv2.circle(canvas, (x, y), max(1, r), (0, 255, 0), 1)
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

    cv2.rectangle(canvas, (10, 10), (min(canvas.shape[1] - 1, 1700), 70), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        title[:190],
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
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
    ap.add_argument("--old_hough_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--split", default="new")

    ap.add_argument("--fixed_radius_ratio", type=float, default=0.46)
    ap.add_argument("--overlap_frac_thr", type=float, default=0.30)

    ap.add_argument("--preview_w", type=int, default=2200)
    ap.add_argument("--clean", action="store_true")

    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    old_hough_root = Path(args.old_hough_root)
    out_root = Path(args.out_root)

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    per_root = out_root / "per_image"
    seed_root = out_root / "05_pitch_clean_seed" / args.split
    overlay_root = out_root / "_COLLECTED_pitch_fixedR046_clean_step1_overlays"
    summary_root = out_root / "summaries"

    for d in [per_root, seed_root, overlay_root, summary_root]:
        d.mkdir(parents=True, exist_ok=True)

    print("[INFO] image_dir:", image_dir)
    print("[INFO] old_hough_root:", old_hough_root)
    print("[INFO] method: existing Hough centers -> pitch -> fixed_r=0.46*pitch -> overlap area filter")
    print("[INFO] edge/white filtering: NOT applied in Step 1")
    print("[INFO] local hex gapfill: NOT applied in Step 1")

    rows = []

    images = list_images(image_dir)

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        print(f"\n[{idx}/{len(images)}] {stem}", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("  [WARN] cannot read image")
            continue

        old_csv = old_hough_root / "per_image" / stem / "hough_circles_nms.csv"

        if not old_csv.exists():
            print("  [WARN] missing old hough:", old_csv)
            continue

        old_df = pd.read_csv(old_csv)

        if len(old_df) == 0:
            print("  [WARN] empty old hough")
            continue

        if not {"x", "y"}.issubset(old_df.columns):
            print("  [WARN] missing x/y columns:", old_csv)
            continue

        old_df = old_df.copy().reset_index(drop=True)
        old_summary = load_old_hough_summary(old_hough_root, stem)

        points = old_df[["x", "y"]].to_numpy(float)

        pitch, pitch_source = estimate_pitch_from_centers(
            points,
            old_df=old_df,
            old_summary=old_summary,
        )

        fixed_r = float(args.fixed_radius_ratio) * float(pitch)

        kept, removed = iterative_remove_overlap_pitch(
            old_df,
            pitch=pitch,
            fixed_r=fixed_r,
            overlap_thr=args.overlap_frac_thr,
        )

        per_dir = per_root / stem
        per_dir.mkdir(parents=True, exist_ok=True)

        old_df.to_csv(per_dir / "01_old_hough_nms_all.csv", index=False)
        kept.to_csv(per_dir / "02_pitch_fixedR046_overlap_clean_kept.csv", index=False)
        removed.to_csv(per_dir / "03_removed_pitch_fixedR046_overlaparea_ge30.csv", index=False)

        out_seed_dir = seed_root / stem
        out_seed_dir.mkdir(parents=True, exist_ok=True)

        final_csv = out_seed_dir / "cell_predictions_8class.csv"
        kept.to_csv(final_csv, index=False)

        max_overlap = float(kept["final_overlap_frac"].max()) if len(kept) else np.nan
        median_overlap = float(kept["final_overlap_frac"].median()) if len(kept) else np.nan

        title = (
            f"{stem} | pitch={pitch:.2f} fixedR={fixed_r:.2f} "
            f"old={len(old_df)} kept={len(kept)} removed={len(removed)} "
            f"maxOv={max_overlap:.3f}"
        )

        overlay_path = overlay_root / f"{stem}_pitch_fixedR046_clean_step1.jpg"

        draw_overlay(
            img,
            old_df,
            kept,
            removed,
            title,
            overlay_path,
            preview_w=args.preview_w,
        )

        print(
            f"  old={len(old_df):5d} "
            f"pitch={pitch:6.2f} "
            f"fixedR={fixed_r:5.2f} "
            f"kept={len(kept):5d} "
            f"removed={len(removed):5d} "
            f"maxOverlap={max_overlap:.3f} "
            f"pitchSource={pitch_source}"
        )

        rows.append({
            "image": stem,
            "old_hough_nms_n": len(old_df),
            "pitch": float(pitch),
            "pitch_source": pitch_source,
            "fixed_radius_ratio": args.fixed_radius_ratio,
            "fixed_r": float(fixed_r),
            "kept_n": len(kept),
            "removed_overlaparea_ge30_n": len(removed),
            "removed_fraction": len(removed) / max(1, len(old_df)),
            "max_final_overlap_frac": max_overlap,
            "median_final_overlap_frac": median_overlap,
            "kept_radius_median": float(kept["r"].median()) if len(kept) else np.nan,
            "final_seed_csv": str(final_csv),
            "overlay": str(overlay_path),
        })

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "pitch_fixedR046_clean_step1_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[DONE]")
    print("summary:", summary_csv)
    print("seed root:", seed_root)
    print("overlays:", overlay_root)


if __name__ == "__main__":
    main()
