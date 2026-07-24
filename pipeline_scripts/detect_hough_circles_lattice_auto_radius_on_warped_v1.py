#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import math

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def configure_opencv(threads=1):
    cv2.setNumThreads(max(1, int(threads)))
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def preprocess_image(img_bgr, mode):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if mode == "gray":
        return gray

    if mode == "gray_clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    if mode == "gray_clahe_blur":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(gray)
        out = cv2.GaussianBlur(out, (5, 5), 1.1)
        return out

    if mode == "labL_unsharp":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        blur = cv2.GaussianBlur(L, (0, 0), 2.0)
        out = cv2.addWeighted(L, 1.65, blur, -0.65, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.GaussianBlur(out, (5, 5), 0.9)
        return out

    if mode == "labL_localnorm":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        bg = cv2.GaussianBlur(L, (0, 0), 25.0)
        norm = L - bg + 128.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(norm)
        out = cv2.GaussianBlur(out, (5, 5), 0.9)
        return out

    raise ValueError(f"Unknown preprocess mode: {mode}")


def hough_detect(pre, dp, min_dist, param1, param2, min_radius, max_radius):
    circles = cv2.HoughCircles(
        pre,
        cv2.HOUGH_GRADIENT,
        dp=float(dp),
        minDist=float(min_dist),
        param1=float(param1),
        param2=float(param2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        return np.empty((0, 3), dtype=float)

    return np.round(circles[0, :], 3).astype(float)


def nms_circles(df, nms_dist, best_radius=None):
    if len(df) == 0:
        return df.copy()

    tmp = df.copy()

    if best_radius is None:
        best_radius = float(tmp["r"].median()) if len(tmp) else 0.0

    tmp["r_dev"] = (pd.to_numeric(tmp["r"], errors="coerce") - float(best_radius)).abs()

    # Prefer the stricter Hough response, then the radius closest to best_radius.
    sort_cols = []
    ascending = []

    if "param2" in tmp.columns:
        sort_cols.append("param2")
        ascending.append(False)

    sort_cols.append("r_dev")
    ascending.append(True)

    tmp = tmp.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    kept = []
    kept_xy = []

    for _, row in tmp.iterrows():
        x, y = float(row["x"]), float(row["y"])

        if not kept_xy:
            kept.append(row)
            kept_xy.append((x, y))
            continue

        arr = np.asarray(kept_xy, dtype=float)
        d = np.sqrt(((arr - np.array([x, y])) ** 2).sum(axis=1))

        if np.min(d) >= float(nms_dist):
            kept.append(row)
            kept_xy.append((x, y))

    if not kept:
        return tmp.iloc[0:0].drop(columns=["r_dev"], errors="ignore").copy()

    out = pd.DataFrame(kept).drop(columns=["r_dev"], errors="ignore").reset_index(drop=True)
    out["nms_rank"] = np.arange(len(out))
    return out


def lattice_score_for_radius(df, radius, img_shape):
    """
    Evaluate whether a radius candidate explains the true cell lattice.
    Key considerations:
    - Whether neighbor distances are near the expected pitch (approximately 2 * radius).
    - Whether each center has supporting neighbors in a hexagonal arrangement.
    - A small penalty for excessively large radii.
    """
    if len(df) < 20:
        return {
            "n": len(df),
            "nn_median": np.nan,
            "expected_pitch": 2.0 * radius,
            "pitch_score": 0.0,
            "support_mean": 0.0,
            "support_frac2": 0.0,
            "support_frac3": 0.0,
            "density_norm": 0.0,
            "score": 0.0,
        }

    xy = df[["x", "y"]].dropna().to_numpy(float)
    if len(xy) < 20:
        return {
            "n": len(xy),
            "nn_median": np.nan,
            "expected_pitch": 2.0 * radius,
            "pitch_score": 0.0,
            "support_mean": 0.0,
            "support_frac2": 0.0,
            "support_frac3": 0.0,
            "density_norm": 0.0,
            "score": 0.0,
        }

    tree = cKDTree(xy)

    expected_pitch = 2.0 * float(radius)

    # Assume that Hough center pitch is approximately twice the radius in a true honeycomb.
    # Use a broad interval to tolerate warping and texture variation.
    low = expected_pitch * 0.72
    high = expected_pitch * 1.35

    # nearest-neighbor median
    d2, _ = tree.query(xy, k=min(2, len(xy)))
    if d2.ndim == 1 or d2.shape[1] < 2:
        nn_median = np.nan
    else:
        nn = d2[:, 1]
        nn = nn[np.isfinite(nn)]
        nn_median = float(np.median(nn)) if len(nn) else np.nan

    # annulus neighbor support
    neigh_lists = tree.query_ball_point(xy, r=high)
    support_counts = []

    for i, neigh in enumerate(neigh_lists):
        cnt = 0
        xi = xy[i]
        for j in neigh:
            if j == i:
                continue
            d = np.linalg.norm(xy[j] - xi)
            if low <= d <= high:
                cnt += 1
        support_counts.append(cnt)

    support_counts = np.asarray(support_counts, dtype=float)
    support_mean = float(np.mean(np.clip(support_counts, 0, 6))) if len(support_counts) else 0.0
    support_frac2 = float(np.mean(support_counts >= 2)) if len(support_counts) else 0.0
    support_frac3 = float(np.mean(support_counts >= 3)) if len(support_counts) else 0.0

    # Evaluate whether the nearest-neighbor distance agrees with the expected pitch.
    if np.isfinite(nn_median) and nn_median > 0:
        ratio = nn_median / expected_pitch
        # A ratio near 1.0 is best; the score decreases rapidly outside 0.70-1.35.
        pitch_score = float(np.exp(-((np.log(ratio) / np.log(1.28)) ** 2)))
    else:
        pitch_score = 0.0

    h, w = img_shape[:2]
    area = max(1.0, float(h * w))

    # hex lattice cell density rough normalization.
    # ideal density ≈ area / (0.866 * pitch^2)
    ideal_n = area / max(1.0, 0.866 * expected_pitch * expected_pitch)
    density_ratio = len(xy) / max(1.0, ideal_n)

    # Too few centers indicate low recall; too many suggest duplicates or false positives.
    # Allow approximately 0.15-1.80, with the preferred range near 0.60-1.20.
    density_norm = float(np.exp(-((np.log(max(density_ratio, 1e-6)) / np.log(2.5)) ** 2)))

    # Apply a small penalty against large-radius bias.
    large_radius_penalty = float(np.exp(-0.035 * max(0.0, radius - 10.0)))

    # Final score.
    score = (
        (support_frac2 ** 1.25)
        * (0.35 + 0.65 * min(support_mean / 4.0, 1.0))
        * (0.35 + 0.65 * pitch_score)
        * (0.25 + 0.75 * density_norm)
        * large_radius_penalty
    )

    return {
        "n": len(xy),
        "nn_median": nn_median,
        "expected_pitch": expected_pitch,
        "pitch_score": pitch_score,
        "support_mean": support_mean,
        "support_frac2": support_frac2,
        "support_frac3": support_frac3,
        "density_ratio": density_ratio,
        "density_norm": density_norm,
        "large_radius_penalty": large_radius_penalty,
        "score": float(score),
    }


def select_radius_from_scores(
    score_df,
    strategy="plateau_median",
    plateau_score_fraction=0.60,
    plateau_min_candidates=3,
    radius_step=1,
):
    """Choose a radius without a fragile winner-take-all argmax.

    ``plateau_median`` first identifies radii whose score is at least a
    prespecified fraction of the maximum, retains the contiguous component
    containing the global maximum, and selects its median candidate. Isolated
    or two-candidate peaks retain the legacy winner.
    """
    fallback = {
        "best_radius": None,
        "best_reason": "fallback_mid",
        "score_max": np.nan,
        "selection_threshold": np.nan,
        "plateau_radius_min": np.nan,
        "plateau_radius_max": np.nan,
        "plateau_candidate_count": 0,
    }
    if len(score_df) == 0:
        return fallback

    valid = score_df.copy()
    valid["score"] = pd.to_numeric(valid["score"], errors="coerce")
    valid["radius_candidate"] = pd.to_numeric(
        valid["radius_candidate"], errors="coerce"
    )
    valid = valid.dropna(subset=["score", "radius_candidate"])
    valid = valid[valid["score"] > 0].copy()
    if len(valid) == 0:
        return fallback

    ranked = valid.sort_values(
        by=["score", "support_frac2", "pitch_score", "radius_candidate"],
        ascending=[False, False, False, True],
    )
    winner = int(ranked.iloc[0]["radius_candidate"])
    maximum = float(ranked.iloc[0]["score"])
    fraction = float(plateau_score_fraction)
    threshold = maximum * fraction

    result = {
        "best_radius": winner,
        "best_reason": "lattice_score_argmax",
        "score_max": maximum,
        "selection_threshold": threshold,
        "plateau_radius_min": winner,
        "plateau_radius_max": winner,
        "plateau_candidate_count": 1,
    }
    if strategy == "winner":
        return result
    if strategy != "plateau_median":
        raise ValueError(f"Unknown radius selection strategy: {strategy}")
    if not 0 < fraction <= 1:
        raise ValueError("plateau_score_fraction must be in (0, 1]")

    eligible = sorted(
        int(value)
        for value in valid.loc[
            valid["score"] >= threshold, "radius_candidate"
        ].unique()
    )
    components = []
    current = []
    for radius in eligible:
        if current and radius - current[-1] > max(1, int(radius_step)):
            components.append(current)
            current = []
        current.append(radius)
    if current:
        components.append(current)

    component = next((part for part in components if winner in part), [winner])
    result.update(
        {
            "plateau_radius_min": int(min(component)),
            "plateau_radius_max": int(max(component)),
            "plateau_candidate_count": int(len(component)),
        }
    )
    if len(component) < int(plateau_min_candidates):
        result["best_reason"] = "lattice_score_argmax_narrow_plateau"
        return result

    median_radius = float(np.median(np.asarray(component, dtype=float)))
    selected = min(component, key=lambda value: (abs(value - median_radius), -value))
    result["best_radius"] = int(selected)
    result["best_reason"] = "lattice_score_plateau_median"
    return result


def draw_overlay(img, raw_df, nms_df, out_path, best_radius=None, preview_w=2200):
    canvas = img.copy()

    # raw: thin gray
    for _, row in raw_df.iterrows():
        x, y, r = int(round(row["x"])), int(round(row["y"])), int(round(row["r"]))
        cv2.circle(canvas, (x, y), r, (150, 150, 150), 1)

    # nms: green circle + red center
    for _, row in nms_df.iterrows():
        x, y, r = int(round(row["x"])), int(round(row["y"])), int(round(row["r"]))
        cv2.circle(canvas, (x, y), r, (0, 255, 0), 1)
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

    if best_radius is not None:
        text = f"best_radius={best_radius}"
        cv2.rectangle(canvas, (10, 10), (360, 55), (255, 255, 255), -1)
        cv2.putText(canvas, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    if preview_w and canvas.shape[1] > preview_w:
        scale = preview_w / canvas.shape[1]
        canvas = cv2.resize(canvas, (preview_w, int(round(canvas.shape[0] * scale))))

    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--preprocess", nargs="+", default=["gray_clahe", "gray_clahe_blur", "labL_unsharp"])

    # radius sweep
    ap.add_argument("--radius_min", type=int, default=7)
    ap.add_argument("--radius_max", type=int, default=15)
    ap.add_argument("--radius_step", type=int, default=1)

    # scoring Hough, loose
    ap.add_argument("--score_dp", type=float, default=1.2)
    ap.add_argument("--score_param1", type=float, default=70)
    ap.add_argument("--score_param2_list", nargs="+", type=float, default=[10, 12, 14])
    ap.add_argument("--score_min_dist_mult", type=float, default=1.25)
    ap.add_argument("--score_nms_mult", type=float, default=1.05)

    # final Hough
    ap.add_argument("--final_dp", type=float, default=1.2)
    ap.add_argument("--final_param1", type=float, default=75)
    ap.add_argument("--final_param2_list", nargs="+", type=float, default=[10, 12, 14, 16])
    ap.add_argument("--final_radius_pad", type=int, default=2)
    ap.add_argument("--final_min_dist_mult", type=float, default=1.40)
    ap.add_argument("--final_nms_mult", type=float, default=1.20)

    ap.add_argument(
        "--radius_selection",
        choices=["winner", "plateau_median"],
        default="winner",
        help="Use plateau_median for the platform-stable workflow.",
    )
    ap.add_argument("--plateau_score_fraction", type=float, default=0.60)
    ap.add_argument("--plateau_min_candidates", type=int, default=3)
    ap.add_argument("--opencv_threads", type=int, default=1)

    ap.add_argument("--preview_w", type=int, default=2200)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    configure_opencv(args.opencv_threads)

    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    per_root = out_root / "per_image"
    overlay_root = out_root / "overlays"
    collected_root = out_root / "_COLLECTED_hough_circle_overlays"
    summary_root = out_root / "summaries"

    for p in [per_root, overlay_root, collected_root, summary_root]:
        ensure_dir(p)

    imgs = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

    print("[INFO] image_dir =", image_dir)
    print("[INFO] images =", len(imgs))
    print("[INFO] out_root =", out_root)
    print("[INFO] radius candidates =", list(range(args.radius_min, args.radius_max + 1, args.radius_step)))
    print(
        "[INFO] radius selection =",
        args.radius_selection,
        "plateau fraction =",
        args.plateau_score_fraction,
    )

    if not imgs:
        raise SystemExit("[ERROR] no images found")

    summary_rows = []

    for idx, img_path in enumerate(imgs, 1):
        print(f"\n[{idx}/{len(imgs)}] {img_path.name}", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("[WARN] failed to read:", img_path)
            summary_rows.append({"image": img_path.stem, "status": "read_failed"})
            continue

        stem = img_path.stem
        per_dir = per_root / stem
        ensure_dir(per_dir)

        pre_by_mode = {}
        for mode in args.preprocess:
            pre = preprocess_image(img, mode)
            pre_by_mode[mode] = pre
            cv2.imwrite(str(per_dir / f"preprocess_{mode}.jpg"), pre)

        radius_score_rows = []
        radius_candidate_dfs = {}

        # 1st pass: radius sweep + lattice scoring
        for r in range(args.radius_min, args.radius_max + 1, args.radius_step):
            raw_rows = []
            score_min_dist = max(4.0, float(r) * args.score_min_dist_mult)
            score_nms_dist = max(4.0, float(r) * args.score_nms_mult)

            for mode, pre in pre_by_mode.items():
                for param2 in args.score_param2_list:
                    circles = hough_detect(
                        pre=pre,
                        dp=args.score_dp,
                        min_dist=score_min_dist,
                        param1=args.score_param1,
                        param2=param2,
                        min_radius=r,
                        max_radius=r,
                    )

                    for j, (x, y, rr) in enumerate(circles):
                        raw_rows.append({
                            "image": stem,
                            "radius_candidate": r,
                            "source": mode,
                            "param2": param2,
                            "x": float(x),
                            "y": float(y),
                            "r": float(rr),
                            "raw_index": j,
                        })

            raw_df = pd.DataFrame(raw_rows)
            nms_df = nms_circles(raw_df, nms_dist=score_nms_dist, best_radius=r)

            score = lattice_score_for_radius(nms_df, radius=r, img_shape=img.shape)
            score.update({
                "image": stem,
                "radius_candidate": r,
                "score_raw_n": len(raw_df),
                "score_nms_n": len(nms_df),
                "score_min_dist": score_min_dist,
                "score_nms_dist": score_nms_dist,
            })
            radius_score_rows.append(score)
            radius_candidate_dfs[r] = nms_df

            print(
                f"  r={r:02d} raw={len(raw_df):5d} nms={len(nms_df):5d} "
                f"nn={score.get('nn_median', np.nan):6.2f} "
                f"support2={score.get('support_frac2', 0):.3f} "
                f"score={score.get('score', 0):.4f}"
            )

        score_df = pd.DataFrame(radius_score_rows)
        score_csv = per_dir / "radius_candidate_lattice_scores.csv"

        if len(score_df) == 0 or score_df["score"].max() <= 0:
            best_radius = int(round((args.radius_min + args.radius_max) / 2))
            best_reason = "fallback_mid"
            selection = {
                "score_max": np.nan,
                "selection_threshold": np.nan,
                "plateau_radius_min": np.nan,
                "plateau_radius_max": np.nan,
                "plateau_candidate_count": 0,
            }
        else:
            selection = select_radius_from_scores(
                score_df,
                strategy=args.radius_selection,
                plateau_score_fraction=args.plateau_score_fraction,
                plateau_min_candidates=args.plateau_min_candidates,
                radius_step=args.radius_step,
            )
            best_radius = int(selection["best_radius"])
            best_reason = str(selection["best_reason"])

        score_df["selection_strategy"] = args.radius_selection
        score_df["selection_threshold"] = selection.get(
            "selection_threshold", np.nan
        )
        score_df["in_selected_plateau"] = (
            pd.to_numeric(score_df["radius_candidate"], errors="coerce")
            .between(
                selection.get("plateau_radius_min", np.nan),
                selection.get("plateau_radius_max", np.nan),
            )
        )
        score_df["selected_radius"] = best_radius
        score_df.to_csv(score_csv, index=False)

        # 2nd pass: final Hough based on best_radius
        min_r = max(1, best_radius - args.final_radius_pad)
        max_r = best_radius + args.final_radius_pad
        final_min_dist = max(4.0, best_radius * args.final_min_dist_mult)
        final_nms_dist = max(4.0, best_radius * args.final_nms_mult)

        print(
            f"  BEST radius={best_radius} reason={best_reason} "
            f"final_radius={min_r}-{max_r} "
            f"final_min_dist={final_min_dist:.2f} final_nms={final_nms_dist:.2f}"
        )

        final_rows = []
        for mode, pre in pre_by_mode.items():
            for param2 in args.final_param2_list:
                circles = hough_detect(
                    pre=pre,
                    dp=args.final_dp,
                    min_dist=final_min_dist,
                    param1=args.final_param1,
                    param2=param2,
                    min_radius=min_r,
                    max_radius=max_r,
                )

                for j, (x, y, rr) in enumerate(circles):
                    final_rows.append({
                        "image": stem,
                        "source": mode,
                        "param2": param2,
                        "x": float(x),
                        "y": float(y),
                        "r": float(rr),
                        "raw_index": j,
                        "best_radius": best_radius,
                        "min_radius_used": min_r,
                        "max_radius_used": max_r,
                        "final_min_dist": final_min_dist,
                        "final_nms_dist": final_nms_dist,
                    })

        raw_df = pd.DataFrame(final_rows)
        raw_csv = per_dir / "hough_circles_raw.csv"
        raw_df.to_csv(raw_csv, index=False)

        nms_df = nms_circles(raw_df, nms_dist=final_nms_dist, best_radius=best_radius)
        nms_csv = per_dir / "hough_circles_nms.csv"
        nms_df.to_csv(nms_csv, index=False)

        overlay_path = overlay_root / f"{stem}_hough_circles_overlay.jpg"
        draw_overlay(img, raw_df, nms_df, overlay_path, best_radius=best_radius, preview_w=args.preview_w)
        shutil.copy2(overlay_path, collected_root / overlay_path.name)

        final_lattice = lattice_score_for_radius(nms_df, radius=best_radius, img_shape=img.shape)

        summary_rows.append({
            "image": stem,
            "status": "ok",
            "best_radius": best_radius,
            "best_reason": best_reason,
            "selection_strategy": args.radius_selection,
            "score_max": selection.get("score_max", np.nan),
            "selection_threshold": selection.get("selection_threshold", np.nan),
            "plateau_radius_min": selection.get("plateau_radius_min", np.nan),
            "plateau_radius_max": selection.get("plateau_radius_max", np.nan),
            "plateau_candidate_count": selection.get("plateau_candidate_count", 0),
            "min_radius_used": min_r,
            "max_radius_used": max_r,
            "final_min_dist": final_min_dist,
            "final_nms_dist": final_nms_dist,
            "raw_n": len(raw_df),
            "nms_n": len(nms_df),
            "radius_median": float(nms_df["r"].median()) if len(nms_df) else np.nan,
            "radius_mean": float(nms_df["r"].mean()) if len(nms_df) else np.nan,
            "final_nn_median": final_lattice.get("nn_median", np.nan),
            "final_support_frac2": final_lattice.get("support_frac2", np.nan),
            "final_support_frac3": final_lattice.get("support_frac3", np.nan),
            "final_lattice_score": final_lattice.get("score", np.nan),
            "score_csv": str(score_csv),
            "raw_csv": str(raw_csv),
            "nms_csv": str(nms_csv),
            "overlay": str(overlay_path),
        })

        print(
            f"  FINAL raw_n={len(raw_df)} nms_n={len(nms_df)} "
            f"r_med={summary_rows[-1]['radius_median']}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = summary_root / "lattice_auto_radius_hough_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[DONE]")
    print("summary:", summary_csv)
    print("overlays:", overlay_root)
    print("collected:", collected_root)


if __name__ == "__main__":
    main()
