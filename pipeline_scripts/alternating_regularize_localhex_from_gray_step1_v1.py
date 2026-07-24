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


def make_tree(points):
    if cKDTree is None or len(points) == 0:
        return None
    return cKDTree(points)


def query_ball(tree, points, q, radius):
    if len(points) == 0:
        return []
    if tree is not None:
        return tree.query_ball_point(q, radius)
    d = np.sqrt(((points - np.asarray(q)) ** 2).sum(axis=1))
    return np.where(d <= radius)[0].tolist()


def query_pairs(points, radius):
    n = len(points)
    if n <= 1:
        return np.empty((0, 2), dtype=int)

    if cKDTree is not None:
        tree = cKDTree(points)
        pairs = np.array(list(tree.query_pairs(radius)), dtype=int)
        if len(pairs) == 0:
            return np.empty((0, 2), dtype=int)
        return pairs

    pairs = []
    for i in range(n):
        d = np.sqrt(((points[i+1:] - points[i]) ** 2).sum(axis=1))
        js = np.where(d <= radius)[0] + i + 1
        for j in js:
            pairs.append((i, int(j)))

    if not pairs:
        return np.empty((0, 2), dtype=int)

    return np.asarray(pairs, dtype=int)


def nearest_distance(points, q):
    if len(points) == 0:
        return np.inf
    d = np.sqrt(((points - np.asarray(q)) ** 2).sum(axis=1))
    return float(d.min())


def load_seed(input_root, split, stem):
    input_root = Path(input_root)

    p = input_root / "05_pitch_clean_seed" / split / stem / "cell_predictions_8class.csv"
    if p.exists():
        return pd.read_csv(p), p

    candidates = sorted(input_root.glob(f"**/{stem}/cell_predictions_8class.csv"))
    if candidates:
        return pd.read_csv(candidates[0]), candidates[0]

    return None, p


def estimate_pitch(points):
    n = len(points)

    if n < 20:
        return 38.0

    if cKDTree is not None:
        tree = cKDTree(points)
        k = min(8, n)
        dists, _ = tree.query(points, k=k)
        vals = dists[:, 1:].reshape(-1)
    else:
        vals = []
        step = max(1, n // 1200)
        for p in points[::step]:
            d = np.sqrt(((points - p) ** 2).sum(axis=1))
            vals.extend(np.sort(d[d > 1e-6])[:7].tolist())
        vals = np.asarray(vals, dtype=float)

    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 5]

    if len(vals) < 30:
        return 38.0

    q20, q80 = np.quantile(vals, [0.20, 0.80])
    vals2 = vals[(vals >= q20 * 0.8) & (vals <= q80 * 1.25)]

    if len(vals2) < 30:
        return float(np.median(vals))

    return float(np.median(vals2))


def get_pitch_fixedr(df, fixed_radius_ratio=0.46):
    if "pitch" in df.columns and pd.to_numeric(df["pitch"], errors="coerce").notna().any():
        pitch = float(pd.to_numeric(df["pitch"], errors="coerce").dropna().median())
    else:
        pitch = estimate_pitch(df[["x", "y"]].to_numpy(float))

    if "fixed_r" in df.columns and pd.to_numeric(df["fixed_r"], errors="coerce").notna().any():
        fixed_r = float(pd.to_numeric(df["fixed_r"], errors="coerce").dropna().median())
    else:
        fixed_r = float(fixed_radius_ratio) * pitch

    return pitch, fixed_r


def initialize_seed_df(df, pitch, fixed_r):
    df = df.copy().reset_index(drop=True)

    if "hough_r_orig" not in df.columns and "r" in df.columns:
        df["hough_r_orig"] = df["r"]

    df["pitch"] = float(pitch)
    df["fixed_r"] = float(fixed_r)
    df["r"] = float(fixed_r)

    if "is_gapfill" not in df.columns:
        df["is_gapfill"] = 0
    else:
        df["is_gapfill"] = df["is_gapfill"].fillna(0).astype(int)

    if "x_initial" not in df.columns:
        df["x_initial"] = df["x"]
    if "y_initial" not in df.columns:
        df["y_initial"] = df["y"]

    return df


def equal_radius_overlap_area_by_distance(d, r):
    d = np.asarray(d, dtype=float)
    r = float(r)

    area = np.zeros_like(d, dtype=float)

    if r <= 0:
        return area

    m0 = d <= 1e-9
    area[m0] = math.pi * r * r

    m = (d > 1e-9) & (d < 2.0 * r)

    if np.any(m):
        dm = d[m]
        x = np.clip(dm / (2.0 * r), -1.0, 1.0)
        area[m] = (
            2.0 * r * r * np.arccos(x)
            - 0.5 * dm * np.sqrt(np.maximum(4.0 * r * r - dm * dm, 0.0))
        )

    return area


def fixed_equal_radius_overlap_matrix(points, fixed_r):
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)

    area = equal_radius_overlap_area_by_distance(d, fixed_r)
    np.fill_diagonal(area, 0.0)

    return area


def final_overlap_cleanup(df, fixed_r, overlap_thr=0.33, phase="cleanup"):
    if len(df) == 0:
        return df.copy(), df.copy()

    work = df.copy().reset_index(drop=True)
    points = work[["x", "y"]].to_numpy(float)

    pair = fixed_equal_radius_overlap_matrix(points, fixed_r)
    circle_area = math.pi * fixed_r * fixed_r

    active = np.ones(len(work), dtype=bool)
    overlap_sum = pair.sum(axis=1)

    is_gapfill = work["is_gapfill"].fillna(0).astype(int).to_numpy() if "is_gapfill" in work.columns else np.zeros(len(work), dtype=int)

    removed_records = []
    step = 0

    while True:
        if active.sum() <= 1:
            break

        frac = overlap_sum / max(circle_area, 1e-9)
        frac[~active] = -1.0

        if float(frac.max()) < overlap_thr:
            break

        candidates = np.where((active) & (frac >= overlap_thr))[0]

        # Remove the cell with the most severe overlap first.
        # If severity is tied, remove a gap-filled cell first.
        order = sorted(candidates, key=lambda i: (-frac[i], -is_gapfill[i]))
        rem_i = int(order[0])

        rec = work.loc[rem_i].to_dict()
        rec["removed_step"] = step
        rec["removed_phase"] = phase
        rec["removed_reason"] = "overlap_ge_threshold"
        rec["overlap_frac_at_removal"] = float(frac[rem_i])
        removed_records.append(rec)

        active[rem_i] = False
        overlap_sum = overlap_sum - pair[:, rem_i]
        overlap_sum[~active] = -1.0

        step += 1

    kept = work.loc[active].copy().reset_index(drop=True)

    kept_idx = np.where(active)[0]
    if len(kept_idx):
        final_pair = pair[np.ix_(kept_idx, kept_idx)]
        final_sum = final_pair.sum(axis=1)
        kept["final_overlap_frac"] = final_sum / max(circle_area, 1e-9)
    else:
        kept["final_overlap_frac"] = np.nan

    kept["r"] = float(fixed_r)
    kept["fixed_r"] = float(fixed_r)
    kept["kept_rank"] = np.arange(len(kept))

    removed = pd.DataFrame(removed_records) if removed_records else work.iloc[0:0].copy()

    return kept, removed


def regularization_energy(points, pitch, fixed_r,
                          neighbor_low_ratio=0.70,
                          neighbor_high_ratio=1.35,
                          close_ratio=0.92,
                          close_weight=8.0):
    pairs = query_pairs(points, neighbor_high_ratio * pitch)

    if len(pairs) == 0:
        return 0.0

    pi = points[pairs[:, 0]]
    pj = points[pairs[:, 1]]
    d = np.sqrt(((pi - pj) ** 2).sum(axis=1))

    e = 0.0

    # Adjust neighbor distances toward the estimated pitch.
    m = (d >= neighbor_low_ratio * pitch) & (d <= neighbor_high_ratio * pitch)
    if np.any(m):
        e += float(np.sum(((d[m] - pitch) / pitch) ** 2))

    # Strongly separate centers that are too close.
    close_target = close_ratio * pitch
    mc = d < close_target
    if np.any(mc):
        e += close_weight * float(np.sum(((close_target - d[mc]) / pitch) ** 2))

    # Also separate circles with excessive fixed-radius overlap.
    overlap_target = 2.0 * fixed_r
    mo = d < overlap_target
    if np.any(mo):
        e += close_weight * 0.5 * float(np.sum(((overlap_target - d[mo]) / pitch) ** 2))

    return float(e)


def compute_forces(points, pitch, fixed_r,
                   neighbor_low_ratio=0.70,
                   neighbor_high_ratio=1.35,
                   close_ratio=0.92,
                   close_weight=3.0):
    forces = np.zeros_like(points, dtype=float)
    pairs = query_pairs(points, neighbor_high_ratio * pitch)

    if len(pairs) == 0:
        return forces

    for i, j in pairs:
        diff = points[i] - points[j]
        d = float(np.linalg.norm(diff))

        if d < 1e-9:
            ang = 2.0 * math.pi * ((i * 13 + j * 7) % 97) / 97.0
            unit = np.array([math.cos(ang), math.sin(ang)], dtype=float)
            d = 1e-6
        else:
            unit = diff / d

        fmag = 0.0

        close_target = close_ratio * pitch
        if d < close_target:
            fmag += close_weight * (close_target - d) / pitch

        if neighbor_low_ratio * pitch <= d <= neighbor_high_ratio * pitch:
            fmag += (pitch - d) / pitch

        overlap_target = 2.0 * fixed_r
        if d < overlap_target:
            fmag += 1.5 * (overlap_target - d) / pitch

        if abs(fmag) < 1e-12:
            continue

        f = unit * fmag
        forces[i] += f
        forces[j] -= f

    return forces


def regularize_centers(df, img_shape, pitch, fixed_r, phase,
                       max_iters=30,
                       max_total_move_ratio=0.12,
                       max_step_ratio=0.025,
                       force_gain=0.25,
                       neighbor_low_ratio=0.70,
                       neighbor_high_ratio=1.35,
                       close_ratio=0.92,
                       close_weight=3.0,
                       energy_tol=1e-5):
    work = df.copy().reset_index(drop=True)

    points_start = work[["x", "y"]].to_numpy(float)
    points = points_start.copy()

    h, w = img_shape[:2]

    max_total_move = max_total_move_ratio * pitch
    max_step = max_step_ratio * pitch

    e0 = regularization_energy(
        points,
        pitch,
        fixed_r,
        neighbor_low_ratio=neighbor_low_ratio,
        neighbor_high_ratio=neighbor_high_ratio,
        close_ratio=close_ratio,
    )

    history = [{
        "phase": phase,
        "iter": -1,
        "accepted": 1,
        "energy": e0,
        "mean_move_px": 0.0,
        "max_move_px": 0.0,
        "step_scale": 0.0,
    }]

    prev_e = e0

    for it in range(max_iters):
        forces = compute_forces(
            points,
            pitch,
            fixed_r,
            neighbor_low_ratio=neighbor_low_ratio,
            neighbor_high_ratio=neighbor_high_ratio,
            close_ratio=close_ratio,
            close_weight=close_weight,
        )

        fnorm = np.sqrt((forces ** 2).sum(axis=1))

        if float(fnorm.max()) < 1e-9:
            break

        disp = np.zeros_like(points)
        m = fnorm > 1e-9

        step_len = np.minimum(max_step, fnorm[m] * force_gain * pitch)
        disp[m] = forces[m] / fnorm[m, None] * step_len[:, None]

        accepted = False
        scale = 1.0

        for attempt in range(8):
            proposed = points + disp * scale

            # Do not apply the edge cut here; only clamp centers to image bounds.
            proposed[:, 0] = np.clip(proposed[:, 0], 0, w - 1)
            proposed[:, 1] = np.clip(proposed[:, 1], 0, h - 1)

            total_vec = proposed - points_start
            total_d = np.sqrt((total_vec ** 2).sum(axis=1))
            too_far = total_d > max_total_move

            if np.any(too_far):
                factor = max_total_move / np.maximum(total_d, 1e-9)
                proposed[too_far] = points_start[too_far] + total_vec[too_far] * factor[too_far, None]

            new_e = regularization_energy(
                proposed,
                pitch,
                fixed_r,
                neighbor_low_ratio=neighbor_low_ratio,
                neighbor_high_ratio=neighbor_high_ratio,
                close_ratio=close_ratio,
            )

            if new_e <= prev_e:
                points = proposed
                accepted = True
                prev_e = new_e

                total_move = np.sqrt(((points - points_start) ** 2).sum(axis=1))

                history.append({
                    "phase": phase,
                    "iter": it,
                    "accepted": 1,
                    "energy": float(new_e),
                    "mean_move_px": float(total_move.mean()),
                    "max_move_px": float(total_move.max()),
                    "step_scale": float(scale),
                })
                break

            scale *= 0.5

        if not accepted:
            break

        if len(history) >= 3:
            if abs(history[-2]["energy"] - history[-1]["energy"]) < energy_tol:
                break

    out = work.copy()
    out[f"x_before_{phase}"] = points_start[:, 0]
    out[f"y_before_{phase}"] = points_start[:, 1]

    out["x"] = points[:, 0]
    out["y"] = points[:, 1]

    out[f"move_dx_{phase}"] = out["x"] - out[f"x_before_{phase}"]
    out[f"move_dy_{phase}"] = out["y"] - out[f"y_before_{phase}"]
    out[f"move_dist_{phase}"] = np.sqrt(out[f"move_dx_{phase}"] ** 2 + out[f"move_dy_{phase}"] ** 2)

    if "x_initial" not in out.columns:
        out["x_initial"] = points_start[:, 0]
    if "y_initial" not in out.columns:
        out["y_initial"] = points_start[:, 1]

    out["move_dx_total_from_initial"] = out["x"] - out["x_initial"]
    out["move_dy_total_from_initial"] = out["y"] - out["y_initial"]
    out["move_dist_total_from_initial"] = np.sqrt(
        out["move_dx_total_from_initial"] ** 2 + out["move_dy_total_from_initial"] ** 2
    )

    out["pitch"] = float(pitch)
    out["fixed_r"] = float(fixed_r)
    out["r"] = float(fixed_r)

    return out, pd.DataFrame(history)


def circular_mean_mod60(angles):
    if len(angles) == 0:
        return 0.0

    z = np.exp(1j * 6.0 * np.asarray(angles, dtype=float)).mean()
    theta = np.angle(z) / 6.0
    theta = theta % (math.pi / 3.0)

    return float(theta)


def estimate_global_theta(points, pitch):
    if len(points) < 5:
        return 0.0

    tree = make_tree(points)
    low = 0.70 * pitch
    high = 1.35 * pitch

    angles = []
    step = max(1, len(points) // 700)

    for p in points[::step]:
        idxs = query_ball(tree, points, p, high)
        for j in idxs:
            q = points[j]
            d = float(np.linalg.norm(q - p))
            if low <= d <= high:
                angles.append(math.atan2(q[1] - p[1], q[0] - p[0]))

    return circular_mean_mod60(angles)


def estimate_local_theta(points, tree, center, pitch, global_theta):
    low = 0.70 * pitch
    high = 1.35 * pitch

    idxs = query_ball(tree, points, center, high)
    angles = []

    c = np.asarray(center, dtype=float)

    for j in idxs:
        p = points[j]
        d = float(np.linalg.norm(p - c))
        if low <= d <= high:
            angles.append(math.atan2(p[1] - c[1], p[0] - c[0]))

    if len(angles) < 2:
        return float(global_theta)

    return circular_mean_mod60(angles)


def support_stats(candidate, current_points, original_mask, current_tree, theta, pitch,
                  low_ratio=0.70, high_ratio=1.35):
    low = low_ratio * pitch
    high = high_ratio * pitch

    idxs = query_ball(current_tree, current_points, candidate, high)

    support_n = 0
    orig_support_n = 0
    dirs = set()

    c = np.asarray(candidate, dtype=float)

    for j in idxs:
        p = current_points[j]
        d = float(np.linalg.norm(p - c))

        if d < low or d > high:
            continue

        support_n += 1

        if original_mask[j]:
            orig_support_n += 1

        ang = math.atan2(p[1] - c[1], p[0] - c[0])
        sector = int(round((ang - theta) / (math.pi / 3.0))) % 6
        dirs.add(sector)

    return int(support_n), int(orig_support_n), int(len(dirs))


def candidate_overlap_ok(candidate, current_points, current_tree, fixed_r,
                         max_sum_frac=0.33, max_pair_frac=0.33):
    if len(current_points) == 0:
        return True, 0.0, 0.0

    idxs = query_ball(current_tree, current_points, candidate, 2.0 * fixed_r)

    if len(idxs) == 0:
        return True, 0.0, 0.0

    pts = current_points[idxs]
    d = np.sqrt(((pts - np.asarray(candidate)) ** 2).sum(axis=1))

    area = equal_radius_overlap_area_by_distance(d, fixed_r)
    circle_area = math.pi * fixed_r * fixed_r

    sum_frac = float(area.sum() / max(circle_area, 1e-9))
    pair_frac = float(area.max() / max(circle_area, 1e-9)) if len(area) else 0.0

    ok = (sum_frac <= max_sum_frac) and (pair_frac <= max_pair_frac)

    return ok, sum_frac, pair_frac


def generate_candidates(current_df, pitch, fixed_r, img_shape, mode, iteration, phase, args):
    current_df = current_df.reset_index(drop=True)

    points = current_df[["x", "y"]].to_numpy(float)
    tree = make_tree(points)

    if "is_gapfill" in current_df.columns:
        original_mask = current_df["is_gapfill"].fillna(0).astype(int).to_numpy() == 0
    else:
        original_mask = np.ones(len(current_df), dtype=bool)

    global_theta = estimate_global_theta(points, pitch)

    h, w = img_shape[:2]

    rows = []

    for i, row in current_df.iterrows():
        center = np.asarray([float(row["x"]), float(row["y"])], dtype=float)
        theta = estimate_local_theta(points, tree, center, pitch, global_theta)

        for k in range(6):
            ang = theta + k * math.pi / 3.0
            cand = center + args.candidate_distance_ratio * pitch * np.asarray(
                [math.cos(ang), math.sin(ang)],
                dtype=float,
            )

            x, y = float(cand[0]), float(cand[1])

            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            if nearest_distance(points, cand) < args.existing_exclude_ratio * pitch:
                continue

            support_n, orig_support_n, support_dirs = support_stats(
                cand,
                points,
                original_mask,
                tree,
                theta,
                pitch,
                low_ratio=args.support_low_ratio,
                high_ratio=args.support_high_ratio,
            )

            if mode == "strict":
                if support_n < args.strict_min_support:
                    continue
                if orig_support_n < args.strict_min_orig_support:
                    continue
                if support_dirs < args.strict_min_support_dirs:
                    continue
            else:
                if support_n < args.recall_min_support:
                    continue
                if orig_support_n < args.recall_min_orig_support:
                    continue
                if support_dirs < args.recall_min_support_dirs:
                    continue

            ok, sum_frac, pair_frac = candidate_overlap_ok(
                cand,
                points,
                tree,
                fixed_r,
                max_sum_frac=args.max_overlap_sum_frac,
                max_pair_frac=args.max_overlap_pair_frac,
            )

            if not ok:
                continue

            rows.append({
                "x": x,
                "y": y,
                "r": float(fixed_r),
                "pitch": float(pitch),
                "fixed_r": float(fixed_r),
                "is_gapfill": 1,
                "gapfill_phase": phase,
                "gapfill_mode": mode,
                "gapfill_iter": int(iteration),
                "gapfill_from_index": int(i),
                "gapfill_dir_k": int(k),
                "gapfill_theta": float(theta),
                "gapfill_support_n": support_n,
                "gapfill_orig_support_n": orig_support_n,
                "gapfill_support_dirs": support_dirs,
                "gapfill_overlap_sum_frac": sum_frac,
                "gapfill_overlap_pair_frac": pair_frac,
                "x_initial": x,
                "y_initial": y,
            })

    if not rows:
        return pd.DataFrame()

    cand_df = pd.DataFrame(rows)

    cand_df = cand_df.sort_values(
        [
            "gapfill_support_n",
            "gapfill_orig_support_n",
            "gapfill_support_dirs",
            "gapfill_overlap_sum_frac",
            "gapfill_overlap_pair_frac",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    accepted = []
    accepted_points = []

    for _, row in cand_df.iterrows():
        cand = np.asarray([row["x"], row["y"]], dtype=float)

        if accepted_points:
            d = np.sqrt(((np.asarray(accepted_points) - cand) ** 2).sum(axis=1))
            if float(d.min()) < args.candidate_nms_ratio * pitch:
                continue

        accepted.append(row.to_dict())
        accepted_points.append(cand)

    return pd.DataFrame(accepted)


def run_gapfill_phase(current, pitch, fixed_r, img_shape, phase, strict_iters, recall_iters, args):
    current = current.copy().reset_index(drop=True)
    history = []

    for it in range(strict_iters):
        cand = generate_candidates(current, pitch, fixed_r, img_shape, "strict", it, phase, args)
        added_n = len(cand)

        if added_n:
            current = pd.concat([current, cand], ignore_index=True)

        history.append({
            "phase": phase,
            "mode": "strict",
            "iter": it,
            "added_n": added_n,
            "current_n": len(current),
        })

    for it in range(recall_iters):
        cand = generate_candidates(current, pitch, fixed_r, img_shape, "recall", it, phase, args)
        added_n = len(cand)

        if added_n:
            current = pd.concat([current, cand], ignore_index=True)

        history.append({
            "phase": phase,
            "mode": "recall",
            "iter": it,
            "added_n": added_n,
            "current_n": len(current),
        })

    return current.reset_index(drop=True), pd.DataFrame(history)


def draw_overlay(img, final_df, removed_df, title, out_path, preview_w=2200):
    canvas = img.copy()

    fixed_r = float(final_df["fixed_r"].median()) if len(final_df) and "fixed_r" in final_df.columns else 17.0

    # movement from original position: blue line
    for _, row in final_df.iterrows():
        if "x_initial" in row and "y_initial" in row:
            x0 = float(row["x_initial"])
            y0 = float(row["y_initial"])
            x1 = float(row["x"])
            y1 = float(row["y"])
            d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if d >= 0.5:
                cv2.line(
                    canvas,
                    (int(round(x0)), int(round(y0))),
                    (int(round(x1)), int(round(y1))),
                    (255, 0, 0),
                    1,
                )

    # removed: orange
    for _, row in removed_df.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["r"]))) if "r" in row and pd.notna(row["r"]) else int(round(fixed_r))
        cv2.circle(canvas, (x, y), max(1, r), (0, 165, 255), 1)

    # original kept: green
    orig = final_df[final_df["is_gapfill"].fillna(0).astype(int) == 0]
    for _, row in orig.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["r"])))
        cv2.circle(canvas, (x, y), max(1, r), (0, 255, 0), 1)
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

    # gapfill kept: magenta
    gap = final_df[final_df["is_gapfill"].fillna(0).astype(int) == 1]
    for _, row in gap.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        r = int(round(float(row["r"])))
        cv2.circle(canvas, (x, y), max(1, r), (255, 0, 255), 1)
        cv2.circle(canvas, (x, y), 2, (255, 0, 255), -1)

    cv2.rectangle(canvas, (10, 10), (min(canvas.shape[1] - 1, 1900), 72), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        title[:220],
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
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

    ap.add_argument("--fixed_radius_ratio", type=float, default=0.46)

    # phase 0 regularization
    ap.add_argument("--phase0_max_move_ratio", type=float, default=0.12)
    ap.add_argument("--phase0_max_iters", type=int, default=30)

    # phase A gapfill
    ap.add_argument("--phaseA_strict_iters", type=int, default=2)
    ap.add_argument("--phaseA_recall_iters", type=int, default=3)

    # phase B regularization
    ap.add_argument("--phaseB_max_move_ratio", type=float, default=0.06)
    ap.add_argument("--phaseB_max_iters", type=int, default=20)

    # phase C gapfill
    ap.add_argument("--phaseC_strict_iters", type=int, default=0)
    ap.add_argument("--phaseC_recall_iters", type=int, default=5)

    # shared regularization
    ap.add_argument("--max_step_ratio", type=float, default=0.025)
    ap.add_argument("--force_gain", type=float, default=0.25)
    ap.add_argument("--neighbor_low_ratio", type=float, default=0.70)
    ap.add_argument("--neighbor_high_ratio", type=float, default=1.35)
    ap.add_argument("--close_ratio", type=float, default=0.92)
    ap.add_argument("--close_weight", type=float, default=3.0)
    ap.add_argument("--energy_tol", type=float, default=1e-5)

    # shared gapfill
    ap.add_argument("--candidate_distance_ratio", type=float, default=1.00)
    ap.add_argument("--existing_exclude_ratio", type=float, default=0.52)
    ap.add_argument("--candidate_nms_ratio", type=float, default=0.52)
    ap.add_argument("--support_low_ratio", type=float, default=0.70)
    ap.add_argument("--support_high_ratio", type=float, default=1.35)

    ap.add_argument("--strict_min_support", type=int, default=3)
    ap.add_argument("--strict_min_orig_support", type=int, default=2)
    ap.add_argument("--strict_min_support_dirs", type=int, default=2)

    ap.add_argument("--recall_min_support", type=int, default=2)
    ap.add_argument("--recall_min_orig_support", type=int, default=0)
    ap.add_argument("--recall_min_support_dirs", type=int, default=2)

    ap.add_argument("--max_overlap_sum_frac", type=float, default=0.33)
    ap.add_argument("--max_overlap_pair_frac", type=float, default=0.33)
    ap.add_argument("--final_overlap_frac_thr", type=float, default=0.33)

    ap.add_argument("--preview_w", type=int, default=2200)
    ap.add_argument("--clean", action="store_true")

    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    per_root = out_root / "per_image"
    seed_root = out_root / "07_alternating_regularized_localhex_seed" / args.split
    overlay_root = out_root / "_COLLECTED_alternating_regularized_localhex_overlays"
    summary_root = out_root / "summaries"

    for d in [per_root, seed_root, overlay_root, summary_root]:
        d.mkdir(parents=True, exist_ok=True)

    print("[INFO] Alternating regularization + local hex gapfill")
    print("[INFO] input_root:", input_root)
    print("[INFO] edge/white cut: NOT applied here")

    rows = []

    images = list_images(image_dir)

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        print(f"\n[{idx}/{len(images)}] {stem}", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("  [WARN] cannot read image")
            continue

        seed_df, input_csv = load_seed(input_root, args.split, stem)

        if seed_df is None:
            print("  [SKIP] missing input seed:", input_csv)
            continue

        if len(seed_df) == 0 or not {"x", "y"}.issubset(seed_df.columns):
            print("  [SKIP] invalid seed")
            continue

        pitch, fixed_r = get_pitch_fixedr(seed_df, fixed_radius_ratio=args.fixed_radius_ratio)
        current = initialize_seed_df(seed_df, pitch, fixed_r)

        per_dir = per_root / stem
        per_dir.mkdir(parents=True, exist_ok=True)

        current.to_csv(per_dir / "01_input_gray_step1_seed.csv", index=False)

        # Phase 0: regularize original Step 1 seed
        current, hist0 = regularize_centers(
            current,
            img.shape,
            pitch,
            fixed_r,
            phase="phase0_reg012",
            max_iters=args.phase0_max_iters,
            max_total_move_ratio=args.phase0_max_move_ratio,
            max_step_ratio=args.max_step_ratio,
            force_gain=args.force_gain,
            neighbor_low_ratio=args.neighbor_low_ratio,
            neighbor_high_ratio=args.neighbor_high_ratio,
            close_ratio=args.close_ratio,
            close_weight=args.close_weight,
            energy_tol=args.energy_tol,
        )

        current, removed0 = final_overlap_cleanup(
            current,
            fixed_r=fixed_r,
            overlap_thr=args.final_overlap_frac_thr,
            phase="phase0_after_reg_cleanup",
        )

        current.to_csv(per_dir / "02_phase0_after_regularization_cleanup.csv", index=False)
        hist0.to_csv(per_dir / "02_phase0_regularization_history.csv", index=False)
        removed0.to_csv(per_dir / "02_phase0_removed_overlap.csv", index=False)

        # Phase A: strict 2 + recall 3
        current, histA = run_gapfill_phase(
            current,
            pitch,
            fixed_r,
            img.shape,
            phase="phaseA_strict2_recall3",
            strict_iters=args.phaseA_strict_iters,
            recall_iters=args.phaseA_recall_iters,
            args=args,
        )

        current.to_csv(per_dir / "03_phaseA_after_gapfill_before_cleanup.csv", index=False)

        current, removedA = final_overlap_cleanup(
            current,
            fixed_r=fixed_r,
            overlap_thr=args.final_overlap_frac_thr,
            phase="phaseA_after_gapfill_cleanup",
        )

        current.to_csv(per_dir / "04_phaseA_after_gapfill_cleanup.csv", index=False)
        histA.to_csv(per_dir / "04_phaseA_gapfill_history.csv", index=False)
        removedA.to_csv(per_dir / "04_phaseA_removed_overlap.csv", index=False)

        # Phase B: weaker regularization
        current, histB = regularize_centers(
            current,
            img.shape,
            pitch,
            fixed_r,
            phase="phaseB_reg006",
            max_iters=args.phaseB_max_iters,
            max_total_move_ratio=args.phaseB_max_move_ratio,
            max_step_ratio=args.max_step_ratio,
            force_gain=args.force_gain,
            neighbor_low_ratio=args.neighbor_low_ratio,
            neighbor_high_ratio=args.neighbor_high_ratio,
            close_ratio=args.close_ratio,
            close_weight=args.close_weight,
            energy_tol=args.energy_tol,
        )

        current, removedB = final_overlap_cleanup(
            current,
            fixed_r=fixed_r,
            overlap_thr=args.final_overlap_frac_thr,
            phase="phaseB_after_reg_cleanup",
        )

        current.to_csv(per_dir / "05_phaseB_after_regularization_cleanup.csv", index=False)
        histB.to_csv(per_dir / "05_phaseB_regularization_history.csv", index=False)
        removedB.to_csv(per_dir / "05_phaseB_removed_overlap.csv", index=False)

        # Phase C: recall 5
        current, histC = run_gapfill_phase(
            current,
            pitch,
            fixed_r,
            img.shape,
            phase="phaseC_recall5",
            strict_iters=args.phaseC_strict_iters,
            recall_iters=args.phaseC_recall_iters,
            args=args,
        )

        current.to_csv(per_dir / "06_phaseC_after_gapfill_before_final_cleanup.csv", index=False)

        final_df, removedC = final_overlap_cleanup(
            current,
            fixed_r=fixed_r,
            overlap_thr=args.final_overlap_frac_thr,
            phase="phaseC_final_cleanup",
        )

        final_df.to_csv(per_dir / "07_final_alternating_regularized_localhex_kept.csv", index=False)
        histC.to_csv(per_dir / "07_phaseC_gapfill_history.csv", index=False)
        removedC.to_csv(per_dir / "08_phaseC_removed_overlap.csv", index=False)

        removed_all = pd.concat(
            [removed0, removedA, removedB, removedC],
            ignore_index=True,
            sort=False,
        )
        removed_all.to_csv(per_dir / "09_all_removed_overlap.csv", index=False)

        out_seed_dir = seed_root / stem
        out_seed_dir.mkdir(parents=True, exist_ok=True)

        final_csv = out_seed_dir / "cell_predictions_8class.csv"
        final_df.to_csv(final_csv, index=False)

        kept_gapfill_n = int((final_df["is_gapfill"].fillna(0).astype(int) == 1).sum())
        kept_original_n = int((final_df["is_gapfill"].fillna(0).astype(int) == 0).sum())

        max_overlap = float(final_df["final_overlap_frac"].max()) if len(final_df) and "final_overlap_frac" in final_df.columns else np.nan
        mean_move = float(final_df["move_dist_total_from_initial"].fillna(0).mean()) if "move_dist_total_from_initial" in final_df.columns else 0.0
        max_move = float(final_df["move_dist_total_from_initial"].fillna(0).max()) if "move_dist_total_from_initial" in final_df.columns else 0.0

        title = (
            f"{stem} | alt reg+hex | pitch={pitch:.2f} fixedR={fixed_r:.2f} "
            f"in={len(seed_df)} final={len(final_df)} gap={kept_gapfill_n} "
            f"rm={len(removed_all)} maxOv={max_overlap:.3f}"
        )

        overlay_path = overlay_root / f"{stem}_alternating_regularized_localhex.jpg"
        draw_overlay(img, final_df, removed_all, title, overlay_path, preview_w=args.preview_w)

        print(
            f"  input={len(seed_df):5d} final={len(final_df):5d} "
            f"orig={kept_original_n:5d} gapfill={kept_gapfill_n:5d} "
            f"removed={len(removed_all):4d} "
            f"pitch={pitch:.2f} fixedR={fixed_r:.2f} "
            f"meanMove={mean_move:.2f} maxMove={max_move:.2f} "
            f"maxOv={max_overlap:.3f}"
        )

        rows.append({
            "image": stem,
            "input_seed_n": len(seed_df),
            "final_seed_n": len(final_df),
            "kept_original_n": kept_original_n,
            "kept_gapfill_n": kept_gapfill_n,
            "removed_total_overlap_n": len(removed_all),
            "removed_phase0_n": len(removed0),
            "removed_phaseA_n": len(removedA),
            "removed_phaseB_n": len(removedB),
            "removed_phaseC_n": len(removedC),
            "pitch": float(pitch),
            "fixed_r": float(fixed_r),
            "phase0_max_move_ratio": args.phase0_max_move_ratio,
            "phaseB_max_move_ratio": args.phaseB_max_move_ratio,
            "mean_move_px": mean_move,
            "max_move_px": max_move,
            "max_final_overlap_frac": max_overlap,
            "input_csv": str(input_csv),
            "final_seed_csv": str(final_csv),
            "overlay": str(overlay_path),
        })

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "alternating_regularized_localhex_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[DONE]")
    print("summary:", summary_csv)
    print("seed root:", seed_root)
    print("overlays:", overlay_root)


if __name__ == "__main__":
    main()
