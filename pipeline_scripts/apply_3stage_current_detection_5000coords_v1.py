#!/usr/bin/env python3
"""
Apply the current 3-stage honey-bee cell-content classifier cascade to detected centers.

Intended current-coordinate setup:
  - Centers are from 3000-px detection images.
  - Classifier crops are extracted from 5000-px frame-cropped images.
  - Use --coord_scale 1.6666667 to scale x/y and estimated_step.

Stages:
  1) cell-only 7-class classifier:
     Egg, Larvae, Capped_group, Pollen, Uncapped_honey, Empty_cell, Unbuilt_cell
  2) capped binary classifier for Stage1 Capped_group:
     Capped_honey vs Capped_pupae
  3) egg-empty binary classifier for Stage1 Egg or Empty_cell:
     Egg vs Empty_cell

Outputs:
  - per-image CSVs with stage predictions and final 8-class labels
  - overlay preview images
  - optional test metrics if --label_root is provided
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP", ".WEBP"}

FINAL_NAMES = [
    "Egg",
    "Larvae",
    "Capped_pupae",
    "Pollen",
    "Capped_honey",
    "Uncapped_honey",
    "Empty_cell",
    "Unbuilt_cell",
]
FINAL_ID = {name: i for i, name in enumerate(FINAL_NAMES)}

OLD_TO_FINAL = {
    0: 0,  # Egg
    1: 1,  # Larvae
    3: 2,  # Capped pupae
    4: 3,  # Pollen
    5: 4,  # Capped honey
    6: 5,  # Uncapped honey
    7: 6,  # Empty cell
    8: 7,  # Unbuilt cell
}

COLORS = {
    "Egg": (0, 255, 255),
    "Larvae": (0, 255, 0),
    "Capped_pupae": (255, 128, 0),
    "Pollen": (0, 128, 255),
    "Capped_honey": (0, 165, 255),
    "Uncapped_honey": (255, 0, 255),
    "Empty_cell": (220, 220, 220),
    "Unbuilt_cell": (255, 255, 0),
}


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def get_model_names(model: YOLO) -> Dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            cc = cand.lower()
            if cc in cl:
                return c
    return None


def get_xy_step(df: pd.DataFrame, coord_scale: float, step_scale: Optional[float]) -> pd.DataFrame:
    xcol = find_col(df, ["x", "cx", "center_x", "x_center", "cell_x", "pred_x", "x_px"])
    ycol = find_col(df, ["y", "cy", "center_y", "y_center", "cell_y", "pred_y", "y_px"])
    if xcol is None or ycol is None:
        raise ValueError(f"Cannot find x/y columns in CSV. Columns={list(df.columns)}")

    out = df.copy()
    out["x_det"] = pd.to_numeric(out[xcol], errors="coerce")
    out["y_det"] = pd.to_numeric(out[ycol], errors="coerce")
    out = out.dropna(subset=["x_det", "y_det"]).reset_index(drop=True)

    out["x"] = out["x_det"].astype(float) * coord_scale
    out["y"] = out["y_det"].astype(float) * coord_scale

    step_col = find_col(out, ["estimated_step", "step", "pitch", "cell_step"])
    if step_col is not None:
        step = pd.to_numeric(out[step_col], errors="coerce").astype(float)
        out["estimated_step_det"] = step
        out["estimated_step"] = step * (coord_scale if step_scale is None else step_scale)
    else:
        xy = out[["x", "y"]].values.astype(float)
        est = estimate_step_from_nn(xy)
        out["estimated_step_det"] = est / coord_scale
        out["estimated_step"] = est

    med = float(np.nanmedian(out["estimated_step"].values)) if len(out) else 61.294 * coord_scale
    if not np.isfinite(med) or med <= 0:
        med = 61.294 * coord_scale
    out["estimated_step"] = out["estimated_step"].fillna(med)
    out.loc[out["estimated_step"] <= 0, "estimated_step"] = med
    return out


def estimate_step_from_nn(xy: np.ndarray, default: float = 61.294) -> float:
    if len(xy) < 10:
        return default
    sample = xy
    if len(sample) > 2500:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), 2500, replace=False)]
    vals = []
    for i in range(len(sample)):
        d = np.sqrt(((sample - sample[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        vals.append(np.min(d))
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return default
    med = float(np.median(vals))
    if not np.isfinite(med) or med <= 0:
        return default
    return med


def crop_cell_bgr(img: np.ndarray, x: float, y: float, step: float, crop_half_step: float, out_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = float(step) * float(crop_half_step)
    x1 = int(math.floor(x - half))
    y1 = int(math.floor(y - half))
    x2 = int(math.ceil(x + half))
    y2 = int(math.ceil(y + half))

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    xx1 = max(0, x1)
    yy1 = max(0, y1)
    xx2 = min(w, x2)
    yy2 = min(h, y2)

    crop = img[yy1:yy2, xx1:xx2]
    if crop.size == 0:
        crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        return crop

    if any(v > 0 for v in [pad_l, pad_t, pad_r, pad_b]):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE)

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def predict_crops(
    model: YOLO,
    img: np.ndarray,
    centers: pd.DataFrame,
    indices: List[int],
    crop_half_step: float,
    imgsz: int,
    batch_size: int,
    device: str,
) -> Tuple[List[str], List[int], List[float]]:
    names = get_model_names(model)
    pred_names: List[str] = []
    pred_ids: List[int] = []
    pred_confs: List[float] = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        pil_batch = []
        for i in batch_idx:
            r = centers.loc[i]
            crop = crop_cell_bgr(img, float(r["x"]), float(r["y"]), float(r["estimated_step"]), crop_half_step, imgsz)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_batch.append(Image.fromarray(crop_rgb))

        results = model.predict(source=pil_batch, imgsz=imgsz, batch=len(pil_batch), device=device, verbose=False)
        for res in results:
            top1 = int(res.probs.top1)
            conf = float(res.probs.top1conf.cpu().item()) if hasattr(res.probs.top1conf, "cpu") else float(res.probs.top1conf)
            pred_ids.append(top1)
            pred_names.append(names.get(top1, str(top1)))
            pred_confs.append(conf)

    return pred_names, pred_ids, pred_confs


def center_csv_for(center_root: Path, split: str, stem: str) -> Optional[Path]:
    exact = center_root / split / stem / "cell_predictions_8class.csv"
    if exact.exists():
        return exact

    # Fallback: strict-ish matching under split
    d = center_root / split
    if not d.exists():
        return None
    candidates = list(d.glob(f"{stem}/cell_predictions_8class.csv"))
    if candidates:
        return candidates[0]
    candidates = list(d.glob(f"*{stem}*/cell_predictions_8class.csv"))
    good = []
    for p in candidates:
        parent = p.parent.name
        if parent == stem or parent.startswith(stem + "_") or parent.startswith("new_" + stem):
            good.append(p)
    return good[0] if good else None


def iter_images(image_root: Path, split: str) -> List[Path]:
    d = image_root / split
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix in IMG_EXTS])


def final_from_stage1(stage1: str) -> str:
    n = norm_name(stage1)
    mapping = {
        "egg": "Egg",
        "larvae": "Larvae",
        "pollen": "Pollen",
        "uncappedhoney": "Uncapped_honey",
        "emptycell": "Empty_cell",
        "unbuiltcell": "Unbuilt_cell",
        "cappedgroup": "Capped_group",
    }
    return mapping.get(n, stage1)


def to_final_class_id(label: str) -> int:
    if label in FINAL_ID:
        return FINAL_ID[label]
    return -1


def read_yolo_label_polys(label_path: Path, w: int, h: int):
    items = []
    if not label_path.exists():
        return items
    for line in label_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            old_cls = int(float(parts[0]))
            vals = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
        except Exception:
            continue
        if old_cls not in OLD_TO_FINAL:
            continue
        final_cls = OLD_TO_FINAL[old_cls]

        if len(vals) == 4:
            xc, yc, bw, bh = vals
            if max(abs(xc), abs(yc), abs(bw), abs(bh)) <= 1.5:
                xc *= w; bw *= w; yc *= h; bh *= h
            x1, x2 = xc - bw/2, xc + bw/2
            y1, y2 = yc - bh/2, yc + bh/2
            poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            poly = vals.reshape(-1, 2)
            if np.nanmax(np.abs(poly)) <= 1.5:
                poly[:,0] *= w
                poly[:,1] *= h
        else:
            continue
        items.append((final_cls, FINAL_NAMES[final_cls], poly))
    return items


def assign_gt_by_center(x: float, y: float, items) -> Tuple[int, str]:
    pt = (float(x), float(y))
    best = None
    best_dist = -1e9
    for cls_id, cls_name, poly in items:
        dist = cv2.pointPolygonTest(poly.astype(np.float32), pt, True)
        if dist >= 0:
            if dist > best_dist:
                best = (cls_id, cls_name)
                best_dist = dist
    if best is None:
        return -1, "Unknown"
    return best


def evaluate_predictions(all_rows: List[dict], out_root: Path):
    labels = FINAL_NAMES
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)  # rows pred, cols true
    no_gt = 0
    for r in all_rows:
        gt = int(r.get("gt_class_id", -1))
        pr = int(r.get("final_class_id", -1))
        if gt < 0:
            no_gt += 1
            continue
        if 0 <= pr < n:
            cm[pr, gt] += 1

    metrics = []
    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total else 0.0

    for i, name in enumerate(labels):
        tp = cm[i, i]
        fp = cm[i, :].sum() - tp
        fn = cm[:, i].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = cm[:, i].sum()
        metrics.append({
            "class": name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    pd.DataFrame(metrics).to_csv(out_root / "test_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=[f"pred_{x}" for x in labels], columns=[f"true_{x}" for x in labels]).to_csv(out_root / "test_confusion_matrix.csv")

    with open(out_root / "test_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["accuracy", acc])
        w.writerow(["n_evaluated", int(total)])
        w.writerow(["n_correct", int(correct)])
        w.writerow(["n_no_gt", int(no_gt)])

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("3-stage cell-level confusion matrix")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_root / "test_confusion_matrix.png", dpi=200)
        plt.close(fig)

        colsum = cm.sum(axis=0, keepdims=True)
        norm = np.divide(cm, colsum, out=np.zeros_like(cm, dtype=float), where=colsum > 0)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("3-stage cell-level confusion matrix normalized by true class")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_root / "test_confusion_matrix_normalized.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print("[WARN] Could not save confusion matrix plots:", e)

    return acc, total, no_gt


def make_overlay(img: np.ndarray, rows: pd.DataFrame, out_path: Path, preview_width: int, label_every: int):
    out = img.copy()
    for idx, r in rows.iterrows():
        cls = str(r.get("final_class", "Unknown"))
        color = COLORS.get(cls, (255, 255, 255))
        x = int(round(float(r["x"])))
        y = int(round(float(r["y"])))
        rad = max(2, int(round(float(r.get("estimated_step", 20)) * 0.30)))
        cv2.circle(out, (x, y), rad, color, 2, lineType=cv2.LINE_AA)
        if label_every > 0 and idx % label_every == 0:
            cv2.putText(out, cls, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    if preview_width and preview_width > 0 and out.shape[1] > preview_width:
        scale = preview_width / out.shape[1]
        out = cv2.resize(out, (preview_width, int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True, type=Path)
    ap.add_argument("--center_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--stage1_model", required=True)
    ap.add_argument("--cap_model", required=True)
    ap.add_argument("--egg_model", required=True)
    ap.add_argument("--label_root", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--coord_scale", type=float, default=1.6666667)
    ap.add_argument("--step_scale", type=float, default=None)
    ap.add_argument("--stage1_half_step", type=float, default=0.80)
    ap.add_argument("--stage1_imgsz", type=int, default=320)
    ap.add_argument("--cap_half_step", type=float, default=1.80)
    ap.add_argument("--cap_imgsz", type=int, default=320)
    ap.add_argument("--egg_half_step", type=float, default=0.45)
    ap.add_argument("--egg_imgsz", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--preview_width", type=int, default=2200)
    ap.add_argument("--label_every", type=int, default=1)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    stage1_model = YOLO(args.stage1_model)
    cap_model = YOLO(args.cap_model)
    egg_model = YOLO(args.egg_model)

    print("Stage1 names:", get_model_names(stage1_model))
    print("Cap names:", get_model_names(cap_model))
    print("Egg names:", get_model_names(egg_model))

    all_eval_rows: List[dict] = []
    summary_rows = []

    for split in args.splits:
        images = iter_images(args.image_root, split)
        print(f"\n===== split={split} images={len(images)} =====")
        for img_path in images:
            stem = img_path.stem
            csv_path = center_csv_for(args.center_root, split, stem)
            if csv_path is None:
                print(f"[SKIP no center] {split}/{img_path.name}")
                summary_rows.append({"split": split, "image": img_path.name, "status": "no_center_csv"})
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[SKIP bad image] {img_path}")
                summary_rows.append({"split": split, "image": img_path.name, "status": "bad_image"})
                continue

            df0 = pd.read_csv(csv_path)
            centers = get_xy_step(df0, coord_scale=args.coord_scale, step_scale=args.step_scale)
            if len(centers) == 0:
                print(f"[SKIP no centers] {split}/{img_path.name}")
                summary_rows.append({"split": split, "image": img_path.name, "status": "no_centers"})
                continue

            indices = list(range(len(centers)))

            s1_names, s1_ids, s1_confs = predict_crops(
                stage1_model, img, centers, indices,
                args.stage1_half_step, args.stage1_imgsz, args.batch_size, args.device
            )
            centers["stage1_class"] = s1_names
            centers["stage1_class_id"] = s1_ids
            centers["stage1_conf"] = s1_confs

            centers["final_class"] = [final_from_stage1(x) for x in s1_names]
            centers["final_conf"] = s1_confs
            centers["cap_class"] = ""
            centers["cap_conf"] = np.nan
            centers["egg_class"] = ""
            centers["egg_conf"] = np.nan

            cap_idx = [i for i, x in enumerate(centers["stage1_class"]) if norm_name(x) == "cappedgroup"]
            if cap_idx:
                cap_names, cap_ids, cap_confs = predict_crops(
                    cap_model, img, centers, cap_idx,
                    args.cap_half_step, args.cap_imgsz, args.batch_size, args.device
                )
                for k, i in enumerate(cap_idx):
                    cname = cap_names[k]
                    # normalize expected names
                    if norm_name(cname) == "cappedhoney":
                        cname = "Capped_honey"
                    elif norm_name(cname) == "cappedpupae":
                        cname = "Capped_pupae"
                    centers.at[i, "cap_class"] = cname
                    centers.at[i, "cap_conf"] = cap_confs[k]
                    centers.at[i, "final_class"] = cname
                    centers.at[i, "final_conf"] = cap_confs[k]

            egg_idx = [i for i, x in enumerate(centers["stage1_class"]) if norm_name(x) in {"egg", "emptycell"}]
            if egg_idx:
                egg_names, egg_ids, egg_confs = predict_crops(
                    egg_model, img, centers, egg_idx,
                    args.egg_half_step, args.egg_imgsz, args.batch_size, args.device
                )
                for k, i in enumerate(egg_idx):
                    ename = egg_names[k]
                    if norm_name(ename) == "egg":
                        ename = "Egg"
                    elif norm_name(ename) == "emptycell":
                        ename = "Empty_cell"
                    centers.at[i, "egg_class"] = ename
                    centers.at[i, "egg_conf"] = egg_confs[k]
                    centers.at[i, "final_class"] = ename
                    centers.at[i, "final_conf"] = egg_confs[k]

            centers["final_class_id"] = [to_final_class_id(x) for x in centers["final_class"]]
            centers["raw_pred_class"] = centers["final_class_id"]
            centers["raw_pred_conf"] = centers["final_conf"]
            centers["display_class"] = centers["final_class"]
            centers["draw_radius"] = centers["estimated_step"] * 0.30

            # Optional GT assignment for quick cell-level test evaluation.
            if args.label_root is not None:
                label_path = args.label_root / split / f"{stem}.txt"
                h, w = img.shape[:2]
                gt_items = read_yolo_label_polys(label_path, w, h)
                gt_ids, gt_names = [], []
                for _, r in centers.iterrows():
                    gid, gname = assign_gt_by_center(float(r["x"]), float(r["y"]), gt_items)
                    gt_ids.append(gid)
                    gt_names.append(gname)
                centers["gt_class_id"] = gt_ids
                centers["gt_class"] = gt_names

            out_dir = args.out_root / split / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "cell_predictions_3stage.csv"
            centers.to_csv(out_csv, index=False)

            # Also write compatibility name for downstream scripts if useful.
            centers.to_csv(out_dir / "cell_predictions_8class.csv", index=False)

            make_overlay(
                img, centers,
                out_dir / f"{stem}_3stage_overlay.jpg",
                preview_width=args.preview_width,
                label_every=args.label_every,
            )

            counts = centers["final_class"].value_counts().to_dict()
            print(f"[OK] {split}/{stem}: n={len(centers)} cap_refined={len(cap_idx)} egg_refined={len(egg_idx)} counts={counts}")
            summary_rows.append({
                "split": split,
                "image": img_path.name,
                "status": "ok",
                "n_centers": len(centers),
                "n_cap_refined": len(cap_idx),
                "n_egg_refined": len(egg_idx),
                "out_csv": str(out_csv),
            })

            if args.label_root is not None:
                all_eval_rows.extend(centers.to_dict("records"))

    pd.DataFrame(summary_rows).to_csv(args.out_root / "apply_3stage_summary.csv", index=False)

    if args.label_root is not None:
        pd.DataFrame(all_eval_rows).to_csv(args.out_root / "all_predictions_with_gt.csv", index=False)
        acc, total, no_gt = evaluate_predictions(all_eval_rows, args.out_root)
        print("\n===== QUICK TEST EVALUATION =====")
        print("accuracy:", acc)
        print("n_evaluated:", total)
        print("n_no_gt:", no_gt)
        print("metrics:", args.out_root / "test_class_metrics.csv")
        print("confusion:", args.out_root / "test_confusion_matrix.png")

    print("\nDONE:", args.out_root)


if __name__ == "__main__":
    main()
