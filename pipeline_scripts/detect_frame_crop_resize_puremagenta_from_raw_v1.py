#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import math
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def configure_opencv(threads=1):
    """Reduce backend-dependent scheduling without changing image values."""
    cv2.setNumThreads(max(1, int(threads)))
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def list_images(image_dir, recursive=False):
    image_dir = Path(image_dir)
    if recursive:
        files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def polygon_mask_from_result(result, image_shape):
    h, w = image_shape[:2]

    masks = []

    if result.masks is not None and result.masks.xy is not None:
        for poly in result.masks.xy:
            if poly is None or len(poly) < 3:
                continue

            pts = np.asarray(poly, dtype=np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_i = np.round(pts).astype(np.int32)

            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts_i], 255)
            area = int((m > 0).sum())
            if area > 0:
                masks.append((m, area))

    if masks:
        masks.sort(key=lambda x: x[1], reverse=True)
        return masks[0][0]

    # Fallback: use the largest detection box as the mask when no polygon mask is available.
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, int(np.floor(x1)))
        y1 = max(0, int(np.floor(y1)))
        x2 = min(w, int(np.ceil(x2)))
        y2 = min(h, int(np.ceil(y2)))

        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        return m

    return None


def bbox_from_mask(mask, margin_ratio=0.02):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    h, w = mask.shape[:2]

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    bw = x2 - x1
    bh = y2 - y1

    margin = int(round(max(bw, bh) * margin_ratio))

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return x1, y1, x2, y2


def quantize_bbox_nearest(bbox, image_shape, quantum=0):
    """Snap ROI edges to a model-stride grid after the safety margin.

    Nearest-grid snapping is intentional. The upstream margin already protects
    the frame, while outward-only snapping can amplify a one-pixel backend
    difference when an edge lies on opposite sides of a grid boundary.
    """
    q = int(quantum)
    if q <= 1:
        return tuple(int(value) for value in bbox)

    h, w = image_shape[:2]

    def snap(value):
        return int(math.floor(float(value) / q + 0.5)) * q

    x1, y1, x2, y2 = [snap(value) for value in bbox]
    x1 = min(max(0, x1), max(0, w - 1))
    y1 = min(max(0, y1), max(0, h - 1))
    x2 = min(max(x1 + 1, x2), w)
    y2 = min(max(y1 + 1, y2), h)
    return x1, y1, x2, y2


def resize_crop_with_magenta(crop_img, crop_mask, target_long, bg_bgr=(255, 0, 255)):
    h, w = crop_img.shape[:2]

    scale = float(target_long) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_mask = resized_mask > 0

    out = resized_img.copy()
    out[~resized_mask] = bg_bgr

    out_mask = resized_mask.astype(np.uint8) * 255

    return out, out_mask, scale


def draw_detection_overlay(img, mask, bbox, out_path, preview_w=2200):
    canvas = img.copy()

    if mask is not None:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 0, 255), 4)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if preview_w and canvas.shape[1] > preview_w:
        scale = preview_w / canvas.shape[1]
        canvas = cv2.resize(canvas, (preview_w, int(round(canvas.shape[0] * scale))))

    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--split", default="new")
    ap.add_argument("--target_longs", type=int, nargs="+", default=[3000, 5000])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin_ratio", type=float, default=0.02)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--preview_w", type=int, default=2200)
    ap.add_argument("--device", default=None, help="Ultralytics device: cpu, 0, etc.")
    ap.add_argument(
        "--bbox_quantum",
        type=int,
        default=0,
        help=(
            "Snap the margin-expanded bbox to the nearest pixel grid. "
            "Use 32 for the platform-stable workflow; 0 preserves legacy behavior."
        ),
    )
    ap.add_argument("--opencv_threads", type=int, default=1)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    configure_opencv(args.opencv_threads)

    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    overlay_root = out_root / "_COLLECTED_frame_detection_bbox_overlays"
    preview_root = out_root / "_COLLECTED_crop3000_previews"
    summary_root = out_root / "summaries"

    overlay_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    rows = []

    images = list_images(image_dir, recursive=args.recursive)

    print("[INFO] image_dir:", image_dir)
    print("[INFO] model:", args.model)
    print("[INFO] out_root:", out_root)
    print("[INFO] n images:", len(images))
    print("[INFO] background: pure magenta BGR=(255,0,255)")
    print("[INFO] output format: PNG to preserve pure magenta")

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        print(f"\n[{idx}/{len(images)}] {stem}", flush=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print("  [WARN] cannot read image")
            continue

        predict_kwargs = {
            "source": str(img_path),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "verbose": False,
            "save": False,
        }
        if args.device not in (None, "", "auto"):
            predict_kwargs["device"] = args.device

        result = model.predict(**predict_kwargs)[0]

        mask = polygon_mask_from_result(result, img.shape)

        if mask is None:
            print("  [WARN] no frame mask detected")
            rows.append({
                "image": stem,
                "status": "no_frame_detected",
                "src_image": str(img_path),
            })
            continue

        raw_bbox = bbox_from_mask(mask, margin_ratio=args.margin_ratio)

        if raw_bbox is None:
            print("  [WARN] empty mask")
            rows.append({
                "image": stem,
                "status": "empty_mask",
                "src_image": str(img_path),
            })
            continue

        bbox = quantize_bbox_nearest(
            raw_bbox,
            img.shape,
            quantum=args.bbox_quantum,
        )
        x1, y1, x2, y2 = bbox
        raw_x1, raw_y1, raw_x2, raw_y2 = raw_bbox

        crop_img = img[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()

        draw_detection_overlay(
            img,
            mask,
            bbox,
            overlay_root / f"{stem}_frame_detection_bbox.jpg",
            preview_w=args.preview_w,
        )

        full_frame_area = int((mask > 0).sum())
        frame_area = int((crop_mask > 0).sum())
        bbox_area = int((x2 - x1) * (y2 - y1))
        outside_frac_in_bbox = 1.0 - frame_area / max(1, bbox_area)

        out_record = {
            "image": stem,
            "status": "ok",
            "src_image": str(img_path),
            "raw_bbox_x1": raw_x1,
            "raw_bbox_y1": raw_y1,
            "raw_bbox_x2": raw_x2,
            "raw_bbox_y2": raw_y2,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_w": x2 - x1,
            "bbox_h": y2 - y1,
            "frame_mask_area_px": frame_area,
            "full_frame_mask_area_px": full_frame_area,
            "bbox_area_px": bbox_area,
            "outside_mask_fraction_in_bbox": outside_frac_in_bbox,
            "bbox_quantum": int(args.bbox_quantum),
            "bbox_quantized": bool(tuple(raw_bbox) != tuple(bbox)),
        }

        for target_long in args.target_longs:
            out_img, out_mask, scale = resize_crop_with_magenta(
                crop_img,
                crop_mask,
                target_long=target_long,
                bg_bgr=(255, 0, 255),
            )

            img_dir = out_root / f"{'00' if target_long == 3000 else '01'}_frame_crop_masked_magenta_{target_long}" / "images" / args.split
            mask_dir = out_root / f"{'00' if target_long == 3000 else '01'}_frame_crop_masked_magenta_{target_long}" / "masks" / args.split

            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            out_img_path = img_dir / f"{stem}.png"
            out_mask_path = mask_dir / f"{stem}.png"

            cv2.imwrite(str(out_img_path), out_img)
            cv2.imwrite(str(out_mask_path), out_mask)

            out_record[f"out_image_{target_long}"] = str(out_img_path)
            out_record[f"out_mask_{target_long}"] = str(out_mask_path)
            out_record[f"scale_{target_long}"] = scale
            out_record[f"out_w_{target_long}"] = out_img.shape[1]
            out_record[f"out_h_{target_long}"] = out_img.shape[0]

            if target_long == 3000:
                preview = out_img.copy()
                if preview.shape[1] > args.preview_w:
                    s = args.preview_w / preview.shape[1]
                    preview = cv2.resize(preview, (args.preview_w, int(round(preview.shape[0] * s))))
                cv2.imwrite(str(preview_root / f"{stem}_crop3000_magenta_preview.jpg"), preview)

        rows.append(out_record)

        raw_text = (
            f" raw_bbox=({raw_x1},{raw_y1})-({raw_x2},{raw_y2})"
            if tuple(raw_bbox) != tuple(bbox)
            else ""
        )
        print(
            f"  bbox=({x1},{y1})-({x2},{y2}){raw_text} "
            f"outside_bbox_frac={outside_frac_in_bbox:.4f}"
        )

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "frame_detection_crop_magenta_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[DONE]")
    print("summary:", summary_csv)
    print("overlays:", overlay_root)
    print("crop previews:", preview_root)


if __name__ == "__main__":
    main()
