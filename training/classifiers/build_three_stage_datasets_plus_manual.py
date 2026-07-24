#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

STAGE1_CLASSES = ["Egg", "Larvae", "Capped_group", "Pollen", "Uncapped_honey", "Empty_cell", "Unbuilt_cell"]
STAGE2_CLASSES = ["Capped_honey", "Capped_pupae"]
STAGE3_CLASSES = ["Egg", "Empty_cell"]


def iter_imgs(d: Path):
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() or p.is_symlink() if p.suffix.lower() in IMG_EXTS])


def link_resolved(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        real = src.resolve(strict=True)
    except Exception:
        return "broken_skip"
    try:
        dst.symlink_to(real)  # absolute symlink; robust after moving/copying dataset root
        return "symlink"
    except Exception:
        shutil.copy2(real, dst)
        return "copy_fallback"


def add_split(src_root: Path, dst_root: Path, split: str, classes: list[str], prefix: str, rows: list[dict]):
    for cls in classes:
        src_dir = src_root / split / cls
        imgs = iter_imgs(src_dir)
        print(f"add {prefix:12s} {split:5s} {cls:16s} n={len(imgs)}")
        for i, src in enumerate(imgs):
            dst = dst_root / split / cls / f"{prefix}__{src.name}"
            action = link_resolved(src, dst)
            rows.append({"stage_root": str(dst_root), "split": split, "class": cls, "source": str(src), "dest": str(dst), "action": action})


def count_valid(root: Path, classes: list[str]):
    for split in ["train", "val", "test"]:
        total = 0
        for cls in classes:
            total += len(iter_imgs(root / split / cls))
        print(f"{root.name}/{split}: {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Project data root containing the base classifier datasets.",
    )
    ap.add_argument("--manual_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    base = args.base
    out_root = args.out_root
    manual = args.manual_root
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Base datasets
    s1_train_base = base / "cell_cls_7class_cellonly_cappedgroup_current_detection_5000crop_crop080_320_hardEgg6_Larvae3"
    s1_eval_base = base / "cell_cls_7class_cellonly_cappedgroup_current_detection_5000crop_crop080_320"
    s2_base = base / "cell_cls_binary_capped_current_detection_5000crop_crop180_320"
    s3_base = base / "cell_cls_binary_egg_empty_current_detection_5000crop_crop045_384"

    stages = {
        "stage1": (STAGE1_CLASSES, s1_train_base, s1_eval_base, manual / "stage1", out_root / "stage1"),
        "stage2": (STAGE2_CLASSES, s2_base, s2_base, manual / "stage2", out_root / "stage2"),
        "stage3": (STAGE3_CLASSES, s3_base, s3_base, manual / "stage3", out_root / "stage3"),
    }

    rows = []
    for stage, (classes, train_base, eval_base, manual_stage, dst) in stages.items():
        print(f"\n===== build {stage} =====")
        # train: base + manual
        add_split(train_base, dst, "train", classes, "base", rows)
        if manual_stage.exists():
            add_split(manual_stage, dst, "train", classes, "manual", rows)
        else:
            print(f"[WARN] no manual crop folder for {stage}: {manual_stage}")
        # val/test: base only, unchanged for comparable validation/testing
        add_split(eval_base, dst, "val", classes, "base", rows)
        add_split(eval_base, dst, "test", classes, "base", rows)
        count_valid(dst, classes)

    manifest = out_root / "dataset_build_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage_root", "split", "class", "source", "dest", "action"])
        w.writeheader(); w.writerows(rows)
    print("\nDONE")
    print("out_root:", out_root)
    print("manifest:", manifest)

if __name__ == "__main__":
    main()
