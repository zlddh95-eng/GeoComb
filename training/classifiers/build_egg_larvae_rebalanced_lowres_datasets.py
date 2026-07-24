#!/usr/bin/env python3
"""
Build rebalanced, low-resolution-augmented Stage 1 and Stage 3 datasets.

Source dataset:
  cell_cls_3stages_plus_new4_new5_manual_v1

Design:
- Stage 1:
    Egg total exposure ~= 4x original count
    Larvae total exposure ~= 2x original count
    Other classes retained without oversampling
- Stage 3:
    Egg total exposure ~= 4x original count
    Empty_cell sampled to approximately 2x the effective Egg count
    Half of selected Empty_cell samples are low-resolution variants
- Validation and test splits are copied unchanged.
- Original files are hard-linked when possible.
- Augmented variants are generated as actual JPEG files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

STAGE1_REQUIRED = {
    "egg",
    "larvae",
    "cappedgroup",
    "pollen",
    "uncappedhoney",
    "emptycell",
    "unbuiltcell",
}
STAGE3_REQUIRED = {"egg", "emptycell"}


def norm(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTS],
        key=lambda p: p.name.casefold(),
    )


def discover_stage_root(source_root: Path, required: set[str]) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in [source_root] + [p for p in source_root.rglob("*") if p.is_dir()]:
        train = path / "train"
        val = path / "val"
        if not train.is_dir() or not val.is_dir():
            continue
        classes = {norm(p.name) for p in train.iterdir() if p.is_dir()}
        if required.issubset(classes):
            candidates.append((len(path.parts), path))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a dataset root under {source_root} "
            f"with classes {sorted(required)}"
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def class_folder(stage_root: Path, split: str, class_norm: str) -> Path:
    split_dir = stage_root / split
    for folder in split_dir.iterdir():
        if folder.is_dir() and norm(folder.name) == class_norm:
            return folder
    raise FileNotFoundError(
        f"Class {class_norm!r} not found in {split_dir}"
    )


def unique_name(source: Path, prefix: str = "") -> str:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in source.stem
    )
    return f"{prefix}{safe_stem}_{digest}{source.suffix.lower()}"


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def lowres_variant(
    source: Path,
    destination: Path,
    rng: random.Random,
) -> dict[str, float | int | str]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size

        factor = rng.uniform(0.25, 0.55)
        low_w = max(8, int(round(width * factor)))
        low_h = max(8, int(round(height * factor)))

        im = im.resize((low_w, low_h), Image.Resampling.BILINEAR)
        im = im.resize((width, height), Image.Resampling.BICUBIC)

        blur_radius = 0.0
        if rng.random() < 0.60:
            blur_radius = rng.uniform(0.25, 1.10)
            im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        brightness = rng.uniform(0.85, 1.15)
        contrast = rng.uniform(0.82, 1.18)
        im = ImageEnhance.Brightness(im).enhance(brightness)
        im = ImageEnhance.Contrast(im).enhance(contrast)

        quality = rng.randint(45, 82)
        im.save(
            destination,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling=2,
        )

    return {
        "source": str(source),
        "destination": str(destination),
        "lowres_factor": factor,
        "blur_radius": blur_radius,
        "brightness": brightness,
        "contrast": contrast,
        "jpeg_quality": quality,
    }


def copy_split_unchanged(
    source_stage: Path,
    output_stage: Path,
    split: str,
) -> list[dict[str, str]]:
    rows = []
    source_split = source_stage / split
    if not source_split.is_dir():
        return rows

    for class_dir in sorted(
        [p for p in source_split.iterdir() if p.is_dir()],
        key=lambda p: p.name.casefold(),
    ):
        output_class = output_stage / split / class_dir.name
        for source in image_files(class_dir):
            destination = output_class / unique_name(source)
            hardlink_or_copy(source, destination)
            rows.append({
                "stage": output_stage.name,
                "split": split,
                "class": class_dir.name,
                "kind": "original",
                "source": str(source),
                "destination": str(destination),
            })
    return rows


def build_stage1(
    source_stage: Path,
    output_stage: Path,
    seed: int,
    egg_multiplier: int,
    larvae_multiplier: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for split in ("train", "val", "test"):
        rows.extend(copy_split_unchanged(source_stage, output_stage, split))

    augmentation_plan = {
        "egg": max(0, egg_multiplier - 1),
        "larvae": max(0, larvae_multiplier - 1),
    }

    for class_norm, extra_variants in augmentation_plan.items():
        source_class = class_folder(source_stage, "train", class_norm)
        output_class = class_folder(output_stage, "train", class_norm)
        originals = image_files(source_class)

        for source in originals:
            for variant_idx in range(extra_variants):
                aug_seed = rng.randrange(0, 2**31 - 1)
                local_rng = random.Random(aug_seed)
                digest = hashlib.sha1(
                    f"{source.resolve()}::{variant_idx}::{aug_seed}".encode("utf-8")
                ).hexdigest()[:10]
                destination = (
                    output_class
                    / f"{source.stem}_lowres_v{variant_idx + 1}_{digest}.jpg"
                )
                meta = lowres_variant(source, destination, local_rng)
                rows.append({
                    "stage": output_stage.name,
                    "split": "train",
                    "class": source_class.name,
                    "kind": "lowres_augmented",
                    "augmentation_seed": aug_seed,
                    **meta,
                })

    return rows


def build_stage3(
    source_stage: Path,
    output_stage: Path,
    seed: int,
    egg_multiplier: int,
    empty_to_egg_ratio: float,
    empty_lowres_probability: float,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    # Validation/test unchanged.
    for split in ("val", "test"):
        rows.extend(copy_split_unchanged(source_stage, output_stage, split))

    source_egg = class_folder(source_stage, "train", "egg")
    source_empty = class_folder(source_stage, "train", "emptycell")
    output_egg = output_stage / "train" / source_egg.name
    output_empty = output_stage / "train" / source_empty.name
    output_egg.mkdir(parents=True, exist_ok=True)
    output_empty.mkdir(parents=True, exist_ok=True)

    egg_originals = image_files(source_egg)
    empty_originals = image_files(source_empty)

    # Egg originals.
    for source in egg_originals:
        destination = output_egg / unique_name(source)
        hardlink_or_copy(source, destination)
        rows.append({
            "stage": output_stage.name,
            "split": "train",
            "class": source_egg.name,
            "kind": "original",
            "source": str(source),
            "destination": str(destination),
        })

    # Egg low-resolution variants.
    for source in egg_originals:
        for variant_idx in range(max(0, egg_multiplier - 1)):
            aug_seed = rng.randrange(0, 2**31 - 1)
            local_rng = random.Random(aug_seed)
            digest = hashlib.sha1(
                f"{source.resolve()}::{variant_idx}::{aug_seed}".encode("utf-8")
            ).hexdigest()[:10]
            destination = (
                output_egg
                / f"{source.stem}_lowres_v{variant_idx + 1}_{digest}.jpg"
            )
            meta = lowres_variant(source, destination, local_rng)
            rows.append({
                "stage": output_stage.name,
                "split": "train",
                "class": source_egg.name,
                "kind": "lowres_augmented",
                "augmentation_seed": aug_seed,
                **meta,
            })

    effective_egg_count = len(egg_originals) * egg_multiplier
    target_empty_count = int(round(effective_egg_count * empty_to_egg_ratio))
    target_empty_count = min(target_empty_count, len(empty_originals))

    selected_empty = rng.sample(empty_originals, target_empty_count)

    for source in selected_empty:
        if rng.random() < empty_lowres_probability:
            aug_seed = rng.randrange(0, 2**31 - 1)
            local_rng = random.Random(aug_seed)
            digest = hashlib.sha1(
                f"{source.resolve()}::empty::{aug_seed}".encode("utf-8")
            ).hexdigest()[:10]
            destination = output_empty / f"{source.stem}_lowres_{digest}.jpg"
            meta = lowres_variant(source, destination, local_rng)
            rows.append({
                "stage": output_stage.name,
                "split": "train",
                "class": source_empty.name,
                "kind": "lowres_augmented_selected_empty",
                "augmentation_seed": aug_seed,
                **meta,
            })
        else:
            destination = output_empty / unique_name(source)
            hardlink_or_copy(source, destination)
            rows.append({
                "stage": output_stage.name,
                "split": "train",
                "class": source_empty.name,
                "kind": "selected_original_empty",
                "source": str(source),
                "destination": str(destination),
            })

    return rows


def summarize_dataset(stage_root: Path) -> list[dict[str, object]]:
    rows = []
    for split_dir in sorted(
        [p for p in stage_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    ):
        for class_dir in sorted(
            [p for p in split_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        ):
            rows.append({
                "stage": stage_root.name,
                "split": split_dir.name,
                "class": class_dir.name,
                "n_images": len(image_files(class_dir)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help="Source root containing the Stage 1 and Stage 3 datasets.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Destination for the rebalanced datasets and manifests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_egg_multiplier", type=int, default=4)
    parser.add_argument("--stage1_larvae_multiplier", type=int, default=2)
    parser.add_argument("--stage3_egg_multiplier", type=int, default=4)
    parser.add_argument("--stage3_empty_to_egg_ratio", type=float, default=2.0)
    parser.add_argument("--stage3_empty_lowres_probability", type=float, default=0.5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    stage1_source = discover_stage_root(source_root, STAGE1_REQUIRED)
    stage3_source = discover_stage_root(source_root, STAGE3_REQUIRED)

    stage1_output = output_root / "stage1_rebalanced_lowres"
    stage3_output = output_root / "stage3_egg_empty_balanced_lowres"

    print(f"[source] Stage 1: {stage1_source}")
    print(f"[source] Stage 3: {stage3_source}")

    manifest_rows = []
    manifest_rows.extend(
        build_stage1(
            stage1_source,
            stage1_output,
            seed=args.seed,
            egg_multiplier=args.stage1_egg_multiplier,
            larvae_multiplier=args.stage1_larvae_multiplier,
        )
    )
    manifest_rows.extend(
        build_stage3(
            stage3_source,
            stage3_output,
            seed=args.seed + 1000,
            egg_multiplier=args.stage3_egg_multiplier,
            empty_to_egg_ratio=args.stage3_empty_to_egg_ratio,
            empty_lowres_probability=args.stage3_empty_lowres_probability,
        )
    )

    manifest_path = output_root / "dataset_manifest.csv"
    fieldnames = sorted({key for row in manifest_rows for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_rows = summarize_dataset(stage1_output) + summarize_dataset(stage3_output)
    summary_path = output_root / "dataset_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "split", "class", "n_images"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n===== Dataset summary =====")
    for row in summary_rows:
        print(
            f"{row['stage']:38s} {row['split']:5s} "
            f"{row['class']:20s} {row['n_images']:8d}"
        )

    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print("Completed successfully.")


if __name__ == "__main__":
    main()
