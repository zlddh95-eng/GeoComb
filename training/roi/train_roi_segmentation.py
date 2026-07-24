#!/usr/bin/env python3
"""Train the GeoComb frame-region segmentation model.

The defaults reproduce the settings reported in Supplementary Table S1.
Dataset paths are supplied by the caller; no workstation-specific path is
embedded in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26m-seg.pt")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="frame_roi_yolo26m_1024")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        device=args.device,
        task="segment",
        imgsz=1024,
        epochs=120,
        batch=4,
        workers=4,
        patience=35,
        optimizer="auto",
        pretrained=True,
        amp=True,
        overlap_mask=True,
        mask_ratio=4,
        degrees=5,
        translate=0.05,
        scale=0.20,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        seed=args.seed,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
