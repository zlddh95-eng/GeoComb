#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   STAGE1_INIT, STAGE2_INIT, STAGE3_INIT
#   STAGE1_DATA, STAGE2_DATA, STAGE3_DATA
#   OUTPUT_ROOT
# Optional:
#   DEVICE (default 0), WORKERS (default 8)

: "${STAGE1_INIT:?Set STAGE1_INIT to the minority-class-refined Stage 1 checkpoint}"
: "${STAGE2_INIT:?Set STAGE2_INIT to the in-house Stage 2 checkpoint}"
: "${STAGE3_INIT:?Set STAGE3_INIT to the in-house Stage 3 checkpoint}"
: "${STAGE1_DATA:?Set STAGE1_DATA to the final rebalanced Stage 1 dataset}"
: "${STAGE2_DATA:?Set STAGE2_DATA to the final combined Stage 2 dataset}"
: "${STAGE3_DATA:?Set STAGE3_DATA to the final rebalanced Stage 3 dataset}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the training-output directory}"

DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"

yolo classify train \
  model="$STAGE1_INIT" data="$STAGE1_DATA" \
  project="$OUTPUT_ROOT" name="stage1_final" \
  imgsz=320 epochs=45 batch=64 workers="$WORKERS" device="$DEVICE" \
  optimizer=AdamW lr0=0.0005 lrf=0.02 weight_decay=0.0005 \
  patience=12 seed=42 deterministic=True \
  degrees=5 translate=0.05 scale=0.15 fliplr=0.5 \
  hsv_h=0.01 hsv_s=0.20 hsv_v=0.20

yolo classify train \
  model="$STAGE2_INIT" data="$STAGE2_DATA" \
  project="$OUTPUT_ROOT" name="stage2_final" \
  imgsz=320 epochs=35 batch=64 workers="$WORKERS" device="$DEVICE" \
  optimizer=auto lr0=0.0007 lrf=0.01 weight_decay=0.0005 \
  patience=10 seed=0 deterministic=True

yolo classify train \
  model="$STAGE3_INIT" data="$STAGE3_DATA" \
  project="$OUTPUT_ROOT" name="stage3_final" \
  imgsz=384 epochs=45 batch=32 workers="$WORKERS" device="$DEVICE" \
  optimizer=AdamW lr0=0.0005 lrf=0.02 weight_decay=0.0005 \
  patience=12 seed=42 deterministic=True \
  degrees=4 translate=0.04 scale=0.12 fliplr=0.5 \
  hsv_h=0.01 hsv_s=0.15 hsv_v=0.18
