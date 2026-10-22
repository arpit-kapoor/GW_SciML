#!/bin/bash


# FNOInterpolate (3D) forcings configuration - mirrors hpc/configs/gino/standard.sh
# with interpolation encoding/decoding in place of the GINO GNO blocks.
# Keeps the FNO AdamW weight decay and cosine warmup to regularize against overfitting.

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head
--epochs 500
--batch-size 168
--input-window-size 5
--output-window-size 5
--learning-rate 1e-3
--scheduler-type cosine
--weight-decay 1e-3
--lr-warmup-epochs 5
--grad-clip-norm 1.0
--save-checkpoint-every 10
--lambda-conc-focus 0.0
--sampling-strategy static
--resolution-ratio 0.3
--forcings-required
--padding-mode border
--device auto
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--batch-size 128
--device auto
--sampling-strategy static
--rolling-sequence
--val-stride 5
--val-only
"

RESOLUTION_RATIOS="1.0"

CHECKPOINT="latest_checkpoint.pth"
