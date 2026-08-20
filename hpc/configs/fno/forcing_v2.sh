#!/bin/bash


# FNOInterpolate forcings configuration - aligned with GINO dataset structure

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head
--epochs 250
--batch-size 32
--learning-rate 5e-4
--scheduler-type exponential
--lr-scheduler-interval 10
--lr-gamma 0.99
--grad-clip-norm 1.0
--input-window-size 10
--output-window-size 10
--train-stride 5
--lambda-conc-focus 0.0
--save-checkpoint-every 10
--padding-mode border
--sampling-strategy static
--resolution-ratio 0.3
--min-resolution-ratio 0.20
--forcings-required
--device auto
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--batch-size 64
--sampling-strategy static
--device auto
--rolling-sequence
--val-stride 10
--val-only
"

RESOLUTION_RATIOS="1.0"

CHECKPOINT="latest_checkpoint.pth"
