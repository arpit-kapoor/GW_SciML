#!/bin/bash


# FNOInterpolate variance-aware loss configuration - aligned with GINO dataset structure

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head
--epochs 300
--batch-size 256
--learning-rate 5e-4
--scheduler-type exponential
--lr-scheduler-interval 10
--lr-gamma 0.99
--grad-clip-norm 1.0
--input-window-size 5
--output-window-size 1
--lambda-conc-focus 0.3
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
--batch-size 256
--sampling-strategy static
--device auto
"

RESOLUTION_RATIOS="0.3 1.0"

CHECKPOINT="latest_checkpoint.pth"
