#!/bin/bash


TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir filter_patch_all_ts
--target-cols mass_concentration head 
--epochs 300
--batch-size 256
--learning-rate 5e-4
--scheduler-type exponential
--input-window-size 5
--output-window-size 1
--lambda-conc-focus 0.0
--save-checkpoint-every 10
--padding-mode border
--min-resolution-ratio 0.20
--forcings-required
--device auto
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--batch-size 256
--min-resolution-ratio 0.20
--metrics-only
--device auto
"

RESOLUTION_RATIOS="0.167"

CHECKPOINT="latest_checkpoint.pth"
