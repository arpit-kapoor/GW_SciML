#!/bin/bash


TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head 
--epochs 1
--batch-size 256 
--input-window-size 5 
--output-window-size 1 
--learning-rate 8e-4 
--scheduler-type exponential 
--lr-scheduler-interval 4
--lr-gamma 0.98
--grad-clip-norm 1.0 
--save-checkpoint-every 5
--lambda-conc-focus 0.3
--resolution-ratio 0.60
--min-resolution-ratio 0.20
--forcings-required
--device auto
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--batch-size 256
--device auto
--min-resolution-ratio 0.20
"

RESOLUTION_RATIOS="0.1"

CHECKPOINT="latest_checkpoint.pth"
