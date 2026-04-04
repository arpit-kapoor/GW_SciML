#!/bin/bash


TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir filter_patch_all_ts
--target-cols mass_concentration head 
--epochs 400
--batch-size 512 
--input-window-size 5 
--output-window-size 1 
--learning-rate 8e-4 
--scheduler-type exponential 
--lr-scheduler-interval 4
--lr-gamma 0.98
--grad-clip-norm 1.0 
--save-checkpoint-every 10
--lambda-conc-focus 0.0
"


PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir filter_patch_all_ts
--batch-size 256
--min-resolution-ratio 0.20
--device auto
"

RESOLUTION_RATIOS="1.0"

CHECKPOINT="latest_checkpoint.pth"
