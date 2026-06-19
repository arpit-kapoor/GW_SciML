#!/bin/bash

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir masked_patch_all_ts
--target-cols mass_concentration head 
--epochs 200
--batch-size 128
--input-window-size 5 
--output-window-size 1 
--learning-rate 8e-4 
--scheduler-type exponential 
--lr-scheduler-interval 20
--lr-gamma 0.98
--grad-clip-norm 1.0 
--save-checkpoint-every 10
--lambda-conc-focus 0
--sampling-strategy static
--resolution-ratio 1.0
--forcings-required
--device auto
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir masked_patch_all_ts
--batch-size 256
--device auto
--sampling-strategy static
"

RESOLUTION_RATIOS="1.0"

CHECKPOINT="latest_checkpoint.pth"
