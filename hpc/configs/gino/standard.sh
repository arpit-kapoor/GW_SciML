#!/bin/bash

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head 
--epochs 100
--batch-size 512
--input-window-size 5 
--output-window-size 1 
--learning-rate 1e-3 
--scheduler-type exponential 
--lr-scheduler-interval 10
--lr-gamma 0.99
--grad-clip-norm 1.0 
--save-checkpoint-every 10
--lambda-conc-focus 0.0
--sampling-strategy static
--resolution-ratio 0.3
--forcings-required
--device auto
--gno-radius 0.2
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--batch-size 256
--device auto
--sampling-strategy static
--gno-radius 0.24
"

RESOLUTION_RATIOS="1.0"

CHECKPOINT="latest_checkpoint.pth"
