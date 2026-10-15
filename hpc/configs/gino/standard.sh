#!/bin/bash

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head
--epochs 300
--batch-size 168
--input-window-size 5
--output-window-size 5
--learning-rate 1e-3
--scheduler-type cosine
--grad-clip-norm 1.0
--save-checkpoint-every 10
--lambda-conc-focus 0.0
--sampling-strategy static
--resolution-ratio 0.3
--forcings-required
--device auto
--gno-radius 0.24
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
