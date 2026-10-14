#!/bin/bash
# Space-time GINO: standard config with the factorized 4D FNO in the latent space

TRAIN_ARGS="
--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--target-cols mass_concentration head
--epochs 100
--batch-size 32
--input-window-size 10
--output-window-size 20
--learning-rate 1e-3
--scheduler-type exponential
--lr-scheduler-interval 10
--lr-gamma 0.98
--grad-clip-norm 1.0
--save-checkpoint-every 10
--lambda-conc-focus 0.0
--sampling-strategy static
--resolution-ratio 0.3
--forcings-required
--space-time
--device auto
--gno-radius 0.21
"

PRED_ARGS="--base-data-dir ${BASE_DATA_DIR}
--patch-data-subdir patch_all_ts
--batch-size 128
--device auto
--sampling-strategy static
--gno-radius 0.28
--rolling-sequence
--val-stride 20
--val-only
"

RESOLUTION_RATIOS="1.0"

CHECKPOINT="latest_checkpoint.pth"
