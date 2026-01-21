#!/bin/bash
TRAIN_ARGS="
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir filter_patch_all_ts
--target-cols mass_concentration head 
--epochs 450
--batch-size 512 
--input-window-size 5 
--output-window-size 1 
--learning-rate 8e-4 
--scheduler-type exponential 
--lr-scheduler-interval 4
--lr-gamma 0.98
--grad-clip-norm 1.0 
--save-checkpoint-every 5
--lambda-conc-focus 0.3
--forcings-required
--device auto
"

PRED_ARGS="--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir patch_all_ts
--batch-size 256
--device auto
--resolution-seed 42
--metrics-only
"

CHECKPOINT="latest_checkpoint.pth"