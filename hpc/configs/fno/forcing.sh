#!/bin/bash
TRAIN_ARGS="
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
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
--forcings-required
--device auto
"

PRED_ARGS="--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir patch_all_ts
--batch-size 256
--resolution-seed 286
--metrics-only
--device auto
"

CHECKPOINT="latest_checkpoint.pth"
