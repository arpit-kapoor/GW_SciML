#!/bin/bash
# Default configuration for GINO training
# Directory: results/GINO/default/training_TIMESTAMP/

TRAIN_ARGS="
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir filter_patch
--target-cols mass_concentration head
--epochs 40
--batch-size 128
--input-window-size 5
--output-window-size 5
--learning-rate 5e-4
--scheduler-type cosine
--grad-clip-norm 1.0
--save-checkpoint-every 10
--lambda-conc-focus 0.5
"
