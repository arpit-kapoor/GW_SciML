#!/bin/bash
TRAIN_ARGS="
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir filter_patch 
--target-cols mass_concentration head 
--epochs 50
--batch-size 512 
--input-window-size 5 
--output-window-size 1 
--learning-rate 5e-4 
--scheduler-type exponential 
--grad-clip-norm 1.0 
--save-checkpoint-every 25
--lambda-conc-focus 0.0
"
