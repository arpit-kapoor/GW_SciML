#!/bin/bash
# Default prediction configuration
# Uses the trained model from default training config

# Path to trained model checkpoint
MODEL_PATH="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO/default/training_20260106_XXXXXX/checkpoints/checkpoint_final.pth"

# Prediction arguments
PREDICT_ARGS="
--model-path $MODEL_PATH
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir filter_patch
--target-col head
--input-window-size 5
--output-window-size 1
--batch-size 128
--device auto
"
