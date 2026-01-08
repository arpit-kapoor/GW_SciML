#!/bin/bash
# GINO Prediction Config: latest
# 
# Generate predictions using the most recent checkpoint from default config

# Model path - automatically find the latest checkpoint
GINO_RESULTS_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO/default"

# Find the latest training directory
LATEST_TRAINING=$(ls -1dt ${GINO_RESULTS_DIR}/training_* 2>/dev/null | head -1)

if [ -z "$LATEST_TRAINING" ]; then
    echo "Error: No training directories found in ${GINO_RESULTS_DIR}"
    exit 1
fi

# Use the final checkpoint
MODEL_PATH="${LATEST_TRAINING}/checkpoints/checkpoint_final.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Final checkpoint not found at ${MODEL_PATH}"
    echo "Attempting to use latest_checkpoint.pth instead..."
    MODEL_PATH="${LATEST_TRAINING}/checkpoints/latest_checkpoint.pth"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: No checkpoint found in ${LATEST_TRAINING}/checkpoints/"
        exit 1
    fi
fi

echo "Using latest checkpoint: $MODEL_PATH"

# Prediction arguments
# Note: target-col is determined from checkpoint metadata, no need to specify
PREDICT_ARGS="
--model-path $MODEL_PATH
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir filter_patch
--input-window-size 5
--output-window-size 1
--batch-size 128
--device auto
"
