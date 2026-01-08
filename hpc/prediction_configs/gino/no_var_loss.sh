#!/bin/bash
# GINO Prediction Config: default
# 
# Generate predictions using the default GINO model (auto-discovers latest training)

# Model path - automatically find the latest trained model
GINO_RESULTS_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO/no_var_loss"

# Find the latest training directory
LATEST_TRAINING=$(ls -1dt ${GINO_RESULTS_DIR}/training_* 2>/dev/null | head -1)

if [ -z "$LATEST_TRAINING" ]; then
    echo "Error: No training directories found in ${GINO_RESULTS_DIR}"
    exit 1
fi

# Use the latest checkpoint (which contains args and full metadata)
MODEL_PATH="${LATEST_TRAINING}/checkpoints/latest_checkpoint.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

echo "Using model: $MODEL_PATH"

# Prediction arguments
# Note: target-col is determined from checkpoint metadata, no need to specify
PREDICT_ARGS="
--model-path $MODEL_PATH
--base-data-dir /srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density
--patch-data-subdir filter_patch
--batch-size 256
--device auto
"
