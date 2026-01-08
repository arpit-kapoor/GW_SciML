#!/bin/bash
# FNO Prediction Config: latest
# 
# Generate predictions using the most recent checkpoint from default config

# Model path - automatically find the latest checkpoint
FNO_RESULTS_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO/default"

# Find the latest training directory
LATEST_TRAINING=$(ls -1dt ${FNO_RESULTS_DIR}/training_* 2>/dev/null | head -1)

if [ -z "$LATEST_TRAINING" ]; then
    echo "Error: No training directories found in ${FNO_RESULTS_DIR}"
    exit 1
fi

# Use the latest checkpoint
MODEL_PATH="${LATEST_TRAINING}/checkpoints/latest_checkpoint.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Latest checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

echo "Using latest checkpoint: $MODEL_PATH"

# Prediction arguments
PREDICT_ARGS="
--model-path $MODEL_PATH
--batch-size 32
--device cuda
"
