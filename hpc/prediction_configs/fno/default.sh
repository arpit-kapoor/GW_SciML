#!/bin/bash
# FNO Prediction Config: default
# 
# Generate predictions using the default FNOInterpolate model
# (mass_concentration + head with bilinear interpolation)

# Model path - automatically find the latest trained model
FNO_RESULTS_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO/default"

# Find the latest training directory
LATEST_TRAINING=$(ls -1dt ${FNO_RESULTS_DIR}/training_* 2>/dev/null | head -1)

if [ -z "$LATEST_TRAINING" ]; then
    echo "Error: No training directories found in ${FNO_RESULTS_DIR}"
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
PREDICT_ARGS="
--model-path $MODEL_PATH
--batch-size 256
--device cuda
"
