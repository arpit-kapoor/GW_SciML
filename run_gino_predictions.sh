#!/bin/bash

# Example script to run GINO predictions
# Make sure to update the paths according to your specific setup

# Path to the trained GINO model (update this to your actual model path)
MODEL_PATH="/srv/scratch/z5370003/projects/src/04_groundwater/variable_density/gino_model.pth"

# Base data directory (update if different)
BASE_DATA_DIR="/Users/arpitkapoor/Library/CloudStorage/OneDrive-UNSW/Shared/Projects/01_PhD/05_groundwater/data/FEFLOW/variable_density"

# Results directory for predictions
RESULTS_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO_predictions"

# Run the prediction script
python generate_gino_predictions.py \
    --model-path "$MODEL_PATH" \
    --base-data-dir "$BASE_DATA_DIR" \
    --results-dir "$RESULTS_DIR" \
    --target-col "mass_concentration" \
    --input-window-size 5 \
    --output-window-size 5 \
    --batch-size 150 \
    --device auto

echo "Prediction generation completed!"
echo "Results saved to: $RESULTS_DIR"

