#!/bin/bash
# Helper script to find the latest trained model for a given config

if [ -z "$1" ]; then
    echo "Usage: ./find_latest_model.sh [CONFIG_NAME] [CHECKPOINT_NAME]"
    echo ""
    echo "Examples:"
    echo "  ./find_latest_model.sh default"
    echo "  ./find_latest_model.sh default checkpoint_epoch_0050.pth"
    echo "  ./find_latest_model.sh high_lr checkpoint_final.pth"
    exit 0
fi

CONFIG_NAME="$1"
CHECKPOINT="${2:-checkpoint_final.pth}"
RESULTS_BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO"

# Find latest training run for this config
LATEST_RUN=$(find "$RESULTS_BASE_DIR/$CONFIG_NAME" -maxdepth 1 -type d -name "training_*" 2>/dev/null | sort -r | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No training runs found for config '$CONFIG_NAME'"
    echo ""
    echo "Available configs:"
    find "$RESULTS_BASE_DIR" -maxdepth 1 -type d ! -path "$RESULTS_BASE_DIR" 2>/dev/null | sed 's|.*/||' | sort
    exit 1
fi

MODEL_PATH="$LATEST_RUN/checkpoints/$CHECKPOINT"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Checkpoint '$CHECKPOINT' not found in latest run"
    echo ""
    echo "Available checkpoints:"
    ls -1 "$LATEST_RUN/checkpoints/" 2>/dev/null
    exit 1
fi

echo "Latest trained model for config '$CONFIG_NAME':"
echo "$MODEL_PATH"
echo ""
echo "Training run: $(basename $LATEST_RUN)"
echo "Checkpoint: $CHECKPOINT"
