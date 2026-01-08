#!/bin/bash
# Resume training from default config's latest checkpoint

# Find latest checkpoint from default config
BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO/default"
LATEST_RUN=$(ls -1td "$BASE_DIR"/training_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No previous runs found in default config"
    echo "Base directory: $BASE_DIR"
    exit 1
fi

CHECKPOINT="$LATEST_RUN/checkpoints/latest_checkpoint.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "========================================="
echo "Resuming from checkpoint:"
echo "$CHECKPOINT"
echo "========================================="

TRAIN_ARGS="
--resume-from $CHECKPOINT
--epochs 100
--batch-size 256
--learning-rate 5e-4
--scheduler-type exponential
--target-cols mass_concentration head
--input-window-size 5
--output-window-size 1
--lambda-conc-focus 0.0
--padding-mode border
"
