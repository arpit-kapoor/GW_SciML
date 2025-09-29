#!/bin/bash

# Script to list and manage GINO experiments
# Usage: ./list_experiments.sh [target_col] [base_results_dir]

TARGET_COL="${1:-mass_concentration}"
RESULTS_BASE_DIR="${2:-/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO}"

TARGET_DIR="$RESULTS_BASE_DIR/$TARGET_COL"

echo "=========================================="
echo "GINO Experiments for target: $TARGET_COL"
echo "=========================================="
echo

if [ ! -d "$TARGET_DIR" ]; then
    echo "No experiments found. Target directory does not exist: $TARGET_DIR"
    exit 1
fi

# Find all experiment directories
EXPERIMENTS=$(find "$TARGET_DIR" -maxdepth 1 -type d -name "exp_*" | sort)

if [ -z "$EXPERIMENTS" ]; then
    echo "No experiments found in: $TARGET_DIR"
    exit 0
fi

echo "Available experiments:"
echo "======================"

for exp_dir in $EXPERIMENTS; do
    exp_name=$(basename "$exp_dir")
    
    # Count training runs in this experiment
    run_count=$(find "$exp_dir" -maxdepth 1 -type d -name "gino_*" | wc -l)
    
    # Check for latest checkpoint
    latest_run=$(find "$exp_dir" -maxdepth 1 -type d -name "gino_*" | sort | tail -1)
    status="No runs"
    
    if [ -n "$latest_run" ]; then
        if [ -f "$latest_run/checkpoints/latest_checkpoint.pth" ]; then
            status="Has checkpoint (resumable)"
        else
            status="No checkpoint"
        fi
        
        # Get latest run timestamp
        run_name=$(basename "$latest_run")
        timestamp=$(echo "$run_name" | sed 's/gino_//')
    fi
    
    echo
    echo "Experiment: $exp_name"
    echo "  Runs: $run_count"
    echo "  Status: $status"
    if [ -n "$latest_run" ]; then
        echo "  Latest run: $timestamp"
        echo "  Path: $latest_run"
    fi
    
done

echo
echo "=========================================="
echo "Usage examples:"
echo "# Resume specific experiment:"
echo "qsub -v \"TARGET_COL=$TARGET_COL,EXPERIMENT_NAME=<exp_name>\" train_gino.pbs"
echo
echo "# Start new experiment with different hyperparameters:"
echo "qsub -v \"TARGET_COL=$TARGET_COL,LEARNING_RATE=1e-3\" train_gino.pbs"
echo "=========================================="
