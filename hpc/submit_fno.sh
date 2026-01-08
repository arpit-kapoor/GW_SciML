#!/bin/bash
# Quick job submission helper for FNOInterpolate with config-based organization
# 
# Usage: 
#   ./submit_fno.sh [config_name] [additional_args]
#   ./submit_fno.sh                         # Show available configs
#   ./submit_fno.sh default                 # Use default config
#   ./submit_fno.sh bilinear                # Use bilinear interpolation config
#   ./submit_fno.sh adhoc --epochs 50 ...   # Use ad-hoc args

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="$SCRIPT_DIR/train_fno_interpolate_multi_col.pbs"

if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "FNOInterpolate Training Job Submission"
    echo "=========================================="
    echo ""
    echo "Usage: ./submit_fno.sh [config_name] [additional_args]"
    echo ""
    echo "Examples:"
    echo "  ./submit_fno.sh default                # Use default config"
    echo "  ./submit_fno.sh bilinear               # Use bilinear interpolation"
    echo "  ./submit_fno.sh nearest                # Use nearest neighbor"
    echo "  ./submit_fno.sh default --epochs 100   # Override specific args"
    echo "  ./submit_fno.sh adhoc --epochs 50 ...  # Custom arguments"
    echo ""
    echo "Available configs:"
    if [ -d "$SCRIPT_DIR/configs/fno" ]; then
        for config in "$SCRIPT_DIR/configs/fno"/*.sh; do
            if [ -f "$config" ]; then
                config_name=$(basename "$config" .sh)
                # Extract comment from config file (first line after shebang)
                config_desc=$(sed -n '2s/^# //p' "$config")
                printf "  %-18s %s\n" "$config_name" "$config_desc"
            fi
        done
    else
        echo "  (no fno configs directory found)"
    fi
    echo ""
    echo "Directory structure: results/FNO/[config_name]/training_TIMESTAMP/"
    echo ""
    echo "Advanced usage:"
    echo "  Override resources:"
    echo "    qsub -l walltime=24:00:00 -v CONFIG=default $PBS_SCRIPT"
    echo "    qsub -l select=1:ngpus=4 -v CONFIG=default $PBS_SCRIPT"
    echo ""
    exit 0
fi

CONFIG_NAME="$1"
shift  # Remove first argument

# Check if config exists (unless it's adhoc)
if [ "$CONFIG_NAME" != "adhoc" ]; then
    CONFIG_FILE="$SCRIPT_DIR/configs/fno/${CONFIG_NAME}.sh"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config '$CONFIG_NAME' not found at $CONFIG_FILE"
        echo ""
        echo "Available configs:"
        ls -1 "$SCRIPT_DIR/configs/fno"/*.sh 2>/dev/null | xargs -n 1 basename | sed 's/.sh$//' | sed 's/^/  - /'
        exit 1
    fi
fi

# Build submission command
if [ $# -gt 0 ]; then
    # Additional args provided
    echo "Submitting FNOInterpolate job with config '$CONFIG_NAME' and additional args: $@"
    qsub -v "CONFIG=${CONFIG_NAME},ARGS=$*" "$PBS_SCRIPT"
else
    # No additional args
    echo "Submitting FNOInterpolate job with config: $CONFIG_NAME"
    qsub -v "CONFIG=${CONFIG_NAME}" "$PBS_SCRIPT"
fi

echo ""
echo "Results will be saved to: results/FNO/${CONFIG_NAME}/"
echo ""
echo "Monitor job status:"
echo "  qstat -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/train_fno_interpolate_*.log"
