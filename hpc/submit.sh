#!/bin/bash
# Unified job submission helper for GINO and FNO with config-based organization
# 
# Usage: 
#   ./submit.sh [model_type] [config_name] [--resume|--predict] [additional_args]
#   ./submit.sh gino default                      # Train GINO with default config
#   ./submit.sh fno default --resume              # Resume FNO training
#   ./submit.sh gino var_loss --predict           # Generate GINO predictions
#   ./submit.sh                                   # Show help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
    echo "=========================================="
    echo "Neural Operator Job Submission"
    echo "=========================================="
    echo ""
    echo "Usage: ./submit.sh [model_type] [config_name] [--resume|--predict] [additional_args]"
    echo ""
    echo "Model types: gino, fno"
    echo ""
    echo "Examples:"
    echo "  ./submit.sh gino default                    # Train GINO with default config"
    echo "  ./submit.sh fno var_loss                    # Train FNO with var_loss config"
    echo "  ./submit.sh gino default --resume           # Resume GINO training from latest checkpoint"
    echo "  ./submit.sh fno default --predict           # Generate FNO predictions using latest model"
    echo "  ./submit.sh gino default --epochs 100       # Override specific args"
    echo "  ./submit.sh fno adhoc --epochs 50 ...       # Custom FNO arguments"
    echo ""
    echo "Available GINO configs:"
    if [ -d "$SCRIPT_DIR/configs/gino" ]; then
        for config in "$SCRIPT_DIR/configs/gino"/*.sh; do
            if [ -f "$config" ]; then
                config_name=$(basename "$config" .sh)
                config_desc=$(sed -n '2s/^# //p' "$config")
                printf "  %-18s %s\n" "$config_name" "$config_desc"
            fi
        done
    else
        echo "  (no gino configs found)"
    fi
    echo ""
    echo "Available FNO configs:"
    if [ -d "$SCRIPT_DIR/configs/fno" ]; then
        for config in "$SCRIPT_DIR/configs/fno"/*.sh; do
            if [ -f "$config" ]; then
                config_name=$(basename "$config" .sh)
                config_desc=$(sed -n '2s/^# //p' "$config")
                printf "  %-18s %s\n" "$config_name" "$config_desc"
            fi
        done
    else
        echo "  (no fno configs found)"
    fi
    echo ""
    echo "Directory structure:"
    echo "  Training: results/[GINO|FNO]/[config_name]/training_TIMESTAMP/"
    echo "  Predictions: results/[GINO|FNO]_predictions/[config_name]/"
    echo ""
}

if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

MODEL_TYPE="$1"
shift

# Validate model type
if [ "$MODEL_TYPE" != "gino" ] && [ "$MODEL_TYPE" != "fno" ]; then
    echo "Error: Invalid model type '$MODEL_TYPE'. Must be 'gino' or 'fno'."
    echo ""
    show_usage
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Error: Config name required"
    echo ""
    show_usage
    exit 1
fi

# Set model-specific paths
if [ "$MODEL_TYPE" == "gino" ]; then
    TRAIN_PBS_SCRIPT="$SCRIPT_DIR/gino_train.pbs"
    PREDICT_PBS_SCRIPT="$SCRIPT_DIR/gino_predict.pbs"
    CONFIG_DIR="$SCRIPT_DIR/configs/gino"
    RESULTS_BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO"
    PREDICTIONS_BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO_predictions"
    TRAIN_LOG_PATTERN="train_gino_*.log"
    PREDICT_LOG_PATTERN="predict_gino_*.log"
else  # fno
    TRAIN_PBS_SCRIPT="$SCRIPT_DIR/fno_train.pbs"
    PREDICT_PBS_SCRIPT="$SCRIPT_DIR/fno_predict.pbs"
    CONFIG_DIR="$SCRIPT_DIR/configs/fno"
    RESULTS_BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO"
    PREDICTIONS_BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO_predictions"
    TRAIN_LOG_PATTERN="train_fno_interpolate_*.log"
    PREDICT_LOG_PATTERN="predict_fno_*.log"
fi

CONFIG_NAME="$1"
shift  # Remove config name argument

# Check for --resume or --predict flag
MODE="train"
if [ "$1" == "--resume" ]; then
    MODE="resume"
    shift  # Remove --resume argument
elif [ "$1" == "--predict" ]; then
    MODE="predict"
    shift  # Remove --predict argument
fi

# Check if config exists (unless it's adhoc)
if [ "$CONFIG_NAME" != "adhoc" ]; then
    CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.sh"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config '$CONFIG_NAME' not found at $CONFIG_FILE"
        echo ""
        echo "Available $MODEL_TYPE configs:"
        ls -1 "$CONFIG_DIR"/*.sh 2>/dev/null | xargs -n 1 basename | sed 's/.sh$//' | sed 's/^/  - /'
        exit 1
    fi
fi

# Handle different modes
if [ "$MODE" == "predict" ]; then
    # Predict mode
    ADDITIONAL_ARGS="$*"
    
    if [ -n "$ADDITIONAL_ARGS" ]; then
        echo "Submitting ${MODEL_TYPE^^} prediction job with config '$CONFIG_NAME'"
        [ -n "$*" ] && echo "  Additional args: $*"
        # Base64 encode the arguments to avoid PBS quoting issues
        ARGS_B64=$(echo "$ADDITIONAL_ARGS" | base64 -w 0)
        qsub -v "CONFIG=${CONFIG_NAME},ARGS_B64=${ARGS_B64}" "$PREDICT_PBS_SCRIPT"
    else
        echo "Submitting ${MODEL_TYPE^^} prediction job with config: $CONFIG_NAME"
        qsub -v "CONFIG=${CONFIG_NAME}" "$PREDICT_PBS_SCRIPT"
    fi
    
    echo ""
    echo "Predictions will be saved to: ${PREDICTIONS_BASE_DIR/${PREDICTIONS_BASE_DIR%/*}\//}/${CONFIG_NAME}/"
    
elif [ "$MODE" == "resume" ]; then
    # Resume mode - find latest checkpoint
    BASE_DIR="${RESULTS_BASE_DIR}/${CONFIG_NAME}"
    LATEST_RUN=$(ls -1td "$BASE_DIR"/training_* 2>/dev/null | head -1)
    
    if [ -z "$LATEST_RUN" ]; then
        echo "Error: No previous runs found for config '$CONFIG_NAME'"
        echo "Base directory: $BASE_DIR"
        exit 1
    fi
    
    CHECKPOINT="$LATEST_RUN/checkpoints/latest_checkpoint.pth"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
    
    echo "========================================="
    echo "Resuming ${MODEL_TYPE^^} training from checkpoint:"
    echo "$CHECKPOINT"
    echo "========================================="
    echo ""
    
    # Add resume argument
    ADDITIONAL_ARGS="--resume-from $CHECKPOINT $*"
    
    echo "Submitting ${MODEL_TYPE^^} job with config '$CONFIG_NAME'"
    echo "  Mode: Resume training"
    [ -n "$*" ] && echo "  Additional args: $*"
    
    # Base64 encode the arguments to avoid PBS quoting issues
    ARGS_B64=$(echo "$ADDITIONAL_ARGS" | base64 -w 0)
    qsub -v "CONFIG=${CONFIG_NAME},ARGS_B64=${ARGS_B64}" "$TRAIN_PBS_SCRIPT"
    
    echo ""
    echo "Results will be saved to: ${RESULTS_BASE_DIR/${RESULTS_BASE_DIR%/*}\//}/${CONFIG_NAME}/"
    
else
    # Train mode
    ADDITIONAL_ARGS="$*"
    
    if [ -n "$ADDITIONAL_ARGS" ]; then
        echo "Submitting ${MODEL_TYPE^^} training job with config '$CONFIG_NAME'"
        [ -n "$*" ] && echo "  Additional args: $*"
        # Base64 encode the arguments to avoid PBS quoting issues
        ARGS_B64=$(echo "$ADDITIONAL_ARGS" | base64 -w 0)
        qsub -v "CONFIG=${CONFIG_NAME},ARGS_B64=${ARGS_B64}" "$TRAIN_PBS_SCRIPT"
    else
        echo "Submitting ${MODEL_TYPE^^} training job with config: $CONFIG_NAME"
        qsub -v "CONFIG=${CONFIG_NAME}" "$TRAIN_PBS_SCRIPT"
    fi
    
    echo ""
    echo "Results will be saved to: ${RESULTS_BASE_DIR/${RESULTS_BASE_DIR%/*}\//}/${CONFIG_NAME}/"
fi

echo ""
echo "Monitor job status:"
echo "  qstat -u \$USER"
echo ""
echo "View logs:"
if [ "$MODE" == "predict" ]; then
    echo "  tail -f logs/$PREDICT_LOG_PATTERN"
else
    echo "  tail -f logs/$TRAIN_LOG_PATTERN"
fi
