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

resolve_with_default() {
    local var_name="$1"
    local default_value="$2"
    local current_value="${!var_name:-}"
    if [ -z "$current_value" ] && [ -n "$default_value" ]; then
        export "$var_name=$default_value"
    fi
}

require_set() {
    local var_name="$1"
    if [ -z "${!var_name:-}" ]; then
        echo "Error: Required variable '$var_name' is not set"
        exit 1
    fi
}

submit_job() {
    local script_path="$1"
    local args_b64="${2:-}"

    if [ "$USE_QSUB" == "1" ]; then
        local qsub_vars="$QSUB_EXPORTS"
        if [ -n "$args_b64" ]; then
            qsub_vars="${qsub_vars},ARGS_B64=${args_b64}"
        fi
        qsub -v "$qsub_vars" "$script_path"
    else
        export CONFIG="${CONFIG_NAME}"
        export PATHS_FILE
        export PYTHON_ENV BASE_DATA_DIR LOG_DIR RESULTS_BASE_DIR PREDICTIONS_BASE_DIR
        export PBS_O_WORKDIR="$SCRIPT_DIR"
        export PBS_JOBID="local_$$"
        if [ -n "$args_b64" ]; then
            export ARGS_B64="$args_b64"
        else
            unset ARGS_B64
        fi
        bash "$script_path"
    fi
}

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
    echo "  ./submit.sh gino default --min-resolution-ratio 0.10  # Set floor for dynamic subsampling"
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
MODEL_TYPE_UPPER=$(echo "$MODEL_TYPE" | tr "[:lower:]" "[:upper:]")
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
    TRAIN_LOG_PATTERN="train_gino_*.log"
    PREDICT_LOG_PATTERN="predict_gino_*.log"
else  # fno
    TRAIN_PBS_SCRIPT="$SCRIPT_DIR/fno_train.pbs"
    PREDICT_PBS_SCRIPT="$SCRIPT_DIR/fno_predict.pbs"
    CONFIG_DIR="$SCRIPT_DIR/configs/fno"
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

PATHS_FILE="${PATHS_FILE:-$SCRIPT_DIR/configs/paths.sh}"
if [ ! -f "$PATHS_FILE" ]; then
    echo "Error: Paths config file not found: $PATHS_FILE"
    exit 1
fi
source "$PATHS_FILE"

USE_QSUB="${USE_QSUB:-1}"
if [ "$USE_QSUB" != "0" ] && [ "$USE_QSUB" != "1" ]; then
    echo "Error: USE_QSUB must be 0 or 1 in $PATHS_FILE"
    exit 1
fi
if [ "$USE_QSUB" == "1" ] && ! command -v qsub >/dev/null 2>&1; then
    echo "Error: USE_QSUB=1 but qsub command not found"
    exit 1
fi

if [ "$MODEL_TYPE" == "gino" ]; then
    resolve_with_default RESULTS_BASE_DIR "${RESULTS_BASE_DIR_GINO:-${RESULTS_BASE_DIR:-}}"
    resolve_with_default PREDICTIONS_BASE_DIR "${PREDICTIONS_BASE_DIR_GINO:-${PREDICTIONS_BASE_DIR:-}}"
else
    resolve_with_default RESULTS_BASE_DIR "${RESULTS_BASE_DIR_FNO:-${RESULTS_BASE_DIR:-}}"
    resolve_with_default PREDICTIONS_BASE_DIR "${PREDICTIONS_BASE_DIR_FNO:-${PREDICTIONS_BASE_DIR:-}}"
fi

require_set PYTHON_ENV
require_set BASE_DATA_DIR
require_set LOG_DIR
require_set RESULTS_BASE_DIR
require_set PREDICTIONS_BASE_DIR

mkdir -p "$LOG_DIR" "$RESULTS_BASE_DIR" "$PREDICTIONS_BASE_DIR"

REQUIRED_VARS=(PYTHON_ENV BASE_DATA_DIR LOG_DIR RESULTS_BASE_DIR PREDICTIONS_BASE_DIR)
MISSING_VARS=()
for required_var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!required_var:-}" ]; then
        MISSING_VARS+=("$required_var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "Error: Missing required path/env vars: ${MISSING_VARS[*]}"
    echo "Hint: define them in $PATHS_FILE."
    exit 1
fi

QSUB_EXPORTS="CONFIG=${CONFIG_NAME},PATHS_FILE=${PATHS_FILE},PYTHON_ENV=${PYTHON_ENV},BASE_DATA_DIR=${BASE_DATA_DIR},LOG_DIR=${LOG_DIR},RESULTS_BASE_DIR=${RESULTS_BASE_DIR},PREDICTIONS_BASE_DIR=${PREDICTIONS_BASE_DIR}"

# Handle different modes
if [ "$MODE" == "predict" ]; then
    # Predict mode
    ADDITIONAL_ARGS="$*"
    
    if [ -n "$ADDITIONAL_ARGS" ]; then
        echo "Submitting ${MODEL_TYPE_UPPER} prediction job with config '$CONFIG_NAME'"
        [ -n "$*" ] && echo "  Additional args: $*"
        # Base64 encode the arguments to avoid PBS quoting issues
        ARGS_B64=$(echo "$ADDITIONAL_ARGS" | base64 -w 0 2>/dev/null || echo "$ADDITIONAL_ARGS" | base64) # fallback for macOS
        submit_job "$PREDICT_PBS_SCRIPT" "$ARGS_B64"
    else
        echo "Submitting ${MODEL_TYPE_UPPER} prediction job with config: $CONFIG_NAME"
        submit_job "$PREDICT_PBS_SCRIPT"
    fi
    
    echo ""
    echo "Predictions will be saved to: $(basename "$PREDICTIONS_BASE_DIR")/${CONFIG_NAME}/"
    
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
    echo "Resuming ${MODEL_TYPE_UPPER} training from checkpoint:"
    echo "$CHECKPOINT"
    echo "========================================="
    echo ""
    
    # Add resume argument
    ADDITIONAL_ARGS="--resume-from $CHECKPOINT $*"
    
    echo "Submitting ${MODEL_TYPE_UPPER} job with config '$CONFIG_NAME'"
    echo "  Mode: Resume training"
    [ -n "$*" ] && echo "  Additional args: $*"
    
    # Base64 encode the arguments to avoid PBS quoting issues
    ARGS_B64=$(echo "$ADDITIONAL_ARGS" | base64 -w 0 2>/dev/null || echo "$ADDITIONAL_ARGS" | base64)
    submit_job "$TRAIN_PBS_SCRIPT" "$ARGS_B64"
    
    echo ""
    echo "Results will be saved to: $(basename "$RESULTS_BASE_DIR")/${CONFIG_NAME}/"
    
else
    # Train mode
    ADDITIONAL_ARGS="$*"
    
    if [ -n "$ADDITIONAL_ARGS" ]; then
        echo "Submitting ${MODEL_TYPE_UPPER} training job with config '$CONFIG_NAME'"
        [ -n "$*" ] && echo "  Additional args: $*"
        # Base64 encode the arguments to avoid PBS quoting issues
        ARGS_B64=$(echo "$ADDITIONAL_ARGS" | base64 -w 0 2>/dev/null || echo "$ADDITIONAL_ARGS" | base64)
        submit_job "$TRAIN_PBS_SCRIPT" "$ARGS_B64"
    else
        echo "Submitting ${MODEL_TYPE_UPPER} training job with config: $CONFIG_NAME"
        submit_job "$TRAIN_PBS_SCRIPT"
    fi
    
    echo ""
    echo "Results will be saved to: $(basename "$RESULTS_BASE_DIR")/${CONFIG_NAME}/"
fi

echo ""
if [ "$USE_QSUB" == "1" ]; then
    echo "Monitor job status:"
    echo "  qstat -u \$USER"
else
    echo "Execution mode: direct (qsub bypassed by USE_QSUB=0)"
fi
echo ""
echo "View logs:"
if [ "$MODE" == "predict" ]; then
    echo "  tail -f logs/$PREDICT_LOG_PATTERN"
else
    echo "  tail -f logs/$TRAIN_LOG_PATTERN"
fi
