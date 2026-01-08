#!/bin/bash
# Helper script to submit FNOInterpolate prediction jobs with different configurations

# Display usage information
show_usage() {
    echo "Usage: ./submit_fno_predictions.sh [CONFIG_NAME]"
    echo ""
    echo "Submit FNOInterpolate prediction jobs using pre-defined configurations."
    echo ""
    echo "Available configurations:"
    if [ -d "prediction_configs/fno" ]; then
        for config in prediction_configs/fno/*.sh; do
            if [ -f "$config" ]; then
                config_name=$(basename "$config" .sh)
                echo "  - $config_name"
            fi
        done
    else
        echo "  (No configs found - create prediction_configs/fno/ directory)"
    fi
    echo ""
    echo "Examples:"
    echo "  ./submit_fno_predictions.sh default      # Use default config"
    echo "  ./submit_fno_predictions.sh bilinear     # Use bilinear config"
    echo "  ./submit_fno_predictions.sh              # Show this help"
    echo ""
}

# Check if config name provided
if [ -z "$1" ]; then
    show_usage
    exit 0
fi

CONFIG_NAME="$1"
CONFIG_FILE="prediction_configs/fno/${CONFIG_NAME}.sh"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration '$CONFIG_NAME' not found"
    echo ""
    show_usage
    exit 1
fi

# Submit job
echo "Submitting FNO prediction job with config: $CONFIG_NAME"
qsub -v CONFIG="$CONFIG_NAME" generate_fno_predictions.pbs

echo ""
echo "Job submitted! Monitor with: qstat -u \$USER"
echo "View logs in: logs/predict_fno_*.log"
