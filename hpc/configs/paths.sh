#!/bin/bash
# Machine-specific paths for job submission and PBS scripts.
# This file is the single source of truth for runtime paths.
# Set to 1 to submit via qsub, or 0 to run PBS scripts directly.
USE_QSUB=1

## Katana
# PYTHON_ENV="/srv/scratch/z5370003/projects/src/04_groundwater/variable_density/.venv/bin/python"
# BASE_DATA_DIR="/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density"
# LOG_DIR="/srv/scratch/z5370003/projects/src/04_groundwater/variable_density/logs"
# RESULTS_BASE_DIR_GINO="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO"
# RESULTS_BASE_DIR_FNO="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO"
# PREDICTIONS_BASE_DIR_GINO="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO_predictions"
# PREDICTIONS_BASE_DIR_FNO="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO_predictions"

## Gadi
PYTHON_ENV="/scratch/yl75/ak4177/src/GW_SciML/.venv/bin/python"
BASE_DATA_DIR="/scratch/yl75/ak4177/data/feflow_data"
LOG_DIR="/scratch/yl75/ak4177/src/GW_SciML/logs"
RESULTS_BASE_DIR_GINO="/scratch/yl75/ak4177/results/groundwater/GINO"
RESULTS_BASE_DIR_FNO="/scratch/yl75/ak4177/results/groundwater/FNO"
PREDICTIONS_BASE_DIR_GINO="/scratch/yl75/ak4177/results/groundwater/GINO_predictions"
PREDICTIONS_BASE_DIR_FNO="/scratch/yl75/ak4177/results/groundwater/FNO_predictions"
