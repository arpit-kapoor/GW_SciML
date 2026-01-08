# Neural Operator Training for Variable-Density Groundwater Modeling

Neural operator training (GINO and FNO) for 3D variable-density groundwater modeling with multi-variable support.

## Overview

This repository contains training and inference scripts for two neural operator architectures:

- **GINO** (Graph-Informed Neural Operator) - Uses graph-based message passing with point cloud data
- **FNO** (Fourier Neural Operator) - Uses spectral methods with interpolation between point clouds and regular grids

Both models support multi-variable predictions for groundwater variables including mass concentration, hydraulic head, and pressure.

## Requirements

### Python Environment

Python 3.11+ with the following main dependencies:

```
torch>=2.0
numpy
pandas
matplotlib
tltorch
open3d (optional, for 3D visualizations)
torch-scatter (optional, for certain graph operations)
```

### System Requirements

- CUDA-capable GPU recommended for training (multi-GPU support via DataParallel)
- ~50GB disk space for data and results
- PBS/Torque job scheduler for HPC cluster submissions (optional)

## Quick Start

### Option 1: Submit PBS Jobs (HPC Cluster)

The recommended way to run training and prediction on HPC systems:

```bash
# View available models and configs
./hpc/submit.sh

# Train GINO with default configuration
./hpc/submit.sh gino default

# Train FNO with variance-aware loss configuration
./hpc/submit.sh fno var_loss

# Resume training from latest checkpoint
./hpc/submit.sh gino default --resume

# Generate predictions using trained model
./hpc/submit.sh fno default --predict

# Override config parameters
./hpc/submit.sh gino default --epochs 100 --batch-size 512
```

### Option 2: Run Scripts Directly (Python)

For local development or interactive sessions:

#### Training

```bash
# Train GINO model
python gino_train.py \
    --base-data-dir /path/to/data \
    --patch-data-subdir filter_patch \
    --epochs 300 \
    --batch-size 512 \
    --learning-rate 8e-4 \
    --target-cols mass_concentration head \
    --input-window-size 5 \
    --output-window-size 1

# Train FNO model
python fno_train.py \
    --base-data-dir /path/to/data \
    --patch-data-subdir filter_patch \
    --epochs 250 \
    --batch-size 256 \
    --learning-rate 5e-4 \
    --target-cols mass_concentration head

# Resume training from checkpoint
python gino_train.py --resume-from /path/to/checkpoint.pth
```

#### Prediction/Inference

```bash
# Generate GINO predictions
python gino_predict.py \
    --model-path /path/to/model.pth \
    --base-data-dir /path/to/data \
    --patch-data-subdir filter_patch \
    --batch-size 256

# Generate FNO predictions
python fno_predict.py \
    --model-path /path/to/model.pth \
    --base-data-dir /path/to/data \
    --batch-size 256 \
    --create-3d-plots  # Optional: create 3D visualizations
```

## Available Configurations

Pre-configured training setups are located in `hpc/configs/{gino,fno}/`:

### GINO Configs

- **default** - Standard training (300 epochs, batch size 512, 2 variables)
- **var_loss** - Variance-aware loss with concentration focus (λ=0.3)

### FNO Configs

- **default** - Standard training (250 epochs, batch size 256, 2 variables)
- **var_loss** - Variance-aware loss with concentration focus (λ=0.3)

## Common Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | 5 |
| `--batch-size` | Batch size for training | 32 |
| `--learning-rate` | Learning rate | 5e-4 |
| `--target-cols` | Target variables to predict | mass_concentration head |
| `--input-window-size` | Number of input timesteps | 10 |
| `--output-window-size` | Number of output timesteps | 10 |
| `--scheduler-type` | LR scheduler (exponential/cosine) | exponential |
| `--lr-gamma` | Exponential LR decay factor | 0.98 |
| `--grad-clip-norm` | Gradient clipping norm | 1.0 |
| `--lambda-conc-focus` | Concentration focus weight (variance-aware loss) | 0.5 |
| `--resume-from` | Path to checkpoint to resume from | None |
| `--device` | Device (cuda/cpu/auto) | auto |

## Monitoring and Results

### Check Job Status (HPC)

```bash
# View running jobs
qstat -u $USER

# View specific job details
qstat -f <job_id>
```

### View Training Logs

```bash
# GINO training logs
tail -f logs/train_gino_*.log

# FNO training logs
tail -f logs/train_fno_interpolate_*.log

# Prediction logs
tail -f logs/predict_gino_*.log
tail -f logs/predict_fno_*.log
```

### Results Directory Structure

Training results are organized by model type and configuration:

```
results/
├── GINO/
│   ├── default/
│   │   └── training_TIMESTAMP/
│   │       ├── checkpoints/
│   │       │   ├── latest_checkpoint.pth
│   │       │   └── epoch_*.pth
│   │       ├── training_curves.png
│   │       └── loss_history.json
│   └── var_loss/
│       └── training_TIMESTAMP/
│           └── ...
├── FNO/
│   ├── default/
│   └── var_loss/
├── GINO_predictions/
│   └── default/
│       ├── metrics/
│       ├── predictions/
│       └── visualizations/
└── FNO_predictions/
    └── default/
        └── ...
```

## Key Features

- **Modular Architecture**: Reusable components for data loading, training, and inference
- **Config-Based Experiments**: Organize experiments with configuration files
- **Automatic Checkpointing**: Save and resume training seamlessly
- **Multi-Variable Support**: Train on multiple groundwater variables simultaneously
- **Variance-Aware Loss**: Focus training on specific variables (e.g., concentration)
- **Multi-GPU Training**: DataParallel support for faster training
- **Comprehensive Metrics**: Automated computation of MSE, MAE, R² per variable
- **Visualization Tools**: Training curves, prediction plots, 3D scatter plots
