# Neural Operator Training for Variable-Density Groundwater Modeling

Neural operator training (GINO and FNO) for 3D variable-density groundwater modeling with multi-variable support.

## Overview

This repository contains training and inference scripts for two neural operator architectures:

- **GINO** (Geometry-Informed Neural Operator) - Uses graph-based message passing with point cloud data
- **FNO** (Fourier Neural Operator) - Uses spectral methods with interpolation between point clouds and regular grids

Both models support multi-variable predictions for groundwater variables including mass concentration, hydraulic head, and pressure.

## Requirements

### Python Environment

Python 3.11+ is required (`requires-python = ">=3.11"` in `pyproject.toml`).

Recommended setup:

```bash
# From repository root
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Core runtime packages used by training/inference scripts:

- `torch`
- `torchvision`
- `torch-geometric`
- `torch-harmonics`
- `tensorly` and `tensorly-torch`
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`
- `tqdm`, `zarr`, `h5py`
- `open3d` (optional, used when creating 3D visualizations)

Dependencies are managed via `uv` and declared in `pyproject.toml`.

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

# Train FNO with forcings-enabled configuration
./hpc/submit.sh fno forcing

# Resume training from latest checkpoint
./hpc/submit.sh gino default --resume

# Generate predictions using trained model
./hpc/submit.sh fno default --predict

# Override config parameters
./hpc/submit.sh gino default --epochs 100 --batch-size 512

# Run with ad-hoc args (no preset config)
./hpc/submit.sh gino adhoc --epochs 50 --patch-data-subdir patch_all_ts
```

### Option 2: Run Scripts Directly (Python)

For local development or interactive sessions:

#### Training

```bash
# Train GINO model
python gino_train.py \
    --base-data-dir /path/to/data \
    --patch-data-subdir filter_patch_all_ts \
    --epochs 300 \
    --batch-size 512 \
    --learning-rate 8e-4 \
    --target-cols mass_concentration head \
    --input-window-size 5 \
    --output-window-size 1

# Train FNO model
python fno_train.py \
    --base-data-dir /path/to/data \
    --patch-data-subdir filter_patch_all_ts \
    --epochs 250 \
    --batch-size 256 \
    --learning-rate 5e-4 \
    --target-cols mass_concentration head

# Resume training from checkpoint
python gino_train.py --resume-from /path/to/checkpoint.pth
```

#### Prediction/Inference

```bash
# Generate GINO predictions (full resolution)
python gino_predict.py \
    --model-path /path/to/model.pth \
    --base-data-dir /path/to/data \
    --patch-data-subdir filter_patch_all_ts \
    --batch-size 256

# Generate FNO predictions (full resolution)
python fno_predict.py \
    --model-path /path/to/model.pth \
    --base-data-dir /path/to/data \
    --batch-size 256 \
    --create-3d-plots  # Optional: create 3D visualizations

# Test at different resolutions (resolution generalization)
python gino_predict.py \
    --model-path /path/to/model.pth \
    --resolution-ratio 0.5  # Test at 50% resolution

python fno_predict.py \
    --model-path /path/to/model.pth \
    --resolution-ratio 0.25 \
    --min-resolution-ratio 0.20  # Test at 25% resolution with dynamic subsampling floor

# Multi-resolution evaluation loop
for ratio in 1.0 0.75 0.5 0.25 0.1; do
    python fno_predict.py \
        --model-path /path/to/model.pth \
        --resolution-ratio $ratio \
        --results-dir results_res_${ratio}
done
```

## Available Configurations

Pre-configured training setups are located in `hpc/configs/{gino,fno}/`:

### GINO Configs

- **default** - Standard 2-variable training on `filter_patch_all_ts`
- **dynamic_sampling** - Dynamic subsampling on `patch_all_ts` (`--resolution-ratio 0.60`, forcings enabled)
- **forcing** - Forcings-enabled training on full filtered patches
- **forcing_standard_loss** - Forcings-enabled training with standard loss weighting (`--lambda-conc-focus 0.0`)
- **updated_lowres** - Low-resolution training (`--resolution-ratio 0.167`, forcings enabled)
- **var_loss** - Variance-aware concentration-focused training
- **local_testing** - Lightweight smoke-test config
- **mass_only** - Single-target (`mass_concentration`) training

### FNO Configs

- **default** - Standard training (250 epochs, batch size 256, 2 variables)
- **forcing** - Forcings-enabled training/prediction pipeline
- **var_loss** - Variance-aware loss with concentration focus (λ=0.3)

### Submission Modes

- `train` (default): `./hpc/submit.sh <gino|fno> <config>`
- `resume`: `./hpc/submit.sh <gino|fno> <config> --resume`
- `predict`: `./hpc/submit.sh <gino|fno> <config> --predict`
- `adhoc` config: pass script flags directly without loading a preset config file

## Common Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base-data-dir` | Base path containing raw and patch subdirectories | model script default |
| `--raw-data-subdir` | Raw data subdirectory under base path | `all` |
| `--patch-data-subdir` | Patch dataset subdirectory under base path | `filter_patch_all_ts` |
| `--epochs` | Number of training epochs | 5 |
| `--batch-size` | Batch size for training | 32 |
| `--learning-rate` | Learning rate | 5e-4 |
| `--target-cols` | Target variables to predict | mass_concentration head |
| `--forcings-required` | Include forcings as model inputs | False |
| `--input-window-size` | Number of input timesteps | 10 |
| `--output-window-size` | Number of output timesteps | 10 |
| `--scheduler-type` | LR scheduler (exponential/cosine) | exponential |
| `--lr-scheduler-interval` | Epoch interval for scheduler updates | 10 |
| `--lr-gamma` | Exponential LR decay factor | 0.98 |
| `--grad-clip-norm` | Gradient clipping norm | 1.0 |
| `--lambda-conc-focus` | Concentration focus weight (variance-aware loss) | 0.5 |
| `--var-aware-alpha` | Variance-aware loss alpha | 0.3 |
| `--var-aware-beta` | Variance-aware loss beta | 2.0 |
| `--resolution-ratio` | Spatial subsampling ratio during training | 1.0 |
| `--min-resolution-ratio` | Lower bound used by dynamic subsampling | 0.20 |
| `--save-checkpoint-every` | Checkpoint frequency (epochs) | 5 |
| `--resume-from` | Path to checkpoint to resume from | None |
| `--checkpoint-dir` | Explicit checkpoint directory | results_dir/checkpoints |
| `--shuffle-within-batches` / `--no-shuffle-within-batches` | Toggle within-batch shuffling | enabled |
| `--shuffle-patches` / `--no-shuffle-patches` | Toggle patch-order shuffling | enabled |
| `--device` | Device (cuda/cpu/auto) | auto |
| `--seed` | Random seed | 42 |

### FNO-Only Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--padding-mode` | `grid_sample` padding mode (`zeros`, `border`, `reflection`) | `border` |
| `--align-corners` | Enable corner alignment in interpolation | False |

### Inference-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path to checkpoint/model file | required |
| `--patch-data-subdir` | Patch dataset subdirectory for inference | `filter_patch` |
| `--batch-size` | Batch size for inference | 32 |
| `--resolution-ratio` | Ratio of nodes to keep (0 < ratio ≤ 1.0) for resolution testing | 1.0 |
| `--min-resolution-ratio` | Lower bound used by dynamic subsampling | 0.20 |
| `--metrics-only` | Save only metrics/metadata (skip arrays/plots) | False |
| `--create-3d-plots` | Create 3D scatter plots and videos | False |
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
tail -f logs/gino_*_train_*.log

# FNO training logs
tail -f logs/fno_*_train_*.log

# Prediction logs
tail -f logs/gino_*_predict_*.log
tail -f logs/fno_*_predict_*.log
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
