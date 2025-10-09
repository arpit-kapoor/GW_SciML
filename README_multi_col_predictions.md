# Multi-Column GINO Prediction Scripts

This document explains the multi-column versions of the GINO prediction scripts, which generate predictions from models trained on multiple target variables simultaneously.

## Overview

The multi-column prediction pipeline consists of:
1. **`generate_gino_predictions_multi_col.py`** - Python script for generating predictions
2. **`hpc/generate_gino_predictions_multi_col.pbs`** - PBS job script for HPC execution

These scripts extend the single-column prediction functionality to handle models trained on multiple target variables (e.g., mass_concentration and head together).

## Key Features

### Data Handling
- Uses `GWPatchDatasetMultiCol` for loading multi-column data
- Automatically reshapes concatenated predictions: `[N_samples, N_points, output_window * n_cols]` → `[N_samples, N_points, output_window, n_cols]`
- Separates predictions by target column for independent analysis

### Visualizations

#### 1. Combined First-Timestep Scatter Plots
- **Location**: `{results_dir}/first_timestep_all_columns.png`
- **Description**: Rows = samples, Columns = target variables
- Shows prediction vs observation for the first timestep of each target column
- Displays correlation coefficient for each subplot

#### 2. Combined 3D Scatter Plots Video ⭐ NEW
- **Location**: `{results_dir}/combined_3d_scatter_plots_video.mp4`
- **Description**: Single video showing all target columns together
- **Layout**: 
  - Rows = target columns
  - Columns = observations, predictions, error
- Each frame shows one sample with all target variables
- 10 fps video cycling through all samples
- Individual plot images saved in `{results_dir}/combined_3d_scatter_plots/`

#### 3. Per-Column Analyses
Each target column gets its own subdirectory with:

**Scatter Plots** (`{col_name}/predictions_vs_observations.png`)
- Shows predictions vs observations at first, middle, and last timesteps
- Multiple samples shown as rows

**Time Series** (`{col_name}/time_series_comparison.png`)
- Spatially-averaged values over time
- Compares predicted vs observed temporal evolution

**Error Analysis** (`{col_name}/error_analysis.png`)
- MAE and RMSE by timestep
- Helps identify error accumulation patterns

## Directory Structure

### Input (Model Location)
```
RESULTS_BASE_DIR/multi_col/{TARGET_COLS_ID}/{EXPERIMENT_NAME}/gino_multi_{TIMESTAMP}/
└── checkpoints/
    ├── latest_checkpoint.pth
    ├── checkpoint_epoch_0010.pth
    └── ...
```

### Output (Default - Separate Storage)
```
PREDICTIONS_BASE_DIR/multi_col/{TARGET_COLS_ID}/{EXPERIMENT_NAME}/gino_multi_{TIMESTAMP}/
├── first_timestep_all_columns.png          # Combined scatter plots (2D)
├── combined_3d_scatter_plots_video.mp4     # ⭐ Single video with all targets
├── combined_3d_scatter_plots/              # Individual frames for video
│   ├── combined_3d_scatter_sample_001.png
│   ├── combined_3d_scatter_sample_002.png
│   └── ...
├── train_predictions.npy                    # Shape: [N_train, N_points, T_out, n_cols]
├── train_targets.npy
├── val_predictions.npy                      # Shape: [N_val, N_points, T_out, n_cols]
├── val_targets.npy
├── train_coords.pkl
├── val_coords.pkl
├── metadata.pkl
├── mass_concentration/                      # Per-column directory
│   ├── predictions_vs_observations.png
│   ├── time_series_comparison.png
│   └── error_analysis.png
└── head/                                    # Another target column
    ├── predictions_vs_observations.png
    ├── time_series_comparison.png
    └── error_analysis.png
```

### Output (Alternative - Store With Model)
Set `STORE_WITH_MODEL=true` to save predictions alongside the model:
```
RESULTS_BASE_DIR/multi_col/{TARGET_COLS_ID}/{EXPERIMENT_NAME}/gino_multi_{TIMESTAMP}/
├── checkpoints/
│   └── ...
└── predictions/
    ├── first_timestep_all_columns.png
    ├── combined_3d_scatter_plots_video.mp4
    ├── combined_3d_scatter_plots/
    ├── mass_concentration/
    └── head/
```

## Usage Examples

### Basic Usage

#### 1. Generate predictions for default experiment
```bash
qsub generate_gino_predictions_multi_col.pbs
```
- Uses: `mass_concentration head` (default target columns)
- Experiment: `exp_lr5e4_cos_bs128` (default)
- Checkpoint: `latest_checkpoint.pth`
- Storage: Separate predictions directory

#### 2. Different target column combination
```bash
qsub -v TARGET_COLS="head pressure" generate_gino_predictions_multi_col.pbs
```

#### 3. All three variables
```bash
qsub -v TARGET_COLS="mass_concentration head pressure" generate_gino_predictions_multi_col.pbs
```

### Advanced Usage

#### 4. Different experiment
```bash
qsub -v EXPERIMENT_NAME=exp_lr1e3_exp_bs256 generate_gino_predictions_multi_col.pbs
```

#### 5. Specific checkpoint
```bash
qsub -v CHECKPOINT=checkpoint_epoch_0050.pth generate_gino_predictions_multi_col.pbs
```

#### 6. Store predictions with model
```bash
qsub -v STORE_WITH_MODEL=true generate_gino_predictions_multi_col.pbs
```

#### 7. Custom predictions directory
```bash
qsub -v PREDICTIONS_BASE_DIR=/custom/path/predictions generate_gino_predictions_multi_col.pbs
```

#### 8. Combined parameters
```bash
qsub -v "TARGET_COLS=mass_concentration head pressure,EXPERIMENT_NAME=exp_lr2e4_cos_bs64,CHECKPOINT=checkpoint_epoch_0040.pth" generate_gino_predictions_multi_col.pbs
```

### Direct Python Usage

```bash
python generate_gino_predictions_multi_col.py \
    --model-path /path/to/checkpoint.pth \
    --base-data-dir /path/to/data \
    --results-dir /path/to/output \
    --target-cols mass_concentration head \
    --input-window-size 5 \
    --output-window-size 5 \
    --batch-size 128 \
    --device cuda
```

## Configuration Parameters

### PBS Script Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_COLS` | `mass_concentration head` | Space-separated list of target columns |
| `EXPERIMENT_NAME` | `exp_lr5e4_cos_bs128` | Name of the experiment to load model from |
| `CHECKPOINT` | `latest_checkpoint.pth` | Checkpoint filename to use |
| `BATCH_SIZE` | `128` | Batch size for inference |
| `INPUT_WINDOW` | `5` | Input window size (must match training) |
| `OUTPUT_WINDOW` | `5` | Output window size (must match training) |
| `STORE_WITH_MODEL` | `false` | Store predictions with model or separately |
| `PREDICTIONS_BASE_DIR` | `/srv/scratch/.../GINO_predictions` | Base directory for predictions |
| `RESULTS_BASE_DIR` | `/srv/scratch/.../GINO` | Base directory where models are stored |

### Python Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | Required | Path to checkpoint file |
| `--target-cols` | `mass_concentration head` | Target column names (space-separated) |
| `--base-data-dir` | See script | Base data directory |
| `--raw-data-subdir` | `all` | Raw data subdirectory |
| `--patch-data-subdir` | `filter_patch` | Patch data subdirectory |
| `--results-dir` | See script | Output directory |
| `--input-window-size` | `10` | Input sequence length |
| `--output-window-size` | `10` | Output sequence length |
| `--batch-size` | `32` | Batch size for inference |
| `--device` | `auto` | Device (cuda/cpu/auto) |

## Target Column Identifiers

The system creates short identifiers for target column combinations:

| Columns | Identifier |
|---------|------------|
| `mass_concentration` | `mass_conc` |
| `head` | `head` |
| `pressure` | `press` |
| `mass_concentration head` | `mass_conc_head` |
| `mass_concentration head pressure` | `mass_conc_head_press` |
| `head pressure` | `head_press` |

These identifiers are used in directory paths to keep them readable.

## Output Files

### Prediction Arrays
- **`train_predictions.npy`**: Training predictions `[N_train, N_points, T_out, n_cols]`
- **`train_targets.npy`**: Training ground truth
- **`val_predictions.npy`**: Validation predictions `[N_val, N_points, T_out, n_cols]`
- **`val_targets.npy`**: Validation ground truth

### Coordinates
- **`train_coords.pkl`**: Spatial coordinates for training samples
- **`val_coords.pkl`**: Spatial coordinates for validation samples

### Metadata
- **`metadata.pkl`**: Contains:
  - `train_metadata`: Batch and sample information for training
  - `val_metadata`: Batch and sample information for validation
  - `args`: Configuration used for prediction generation

### Combined Video Layout

The `combined_3d_scatter_plots_video.mp4` shows all target columns in a single video:

**For 2 target columns (e.g., mass_concentration + head):**
```
┌─────────────────────────────────────────────────────────┐
│  mass_concentration    │  mass_concentration   │  Error  │
│    Observations        │    Predictions        │ (Obs-P) │
├─────────────────────────────────────────────────────────┤
│       head             │       head            │  Error  │
│    Observations        │    Predictions        │ (Obs-P) │
└─────────────────────────────────────────────────────────┘
```

**For 3 target columns (e.g., mass_concentration + head + pressure):**
```
┌─────────────────────────────────────────────────────────┐
│  mass_concentration    │  mass_concentration   │  Error  │
│    Observations        │    Predictions        │ (Obs-P) │
├─────────────────────────────────────────────────────────┤
│       head             │       head            │  Error  │
│    Observations        │    Predictions        │ (Obs-P) │
├─────────────────────────────────────────────────────────┤
│     pressure           │     pressure          │  Error  │
│    Observations        │    Predictions        │ (Obs-P) │
└─────────────────────────────────────────────────────────┘
```

Each frame in the video represents one sample, cycling through all validation samples at 10 fps.

## Typical Workflow

1. **Train multi-column model** using `train_gino_multi_col.pbs`
2. **Wait for training completion** (checkpoints saved periodically)
3. **Generate predictions** using `generate_gino_predictions_multi_col.pbs`
4. **Analyze results** by examining:
   - Combined first-timestep scatter plot
   - Per-column error statistics
   - 3D spatial distributions
   - Time series evolution
5. **Compare models** by generating predictions for different experiments

## Troubleshooting

### Error: No training runs found
```bash
echo "Available experiments:"
ls -d /srv/scratch/.../GINO/multi_col/*/exp_*
```

### Error: Checkpoint not found
```bash
# List available checkpoints
ls -1 /path/to/experiment/gino_multi_*/checkpoints/
```

### Dimension mismatch errors
- Ensure `INPUT_WINDOW` and `OUTPUT_WINDOW` match training configuration
- Verify `TARGET_COLS` matches the model (order matters)

### Memory issues
- Reduce `BATCH_SIZE` (e.g., from 128 to 64 or 32)
- Process fewer samples by modifying the script

## Performance Considerations

- **GPU Usage**: Predictions are faster on GPU (automatic with `device=auto`)
- **Batch Size**: Larger batches are more efficient but require more memory
- **Video Generation**: Can be slow for many samples; consider running separately
- **3D Plots**: Memory-intensive for large point clouds; may need to sample points

## Comparison with Single-Column Scripts

| Feature | Single-Column | Multi-Column |
|---------|---------------|--------------|
| Dataset | `GWPatchDataset` | `GWPatchDatasetMultiCol` |
| Target specification | `--target-col` (single) | `--target-cols` (multiple) |
| Prediction shape | `[N, P, T]` | `[N, P, T, C]` |
| Output structure | Flat | Per-column subdirectories |
| Combined visualizations | No | Yes (first timestep) |
| Directory naming | `{target_col}/` | `multi_col/{cols_id}/` |

## See Also

- **Training**: `README_multi_col_training.md` (if available)
- **Single-column predictions**: `generate_gino_predictions.py`
- **Dataset documentation**: `src/data/patch_dataset_multi_col.py`

