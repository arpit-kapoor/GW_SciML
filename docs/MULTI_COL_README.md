# Multi-Column GINO Training

This document explains the multi-column extension for training GINO on multiple target variables simultaneously.

## Overview

The multi-column version extends the original single-column training to support multiple target variables (e.g., mass_concentration, head, pressure) in a single model. This is achieved by concatenating values from all target columns along the last dimension of input/output sequences.

## New Files

1. **`train_gino_on_patches_multi_col.py`**: Main training script with multi-column support
2. **`src/data/patch_dataset_multi_col.py`**: Dataset class that handles multi-column data loading and concatenation

## Key Differences from Single-Column Version

### 1. Command Line Arguments

**Single-column:**
```bash
--target-col mass_concentration
```

**Multi-column:**
```bash
--target-cols mass_concentration head pressure
```

### 2. Model Channel Dimensions

The key insight is that we concatenate multiple variables along the last dimension without adding new tensor dimensions.

**Example:** For 2 target columns and 5 time steps:
- **Single column**: `[N_points, 5]` per variable
- **Multi-column**: `[N_points, 10]` (5 timesteps × 2 variables)

**Model parameters:**
- `in_gno_out_channels = input_window_size * n_target_cols`
- `out_channels = output_window_size * n_target_cols`

### 3. Data Concatenation Strategy

The dataset concatenates sequences as follows:

**Input format:** `[time_steps, n_points, n_target_cols]`

**Concatenation process:**
1. Extract time window: `[window_size, n_points, n_target_cols]`
2. Transpose: `[n_points, window_size, n_target_cols]`
3. Flatten: `[n_points, window_size * n_target_cols]`

**Example with 2 variables (mass_concentration, head) and 3 timesteps:**
```
Original per variable:
  mass_conc: [[t0], [t1], [t2]]  # shape: [3, n_points]
  head:      [[t0], [t1], [t2]]  # shape: [3, n_points]

Combined:
  [time_steps, n_points, 2]

After concatenation:
  [n_points, 6]  # [mass_t0, head_t0, mass_t1, head_t1, mass_t2, head_t2]
```

## Usage

### Basic Training

Train on two variables (mass_concentration and head):
```bash
python train_gino_on_patches_multi_col.py \
    --target-cols mass_concentration head \
    --input-window-size 10 \
    --output-window-size 10 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 5e-4
```

Train on all three variables:
```bash
python train_gino_on_patches_multi_col.py \
    --target-cols mass_concentration head pressure \
    --input-window-size 10 \
    --output-window-size 10 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 5e-4
```

### Resume Training

```bash
python train_gino_on_patches_multi_col.py \
    --resume-from /path/to/checkpoint.pth \
    --target-cols mass_concentration head \
    --epochs 150
```

## Tensor Shapes

Throughout the training pipeline:

| Component | Single Column | Multi-Column (n=2) |
|-----------|---------------|-------------------|
| Point coords | `[N_points, 3]` | `[N_points, 3]` |
| Latent queries | `[Qx, Qy, Qz, 3]` | `[Qx, Qy, Qz, 3]` |
| Input (x) | `[B, N_points, 10]` | `[B, N_points, 20]` |
| Output (y) | `[B, N_points, 10]` | `[B, N_points, 20]` |
| Model output | `[B, N_points, 10]` | `[B, N_points, 20]` |

*Assuming input_window_size = output_window_size = 10*

## Implementation Details

### GWPatchDatasetMultiCol

The dataset class handles:
1. Loading patch data with multiple observation columns
2. Applying normalization transforms to all columns
3. Selecting specified target columns via indices
4. Concatenating sequences across time and variable dimensions

**Key method: `_concat_sequence()`**
```python
def _concat_sequence(self, seq):
    """
    Concatenate sequence across time and variable dimensions.
    
    Args:
        seq: Array of shape [window_size, n_points, n_target_cols]
        
    Returns:
        Array of shape [n_points, window_size * n_target_cols]
    """
    # Transpose to [n_points, window_size, n_target_cols]
    seq = seq.transpose(1, 0, 2)
    
    # Reshape to [n_points, window_size * n_target_cols]
    n_points = seq.shape[0]
    seq_flat = seq.reshape(n_points, -1)
    
    return seq_flat
```

### Model Configuration

The `define_model_parameters()` function automatically adjusts channel dimensions:

```python
# Number of target columns
args.n_target_cols = len(args.target_cols)

# Input channels scaled by number of variables
args.in_gno_out_channels = args.input_window_size * args.n_target_cols

# Output channels scaled by number of variables
args.out_channels = args.output_window_size * args.n_target_cols
```

## Checkpoint Compatibility

Checkpoints include validation for multi-column parameters:
- `target_col_indices`: Must match when resuming
- `n_target_cols`: Must match when resuming
- `in_gno_out_channels`: Must match (derived from n_target_cols)
- `out_channels`: Must match (derived from n_target_cols)

## Debugging

The training script includes shape debugging:
```
DEBUG - Step 1:
  point_coords shape: torch.Size([N_points, 3])
  latent_queries shape: torch.Size([32, 32, 24, 3])
  x shape: torch.Size([B, N_points, input_window_size * n_target_cols])
  y shape: torch.Size([B, N_points, output_window_size * n_target_cols])
  outputs shape: torch.Size([B, N_points, output_window_size * n_target_cols])
```

## Benefits

1. **Single model for multiple variables**: One model learns correlations between variables
2. **Efficient training**: Shared computations across variables
3. **No additional dimensions**: Maintains compatibility with existing GINO architecture
4. **Flexible**: Can train on any subset of available variables

## Limitations

1. **Channel growth**: More variables → larger final layers
2. **Assumes similar scales**: All variables should be normalized
3. **Coupled predictions**: Cannot predict variables independently

## Comparison with Alternatives

### Alternative 1: Separate Models
- ❌ No shared learning between variables
- ❌ Multiple training runs required
- ✅ Independent predictions

### Alternative 2: Additional Tensor Dimension
- ❌ Requires architecture changes
- ❌ More complex implementation
- ✅ Cleaner separation of variables

### Chosen Approach: Channel Concatenation
- ✅ No architecture changes needed
- ✅ Shared learning between variables
- ✅ Single training run
- ⚠️ Predictions are coupled

## Tips

1. **Start small**: Begin with 2 variables to verify correct shapes
2. **Check normalization**: Ensure all variables are on similar scales
3. **Monitor shapes**: Use the debug prints to verify tensor dimensions
4. **Compare losses**: Single vs multi-column to assess benefit

