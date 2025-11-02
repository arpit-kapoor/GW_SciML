# GFNO Training on 2D Plane Sequences

This directory contains scripts for training the Graph-Fourier Neural Operator (GFNO) model on 2D plane sequences with multi-GPU support.

## Model Architecture

The GFNO combines two powerful neural operator architectures:

1. **GNO (Graph Neural Operator)**: Encodes boundary conditions from irregular point clouds
2. **FNO (Fourier Neural Operator)**: Processes features in latent space on regular grids

### Key Features

- **Irregular input geometry**: Boundary conditions on arbitrary point clouds
- **Regular output**: Predictions on uniform latent grids
- **Multi-GPU support**: Uses PyTorch DataParallel for distributed training
- **Checkpoint/resume**: Full training state persistence

## Data Format

### Expected Directory Structure

```
2d_plane_sequences/
    plane_00/
        input_geom.npy          # [n_sequences, n_bc_points*alpha, 3] - S, Z, T coords
        input_data.npy          # [n_sequences, n_bc_points*alpha, 2] - head, mass_conc
        latent_geom.npy         # [n_sequences, alpha, H, W, 3] - grid coordinates
        latent_features.npy     # [n_sequences, alpha, H, W, 4] - X, Y, head, mass_conc
        output_latent_geom.npy  # [n_sequences, alpha, H, W, 3] - output grid coords
        output_latent_features.npy  # [n_sequences, alpha, H, W, 4] - target values
    plane_01/
        ...
```

### Tensor Shapes

- **Input geometry**: `[B, N_bc_points, 3]` - Boundary condition coordinates (S, Z, T)
- **Input data**: `[B, N_bc_points, 2]` - Boundary values (head, mass_concentration)
- **Latent queries**: `[B, alpha, H, W, 3]` - Latent grid coordinates
- **Latent features**: `[B, alpha, H, W, 4]` - Latent grid features (X, Y, head, mass_conc)
- **Output**: `[B, alpha, H, W, 2]` - Predictions (head, mass_concentration)

Where:
- `B`: Batch size
- `N_bc_points`: Number of boundary condition points
- `alpha`: Temporal window size (e.g., 5 timesteps)
- `H, W`: Spatial grid dimensions (e.g., 32 x 32)

## Training Scripts

### Main Training Script

**File**: `train_gfno_2d_planes.py`

Implements the full training pipeline with:
- Multi-GPU support via DataParallel
- Checkpoint saving and resuming
- Loss tracking and visualization
- Learning rate scheduling
- Gradient clipping

### PBS Script for HPC

**File**: `hpc/train_gfno_2d_planes.pbs`

Configured for Katana HPC cluster with:
- 2 GPUs for parallel training
- 12 CPUs, 64GB RAM
- 24-hour walltime
- Email notifications

## Usage

### Local Training

```bash
python train_gfno_2d_planes.py \
    --data-dir /path/to/2d_plane_sequences \
    --results-dir /path/to/results \
    --batch-size 64 \
    --epochs 100 \
    --learning-rate 1e-3 \
    --use-multi-gpu
```

### HPC Training

```bash
# Submit job
qsub hpc/train_gfno_2d_planes.pbs

# Check job status
qstat -u $USER

# Monitor output
tail -f train_gfno_2d_planes.o*
```

### Resume Training

```bash
python train_gfno_2d_planes.py \
    --data-dir /path/to/2d_plane_sequences \
    --resume-from /path/to/checkpoint.pth \
    --epochs 200
```

## Command-Line Arguments

### Data Arguments

- `--data-dir`: Directory containing 2D plane sequence data (required)
- `--results-dir`: Directory to save results (default: `/srv/scratch/.../GFNO`)

### Model Arguments

Model architecture is configured automatically with sensible defaults:
- GNO encoder: 3D coordinates, radius=0.15, 16 output channels
- FNO core: 4 layers, (6,8,8) modes, 64 hidden channels
- Output: 2 channels (head, mass_concentration)

### Training Arguments

- `--batch-size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--learning-rate`: Initial learning rate (default: 1e-3)
- `--lr-gamma`: LR decay factor (default: 0.95)
- `--lr-scheduler-interval`: Epochs between LR updates (default: 5)
- `--grad-clip-norm`: Gradient clipping norm (default: 1.0)
- `--scheduler-type`: LR scheduler type (`exponential` or `cosine`, default: `exponential`)

### Shuffling Arguments

- `--shuffle-within-batches` / `--no-shuffle-within-batches`: Shuffle examples within batches (default: True)
- `--shuffle-patches` / `--no-shuffle-patches`: Shuffle patch order between epochs (default: True)

### Checkpoint Arguments

- `--resume-from`: Path to checkpoint file to resume from
- `--checkpoint-dir`: Directory to save checkpoints (default: `results_dir/checkpoints`)
- `--save-checkpoint-every`: Save checkpoint every N epochs (default: 10)

### Other Arguments

- `--device`: Device to use (`cuda`, `cpu`, or `auto`, default: `auto`)
- `--use-multi-gpu`: Enable multi-GPU training with DataParallel (default: True)
- `--seed`: Random seed for reproducibility (default: 42)

## Model Configuration

### Default Hyperparameters

```python
# GNO Encoder
coord_dim = 3  # S, Z, T coordinates
gno_radius = 0.15
gno_out_channels = 16
gno_channel_mlp_layers = [32, 64, 32]
gno_pos_embed_type = 'transformer'
gno_pos_embed_channels = 32

# Latent Features
latent_feature_channels = 4  # X, Y, head, mass_conc

# FNO Core
fno_n_layers = 4
fno_n_modes = (6, 8, 8)  # alpha, S, Z dimensions
fno_hidden_channels = 64
lifting_channels = 64

# Output
out_channels = 2  # head, mass_concentration
projection_channel_ratio = 2
```

## Multi-GPU Training

### DataParallel Implementation

The script uses `torch.nn.DataParallel` for multi-GPU training:

1. **Automatic detection**: Detects available GPUs automatically
2. **Data splitting**: Batches are split across GPUs automatically
3. **Gradient aggregation**: Gradients are averaged across GPUs
4. **Efficient handling**: Static geometries are broadcast efficiently

### Performance Considerations

- **Batch size**: Increase proportionally with number of GPUs
- **Memory**: Monitor GPU memory usage to avoid OOM errors
- **Speedup**: Expect ~1.8x speedup with 2 GPUs (not quite 2x due to communication overhead)

### Example Multi-GPU Configuration

```bash
# 2 GPUs, batch size 128 (64 per GPU)
python train_gfno_2d_planes.py \
    --batch-size 128 \
    --use-multi-gpu

# 4 GPUs, batch size 256 (64 per GPU)
python train_gfno_2d_planes.py \
    --batch-size 256 \
    --use-multi-gpu
```

## Output Files

Training produces the following outputs in the results directory:

```
results_dir/
    run_YYYYMMDD_HHMMSS/
        checkpoints/
            checkpoint_epoch_0010.pth
            checkpoint_epoch_0020.pth
            ...
            latest_checkpoint.pth
            final_checkpoint.pth
            loss_history.json
        training_curves.png
        loss_history.json
```

### Checkpoint Contents

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `train_losses`: Training loss history
- `val_losses`: Validation loss history
- `epoch`: Current epoch number
- `args`: Full training configuration

### Loss History JSON

```json
{
    "train_losses": [0.123, 0.098, ...],
    "val_losses": [0.145, 0.112, ...],
    "total_epochs": 100,
    "last_updated": "2025-11-03T10:30:00"
}
```

## Monitoring Training

### During Training

Watch the output logs:
```bash
# Local training
# Watch terminal output

# HPC training
tail -f train_gfno_2d_planes.o*
```

### Training Progress

The script prints:
- Batch-level loss every 10 batches
- Epoch-level summaries (train/val loss, learning rate)
- Checkpoint saving confirmations
- GPU memory usage (if available)

### Loss Curves

Training curves are automatically plotted and saved:
- Updated every 5 epochs
- Shows both training and validation loss
- Indicates resume points for continued training
- Includes summary statistics

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce `--batch-size`
2. Reduce model size (modify `define_model_parameters()`)
3. Use gradient accumulation (not currently implemented)

### Slow Training

If training is slow:
1. Increase `--batch-size` (if memory allows)
2. Reduce validation frequency
3. Use more GPUs
4. Profile with PyTorch profiler

### Checkpoint Compatibility

If resuming fails:
1. Verify checkpoint path is correct
2. Check model architecture hasn't changed
3. Ensure CUDA/device compatibility
4. Review error message for specific parameter mismatch

## Advanced Usage

### Custom Model Configuration

Edit `define_model_parameters()` in `train_gfno_2d_planes.py`:

```python
def define_model_parameters(args):
    # Modify these values
    args.gno_radius = 0.2  # Increase neighborhood size
    args.fno_n_layers = 6  # Deeper FNO
    args.fno_hidden_channels = 128  # Wider FNO
    # ... etc
```

### Custom Loss Functions

Replace the loss function in `train_gfno()`:

```python
# Current: Relative L2 loss
loss_fn = LpLoss(d=1, p=2, reduce_dims=[0, 1], reductions='mean')

# Alternative: H1 loss (includes derivatives)
from src.models.neuralop.losses import H1Loss
loss_fn = H1Loss(d=3)
```

### Learning Rate Schedules

The script supports two schedulers:
- `exponential`: Decay by factor `gamma` every `interval` epochs
- `cosine`: Cosine annealing over all epochs

Switch with `--scheduler-type`:
```bash
python train_gfno_2d_planes.py --scheduler-type cosine
```

## Differences from GINO Training

| Aspect | GINO | GFNO |
|--------|------|------|
| Input | Irregular points | Irregular boundary points |
| Processing | GNO → FNO → GNO | GNO → FNO |
| Output | Irregular points | Regular grid |
| Target | Point-wise predictions | Grid predictions |
| Loss | Computed on output points | Computed on grid |
| Use case | General operator learning | Boundary-to-field problems |

## Performance Benchmarks

Expected training performance on Katana:

| Configuration | Time/Epoch | GPU Memory | Throughput |
|--------------|------------|------------|------------|
| 1x GPU, BS=64 | ~5 min | ~8 GB | ~12 batch/s |
| 2x GPU, BS=128 | ~3 min | ~8 GB/GPU | ~21 batch/s |
| 4x GPU, BS=256 | ~2 min | ~8 GB/GPU | ~42 batch/s |

*Benchmarks are approximate and depend on data size and model configuration*

## Related Files

- `src/models/gfno.py`: GFNO model implementation
- `src/data/plane_dataset.py`: Dataset implementation for 2D planes
- `src/data/batch_sampler.py`: Batch sampler for plane-based sampling
- `notebooks/test_gfno_model.ipynb`: Testing notebook for GFNO

## References

- **GINO Paper**: [Graph Neural Operator for Irregular Geometries](https://arxiv.org/abs/2206.14127)
- **FNO Paper**: [Fourier Neural Operator](https://arxiv.org/abs/2010.08895)
- **NeuralOperator Library**: [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)

## Contact

For questions or issues:
- Author: Arpit Kapoor (z5370003@unsw.edu.au)
- Project: Groundwater Modeling with Scientific Machine Learning
