# Plane Dataset and Batch Sampler

This document describes how to use the `GWPlaneDataset` and `PatchBatchSampler` classes for training models on 2D plane data.

## Overview

The plane dataset classes are designed to handle temporal sequences of groundwater data organized by 2D planes. The key feature is that samples from the same plane are grouped together in batches, which is important for proper model training.

## Dataset Classes

### 1. GWPlaneDataset (In-Memory)

Use this when your data fits in memory. It loads all sequences at initialization for fast access during training.

```python
from src.data.plane_dataset import GWPlaneDataset

dataset = GWPlaneDataset(
    input_sequences=input_sequences,    # Dict mapping plane_id to input data
    output_sequences=output_sequences,  # Dict mapping plane_id to output data
    coord_transform=None,               # Optional coordinate transform
    obs_transform=None,                 # Optional observation transform
    fill_nan_value=-999.0              # Value to replace NaN with
)
```

**Input Data Structure:**

Each plane in `input_sequences` should have:
- `'input_geom'`: (n_sequences, n_bc_nodes * alpha, 3) - Boundary condition coordinates (S, Z, T)
- `'input_data'`: (n_sequences, n_bc_nodes * alpha, 2) - Boundary values (head, mass_conc)
- `'latent_geom'`: (n_sequences, alpha, n_nodes, 3) - Latent grid coordinates (S, Z, T)
- `'latent_features'`: (n_sequences, alpha, n_nodes, 4) - Latent features (X, Y, head, mass_conc)

Each plane in `output_sequences` should have:
- `'latent_geom'`: (n_sequences, alpha, n_nodes, 3) - Output coordinates
- `'latent_features'`: (n_sequences, alpha, n_nodes, 4) - Output features

### 2. GWPlaneDatasetFromFiles (On-Disk)

Use this for large datasets that don't fit in memory. It loads data on-demand from disk.

```python
from src.data.plane_dataset import GWPlaneDatasetFromFiles

dataset = GWPlaneDatasetFromFiles(
    data_dir='/path/to/data',
    coord_transform=None,
    obs_transform=None,
    fill_nan_value=-999.0
)
```

**Expected Directory Structure:**

```
data_dir/
    plane_00/
        input_geom.npy
        input_data.npy
        latent_geom.npy
        latent_features.npy
        output_latent_geom.npy
        output_latent_features.npy
    plane_01/
        ...
```

## Batch Sampler

The `PatchBatchSampler` (from `src.data.batch_sampler`) ensures that all samples in a batch come from the same plane. This is crucial for models that expect spatial/temporal consistency within batches.

```python
from src.data.batch_sampler import PatchBatchSampler

batch_sampler = PatchBatchSampler(
    dataset=dataset,
    batch_size=8,
    shuffle_within_batches=True,  # Shuffle sequences within each batch
    shuffle_patches=True,          # Shuffle the order of planes between epochs
    seed=42                        # Random seed for reproducibility
)
```

**Key Features:**
- **Plane Grouping**: All samples in a batch come from the same plane
- **Shuffling**: Can shuffle both the order of planes and sequences within batches
- **Efficient Caching**: Pre-builds batch structure for fast epoch iterations
- **Reshuffling**: Automatically reshuffles between epochs

## Creating a DataLoader

Combine the dataset and batch sampler to create a PyTorch DataLoader:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=4,  # Use multiple workers for parallel loading
    pin_memory=True  # For faster GPU transfer
)
```

**Note**: When using `batch_sampler`, don't specify `batch_size` or `shuffle` in the DataLoader constructor.

## Usage Example

```python
# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # All samples in batch are from the same plane
        plane_ids = batch['plane_id']  # All values will be identical
        input_geom = batch['input_geom']  # (batch_size, n_bc_nodes * alpha, 3)
        input_data = batch['input_data']  # (batch_size, n_bc_nodes * alpha, 2)
        latent_geom = batch['latent_geom']  # (batch_size, alpha, n_nodes, 3)
        latent_features = batch['latent_features']  # (batch_size, alpha, n_nodes, 4)
        output_latent_geom = batch['output_latent_geom']  # (batch_size, alpha, n_nodes, 3)
        output_latent_features = batch['output_latent_features']  # (batch_size, alpha, n_nodes, 4)
        
        # Forward pass
        predictions = model(
            input_geom=input_geom,
            x=input_data,
            latent_queries=latent_geom,
            latent_features=latent_features
        )
        
        # Compute loss
        loss = criterion(predictions, output_latent_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Dataset Output Format

Each sample from the dataset is a dictionary containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `plane_id` | scalar | ID of the plane (0-31) |
| `input_geom` | (n_bc * alpha, 3) | Boundary condition coordinates (S, Z, T) |
| `input_data` | (n_bc * alpha, 2) | Boundary values (head, mass_conc) |
| `latent_geom` | (alpha, n_nodes, 3) | Input latent grid coordinates |
| `latent_features` | (alpha, n_nodes, 4) | Input latent features (X, Y, head, mass_conc) |
| `output_latent_geom` | (alpha, n_nodes, 3) | Output latent grid coordinates |
| `output_latent_features` | (alpha, n_nodes, 4) | Output latent features to predict |

## Advanced Features

### Custom Transforms

You can apply custom transformations to coordinates and observations:

```python
def normalize_coords(coords):
    """Normalize coordinates to [-1, 1]"""
    return (coords - coords.mean(dim=0)) / coords.std(dim=0)

def normalize_obs(obs):
    """Normalize observations"""
    return (obs - obs.mean()) / obs.std()

dataset = GWPlaneDataset(
    input_sequences=input_sequences,
    output_sequences=output_sequences,
    coord_transform=normalize_coords,
    obs_transform=normalize_obs
)
```

### Manual Reshuffling

You can manually trigger reshuffling of the batch sampler:

```python
# Reshuffle before starting a new epoch
batch_sampler.reshuffle()

# Check current epoch count
epoch_count = batch_sampler.get_epoch_count()
```

### Optimized Plane ID Access

Both dataset classes implement `get_all_patch_ids()` for efficient batch sampler initialization:

```python
# This is called automatically by PatchBatchSampler
plane_ids = dataset.get_all_patch_ids()  # Returns cached numpy array
```

## Performance Tips

1. **In-Memory vs On-Disk**: Use `GWPlaneDataset` for faster training if memory allows, otherwise use `GWPlaneDatasetFromFiles`
2. **Num Workers**: Set `num_workers` in DataLoader to leverage multi-core CPUs (try 4-8 workers)
3. **Pin Memory**: Enable `pin_memory=True` when using GPU for faster data transfer
4. **Batch Size**: Larger batches are more efficient but require more memory
5. **Shuffling**: Disable shuffling during validation/testing for reproducibility

## Troubleshooting

**Issue**: Batches contain samples from different planes
- **Solution**: Make sure you're using `batch_sampler` parameter in DataLoader, not `batch_size`

**Issue**: Out of memory errors
- **Solution**: Reduce `batch_size` or switch to `GWPlaneDatasetFromFiles`

**Issue**: Slow data loading
- **Solution**: Increase `num_workers` in DataLoader or switch to `GWPlaneDataset` (in-memory)

**Issue**: NaN values in data
- **Solution**: Set appropriate `fill_nan_value` (default is -999.0) or preprocess data

## See Also

- `notebooks/2dplane_data.ipynb` - Example usage with your data
- `src/data/batch_sampler.py` - Implementation details of PatchBatchSampler
- `src/data/plane_dataset.py` - Implementation of dataset classes
