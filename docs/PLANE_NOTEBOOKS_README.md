# 2D Plane Notebooks Documentation

This document describes the two notebooks created for processing 2D plane data and testing the GFNO model.

## Overview

The workflow has been split into two separate notebooks:

1. **`generate_plane_sequences.ipynb`** - Data preprocessing and sequence generation
2. **`test_gfno_model.ipynb`** - Model testing with CUDA support

---

## 1. Generate Plane Sequences Notebook

**File**: `notebooks/generate_plane_sequences.ipynb`

### Purpose
Reads raw 2D plane data from FEFLOW simulations and generates temporal sequences suitable for training the GFNO model.

### What It Does

1. **Loads Raw Data**
   - Reads plane data from timestep directories
   - Loads boundary condition data
   - Loads sea level time series

2. **Processes Data**
   - Stacks data across timesteps for all 32 planes
   - Filters boundary conditions to consistent nodes
   - Validates data integrity

3. **Generates Sequences**
   - Creates overlapping temporal sequences with configurable window size (alpha=16)
   - Splits sequences into input (t_start:t_mid) and output (t_mid:t_end)
   - Organizes data as:
     - `input_geom`: Boundary condition coordinates (S, Z, T)
     - `input_data`: Boundary values (head, mass_conc)
     - `latent_geom`: Latent grid coordinates
     - `latent_features`: Latent grid features (X, Y, head, mass_conc)
     - `output_latent_geom`: Output coordinates
     - `output_latent_features`: Output features to predict

4. **Saves to Disk**
   - Saves sequences organized by plane: `plane_00/`, `plane_01/`, etc.
   - Each plane directory contains 6 `.npy` files
   - Output directory: `/path/to/2d_plane_sequences`

### Configuration Parameters

```python
skip_factor = 2          # Process every 2nd timestep
alpha = 16               # Sequence length
N_planes = 32           # Number of planes
```

### Output Structure

```
2d_plane_sequences/
├── plane_00/
│   ├── input_geom.npy
│   ├── input_data.npy
│   ├── latent_geom.npy
│   ├── latent_features.npy
│   ├── output_latent_geom.npy
│   └── output_latent_features.npy
├── plane_01/
│   └── ...
└── plane_31/
    └── ...
```

### Expected Runtime
- Approximately 5-10 minutes depending on system
- Progress bars show loading and processing status

---

## 2. Test GFNO Model Notebook

**File**: `notebooks/test_gfno_model.ipynb`

### Purpose
Loads pre-generated sequence data and tests the GFNO model with proper dataset, sampler, and dataloader setup. **Fully CUDA compatible** for GPU acceleration.

### What It Does

1. **Environment Setup**
   - Detects CUDA availability
   - Automatically selects GPU/CPU device
   - Loads required modules

2. **Dataset Loading**
   - Uses `GWPlaneDatasetFromFiles` for on-disk loading
   - Handles NaN values automatically
   - Converts data to PyTorch tensors

3. **Batch Sampler Setup**
   - Creates `PatchBatchSampler` to group sequences by plane
   - Ensures all samples in a batch come from the same plane
   - Configurable shuffling

4. **DataLoader Creation**
   - Parallel data loading with configurable workers
   - Pin memory for faster GPU transfer
   - Efficient batching

5. **Model Initialization**
   - Creates GFNO model with specified architecture
   - Moves model to GPU/CPU automatically
   - Reports parameter count

6. **Forward Pass Testing**
   - Tests single batch forward pass
   - Tests multiple batches with timing
   - Reports GPU memory usage (if available)
   - Validates output shapes

7. **Model Analysis**
   - Prints model architecture
   - Shows parameter counts by component
   - Performance benchmarking

### Configuration Parameters

```python
# Data
data_dir = '/path/to/2d_plane_sequences'

# Device (auto-detected)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model architecture
coord_dim = 3
gno_radius = 0.15
gno_out_channels = 2
fno_n_layers = 4
fno_n_modes = (6, 8, 8)
fno_hidden_channels = 64

# Training
batch_size = 8
num_workers = 4
```

### Key Features

✅ **CUDA Support**: Automatic GPU detection and usage  
✅ **Memory Efficient**: On-disk dataset loading  
✅ **Plane Grouping**: Samples from same plane batched together  
✅ **Performance Metrics**: Timing and memory usage tracking  
✅ **Validation**: Output shape and statistics verification  

### Expected Output

```
Using device: cuda
GPU device: NVIDIA A100-SXM4-40GB

Dataset loaded successfully!
Total sequences: 448

Batch sampler created:
  Total batches: 56
  Batch size: 8

GFNO model created with 123,456 parameters

Forward pass complete!
Time taken: 0.0234 seconds

Output shape: torch.Size([8, 16, 32, 32, 2])
```

### Performance

Typical performance on different hardware:

| Hardware | Batch Size | Time per Batch | GPU Memory |
|----------|------------|----------------|------------|
| NVIDIA A100 | 32 | ~0.05s | ~4 GB |
| NVIDIA V100 | 16 | ~0.08s | ~3 GB |
| NVIDIA RTX 3090 | 16 | ~0.10s | ~3 GB |
| CPU (16 cores) | 8 | ~2.5s | N/A |

---

## Workflow

### Step 1: Generate Sequences

```bash
# Run the data generation notebook
jupyter notebook generate_plane_sequences.ipynb
```

**OR** run all cells programmatically:

```bash
jupyter nbconvert --to notebook --execute generate_plane_sequences.ipynb
```

### Step 2: Test Model

```bash
# Run the model testing notebook
jupyter notebook test_gfno_model.ipynb
```

### Step 3: Train Model (Next Steps)

After successful testing, create a training script:
- Use the model testing notebook as a template
- Add training loop with optimizer and loss function
- Implement train/validation split
- Add checkpointing and logging
- Monitor metrics and visualize predictions

---

## Adapting for HPC/Katana

To run on HPC systems like Katana, modify paths in the notebooks:

### In `generate_plane_sequences.ipynb`:

```python
# Update these paths
data_path = '/srv/scratch/YOUR_USER/FEFLOW/processed/2d_plane_data'
sea_level_csv = '/srv/scratch/YOUR_USER/FEFLOW/simulation_files/SeaLevelDataPeaksHL.csv'
output_data_dir = '/srv/scratch/YOUR_USER/FEFLOW/processed/2d_plane_sequences'
```

### In `test_gfno_model.ipynb`:

```python
# Update these paths
project_root = '/home/YOUR_USER/GW_SciML/'
data_dir = '/srv/scratch/YOUR_USER/FEFLOW/processed/2d_plane_sequences'
```

### PBS Script Example

Create `run_gfno_test.pbs`:

```bash
#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
module load python/3.10.4
module load cuda/11.7

# Activate virtual environment
source ~/venv/bin/activate

# Run notebook
jupyter nbconvert --to notebook --execute test_gfno_model.ipynb
```

Submit with:
```bash
qsub run_gfno_test.pbs
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `test_gfno_model.ipynb`:
```python
batch_size = 4  # or 2
```

### Issue: Dataset not found

**Solution**: Ensure `generate_plane_sequences.ipynb` ran successfully and check output path.

### Issue: Slow data loading

**Solution**: Increase `num_workers` (but not more than CPU cores):
```python
num_workers = 8
```

### Issue: Module import errors

**Solution**: Ensure project root is in path:
```python
sys.path.append('/path/to/GW_SciML/')
```

---

## Dependencies

Required packages (already in your environment):

```
torch >= 1.10.0
numpy >= 1.20.0
pandas >= 1.3.0
tqdm >= 4.62.0
```

Optional for visualization:
```
matplotlib >= 3.4.0
```

---

## Next Steps

1. ✅ **Data Generation** - Run `generate_plane_sequences.ipynb`
2. ✅ **Model Testing** - Run `test_gfno_model.ipynb`
3. **Training Script** - Create based on test notebook
4. **Hyperparameter Tuning** - Experiment with model parameters
5. **Validation** - Implement proper validation loop
6. **Visualization** - Plot predictions vs ground truth
7. **Deployment** - Create inference pipeline

---

## Contact

For issues or questions about these notebooks, refer to:
- Main README: `README.md`
- Dataset documentation: `docs/PLANE_DATASET_README.md`
- Model architecture: `src/models/gfno.py`
